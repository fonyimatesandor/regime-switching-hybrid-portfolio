import inspect

import numpy as np
import pandas as pd

from typing import Literal

import sklearn.cluster as cls
from sklearn.metrics import calinski_harabasz_score
from sklearn import config_context

from src.backtest.engine import BaseStrategy

from src.models.EW import EqualWeightPortfolio


class NestedClusteredOptimizationPortfolio(BaseStrategy):
    """Nested Clustered Optimization (NCO) Strategy"""

    def __init__(
        self,
        assets: pd.DataFrame,
        initial_capital: float = 1000000.0,
        rebalance_frequency: int = 20,
        integer_sizing: bool = False,
        use_costs: bool = False,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0002,
        allocation_bounds: list[tuple] = None,
        static_constraints: list[dict] = None,
        dynamic_constraints: dict = None,
        lookback_period: int = 252,
        clustering_method: Literal["kmeans", "hierarchical", "spectral"] = "kmeans",
        inner_optimizer: BaseStrategy | dict | None = None,
        outer_optimizer: BaseStrategy | dict | None = None,
    ) -> None:

        super().__init__(
            assets,
            initial_capital,
            rebalance_frequency,
            integer_sizing,
            use_costs,
            commission_rate,
            slippage_rate,
            allocation_bounds,
            static_constraints,
            dynamic_constraints,
        )

        self.lookback_period = lookback_period

        self.clustering_method = clustering_method

        if inner_optimizer is None:
            self.inner_optimizer_class = EqualWeightPortfolio
            self.inner_params = {
                "initial_capital": initial_capital,
                "rebalance_frequency": rebalance_frequency,
                "integer_sizing": integer_sizing,
                "use_costs": use_costs,
                "commission_rate": commission_rate,
                "slippage_rate": slippage_rate,
            }
        elif isinstance(inner_optimizer, dict):
            self.inner_optimizer_class = inner_optimizer["class"]
            self.inner_params = inner_optimizer["kwargs"].copy()
        else:
            self.inner_optimizer_class = inner_optimizer.__class__
            inner_optimizer_sig = inspect.signature(inner_optimizer.__init__)
            inner_expected_args = [
                p for p in inner_optimizer_sig.parameters if p != "self"
            ]

            self.inner_params = {
                key: getattr(inner_optimizer, key)
                for key in inner_expected_args
                if hasattr(inner_optimizer, key)
                and key
                not in {
                    "assets",
                    "allocation_bounds",
                    "static_constraints",
                    "dynamic_constraints",
                }
            }

        if outer_optimizer is None:
            self.outer_optimizer_class = EqualWeightPortfolio
            self.outer_params = {
                "initial_capital": initial_capital,
                "rebalance_frequency": rebalance_frequency,
                "integer_sizing": integer_sizing,
                "use_costs": use_costs,
                "commission_rate": commission_rate,
                "slippage_rate": slippage_rate,
            }
        elif isinstance(outer_optimizer, dict):
            self.outer_optimizer_class = outer_optimizer["class"]
            self.outer_params = outer_optimizer["kwargs"].copy()
        else:
            self.outer_optimizer_class = outer_optimizer.__class__
            outer_optimizer_sig = inspect.signature(outer_optimizer.__init__)
            outer_expected_args = [
                p for p in outer_optimizer_sig.parameters if p != "self"
            ]

            self.outer_params = {
                key: getattr(outer_optimizer, key)
                for key in outer_expected_args
                if hasattr(outer_optimizer, key)
                and key
                not in {
                    "assets",
                    "allocation_bounds",
                    "static_constraints",
                    "dynamic_constraints",
                }
            }

    def _localize_allocation_bounds(self, cluster_assets: pd.DataFrame):
        cluster_tickers = list(cluster_assets.columns)
        if self.allocation_bounds is None:
            return [(0.0, 1.0) for _ in cluster_tickers]

        full_tickers = list(self.assets.columns)
        ticker_to_position = {ticker: idx for idx, ticker in enumerate(full_tickers)}

        return [
            (
                self.allocation_bounds[ticker_to_position[ticker]]
                if ticker in ticker_to_position
                else (0.0, 1.0)
            )
            for ticker in cluster_tickers
        ]

    def _localize_outer_allocation_bounds(
        self, num_clusters: int, clusters: np.ndarray
    ) -> list[tuple]:
        """Aggregate individual asset bounds into cluster bounds by summing them."""
        if self.allocation_bounds is None:
            return [(0.0, 1.0) for _ in range(num_clusters)]

        cluster_bounds = [(0.0, 0.0) for _ in range(num_clusters)]

        for asset_idx, (lower, upper) in enumerate(self.allocation_bounds):
            cluster_idx = clusters[asset_idx]
            current_lower, current_upper = cluster_bounds[cluster_idx]
            cluster_bounds[cluster_idx] = (current_lower + lower, current_upper + upper)

        return cluster_bounds

    def _localize_static_constraints(self, cluster_assets: pd.DataFrame):
        cluster_tickers = list(cluster_assets.columns)
        if not getattr(self, "static_constraints", None):
            return []

        local_constraints = []
        full_tickers = list(self.assets.columns)

        for constraint in self.static_constraints:
            metadata = constraint.get("metadata")
            if metadata is None:
                continue

            global_index = metadata.get("global_index", [])
            if not global_index:
                continue

            relevant = []
            for global_pos in global_index:
                if global_pos >= len(full_tickers):
                    continue
                ticker = full_tickers[global_pos]
                if ticker in cluster_tickers:
                    relevant.append((global_pos, cluster_tickers.index(ticker)))

            if not relevant:
                continue

            local_positions = np.array(
                [local_pos for _, local_pos in relevant], dtype=int
            )
            global_positions = np.array(
                [global_pos for global_pos, _ in relevant], dtype=int
            )
            jacobian = np.asarray(
                metadata.get("global_jac", np.zeros(len(full_tickers))), dtype=float
            )

            local_jacobian = np.zeros(len(cluster_tickers), dtype=float)
            local_jacobian[local_positions] = jacobian[global_positions]

            limit = float(metadata["limit"])
            constraint_kind = metadata.get("constraint_kind")

            if constraint_kind == "max_weight":
                local_constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda w, idx=local_positions, limit=limit: limit
                        - w[idx].sum(),
                        "jac": lambda w, j=local_jacobian: j,
                    }
                )
            elif constraint_kind == "min_weight":
                local_constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda w, idx=local_positions, limit=limit: w[idx].sum()
                        - limit,
                        "jac": lambda w, j=local_jacobian: j,
                    }
                )

        return local_constraints

    def _compute_target_weights(self, period) -> np.ndarray:

        period_start = max(0, period - self.lookback_period)
        price_window = self.prices[period_start:period]

        returns = self._get_returns(price_window)
        standardized_returns = self._standardize_returns(returns)

        clusters, _ = self._create_clusters(standardized_returns)
        
        unique_clusters = np.unique(clusters)
        num_of_clusters = len(unique_clusters)
        
        cluster_map = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
        mapped_clusters = np.array([cluster_map[c] for c in clusters])

        cluster_df = pd.DataFrame(
            columns=[f"cluster_{i}" for i in range(num_of_clusters)]
        )

        inner_data = {}

        for k in range(num_of_clusters):
            cluster_mask = mapped_clusters == k
            cluster_assets = self.assets.loc[:, cluster_mask]

            params = self.inner_params.copy()

            params["assets"] = cluster_assets
            params["allocation_bounds"] = self._localize_allocation_bounds(
                cluster_assets
            )
            params["static_constraints"] = self._localize_static_constraints(
                cluster_assets
            )
            params["dynamic_constraints"] = self.dynamic_constraints.copy()

            inner_portfolio = self.inner_optimizer_class(**params)

            cluster_weights = self.weights[:, cluster_mask]

            inner_portfolio.weights = cluster_weights

            optimized_weights = inner_portfolio._compute_target_weights(period)

            portfolio_value = np.dot(cluster_assets, optimized_weights)

            cluster_df[f"cluster_{k}"] = portfolio_value

            inner_data[k] = (cluster_mask, optimized_weights)

        cluster_level_weights = np.zeros((self.weights.shape[0], num_of_clusters))
        for k, (cluster_mask, inner_weights) in inner_data.items():
            cluster_level_weights[:, k] = self.weights[:, cluster_mask].sum(axis=1)

        params = self.outer_params.copy()
        params["assets"] = cluster_df
        params["allocation_bounds"] = self._localize_outer_allocation_bounds(
            num_of_clusters, mapped_clusters
        )
        params["static_constraints"] = None
        params["dynamic_constraints"] = None

        outer_portfolio = self.outer_optimizer_class(**params)
        outer_portfolio.weights = cluster_level_weights

        outer_weights = outer_portfolio._compute_target_weights(period)

        final_weights = np.zeros(self.num_assets)
        for k, (cluster_mask, inner_weights) in inner_data.items():
            final_weights[cluster_mask] = outer_weights[k] * inner_weights

        return final_weights

    def _create_clusters(self, returns: np.ndarray) -> tuple[np.ndarray, int]:

        max_num_of_clusters = self.num_assets // 2

        calinski_harabasz_scores = []
        labels_cache = {}

        ks = np.arange(2, max_num_of_clusters, 1)

        for k in ks:

            labels = self._cluster(returns.T, k)
            labels_cache[k] = labels

            calinski_harabasz_scores.append(calinski_harabasz_score(returns.T, labels))

        best_K = ks[np.argmax(calinski_harabasz_scores)]

        return labels_cache[best_K], best_K

    def _cluster(self, returns: np.ndarray, num_of_clusters: int) -> np.ndarray:

        with config_context(assume_finite=True, skip_parameter_validation=True):

            if self.clustering_method == "kmeans":

                clusterer = cls.KMeans(n_clusters=num_of_clusters, random_state=0)

            elif self.clustering_method == "hierarchical":

                clusterer = cls.AgglomerativeClustering(
                    n_clusters=num_of_clusters, metric="euclidean", linkage="ward"
                )

            elif self.clustering_method == "spectral":
                clusterer = cls.SpectralClustering(
                    n_clusters=num_of_clusters, random_state=0
                )

            clusterer.fit(returns)

            return clusterer.labels_

    def _get_returns(self, price_window: np.ndarray) -> np.ndarray:

        clipped_prices = np.clip(price_window, 1e-8, None)
        returns = clipped_prices[1:] / clipped_prices[:-1] - 1

        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    def _standardize_returns(self, returns: np.ndarray) -> np.ndarray:

        standardized_returns = returns - np.mean(returns, axis=0)

        stdevs = np.std(standardized_returns, axis=0)
        stdevs = np.where(stdevs == 0, 1e-8, stdevs)

        standardized_returns = standardized_returns / stdevs
        
        standardized_returns += np.random.normal(0, 1e-8, standardized_returns.shape)

        return standardized_returns

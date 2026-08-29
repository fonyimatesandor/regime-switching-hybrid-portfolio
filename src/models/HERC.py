import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from src.backtest.engine import BaseStrategy

from src.estimators.covariance import CovarianceEstimator, HistoricalCovarianceEstimator


class HierarchicalEqualRiskContributionPortfolio(BaseStrategy):
    """Hierarchical Equal Risk Contribution Portfolio (HERC) Portfolio Strategy"""

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
        covariance_estimator: CovarianceEstimator = None,
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

        if covariance_estimator is None:
            self.covariance_estimator = HistoricalCovarianceEstimator()
        else:
            self.covariance_estimator = covariance_estimator

    def _compute_target_weights(self, period) -> np.ndarray:

        period_start = max(0, period - self.lookback_period)
        price_window = self.prices[period_start:period]

        if price_window.shape[1] == 1:
            return np.array([1.0])

        cov_matrix = self.covariance_estimator.estimate(
            price_window, period, self.lookback_period
        )

        std_devs = np.sqrt(np.diag(cov_matrix))

        std_devs = np.where(std_devs <= 0, 1e-8, std_devs)

        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        np.fill_diagonal(corr_matrix, 1.0)

        dist = self._correlation_distance(corr_matrix)

        linkage = sch.linkage(dist, method="single")

        k = self._get_optimal_k(linkage)
        k = max(2, min(k, len(cov_matrix) - 1))

        cluster_labels = sch.fcluster(linkage, k, criterion="maxclust")

        herc_weights = self._allocate_herc(cov_matrix, cluster_labels)

        return herc_weights

    def _get_optimal_k(self, linkage) -> int:
        """
        Determines optimal k using the Two Difference Gap Index (TDI).
        Evaluates the acceleration in linkage distances (second derivative).
        """
        n_assets = len(linkage) + 1
        max_k = n_assets - 1

        if n_assets < 4 or max_k < 3:
            return 2

        distances = linkage[::-1, 2]

        gaps = -np.diff(distances)

        tdi = -np.diff(gaps)

        search_space = tdi[: max_k - 1]

        optimal_k = np.argmax(search_space) + 2

        return int(optimal_k)

    def _allocate_herc(self, cov_matrix, cluster_labels) -> np.ndarray:
        n_assets = len(cov_matrix)
        w = np.zeros(n_assets, dtype=float)

        unique_clusters = np.unique(cluster_labels)

        cluster_variances = {}
        cluster_inner_weights = {}
        cluster_indices = {}

        for c in unique_clusters:
            c_idx = np.where(cluster_labels == c)[0]
            cluster_indices[c] = c_idx

            cov_slice = cov_matrix[c_idx][:, c_idx]

            ivp_c = self.get_ivp(cov_slice)
            cluster_inner_weights[c] = ivp_c

            c_var = ivp_c @ cov_slice @ ivp_c
            cluster_variances[c] = c_var

        inv_cluster_vars = {c: 1.0 / var for c, var in cluster_variances.items()}
        sum_inv_cluster_vars = sum(inv_cluster_vars.values())
        cluster_weights = {
            c: inv / sum_inv_cluster_vars for c, inv in inv_cluster_vars.items()
        }

        for c in unique_clusters:
            c_idx = cluster_indices[c]
            w[c_idx] = cluster_inner_weights[c] * cluster_weights[c]

        return w

    def _correlation_distance(self, corr_matrix) -> np.ndarray:
        corr_matrix = np.clip(corr_matrix, -1, 1)
        dist = np.sqrt(0.5 * (1 - corr_matrix))
        dist = ssd.squareform(dist, force="tovector", checks=False)

        return dist

    def get_ivp(self, cov) -> np.ndarray:
        if cov.shape[0] == 1:
            return np.array([1.0])

        ivp = 1.0 / cov.diagonal()
        ivp /= ivp.sum()
        return ivp

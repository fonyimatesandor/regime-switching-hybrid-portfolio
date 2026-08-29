import numpy as np
import pandas as pd

from src.backtest.engine import BaseStrategy
from src.hmm.hmm_model import HMMModel
from src.hmm.feature_extraction import HMMFeatureExtractor


class rHMM(BaseStrategy):
    def __init__(
        self,
        assets: pd.DataFrame,
        initial_capital: float = 1000000.0,
        rebalance_frequency: int = 20,
        integer_sizing: bool = False,
        use_costs: bool = False,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0002,
        allocation_bounds: list[tuple] | None = None,
        static_constraints: list[dict] | None = None,
        dynamic_constraints: dict | None = None,
        hmm_model: HMMModel | None = None,
        hmm_feature_extractor: HMMFeatureExtractor | None = None,
        model_low_vol: BaseStrategy | None = None,
        model_high_vol: BaseStrategy | None = None,
        regime_threshold: float = 0.6,
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

        self.hmm_model = hmm_model
        self.model_low_vol = model_low_vol
        self.model_high_vol = model_high_vol
        self.regime_threshold = float(regime_threshold)

        volatility_means = self.hmm_model.means_[:, 1]

        self.low_vol_state = np.argmin(volatility_means)
        self.high_vol_state = np.argmax(volatility_means)

        self.hmm_feature_extractor = hmm_feature_extractor

        self._feature_start_period = max(
            self.hmm_feature_extractor.corr_window,
            self.hmm_feature_extractor.vol_window,
        )
        valid_feature_count = max(0, len(assets) - self._feature_start_period)
        self.features = np.empty((valid_feature_count, 4))
        self._features_filled_until = self._feature_start_period - 1

        self.states = np.zeros((len(assets), 3))

    def _compute_target_weights(self, period) -> np.ndarray:

        self._set_model_state()

        if period < max(
            self.hmm_feature_extractor.corr_window,
            self.hmm_feature_extractor.vol_window,
        ):
            return self.model_low_vol._compute_target_weights(period)

        self._fill_features_through(period)
        feature_count = period - self._feature_start_period + 1
        posterior = self.hmm_model.predict_proba(
            self.features[:feature_count]
        )[-1]

        prob_low = posterior[self.low_vol_state]
        prob_high = posterior[self.high_vol_state]

        self.states[period] = np.array([period, prob_low, prob_high])

        self.states = self.backfill_zeros(self.states)

        low_weights = self.model_low_vol._compute_target_weights(period)
        high_weights = self.model_high_vol._compute_target_weights(period)

        if prob_high >= self.regime_threshold and prob_high >= prob_low:
            return high_weights

        if prob_low >= self.regime_threshold and prob_low >= prob_high:
            return low_weights

        blend_weight = prob_low / (prob_low + prob_high + 1e-12)
        return blend_weight * low_weights + (1.0 - blend_weight) * high_weights

    def _fill_features_through(self, period: int) -> None:
        feature_start = self._features_filled_until + 1
        if feature_start > period:
            return

        features = self._compute_current_features(period, feature_start)
        matrix_start = feature_start - self._feature_start_period
        matrix_end = period - self._feature_start_period + 1
        self.features[matrix_start:matrix_end] = features
        self._features_filled_until = period

    def _compute_current_features(
        self, period: int, feature_start: int
    ) -> np.ndarray:
        """Build the newly available feature rows ending at ``period``."""
        if period < 0:
            raise ValueError("period must be non-negative")

        max_window = max(
            self.hmm_feature_extractor.vol_window,
            self.hmm_feature_extractor.corr_window,
            self.hmm_feature_extractor.draw_window,
        )
        window_start = max(0, feature_start - max_window + 1)
        price_window = self.assets.iloc[window_start : period + 1]

        if getattr(self.hmm_feature_extractor, "scaler_", None) is None:
            features = self.hmm_feature_extractor.extract_features(price_window)
        else:
            features = self.hmm_feature_extractor.transform(price_window)

        if features.shape[0] == 0:
            raise RuntimeError(
                f"Feature window ending at period {period} produced no rows."
            )

        if feature_start is None:
            return features[-1]

        rows_needed = period - feature_start + 1
        if features.shape[0] < rows_needed:
            raise RuntimeError(
                f"Feature window ending at period {period} produced "
                f"{features.shape[0]} rows; expected at least {rows_needed}."
            )

        return features[-rows_needed:]

    def _set_model_state(self) -> None:

        self.model_low_vol.num_periods, self.model_low_vol.num_assets = (
            self.prices.shape
        )
        self.model_low_vol.portfolio_value = self.portfolio_value
        self.model_low_vol.asset_shares = self.asset_shares
        self.model_low_vol.asset_values = self.asset_values
        self.model_low_vol.cash = self.cash
        self.model_low_vol.costs = self.costs

        self.model_low_vol.weights = self.weights

        self.model_high_vol.num_periods, self.model_high_vol.num_assets = (
            self.prices.shape
        )
        self.model_high_vol.portfolio_value = self.portfolio_value
        self.model_high_vol.asset_shares = self.asset_shares
        self.model_high_vol.asset_values = self.asset_values
        self.model_high_vol.cash = self.cash
        self.model_high_vol.costs = self.costs

        self.model_high_vol.weights = self.weights

    def backfill_zeros(self, arr):
        rev_arr = arr[::-1]

        valid = rev_arr != 0

        idx = np.arange(rev_arr.shape[0])[:, None]

        idx_filled = np.maximum.accumulate(np.where(valid, idx, 0), axis=0)

        cols = np.arange(rev_arr.shape[1])
        res_rev = rev_arr[idx_filled, cols]

        return res_rev[::-1]

import numpy as np
import pandas as pd

from src.backtest.engine import BaseStrategy


class EqualWeightPortfolio(BaseStrategy):
    """Implements an equal-weighted portfolio strategy."""

    def _compute_target_weights(self, period: int) -> np.ndarray:
        return np.ones(self.num_assets) / self.num_assets

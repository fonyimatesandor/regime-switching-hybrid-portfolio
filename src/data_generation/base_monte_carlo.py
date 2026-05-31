from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseMonteCarloSimulator(ABC):
    def __init__(self) -> None:
        self.is_fitted = False

    @abstractmethod
    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame) -> None:
        """Fits the statistical model to the data. Stores parameters as internal state."""

        pass

    @abstractmethod
    def simulate(
        self, starting_prices: np.ndarray, num_simulations: int, num_steps: int
    ) -> tuple:
        """
        Simulates future paths based on the fitted model.
        Returns: (simulated_asset_prices, simulated_simple_factors)
        """

        pass

    def _get_price_log_returns(self, asset_prices: np.ndarray) -> np.ndarray:
        """Translates Prices into Log Returns. Drops the first row (T -> T-1)"""

        clipped_prices = np.clip(asset_prices, 1e-8, None)
        log_returns = np.log(clipped_prices[1:] / clipped_prices[:-1])
        return np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_factor_log_returns(self, factors: np.ndarray) -> np.ndarray:
        """Translates Simple Returns into Log Returns using ln(1 + R)"""

        clipped_factors = np.clip(factors, -0.999, None)
        log_returns = np.log(1.0 + clipped_factors)
        return np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)

import numpy as np
from abc import ABC, abstractmethod


class ReturnEstimator(ABC):
    """Base class for return estimation methods."""

    def _get_returns(self, price_window: np.ndarray) -> np.ndarray:
        clipped_prices = np.clip(price_window, 1e-8, None)
        returns = clipped_prices[1:] / clipped_prices[:-1] - 1
        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    @abstractmethod
    def estimate(
        self, price_window: np.ndarray, t: int, lookback: int, cov_matrix: np.ndarray
    ) -> np.ndarray:
        pass


class HistoricalReturnEstimator(ReturnEstimator):
    """Estimates expected returns using historical average returns, adjusted for volatility."""

    def estimate(
        self, price_window: np.ndarray, t: int, lookback: int, cov_matrix: np.ndarray
    ) -> np.ndarray:
        returns = self._get_returns(price_window)

        mean_daily_returns = np.mean(returns, axis=0)
        annualized_returns = ((1 + mean_daily_returns) ** 252) - 1.0

        annualized_variance = np.diag(cov_matrix)

        adjusted_mu = np.exp(annualized_returns + 0.5 * annualized_variance) - 1

        return adjusted_mu


class FactorReturnEstimator(ReturnEstimator):
    """Estimates expected returns using a factor model."""

    def __init__(self, factors: np.ndarray) -> None:
        """factors: 2D array of shape (T, K+1) where the first column is the risk-free rate
        and the remaining K columns are factor returns
        """
        self.factors = factors

    def estimate(
        self, price_window: np.ndarray, t: int, lookback: int, cov_matrix: np.ndarray
    ) -> np.ndarray:
        returns = self._get_returns(price_window)

        factors_slice = self.factors[t - lookback + 1 : t]

        RF = factors_slice[:, 0]
        F = factors_slice[:, 1:]

        excess_returns = returns - RF[:, np.newaxis]

        res = np.linalg.lstsq(F, excess_returns, rcond=None)
        betas = res[0]

        mean_factor_returns = np.mean(F, axis=0)
        mean_rf = np.mean(RF)

        daily_log_mu = mean_rf + (betas.T @ mean_factor_returns)  # Shape: (N,)

        annualized_log_mu = daily_log_mu * 252

        annualized_variance = np.diag(cov_matrix)

        adjusted_mu = np.exp(annualized_log_mu + (0.5 * annualized_variance)) - 1.0

        return adjusted_mu


class EquilibriumReturnEstimator(ReturnEstimator):
    """Estimates expected returns using the Capital Asset Pricing Model (CAPM) approach."""

    def __init__(
        self,
        market_index_prices: np.ndarray,
        risk_free_rates: np.ndarray,
        market_caps: np.ndarray,
    ) -> None:
        """
        market_index_prices: 1D array of shape (T,)
        risk_free_rates: 1D array of shape (T,) containing daily decimal rates
        market_caps: 2D array of shape (T, N)
        """
        self.market_index = market_index_prices
        self.risk_free_rate = risk_free_rates
        self.market_caps = market_caps

    def estimate(
        self, price_window: np.ndarray, t: int, lookback: int, cov_matrix: np.ndarray
    ) -> np.ndarray:
        market_slice = self.market_index[t - lookback : t]
        rf_slice = self.risk_free_rate[t - lookback + 1 : t]

        market_simple_returns = (market_slice[1:] / market_slice[:-1]) - 1.0
        mean_mkt_daily = np.mean(market_simple_returns)
        ann_mkt_return = ((1 + mean_mkt_daily) ** 252) - 1.0

        market_log_returns = np.log(market_slice[1:] / market_slice[:-1])
        ann_mkt_variance = np.var(market_log_returns, ddof=1) * 252

        mean_rf_daily = np.mean(rf_slice)
        ann_rf_return = ((1 + mean_rf_daily) ** 252) - 1.0

        if ann_mkt_variance == 0:
            delta = 0.0
        else:
            raw_delta = (ann_mkt_return - ann_rf_return) / ann_mkt_variance
            delta = max(0.0, raw_delta)

        current_market_caps = self.market_caps[t - 1]
        w_mkt = current_market_caps / np.sum(current_market_caps)

        equilibrium_returns = ann_rf_return + delta * (cov_matrix @ w_mkt)

        return equilibrium_returns

import numpy as np
from abc import ABC, abstractmethod
from sklearn.covariance import LedoitWolf


class CovarianceEstimator(ABC):
    def _get_returns(self, price_window: np.ndarray) -> np.ndarray:
        clipped_prices = np.clip(price_window, 1e-8, None)
        returns = clipped_prices[1:] / clipped_prices[:-1] - 1
        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    @abstractmethod
    def estimate(self, price_window: np.ndarray,t: int, lookback: int) -> np.ndarray:
        pass
    
    
class HistoricalCovarianceEstimator(CovarianceEstimator):
    def estimate(self, price_window: np.ndarray,t: int, lookback: int) -> np.ndarray:
        returns = self._get_returns(price_window)
        covariance_matrix = np.cov(returns, rowvar=False) * 252
        return covariance_matrix
    
    
class LedoitWolfCovarianceEstimator(CovarianceEstimator):
    def estimate(self, price_window: np.ndarray,t: int, lookback: int) -> np.ndarray:
        returns = self._get_returns(price_window)
        lw = LedoitWolf()
        lw.fit(returns)
        covariance_matrix = lw.covariance_
        return covariance_matrix * 252
    
class FactorCovarianceEstimator(CovarianceEstimator):
    def __init__(self, factors: np.ndarray):
        ''' factors: 2D array of shape (T, K+1) where the first column is the risk-free rate and the remaining K columns are factor returns
        '''
        self.factors = factors
    
    
    def estimate(self, price_window: np.ndarray, t: int, lookback: int) -> np.ndarray:
        returns = self._get_returns(price_window)

        factors_slice = self.factors[t - lookback + 1:t]

        RF = factors_slice[:, 0]
        F = factors_slice[:, 1:]
        
        factor_cov = np.cov(F, rowvar=False) * 252
        excess_returns = returns - RF[:, np.newaxis]
        
        
        res = np.linalg.lstsq(F, excess_returns, rcond=None)
        betas = res[0]
        
        predicted_excess_returns = F @ betas
        residuals = excess_returns - predicted_excess_returns
        idiosyncratic_cov = np.cov(residuals, rowvar=False) * 252
        
        cov = betas.T @ factor_cov @ betas + idiosyncratic_cov

        return cov


class EWMACovarianceEstimator(CovarianceEstimator):
    def __init__(self, decay_factor: float = 0.94):
        self.decay_factor = decay_factor

    def estimate(self, price_window: np.ndarray, t: int, lookback: int) -> np.ndarray:
        returns = self._get_returns(price_window)
        n, m = returns.shape

        weights = self.decay_factor ** np.arange(n - 1, -1, -1)
        weights = weights / weights.sum()

        weighted_returns = returns * np.sqrt(weights)[:, np.newaxis]
        cov = weighted_returns.T @ weighted_returns
        return cov * 252

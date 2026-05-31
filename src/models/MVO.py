import numpy as np
import pandas as pd
from typing import Literal
from scipy.optimize import minimize

from .engine import BaseStrategy

from src.estimators.covariance import CovarianceEstimator, HistoricalCovarianceEstimator
from src.estimators.returns import ReturnEstimator, HistoricalReturnEstimator

class MeanVariancePortfolio(BaseStrategy):
    def __init__(self, 
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
                 return_estimator: ReturnEstimator = None,
                 objective: Literal['min_variance', 'max_return', 'max_sharpe', 'risk_aversion'] = 'max_sharpe',
                 risk_aversion_lambda: float = 1.0
                 ):
        super().__init__(assets, initial_capital, rebalance_frequency, integer_sizing, use_costs, commission_rate, slippage_rate, allocation_bounds, static_constraints, dynamic_constraints)

        self.lookback_period = lookback_period
        
        if covariance_estimator is None:
            self.covariance_estimator = HistoricalCovarianceEstimator()
        else:
            self.covariance_estimator = covariance_estimator
            
        if return_estimator is None:
            self.return_estimator = HistoricalReturnEstimator()
        else:
            self.return_estimator = return_estimator
            
        self.objective = objective
        self.risk_aversion_lambda = risk_aversion_lambda
    
    def _compute_target_weights(self, period: int) -> np.ndarray:
        
        period_start = max(0, period - self.lookback_period)
        
        price_window = self.prices[period_start:period]
        
        cov_matrix = self.covariance_estimator.estimate(price_window, period, self.lookback_period)
        
        expected_returns = self.return_estimator.estimate(price_window, period, self.lookback_period, cov_matrix)
        
        
        def objective_function(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = weights.T @ cov_matrix @ weights
            
            if self.objective == 'min_variance':
                return portfolio_variance
            elif self.objective == 'max_return':
                return -portfolio_return
            elif self.objective == 'max_sharpe':
                return -portfolio_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else np.inf
            elif self.objective == 'risk_aversion':
                return -portfolio_return + self.risk_aversion_lambda * portfolio_variance
        
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        constraints.extend(self.static_constraints)
        
        dynamic_constraints = []
        
        if 'max_turnover' in self.dynamic_constraints:
            max_turnover = self.dynamic_constraints['max_turnover']
            turnover_constraint = {
                'type': 'ineq',
                'fun': lambda w, p=period, m=max_turnover: m - np.sum(np.abs(w - self.weights[p - 1]))
            }
            dynamic_constraints.append(turnover_constraint)
            
        constraints.extend(dynamic_constraints)
        
        result = minimize(objective_function, 
                          x0=np.ones(self.num_assets) / self.num_assets,
                          bounds=self.allocation_bounds,
                          constraints=constraints)
        
        if result.success:
            return result.x
        else:
            return np.ones(self.num_assets) / self.num_assets
        
        
        
        
        
        
        
        
        

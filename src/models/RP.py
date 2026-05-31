import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .engine import BaseStrategy

from src.estimators.covariance import CovarianceEstimator, HistoricalCovarianceEstimator

class RiskParityPortfolio(BaseStrategy):
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
                 ):
        super().__init__(assets, initial_capital, rebalance_frequency, integer_sizing, use_costs, commission_rate, slippage_rate, allocation_bounds, static_constraints, dynamic_constraints)
        
        self.lookback_period = lookback_period
        
        if covariance_estimator is None:
            self.covariance_estimator = HistoricalCovarianceEstimator()
        else:
            self.covariance_estimator = covariance_estimator
            
            
    def _compute_target_weights(self, period):
        
        period_start = max(0, period - self.lookback_period)
        price_window = self.prices[period_start:period]
        
        cov_matrix = self.covariance_estimator.estimate(price_window, period, self.lookback_period)  
            
        def objective(weights):
            
            portfolio_var = weights.T @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights

            result = weights * marginal_contrib - portfolio_var  / self.num_assets
            
            return np.sum(result ** 2)
        
        def jacobian(weights):
            portfolio_var = weights.T @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            
            grad = 2 * (weights * marginal_contrib - portfolio_var / self.num_assets) * (marginal_contrib + cov_matrix @ weights - 2 * portfolio_var / self.num_assets)
            
            return grad
                
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}] + self.static_constraints
        
        result = minimize(objective, jac=jacobian, x0=np.ones(self.num_assets) / self.num_assets, bounds=self.allocation_bounds, constraints=constraints)
        
        return result.x
        
            
        
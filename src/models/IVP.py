import numpy as np
import pandas as pd

from .engine import BaseStrategy

class InverseVariancePortfolio(BaseStrategy):
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
                 lookback_period: int = 252
                 ):        
        super().__init__(assets, initial_capital, rebalance_frequency, integer_sizing, use_costs, commission_rate, slippage_rate, allocation_bounds, static_constraints, dynamic_constraints)
        self.lookback_period = lookback_period
     
    def _compute_target_weights(self, period: int) -> np.ndarray:

        period_start = max(0, period - self.lookback_period)
        
        returns = np.diff(self.prices[period_start:period], axis=0) / self.prices[period_start:period - 1]
        
        variances = np.var(returns, axis=0)
        
        inv_variances = np.where(variances > 0, 1 / variances, 0)
        
        weights = inv_variances / np.sum(inv_variances) if np.sum(inv_variances) > 0 else np.ones(self.num_assets) / self.num_assets
        
        return weights
        
        
        
        
        
    
    
    


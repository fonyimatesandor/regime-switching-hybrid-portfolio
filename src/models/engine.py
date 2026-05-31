from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class BaseStrategy(ABC):
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
                 dynamic_constraints: dict = None
                 ):
        
        self.assets = assets
        self.prices = assets.values
        self.dates = assets.index
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        
        self.integer_sizing = integer_sizing
        self.use_costs = use_costs
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        if allocation_bounds is not None:
            self.allocation_bounds = allocation_bounds
        else:
            self.allocation_bounds = [(0.0, 1.0) for _ in range(self.prices.shape[1])]
        
        if static_constraints is not None:
            self.static_constraints = static_constraints
        else: 
            self.static_constraints = []    
            
        if dynamic_constraints is not None:
            self.dynamic_constraints = dynamic_constraints
        else:
            self.dynamic_constraints = {}
        
        
    def run_backtest(self):
        
        self.num_periods, self.num_assets = self.prices.shape
        self.portfolio_value = np.zeros(self.num_periods)
        self.asset_shares = np.zeros((self.num_periods, self.num_assets))
        self.asset_values = np.zeros((self.num_periods, self.num_assets))
        self.cash = np.zeros(self.num_periods)
        self.costs = np.zeros(self.num_periods)

        self.weights = np.zeros((self.num_periods, self.num_assets))
        
        self.cash[0] = self.initial_capital
        
        self._initialize_portfolio()
        
        for period in range(1, self.num_periods):
            print(f"Processing period {period}/{self.num_periods - 1}", end='\r')
            if period % self.rebalance_frequency == 0:
                target_weights = self._compute_target_weights(period)
                self._rebalance_portfolio(period, target_weights)

            else:
                self.asset_shares[period] = self.asset_shares[period - 1]
                self.cash[period] = self.cash[period - 1]

                self.asset_values[period] = self.asset_shares[period] * self.prices[period]
                self.portfolio_value[period] = self.cash[period] + np.sum(self.asset_values[period])
                self.weights[period] = np.divide(self.asset_values[period], self.portfolio_value[period], out=np.zeros_like(self.asset_values[period]), where=self.portfolio_value[period] != 0)

    def _initialize_portfolio(self):
        
        target_weights = np.ones(self.num_assets) / self.num_assets
        self._rebalance_portfolio(0, target_weights)
        
    def _rebalance_portfolio(self, period: int, target_weights: np.ndarray):
        
        target_weights = self._validate_weights(period, target_weights)
        
        current_prices = self.prices[period]
        current_shares = self.asset_shares[period - 1] if period > 0 else np.zeros(self.num_assets)
        current_cash = self.cash[period - 1] if period > 0 else self.initial_capital
        
        current_values = current_shares * current_prices
        total_portfolio_value = current_cash + np.sum(current_values)
        
        if self.use_costs:
            fee_rate = self.commission_rate + self.slippage_rate
            
            naive_target_values = target_weights * total_portfolio_value
            trade_sign = np.sign(naive_target_values - current_values)
            
            numerator = total_portfolio_value + fee_rate * np.sum(trade_sign * current_values)
            denominator = 1 + fee_rate * np.sum(trade_sign * target_weights)
            
            net_portfolio_value = numerator / denominator
        else:
            net_portfolio_value = total_portfolio_value
            
        target_values = target_weights * net_portfolio_value
        target_shares = target_values / current_prices
        
        if self.integer_sizing:
            target_shares = np.floor(target_shares)
            
        trade_shares = target_shares - current_shares
        trade_values = trade_shares * current_prices
        
        if self.use_costs:
            trade_costs = np.abs(trade_values) * fee_rate        
            self.costs[period] = np.sum(trade_costs)
            current_cash -= np.sum(trade_costs)
            
        self.asset_shares[period] = current_shares + trade_shares
        self.asset_values[period] = self.asset_shares[period] * current_prices
        
        self.cash[period] = current_cash - np.sum(trade_values)
        self.portfolio_value[period] = self.cash[period] + np.sum(self.asset_values[period])
    
        self.weights[period] = np.divide(self.asset_values[period], self.portfolio_value[period], out=np.zeros_like(self.asset_values[period]), where=self.portfolio_value[period] != 0)
       
       
    def _validate_weights(self, period: int, weights: np.ndarray) -> np.ndarray:
    
        bound_constraints = []
        for i, (min_bound, max_bound) in enumerate(self.allocation_bounds):
            bound_constraints.append({'type': 'ineq', 'fun': lambda w, i=i, min_bound=min_bound: w[i] - min_bound})
            bound_constraints.append({'type': 'ineq', 'fun': lambda w, i=i, max_bound=max_bound: max_bound - w[i]})
    
        constraints_to_check = self.static_constraints + bound_constraints + [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        dynamic_constraints = []
        
        if 'max_turnover' in self.dynamic_constraints:
            max_turnover = self.dynamic_constraints['max_turnover']
            turnover_constraint = {
                'type': 'ineq',
                'fun': lambda w, p=period, m=max_turnover: m - np.sum(np.abs(w - self.weights[p - 1]))
            }
            dynamic_constraints.append(turnover_constraint)
            
        if 'min_diff_to_rebalance' in self.dynamic_constraints:
            min_diff = self.dynamic_constraints['min_diff_to_rebalance']
            diff = np.abs(weights - self.weights[period - 1])

            for i in range(self.num_assets):
                if diff[i] < min_diff:
                    dynamic_constraints.append({
                        'type': 'eq',
                        'fun': lambda w, i=i, p=period: w[i] - self.weights[p - 1][i]
                    })
                else:
                    dynamic_constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, i=i, p=period, d=min_diff: np.abs(w[i] - self.weights[p - 1][i]) - d
                    })
                
        constraints_to_check += dynamic_constraints
        
        truth_mask = []
        
        for constraint in constraints_to_check:
            if constraint['type'] == 'eq':
                truth_mask.append(np.isclose(constraint['fun'](weights), 0.0))
            elif constraint['type'] == 'ineq':
                truth_mask.append(constraint['fun'](weights) >= 0.0)
        
        
        if not all(truth_mask):
            
            def objective(w):
                return np.sum((w - weights) ** 2)
            
            result = minimize(objective, weights, constraints=constraints_to_check)
            
            if result.success:
                return result.x
            else:
                return weights
        else:
            return weights
        
        
    @abstractmethod
    def _compute_target_weights(self, period: int) -> np.ndarray:
        pass
        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    







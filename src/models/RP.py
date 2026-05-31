import numpy as np
import pandas as pd
import cvxpy as cp

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
    
    
        self._setup_cvxpy_problem()

    def _compute_target_weights(self, period):
        
        period_start = max(0, period - self.lookback_period)
        price_window = self.prices[period_start:period]
        
        cov_matrix = self.covariance_estimator.estimate(price_window, period, self.lookback_period)  
        
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
        min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
        if min_eig < 0:
            cov_matrix -= 10 * min_eig * np.eye(self.num_assets)
            
        self.cov_param.value = cov_matrix
        
        try:
            self.problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            pass 
        
        if self.x.value is None:
            return np.ones(self.num_assets) / self.num_assets
            
        weights = self.x.value / np.sum(self.x.value)
        
        return weights
    
    
    def _setup_cvxpy_problem(self):
        N = self.num_assets
        
        self.x = cp.Variable(N)
        self.cov_param = cp.Parameter((N, N), PSD=True) 
        
        sum_x = cp.sum(self.x)
        risk_budget = np.ones(N) / N
        
        objective = cp.Minimize(
            0.5 * cp.quad_form(self.x, self.cov_param) - risk_budget @ cp.log(self.x)
        )
        
        constraints = [
            self.x >= self.lb * sum_x,
            self.x <= self.ub * sum_x
        ]
        
        for c in self.osqp_static:
            A = c['A']
            l = c['l']
            u = c['u']
            
            for i in range(len(l)):
                if l[i] > -1e15: 
                    constraints.append(A[i] @ self.x >= l[i] * sum_x)
                if u[i] < 1e15:  
                    constraints.append(A[i] @ self.x <= u[i] * sum_x)

        self.problem = cp.Problem(objective, constraints)
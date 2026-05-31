import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from .engine import BaseStrategy

from src.estimators.covariance import CovarianceEstimator, HistoricalCovarianceEstimator

class HierarchicalRiskParityPortfolio(BaseStrategy):
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
        
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)    
    
        dist = self._correlation_distance(corr_matrix)
        
        linkage = sch.linkage(dist, method='single')
        
        sortIX = self.get_quasi_diag(linkage)
        
        hrp = self.get_rec_bipart(cov_matrix, sortIX)

        return hrp
    
    
    def _correlation_distance(self, corr_matrix):
        corr_matrix = np.clip(corr_matrix, -1, 1)
        dist = np.sqrt(0.5 * (1 - corr_matrix))
        dist = ssd.squareform(dist, force='tovector', checks=False)
        
        return dist

    def get_quasi_diag(self, linkage):
        
        return sch.leaves_list(linkage).tolist()

    def get_rec_bipart(self, cov, sortIx):
        w = np.ones(len(cov), dtype=float)
        sortIx = np.array(sortIx)
        
        def _recurse(cluster):
            if len(cluster) <= 1:
                return
            
            split_idx = len(cluster) // 2
            c0 = cluster[:split_idx]
            c1 = cluster[split_idx:]
            
            cVar0 = self.get_cluster_var(cov, c0)
            cVar1 = self.get_cluster_var(cov, c1)
            
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            
            w[c0] *= alpha
            w[c1] *= (1 - alpha)
            
            _recurse(c0)
            _recurse(c1)
            
        _recurse(sortIx)
        
        return w


    def get_cluster_var(self, cov, cItems):
                
        c_idx = np.array(cItems)
        cov_slice = cov[c_idx][:, c_idx]    
            
        w_ = self.get_ivp(cov_slice) 
        cvar = w_ @ cov_slice @ w_
        
        return cvar
        
        
    def get_ivp(self, cov):
        ivp = 1. / cov.diagonal()
        ivp /= ivp.sum()
        return ivp
from src.data_generation.base_monte_carlo import BaseMonteCarloSimulator
import numpy as np
import pandas as pd

class StaticNormalSimulator(BaseMonteCarloSimulator):
    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame):
        
        self.asset_prices = asset_prices
        self.factors = factors
        
        asset_log_returns = self._get_price_log_returns(self.asset_prices.values)
        factor_log_returns = self._get_factor_log_returns(self.factors.values[1:])
        
        self.n_assets = asset_log_returns.shape[1]
        self.n_factors = factor_log_returns.shape[1]
        
        
        joint_log_returns = np.concatenate((asset_log_returns, factor_log_returns), axis=1)
        
        self.means = np.mean(joint_log_returns, axis=0)
        self.cov_matrix = np.cov(joint_log_returns, rowvar=False)
        self.is_fitted = True
        
       
    def simulate(self, starting_prices: np.ndarray, num_simulations: int, num_steps: int) -> tuple:
        
        joint_sim_log_returns = np.random.multivariate_normal(self.means, self.cov_matrix, size=(num_simulations, num_steps - 1))

        asset_sim_log = joint_sim_log_returns[:, :, :self.n_assets]
        factor_sim_log = joint_sim_log_returns[:, :, self.n_assets:]        

        zeros = np.zeros((num_simulations, 1, self.n_assets))
        asset_sim_log_aligned = np.concatenate((zeros, asset_sim_log), axis=1)
        simulated_prices = np.exp(asset_sim_log_aligned.cumsum(axis=1)) * starting_prices

        simulated_simple_factors = np.exp(factor_sim_log) - 1.0

        return simulated_prices, simulated_simple_factors
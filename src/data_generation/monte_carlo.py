from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import t, multivariate_t
from scipy.optimize import minimize_scalar, minimize
from arch import arch_model

import sys
sys.path.append('../')

from src.utils.hansen import HansenSkewedT



class BaseMonteCarloSimulator(ABC):
    @abstractmethod
    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame):
        '''Fits the statistical model to the data. Stores parameters as internal state.'''
        pass
    
    @abstractmethod
    def simulate(self, starting_prices: np.ndarray, starting_factors: np.ndarray, num_simulations: int, num_steps: int) -> tuple:
        '''
        Simulates future paths based on the fitted model.
        Returns: (simulated_asset_prices, simulated_simple_factors)
        '''
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
    
    


class StaticNormalSimulator(BaseMonteCarloSimulator):
    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame):
        
        asset_log_returns = self._get_price_log_returns(asset_prices.values)
        factor_log_returns = self._get_factor_log_returns(factors.values[1:])
        
        self.n_assets = asset_log_returns.shape[1]
        self.n_factors = factor_log_returns.shape[1]
        
        
        joint_log_returns = np.concatenate((asset_log_returns, factor_log_returns), axis=1)
        
        self.means = np.mean(joint_log_returns, axis=0)
        self.cov_matrix = np.cov(joint_log_returns, rowvar=False)
       
    def simulate(self, starting_prices: np.ndarray, starting_factors: np.ndarray, num_simulations: int, num_steps: int) -> tuple:
        n_factors = starting_factors.shape[0]
        n_assets = self.n_assets
        
        joint_sim_log_returns = np.random.multivariate_normal(self.means, self.cov_matrix, size=(num_simulations, num_steps - 1))

        asset_sim_log = joint_sim_log_returns[:, :, :self.n_assets]
        factor_sim_log = joint_sim_log_returns[:, :, self.n_assets:]        

        zeros = np.zeros((num_simulations, 1, self.n_assets))
        asset_sim_log_aligned = np.concatenate((zeros, asset_sim_log), axis=1)
        simulated_prices = np.exp(asset_sim_log_aligned.cumsum(axis=1)) * starting_prices

        simulated_simple_factors = np.exp(factor_sim_log) - 1.0

        return simulated_prices, simulated_simple_factors
   
   
class StaticStudentTSimulator(BaseMonteCarloSimulator):
    
    def __init__(self, maxiter = 1000, tol = 1e-8):
        self.maxiter = maxiter
        self.tol = tol
    
    
    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame):
        
        asset_log_returns = self._get_price_log_returns(asset_prices.values)
        factor_log_returns = self._get_factor_log_returns(factors.values[1:])
        
        self.n_assets = asset_log_returns.shape[1]
        self.n_factors = factor_log_returns.shape[1]
        
        joint_log_returns = np.concatenate((asset_log_returns, factor_log_returns), axis=1)
        n, d = joint_log_returns.shape
        
        self.marginal_params = []
        uniform_data = np.zeros_like(joint_log_returns)
        
        for i in range(d):
            params = t.fit(joint_log_returns[:, i])
            self.marginal_params.append(params)
            uniform_data[:, i] = t.cdf(joint_log_returns[:, i], *params)
            
        
        eps = 1e-6
        uniform_clipped = np.clip(uniform_data, eps, 1.0 - eps)

        def nll(nu):
            if nu <= 2.01: return np.inf
            
            Y = t.ppf(uniform_clipped, df=nu)
            
            Sigma = np.corrcoef(Y, rowvar=False) 
            
            for _ in range(self.maxiter): 
                try:
                    inv_Sigma = np.linalg.pinv(Sigma)
                except np.linalg.LinAlgError:
                    return np.inf
                
                delta_sq = np.sum((Y @ inv_Sigma) * Y, axis=1)
                weights = (nu + d) / (nu + delta_sq)
                
                mu = np.average(Y, weights=weights, axis=0)
                Y_c = Y - mu
                S_new = (Y_c.T @ (Y_c * weights[:, np.newaxis])) / n
                
                inv_sqrt_diag = np.diag(1.0 / np.sqrt(np.diag(S_new)))
                Sigma_new = inv_sqrt_diag @ S_new @ inv_sqrt_diag
                
                if np.max(np.abs(Sigma_new - Sigma)) < self.tol:
                    Sigma = Sigma_new
                    break
                Sigma = Sigma_new

            try:
                log_pdf_joint = multivariate_t.logpdf(Y, shape=Sigma, df=nu)
                log_pdf_margins = np.sum(t.logpdf(Y, df=nu), axis=1)
                
                ll = np.sum(log_pdf_joint - log_pdf_margins)
                return -ll
            except Exception:
                return np.inf

        res = minimize_scalar(nll, bounds=(2.5, 50.0), method='bounded')
        
        self.copula_df = res.x
        
        Y_final = t.ppf(uniform_clipped, df=self.copula_df)
        Sigma_final = np.corrcoef(Y_final, rowvar=False)
        for _ in range(self.maxiter):
            inv_Sigma = np.linalg.pinv(Sigma_final)
            delta_sq = np.sum((Y_final @ inv_Sigma) * Y_final, axis=1)
            weights = (self.copula_df + d) / (self.copula_df + delta_sq)
            S_new = (Y_final.T @ (Y_final * weights[:, np.newaxis])) / n
            inv_sqrt_diag = np.diag(1.0 / np.sqrt(np.diag(S_new)))
            Sigma_new = inv_sqrt_diag @ S_new @ inv_sqrt_diag
            if np.max(np.abs(Sigma_new - Sigma_final)) < self.tol:
                Sigma_final = Sigma_new
                break
            Sigma_final = Sigma_new
            
        self.copula_corr = Sigma_final
              
    def simulate(self, starting_prices: np.ndarray, starting_factors: np.ndarray, num_simulations: int, num_steps: int) -> tuple:

        d = self.n_assets + self.n_factors
        
        safe_corr = self.copula_corr + np.eye(d) * 1e-8
        
        Z_sim = multivariate_t.rvs(shape=safe_corr, df=self.copula_df, size=(num_simulations, num_steps - 1))
        if num_simulations == 1: Z_sim = Z_sim[np.newaxis, :, :]
            
        U_sim = t.cdf(Z_sim, df=self.copula_df)
        joint_sim_log_returns = np.zeros_like(U_sim)
        
        for i in range(d):
            params = self.marginal_params[i]
            joint_sim_log_returns[:, :, i] = t.ppf(U_sim[:, :, i], *params)
            
        asset_sim_log = joint_sim_log_returns[:, :, :self.n_assets]
        factor_sim_log = joint_sim_log_returns[:, :, self.n_assets:]        

        zeros = np.zeros((num_simulations, 1, self.n_assets))
        asset_sim_log_aligned = np.concatenate((zeros, asset_sim_log), axis=1)
        simulated_prices = np.exp(asset_sim_log_aligned.cumsum(axis=1)) * starting_prices
        simulated_simple_factors = np.exp(factor_sim_log) - 1.0

        return simulated_prices, simulated_simple_factors



class StaticSkewedTSimulator(BaseMonteCarloSimulator):    
    def __init__(self, maxiter = 1000, tol = 1e-8):
        self.maxiter = maxiter
        self.tol = tol
    
    
    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame):
        
        asset_log_returns = self._get_price_log_returns(asset_prices.values)
        factor_log_returns = self._get_factor_log_returns(factors.values[1:])
        
        self.n_assets = asset_log_returns.shape[1]
        self.n_factors = factor_log_returns.shape[1]
        
        joint_log_returns = np.concatenate((asset_log_returns, factor_log_returns), axis=1)
        n, d = joint_log_returns.shape
        
        self.marginal_params = []
        uniform_data = np.zeros_like(joint_log_returns)
        
        self.ppf_interpolators = []
        
        for i in range(d):
            params = HansenSkewedT.fit(joint_log_returns[:, i])
            self.marginal_params.append(params)
            uniform_data[:, i] = HansenSkewedT.cdf(joint_log_returns[:, i], *params)
            
            u_grid = np.linspace(1e-6, 1.0 - 1e-6, 10000)
            z_grid = HansenSkewedT.ppf(u_grid, *params)
            self.ppf_interpolators.append((u_grid, z_grid))

        eps = 1e-6
        uniform_clipped = np.clip(uniform_data, eps, 1.0 - eps)

        def nll(nu):
            if nu <= 2.01: return np.inf
            
            Y = t.ppf(uniform_clipped, df=nu)
            
            Sigma = np.corrcoef(Y, rowvar=False) 
            
            for _ in range(self.maxiter): 
                try:
                    inv_Sigma = np.linalg.pinv(Sigma)
                except np.linalg.LinAlgError:
                    return np.inf
                
                delta_sq = np.sum((Y @ inv_Sigma) * Y, axis=1)
                weights = (nu + d) / (nu + delta_sq)
                
                mu = np.average(Y, weights=weights, axis=0)
                Y_c = Y - mu
                S_new = (Y_c.T @ (Y_c * weights[:, np.newaxis])) / n
                
                inv_sqrt_diag = np.diag(1.0 / np.sqrt(np.diag(S_new)))
                Sigma_new = inv_sqrt_diag @ S_new @ inv_sqrt_diag
                
                if np.max(np.abs(Sigma_new - Sigma)) < self.tol:
                    Sigma = Sigma_new
                    break
                Sigma = Sigma_new

            try:
                log_pdf_joint = multivariate_t.logpdf(Y, shape=Sigma, df=nu)
                log_pdf_margins = np.sum(t.logpdf(Y, df=nu), axis=1)
                
                ll = np.sum(log_pdf_joint - log_pdf_margins)
                return -ll
            except Exception:
                return np.inf

        res = minimize_scalar(nll, bounds=(2.5, 50.0), method='bounded')
        
        self.copula_df = res.x
        
        Y_final = t.ppf(uniform_clipped, df=self.copula_df)
        Sigma_final = np.corrcoef(Y_final, rowvar=False)
        for _ in range(self.maxiter):    
            inv_Sigma = np.linalg.pinv(Sigma_final)
            delta_sq = np.sum((Y_final @ inv_Sigma) * Y_final, axis=1)
            weights = (self.copula_df + d) / (self.copula_df + delta_sq)
            S_new = (Y_final.T @ (Y_final * weights[:, np.newaxis])) / n
            inv_sqrt_diag = np.diag(1.0 / np.sqrt(np.diag(S_new)))
            Sigma_new = inv_sqrt_diag @ S_new @ inv_sqrt_diag
            if np.max(np.abs(Sigma_new - Sigma_final)) < self.tol:
                Sigma_final = Sigma_new
                break
            Sigma_final = Sigma_new
            
        self.copula_corr = Sigma_final
        
    def simulate(self, starting_prices: np.ndarray, starting_factors: np.ndarray, num_simulations: int, num_steps: int) -> tuple:

        d = self.n_assets + self.n_factors
        
        safe_corr = self.copula_corr + np.eye(d) * 1e-8
        
        Z_sim = multivariate_t.rvs(shape=safe_corr, df=self.copula_df, size=(num_simulations, num_steps - 1))
        if num_simulations == 1: Z_sim = Z_sim[np.newaxis, :, :]
            
        U_sim = t.cdf(Z_sim, df=self.copula_df)
        joint_sim_log_returns = np.zeros_like(U_sim)
        
        for i in range(d):
            params = self.marginal_params[i]
            flat_u = U_sim[:, :, i].ravel()
            u_grid, z_grid = self.ppf_interpolators[i]
            flat_r = np.interp(flat_u, u_grid, z_grid)
            joint_sim_log_returns[:, :, i] = flat_r.reshape(num_simulations, num_steps - 1)
            
        asset_sim_log = joint_sim_log_returns[:, :, :self.n_assets]
        factor_sim_log = joint_sim_log_returns[:, :, self.n_assets:]        

        zeros = np.zeros((num_simulations, 1, self.n_assets))
        asset_sim_log_aligned = np.concatenate((zeros, asset_sim_log), axis=1)
        simulated_prices = np.exp(asset_sim_log_aligned.cumsum(axis=1)) * starting_prices
        simulated_simple_factors = np.exp(factor_sim_log) - 1.0

        return simulated_prices, simulated_simple_factors
    
    
    
class DynamicSkewedTSimulator(BaseMonteCarloSimulator):
    def __init__(self, maxiter = 1000, tol = 1e-8):
        self.maxiter = maxiter
        self.tol = tol
        
    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame):
        
        asset_log_returns = self._get_price_log_returns(asset_prices.values)
        factor_log_returns = self._get_factor_log_returns(factors.values[1:])
        
        self.n_assets = asset_log_returns.shape[1]
        self.n_factors = factor_log_returns.shape[1]
        
        joint_log_returns = np.concatenate((asset_log_returns, factor_log_returns), axis=1) * 100.0
        n, d = joint_log_returns.shape
        
        self.garch_params = []
        self.marginal_params = []
        std_residuals = np.zeros_like(joint_log_returns)
        uniform_residuals = np.zeros_like(joint_log_returns)
        
        self.ppf_interpolators = []
        
        
        for i in range(d):
            model = arch_model(joint_log_returns[:, i], mean='Zero', vol='GARCH', p=1, o=1, q=1, dist='skewt', rescale=False)
            
            res = model.fit(disp='off', options={'maxiter': self.maxiter}, show_warning=False)

            if res.optimization_result.status != 0:
                model = arch_model(joint_log_returns[:, i], mean='Zero', vol='GARCH', p=1, o=0, q=1, dist='skewt', rescale=False)
                res = model.fit(disp='off', options={'maxiter': self.maxiter}, show_warning=False)
                
                omega = res.params['omega']
                alpha = res.params['alpha[1]']
                gamma = 0.0  
                beta  = res.params['beta[1]']
            else:
                omega = res.params['omega']
                alpha = res.params['alpha[1]']
                gamma = res.params['gamma[1]']
                beta  = res.params['beta[1]']

            self.garch_params.append((omega, alpha, gamma, beta))

            lam = res.params['lambda']
            nu  = res.params['eta']
            self.marginal_params.append((nu, lam))  

            std_residuals[:, i] = res.resid / res.conditional_volatility
            uniform_residuals[:, i] = HansenSkewedT.cdf(std_residuals[:, i], nu, lam)
                    
            u_grid = np.linspace(1e-6, 1.0 - 1e-6, 10000)
            z_grid = HansenSkewedT.ppf(u_grid, nu, lam)
            self.ppf_interpolators.append((u_grid, z_grid))
            
            
                    
        eps = 1e-6
        uniform_clipped = np.clip(uniform_residuals, eps, 1.0 - eps)

        def nll(nu):
            if nu <= 2.01: return np.inf
            
            Y = t.ppf(uniform_clipped, df=nu)
            
            Sigma = np.corrcoef(Y, rowvar=False) 
            
            for _ in range(self.maxiter): 
                try:
                    inv_Sigma = np.linalg.pinv(Sigma)
                except np.linalg.LinAlgError:
                    return np.inf
                
                delta_sq = np.sum((Y @ inv_Sigma) * Y, axis=1)
                weights = (nu + d) / (nu + delta_sq)
                
                mu = np.average(Y, weights=weights, axis=0)
                Y_c = Y - mu
                S_new = (Y_c.T @ (Y_c * weights[:, np.newaxis])) / n
                
                inv_sqrt_diag = np.diag(1.0 / np.sqrt(np.diag(S_new)))
                Sigma_new = inv_sqrt_diag @ S_new @ inv_sqrt_diag
                
                if np.max(np.abs(Sigma_new - Sigma)) < self.tol:
                    Sigma = Sigma_new
                    break
                Sigma = Sigma_new

            try:
                log_pdf_joint = multivariate_t.logpdf(Y, shape=Sigma, df=nu)
                log_pdf_margins = np.sum(t.logpdf(Y, df=nu), axis=1)
                
                ll = np.sum(log_pdf_joint - log_pdf_margins)
                return -ll
            except Exception:
                return np.inf

        res_copula = minimize_scalar(nll, bounds=(2.5, 50.0), method='bounded')
        
        self.copula_df = res_copula.x
        
        Y_final = t.ppf(uniform_clipped, df=self.copula_df)
        Sigma_final = np.corrcoef(Y_final, rowvar=False)
        for _ in range(self.maxiter):    
            inv_Sigma = np.linalg.pinv(Sigma_final)
            delta_sq = np.sum((Y_final @ inv_Sigma) * Y_final, axis=1)
            weights = (self.copula_df + d) / (self.copula_df + delta_sq)
            S_new = (Y_final.T @ (Y_final * weights[:, np.newaxis])) / n
            inv_sqrt_diag = np.diag(1.0 / np.sqrt(np.diag(S_new)))
            Sigma_new = inv_sqrt_diag @ S_new @ inv_sqrt_diag
            if np.max(np.abs(Sigma_new - Sigma_final)) < self.tol:
                Sigma_final = Sigma_new
                break
            Sigma_final = Sigma_new
            
        self.Qbar = Sigma_final
        
        self.dcc_a, self.dcc_b = self._estimate_dcc_params(Y_final, self.copula_df, self.Qbar)
        
        
    def _estimate_dcc_params(self, Y, nu, Qbar):
        T, d = Y.shape
        
        def dcc_loglik(params):
            a, b = params
            if a <= 0 or b <= 0 or a + b >= 0.9999:
                return 1e10
                
            ll = 0.0
            Q = Qbar.copy()
            
            for t_step in range(1, T):
                Y_prev = Y[t_step-1, :]
                Q = (1 - a - b) * Qbar + a * np.outer(Y_prev, Y_prev) + b * Q
                
                inv_sqrt_d = np.diag(1.0 / np.sqrt(np.diag(Q)))
                R = inv_sqrt_d @ Q @ inv_sqrt_d
                
                det_sign, logdet = np.linalg.slogdet(R)
                if det_sign <= 0:
                    return 1e10
                    
                try:
                    invR = np.linalg.inv(R)
                except np.linalg.LinAlgError:
                    return 1e10
                
                y_t = Y[t_step, :]
                quad = y_t @ invR @ y_t
                
                ll += 0.5 * logdet + ((nu + d) / 2.0) * np.log(1.0 + quad / nu)
                
            return ll 

        res = minimize(dcc_loglik, x0=[0.02, 0.95], bounds=[(1e-6, 0.5), (0.5, 0.99)], method='L-BFGS-B')
        
        return res.x[0], res.x[1]
    
    def simulate(self, starting_prices: np.ndarray, starting_factors: np.ndarray, num_simulations: int, num_steps: int) -> tuple:
        
        joint_sim_log_returns = np.zeros((num_simulations, num_steps - 1, self.n_assets + self.n_factors))
        
        d = self.n_assets + self.n_factors
        
        uncond_sigma2 = np.zeros(d)
        for i in range(d):
            omega, alpha, gamma, beta = self.garch_params[i]
            persistence = min(alpha + beta + (gamma / 2.0), 0.999) 
            uncond_sigma2[i] = omega / (1.0 - persistence)

        for sim in range(num_simulations):
            
            Q_t = self.Qbar.copy()
            Y_prev = np.zeros(d)
            ret_prev = np.zeros(d)
            sigma2_t = uncond_sigma2.copy()
            
            current_prices = starting_prices.copy()
            
            for step in range(num_steps - 1):  
                
                for i in range(d):
                    omega, alpha, gamma, beta = self.garch_params[i]
                    asym_shock = (ret_prev[i] ** 2) if ret_prev[i] < 0 else 0.0
                    sigma2_t[i] = omega + alpha * (ret_prev[i] ** 2) + gamma * asym_shock + beta * sigma2_t[i]
                
                Q_t = (1 - self.dcc_a - self.dcc_b) * self.Qbar + self.dcc_a * np.outer(Y_prev, Y_prev) + self.dcc_b * Q_t
                
                d_inv = 1.0 / np.sqrt(np.diag(Q_t))
                R_t = np.diag(d_inv) @ Q_t @ np.diag(d_inv)
                
                R_t = (R_t + R_t.T) / 2.0  
                np.fill_diagonal(R_t, 1.0)
                
                L_t = np.linalg.cholesky(R_t)
                Z = np.random.standard_normal(d)
                X = L_t @ Z
                W = np.random.chisquare(self.copula_df)
                
                Y_t = X * np.sqrt(self.copula_df / W)
                
                U_t = np.clip(t.cdf(Y_t, df=self.copula_df), 1e-6, 1.0 - 1e-6)
                z_t = np.zeros(d)
                
                for i in range(d):
                    nu_i, lam_i = self.marginal_params[i]
                    u_grid, z_grid = self.ppf_interpolators[i]
                    val = np.interp(U_t[i], u_grid, z_grid)
                    z_t[i] = val
                    
                raw_returns_scaled = z_t * np.sqrt(sigma2_t)
                raw_returns_true = raw_returns_scaled / 100.0

                joint_sim_log_returns[sim, step, :] = raw_returns_true

                Y_prev = Y_t
                ret_prev = raw_returns_scaled

        asset_sim_log = joint_sim_log_returns[:, :, :self.n_assets]
        factor_sim_log = joint_sim_log_returns[:, :, self.n_assets:]        

        zeros_assets = np.zeros((num_simulations, 1, self.n_assets))
        zeros_factors = np.zeros((num_simulations, 1, self.n_factors))
        asset_sim_log_aligned = np.concatenate((zeros_assets, asset_sim_log), axis=1)
        factor_sim_log_aligned = np.concatenate((zeros_factors, factor_sim_log), axis=1)
        simulated_prices = np.exp(asset_sim_log_aligned.cumsum(axis=1)) * starting_prices
        simulated_simple_factors = np.exp(factor_sim_log_aligned) - 1.0
           


        return simulated_prices, simulated_simple_factors
    

class HistoricalBootstrapSimulator(BaseMonteCarloSimulator):
    def __init__(self, block_size: int = 5):
        self.block_size = block_size
        
    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame):
        asset_log_returns = self._get_price_log_returns(asset_prices.values)
        factor_raw_returns = factors.values[1:] 
        
        self.n_assets = asset_log_returns.shape[1]
        self.n_factors = factor_raw_returns.shape[1]
        
        self.historical_joint_returns = np.concatenate((asset_log_returns, factor_raw_returns), axis=1)
        self.T_history = self.historical_joint_returns.shape[0]

    def simulate(self, starting_prices: np.ndarray, starting_factors: np.ndarray, num_simulations: int, num_steps: int) -> tuple:
        
        d = self.n_assets + self.n_factors
        joint_sim_returns = np.zeros((num_simulations, num_steps, d))
        
        num_blocks = int(np.ceil(num_steps / self.block_size))
        
        max_start_idx = self.T_history - self.block_size
        
        for sim in range(num_simulations):
            
            random_start_indices = np.random.randint(0, max_start_idx + 1, size=num_blocks)
            
            path_returns = []
            for start_idx in random_start_indices:
                block = self.historical_joint_returns[start_idx : start_idx + self.block_size, :]
                path_returns.append(block)
                
            full_path = np.concatenate(path_returns, axis=0)[:num_steps, :]
            joint_sim_returns[sim, :, :] = full_path

        asset_sim_log = joint_sim_returns[:, :, :self.n_assets]
        factor_sim_raw = joint_sim_returns[:, :, self.n_assets:]
        
        simulated_prices = np.exp(asset_sim_log.cumsum(axis=1)) * starting_prices
        
        simulated_simple_factors = factor_sim_raw

        return simulated_prices, simulated_simple_factors
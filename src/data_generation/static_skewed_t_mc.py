from src.data_generation.base_monte_carlo import BaseMonteCarloSimulator
import numpy as np
import pandas as pd
from scipy.stats import t, multivariate_t
from scipy.optimize import minimize_scalar
from src.utils.hansen import HansenSkewedT


class StaticSkewedTSimulator(BaseMonteCarloSimulator):
    """Monte Carlo simulator that assumes joint skewed t distribution of asset and factor log returns, with static parameters estimated from historical data."""

    def __init__(self, maxiter=1000, tol=1e-8) -> None:
        super().__init__()
        self.maxiter = maxiter
        self.tol = tol

    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame) -> None:

        self.asset_prices = asset_prices
        self.factors = factors

        asset_log_returns = self._get_price_log_returns(self.asset_prices.values)
        factor_log_returns = self._get_factor_log_returns(self.factors.values[1:])

        self.n_assets = asset_log_returns.shape[1]
        self.n_factors = factor_log_returns.shape[1]

        joint_log_returns = np.concatenate(
            (asset_log_returns, factor_log_returns), axis=1
        )
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
            if nu <= 2.01:
                return np.inf

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

        res = minimize_scalar(nll, bounds=(2.5, 50.0), method="bounded")

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

        self.is_fitted = True

    def simulate(
        self, starting_prices: np.ndarray, num_simulations: int, num_steps: int
    ) -> tuple:

        d = self.n_assets + self.n_factors

        safe_corr = self.copula_corr + np.eye(d) * 1e-8

        Z_sim = multivariate_t.rvs(
            shape=safe_corr, df=self.copula_df, size=(num_simulations, num_steps - 1)
        )
        if num_simulations == 1:
            Z_sim = Z_sim[np.newaxis, :, :]

        U_sim = t.cdf(Z_sim, df=self.copula_df)
        joint_sim_log_returns = np.zeros_like(U_sim)

        for i in range(d):
            flat_u = U_sim[:, :, i].ravel()
            u_grid, z_grid = self.ppf_interpolators[i]
            flat_r = np.interp(flat_u, u_grid, z_grid)
            joint_sim_log_returns[:, :, i] = flat_r.reshape(
                num_simulations, num_steps - 1
            )

        asset_sim_log = joint_sim_log_returns[:, :, : self.n_assets]
        factor_sim_log = joint_sim_log_returns[:, :, self.n_assets :]

        zeros = np.zeros((num_simulations, 1, self.n_assets))
        asset_sim_log_aligned = np.concatenate((zeros, asset_sim_log), axis=1)
        simulated_prices = (
            np.exp(asset_sim_log_aligned.cumsum(axis=1)) * starting_prices
        )
        simulated_simple_factors = np.exp(factor_sim_log) - 1.0

        return simulated_prices, simulated_simple_factors

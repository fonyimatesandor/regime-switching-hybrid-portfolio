from src.data_generation.base_monte_carlo import BaseMonteCarloSimulator
import numpy as np
import pandas as pd
from scipy.stats import t, multivariate_t
from scipy.optimize import minimize_scalar


from src.data_generation.base_monte_carlo import BaseMonteCarloSimulator
import numpy as np
import pandas as pd
from scipy.stats import t, multivariate_t
from scipy.optimize import minimize_scalar

MAX_LOG_RET = 0.5


class StaticStudentTSimulator(BaseMonteCarloSimulator):
    """Monte Carlo simulator that assumes joint Student's t distribution of asset and factor log returns,
    with static parameters estimated from historical data.
    """

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

        for i in range(d):
            params = t.fit(joint_log_returns[:, i])
            self.marginal_params.append(params)
            uniform_data[:, i] = t.cdf(joint_log_returns[:, i], *params)

        eps = 1e-6
        uniform_clipped = np.clip(uniform_data, eps, 1.0 - eps)

        def copula_neg_log_likelihood(nu):
            """Negative log‑likelihood of the t‑copula for a given df nu."""
            if nu <= 2.01:
                return np.inf

            Y = t.ppf(uniform_clipped, df=nu)

            Sigma = np.corrcoef(Y, rowvar=False)
            Sigma += np.eye(d) * 1e-6

            for _ in range(self.maxiter):
                try:
                    inv_Sigma = np.linalg.pinv(Sigma)
                except np.linalg.LinAlgError:
                    return np.inf

                delta_sq = np.sum((Y @ inv_Sigma) * Y, axis=1)
                w = (nu + d) / (nu + delta_sq)

                S = (Y.T @ (Y * w[:, np.newaxis])) / n

                inv_sqrt_diag = np.diag(1.0 / np.sqrt(np.diag(S)))
                Sigma_new = inv_sqrt_diag @ S @ inv_sqrt_diag

                if np.max(np.abs(Sigma_new - Sigma)) < self.tol:
                    Sigma = Sigma_new
                    break
                Sigma = Sigma_new

            try:
                log_joint = multivariate_t.logpdf(Y, loc=None, shape=Sigma, df=nu)
                log_margins = np.sum(t.logpdf(Y, df=nu), axis=1)
                ll = np.sum(log_joint - log_margins)
                return -ll
            except Exception:
                return np.inf

        res = minimize_scalar(
            copula_neg_log_likelihood, bounds=(2.5, 50.0), method="bounded"
        )
        self.copula_df = res.x

        Y_final = t.ppf(uniform_clipped, df=self.copula_df)
        Sigma_final = np.corrcoef(Y_final, rowvar=False)
        Sigma_final += np.eye(d) * 1e-6

        for _ in range(self.maxiter):
            inv_Sigma = np.linalg.pinv(Sigma_final)
            delta_sq = np.sum((Y_final @ inv_Sigma) * Y_final, axis=1)
            w = (self.copula_df + d) / (self.copula_df + delta_sq)
            S = (Y_final.T @ (Y_final * w[:, np.newaxis])) / n
            inv_sqrt_diag = np.diag(1.0 / np.sqrt(np.diag(S)))
            Sigma_new = inv_sqrt_diag @ S @ inv_sqrt_diag
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
        U_sim = np.clip(U_sim, 1e-6, 1.0 - 1e-6)

        joint_sim_log_returns = np.zeros_like(U_sim)
        for i in range(d):
            params = self.marginal_params[i]
            joint_sim_log_returns[:, :, i] = t.ppf(U_sim[:, :, i], *params)

        asset_sim_log = joint_sim_log_returns[:, :, : self.n_assets]
        factor_sim_log = joint_sim_log_returns[:, :, self.n_assets :]

        zeros_assets = np.zeros((num_simulations, 1, self.n_assets))
        zeros_factors = np.zeros((num_simulations, 1, self.n_factors))

        asset_sim_log_aligned = np.concatenate((zeros_assets, asset_sim_log), axis=1)
        factor_sim_log_aligned = np.concatenate((zeros_factors, factor_sim_log), axis=1)

        asset_sim_log_aligned = np.clip(
            asset_sim_log_aligned, -MAX_LOG_RET, MAX_LOG_RET
        )
        factor_sim_log_aligned = np.clip(
            factor_sim_log_aligned, -MAX_LOG_RET, MAX_LOG_RET
        )

        simulated_prices = (
            np.exp(asset_sim_log_aligned.cumsum(axis=1)) * starting_prices
        )
        simulated_simple_factors = np.exp(factor_sim_log_aligned) - 1.0

        return simulated_prices, simulated_simple_factors

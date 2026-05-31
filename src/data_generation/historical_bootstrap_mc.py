from src.data_generation.base_monte_carlo import BaseMonteCarloSimulator
import numpy as np
import pandas as pd


class HistoricalBootstrapSimulator(BaseMonteCarloSimulator):
    """Monte Carlo simulator that generates paths by bootstrapping historical returns in blocks."""

    def __init__(self, block_size: int = 5) -> None:
        super().__init__()
        self.block_size = block_size

    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame) -> None:
        self.asset_prices = asset_prices
        self.factors = factors

        asset_log_returns = self._get_price_log_returns(self.asset_prices.values)
        factor_raw_returns = self.factors.values[1:]

        self.n_assets = asset_log_returns.shape[1]
        self.n_factors = factor_raw_returns.shape[1]

        self.historical_joint_returns = np.concatenate(
            (asset_log_returns, factor_raw_returns), axis=1
        )
        self.T_history = self.historical_joint_returns.shape[0]
        self.is_fitted = True

    def simulate(
        self, starting_prices: np.ndarray, num_simulations: int, num_steps: int
    ) -> tuple:

        d = self.n_assets + self.n_factors
        joint_sim_returns = np.zeros((num_simulations, num_steps, d))

        num_blocks = int(np.ceil(num_steps / self.block_size))

        max_start_idx = self.T_history - self.block_size

        for sim in range(num_simulations):

            random_start_indices = np.random.randint(
                0, max_start_idx + 1, size=num_blocks
            )

            path_returns = []
            for start_idx in random_start_indices:
                block = self.historical_joint_returns[
                    start_idx : start_idx + self.block_size, :
                ]
                path_returns.append(block)

            full_path = np.concatenate(path_returns, axis=0)[:num_steps, :]
            joint_sim_returns[sim, :, :] = full_path

        asset_sim_log = joint_sim_returns[:, :, : self.n_assets]
        factor_sim_raw = joint_sim_returns[:, :, self.n_assets :]

        simulated_prices = np.exp(asset_sim_log.cumsum(axis=1)) * starting_prices

        simulated_simple_factors = factor_sim_raw

        return simulated_prices, simulated_simple_factors

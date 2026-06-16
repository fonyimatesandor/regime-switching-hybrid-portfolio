import pandas as pd
import numpy as np

from src.data_generation.base_monte_carlo import BaseMonteCarloSimulator
from src.hmm.hmm_model import HMMModel
from src.hmm.feature_extraction import HMMFeatureExtractor


class HMMMonteCarloSimulator(BaseMonteCarloSimulator):
    def __init__(
        self,
        n_components=2,
        n_iter=1000,
        tol=1e-4,
        df_bounds=(2.01, 50.0),
        vol_window=21,
        corr_window=21,
        draw_window=252,
        simulator_type=type[BaseMonteCarloSimulator],
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.df_bounds = df_bounds

        self.hmm_model = HMMModel(
            n_components=n_components,
            n_iter=n_iter,
            tol=tol,
            df_bounds=df_bounds,
        )

        self.vol_window = vol_window
        self.corr_window = corr_window
        self.draw_window = draw_window

        self.simulator_type = simulator_type

    def fit(self, asset_prices: pd.DataFrame, factors: pd.DataFrame) -> None:

        self.n_assets = asset_prices.shape[1]
        self.n_factors = factors.shape[1]

        self.extractor = HMMFeatureExtractor(
            vol_window=self.vol_window,
            corr_window=self.corr_window,
            draw_window=self.draw_window,
        )
        features = self.extractor.extract_features(asset_prices)
        self.hmm_model.fit(features)

        self.states = self.hmm_model.predict(features)

        self.simulators = {}

        aligned_asset_prices = asset_prices.iloc[-len(self.states) :]
        aligned_factors = factors.iloc[-len(self.states) :]

        for state in range(self.n_components):
            state_mask = self.states == state
            state_asset_prices = aligned_asset_prices[state_mask]
            state_factors = aligned_factors[state_mask]

            simulator = self.simulator_type()
            simulator.fit(state_asset_prices, state_factors)
            self.simulators[state] = simulator

        self.is_fitted = True

    def simulate(
        self, starting_prices: pd.DataFrame, num_simulations: int, num_steps: int
    ) -> tuple:

        joint_sim_log_returns = np.zeros(
            (num_simulations, num_steps - 1, self.n_assets + self.n_factors)
        )

        simulated_states = np.zeros((num_simulations, num_steps), dtype=int)

        for sim in range(num_simulations):

            starting_state = self.states[0]

            X, state_sequence = self.hmm_model.sample(
                num_steps, currstate=starting_state
            )

            return_states = state_sequence[1:]

            for state in range(self.n_components):
                state_mask = return_states == state
                num_state_steps = state_mask.sum()

                if num_state_steps > 0:
                    sim_prices, sim_factors = self.simulators[state].simulate(
                        np.asarray(starting_prices).flatten(),
                        num_simulations=1,
                        num_steps=num_state_steps + 1,
                    )

                    sim_price_log_returns = np.log(
                        sim_prices[:, 1:] / sim_prices[:, :-1]
                    )

                    sim_factor_log_returns = np.log(1.0 + sim_factors[:, 1:])

                    joint_sim_log_returns[sim, state_mask] = np.concatenate(
                        (sim_price_log_returns[0], sim_factor_log_returns[0]), axis=1
                    )

            simulated_states[sim] = state_sequence

        simulated_prices = (
            np.exp(joint_sim_log_returns[:, :, : self.n_assets].cumsum(axis=1))
            * np.asarray(starting_prices).flatten()
        )
        simulated_simple_factors = (
            np.exp(joint_sim_log_returns[:, :, self.n_assets :]) - 1.0
        )
        self.simulated_states = simulated_states
        return simulated_prices, simulated_simple_factors

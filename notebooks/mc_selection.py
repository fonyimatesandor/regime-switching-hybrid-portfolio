import numpy as np
import pandas as pd
import pickle as pkl

from scipy.linalg import sqrtm
import scipy.stats as stats

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().resolve().parent))

from src.data_generation.base_monte_carlo import BaseMonteCarloSimulator

simulator_names = [
    "static_normal",
    "static_student_t",
    "static_skewed_t",
    "dynamic_skewed_t",
    "historical_bootstrap",
    "static_normal_hmm",
    "static_student_t_hmm",
    "static_skewed_t_hmm",
    "dynamic_skewed_t_hmm",
]

asset_prices_learning = pd.read_csv(
    "../data/raw/stock_data_learning.csv", index_col=0, parse_dates=True
)


def calculate_distances(
    mc_simulator: BaseMonteCarloSimulator, N_simulations
) -> tuple[float, float]:

    asset_distances_outer = []
    cov_distances = []

    for _ in range(N_simulations):

        sample_stock_data, sample_factor_data = mc_simulator.simulate(
            asset_prices_learning.values[0, :],
            N_simulations,
            len(asset_prices_learning),
        )

        sample_stock_data = sample_stock_data.reshape(
            -1, len(asset_prices_learning.columns)
        )

        asset_distances = np.zeros(len(asset_prices_learning.columns))

        historical_returns = (
            asset_prices_learning.values[1:] / asset_prices_learning.values[:-1] - 1
        )

        model_returns = sample_stock_data[1:] / sample_stock_data[:-1] - 1

        for i in range(len(asset_distances)):

            asset_distances[i] = stats.wasserstein_distance(
                historical_returns[:, i], model_returns[:, i]
            )

        historical_cov_matrix = np.cov(historical_returns, rowvar=False)

        model_cov_matrix = np.cov(model_returns, rowvar=False)

        cov_distance = frechet_distance(model_cov_matrix, historical_cov_matrix)

        asset_distances_outer.append(np.mean(asset_distances))
        cov_distances.append(cov_distance)

    return np.mean(asset_distances), np.mean(cov_distances)


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:

    sqrt_a = sqrtm(a).real
    middle_sqrt = sqrtm(sqrt_a @ b @ sqrt_a).real
    distance_squared = np.trace(a) + np.trace(b) - 2 * np.trace(middle_sqrt)

    return float(np.sqrt(max(distance_squared, 0.0)))


def distance_summary() -> pd.DataFrame:

    distance_dict = {}

    for name in simulator_names:

        with open(f"../data/synthetic/{name}_simulator.pkl", "rb") as f:
            simulator = pkl.load(f)

        tup = calculate_distances(simulator, 1)

        distance_dict[name] = tup

        distance_df = pd.DataFrame(distance_dict)

        distance_df = distance_df.transpose()

        distance_df.columns = ["Average asset distance", "Covariance distance"]

    return distance_df

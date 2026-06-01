import numpy
import pandas as pd
import pickle as pkl

from src.utils.config_loader import load_config

from src.data_generation.static_normal_mc import StaticNormalSimulator
from src.data_generation.static_student_t_mc import StaticStudentTSimulator
from src.data_generation.static_skewed_t_mc import StaticSkewedTSimulator
from src.data_generation.dynamic_skewed_t_mc import DynamicSkewedTSimulator
from src.data_generation.historical_bootstrap_mc import HistoricalBootstrapSimulator

config = load_config("./config.yaml")

simulators = {
    "static_normal": StaticNormalSimulator(),
    "static_student_t": StaticStudentTSimulator(),
    "static_skewed_t": StaticSkewedTSimulator(),
    "dynamic_skewed_t": DynamicSkewedTSimulator(),
    "historical_bootstrap": HistoricalBootstrapSimulator(),
}

asset_prices = pd.read_csv(
    "./data/raw/stock_data_05_25.csv", index_col=0, parse_dates=True
)

factors = pd.read_csv("./data/raw/FF_factor_data.csv", index_col=0, parse_dates=True)

for name, simulator in simulators.items():
    print(f"Fitting {name} simulator...")
    simulator.fit(asset_prices=asset_prices, factors=factors)

    print(f"Simulating paths with {name} simulator...")
    simulated_prices, simulated_factors = simulator.simulate(
        starting_prices=asset_prices.iloc[-1].values,
        num_simulations=config["monte_carlo"]["num_simulations"],
        num_steps=len(asset_prices),
    )

    output = {
        "simulated_prices": simulated_prices,
        "simulated_factors": simulated_factors,
    }

    with open(f"./data/synthetic/{name}_simulated_data.pkl", "wb") as f:
        pkl.dump((simulator, output), f)

    with open(f"./data/synthetic/{name}_simulator.pkl", "wb") as f:
        pkl.dump(simulator, f)

    print(f"{name} simulator data saved.\n")

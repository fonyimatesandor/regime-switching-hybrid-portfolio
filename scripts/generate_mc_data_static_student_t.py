import pickle as pkl

from scripts.mc_backtest_setup import (
    config,
    simulators,
    asset_prices_learning,
    asset_prices_comparison,
    factors_learning,
    factors_comparison,
)

simulator_name = "static_student_t"

simulator = simulators[simulator_name]

print(f"Fitting {simulator_name} simulator...")
simulator.fit(asset_prices=asset_prices_learning, factors=factors_learning)

print(f"Simulating paths with {simulator_name} simulator...")
simulated_prices, simulated_factors = simulator.simulate(
    starting_prices=asset_prices_comparison.iloc[-1].values,
    num_simulations=config["monte_carlo"]["num_precomputed_paths"],
    num_steps=len(asset_prices_comparison),
)

output = {
    "simulated_prices": simulated_prices,
    "simulated_factors": simulated_factors,
}

with open(f"./data/synthetic/{simulator_name}_simulated_data.pkl", "wb") as f:
    pkl.dump((simulator, output), f)

with open(f"./data/synthetic/{simulator_name}_simulator.pkl", "wb") as f:
    pkl.dump(simulator, f)

print(f"{simulator_name} simulator data saved.\n")

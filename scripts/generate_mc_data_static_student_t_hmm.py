import pickle as pkl

from scripts.mc_backtest_setup import config, simulators, asset_prices, factors

simulator_name = "static_student_t_hmm"

simulator = simulators[simulator_name]

print(f"Fitting {simulator_name} simulator...")
simulator.fit(asset_prices=asset_prices, factors=factors)

print(f"Simulating paths with {simulator_name} simulator...")
simulated_prices, simulated_factors = simulator.simulate(
    starting_prices=asset_prices.iloc[-1],
    num_simulations=config["monte_carlo"]["num_precomputed_paths"],
    num_steps=len(asset_prices),
)

output = {
    "simulated_prices": simulated_prices,
    "simulated_factors": simulated_factors,
    "simulated_states": simulator.simulated_states,
}

with open(f"./data/synthetic/{simulator_name}_simulated_data.pkl", "wb") as f:
    pkl.dump((simulator, output), f)

with open(f"./data/synthetic/{simulator_name}_simulator.pkl", "wb") as f:
    pkl.dump(simulator, f)

print(f"{simulator_name} simulator data saved.\n")

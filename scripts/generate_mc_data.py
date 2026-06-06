import pickle as pkl

from scripts.mc_backtest_setup import config, simulators, asset_prices, factors

for name, simulator in simulators.items():
    print(f"Fitting {name} simulator...")
    simulator.fit(asset_prices=asset_prices, factors=factors)

    print(f"Simulating paths with {name} simulator...")
    simulated_prices, simulated_factors = simulator.simulate(
        starting_prices=asset_prices.iloc[-1].values,
        num_simulations=config["monte_carlo"]["num_precomputed_paths"],
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

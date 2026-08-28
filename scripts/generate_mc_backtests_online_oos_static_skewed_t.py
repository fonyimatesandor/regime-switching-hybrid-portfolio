import pickle as pkl
import json

from scripts.mc_backtest_setup import (
    config,
    simulators,
    asset_prices_testing,
    choosen_classical_models,
)

simulator_name = "static_skewed_t_oos"

with open(f"data/synthetic/{simulator_name}_simulated_data.pkl", "rb") as f:
    simulator, simulated_paths = pkl.load(f)

simulated_prices = simulated_paths["simulated_prices"]
simulated_factors = simulated_paths["simulated_factors"]

for model in choosen_classical_models:

    print(
        f"Running MC backtest for model {model['name']} on simulator {simulator_name}..."
    )

    testing_model = model["class"](assets=asset_prices_testing, **model["kwargs"])
    testing_model.run_backtest()

    mc_results = testing_model.run_MC_backtest(
        simulator=simulator,
        num_simulations=config["monte_carlo"]["num_simulations"],
        workers=-1,
        batch_size=config["monte_carlo"]["chunk_size"],
        return_metrics=True,
    )

    with open(
        f"data/oos_mc_metrics/{model['name']}_{simulator_name}_MC_backtest_results.pkl",
        "wb",
    ) as f:
        pkl.dump(mc_results, f)


json.dump(
    {
        "models_tested": [model["name"] for model in choosen_classical_models],
        "simulators_tested": [simulator_name],
        "config": config,
    },
    open(f"data/oos_mc_metrics/mc_backtest_summary_{simulator_name}.json", "w"),
    indent=4,
)

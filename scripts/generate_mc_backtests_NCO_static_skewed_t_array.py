import sys
import pickle as pkl
import json

from scripts.mc_backtest_setup import (
    config,
    simulators,
    asset_prices_comparison,
    NCO_models,
    choosen_classical_models,
)

if len(sys.argv) < 2:
    print("Usage: python -m scripts.generate_mc_backtests_NCO_static_skewed_t_array <array_index>")
    sys.exit(1)

array_index = int(sys.argv[1]) - 1
if array_index < 0 or array_index >= len(choosen_classical_models):
    print(f"Error: Index {array_index + 1} is out of bounds (1-{len(choosen_classical_models)})")
    sys.exit(1)

model_1_name = choosen_classical_models[array_index]["name"]

filtered_models = [
    model for model in NCO_models 
    if model["kwargs"]["inner_optimizer"]["name"] == model_1_name
]

print(f"Array job for model_1: {model_1_name} | Running {len(filtered_models)} models")

simulator_name = "static_skewed_t"

with open(f"data/synthetic/{simulator_name}_simulated_data.pkl", "rb") as f:
    simulator, simulated_paths = pkl.load(f)

simulated_prices = simulated_paths["simulated_prices"]
simulated_factors = simulated_paths["simulated_factors"]

for model in filtered_models:

    print(
        f"Running MC backtest for model {model['name']} on simulator {simulator_name}..."
    )

    testing_model = model["class"](assets=asset_prices_comparison, **model["kwargs"])
    testing_model.run_backtest()

    mc_results = testing_model.run_MC_backtest(
        precomputed_prices=simulated_prices,
        num_simulations=config["monte_carlo"]["num_simulations"],
        workers=-1,
        batch_size=config["monte_carlo"]["chunk_size"],
    )

    with open(
        f"data/processed/{model['name']}_{simulator_name}_MC_backtest_results.pkl",
        "wb",
    ) as f:
        pkl.dump(mc_results, f)

# We should output the summary json per-array job so they don't overwrite each other simultaneously
json.dump(
    {
        "models_tested": [model["name"] for model in filtered_models],
        "simulators_tested": [simulator_name],
        "config": config,
    },
    open(f"data/processed/mc_backtest_summary_NCO_{simulator_name}_{model_1_name}.json", "w"),
    indent=4,
)

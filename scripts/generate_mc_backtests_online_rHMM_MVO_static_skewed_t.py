import pickle as pkl
import json

from scripts.mc_backtest_setup import (
    config,
    simulators,
    asset_prices_comparison,
    rHMM_MVO_models,
    rHMM_MVO_comparison_models,
)

with open("data/hmm_model/hmm_model.pkl", "rb") as f:
    hmm_model = pkl.load(f)

with open("data/hmm_model/hmm_feature_extractor.pkl", "rb") as f:
    feature_extractor = pkl.load(f)

simulator_name = "static_skewed_t"

with open(f"data/synthetic/{simulator_name}_simulated_data.pkl", "rb") as f:
    simulator, simulated_paths = pkl.load(f)

simulated_prices = simulated_paths["simulated_prices"]
simulated_factors = simulated_paths["simulated_factors"]

for model in rHMM_MVO_models:
    print(
        f"Running MC backtest for model {model['name']} on simulator {simulator_name}..."
    )
    testing_model = model["class"](
        assets=asset_prices_comparison,
        **{
            **model["kwargs"],
            "hmm_model": hmm_model,
            "hmm_feature_extractor": feature_extractor,
        },
    )
    testing_model.run_backtest()
    mc_results = testing_model.run_MC_backtest(
        simulator=simulator,
        num_simulations=config["monte_carlo"]["num_simulations"],
        workers=-1,
        batch_size=config["monte_carlo"]["chunk_size"],
        return_metrics=True,
    )
    with open(
        f"data/rhmm_mvo_mc_metrics/{model['name']}_{simulator_name}_MC_backtest_results.pkl",
        "wb",
    ) as f:
        pkl.dump(mc_results, f)


for model in rHMM_MVO_comparison_models:

    print(
        f"Running MC backtest for model {model['name']} on simulator {simulator_name}..."
    )

    testing_model = model["class"](assets=asset_prices_comparison, **model["kwargs"])
    testing_model.run_backtest()

    mc_results = testing_model.run_MC_backtest(
        simulator=simulator,
        num_simulations=config["monte_carlo"]["num_simulations"],
        workers=-1,
        batch_size=config["monte_carlo"]["chunk_size"],
        return_metrics=True,
    )

    with open(
        f"data/rhmm_mvo_mc_metrics/{model['name']}_{simulator_name}_MC_backtest_results.pkl",
        "wb",
    ) as f:
        pkl.dump(mc_results, f)


json.dump(
    {
        "models_tested": [model["name"] for model in rHMM_MVO_models]
        + [model["name"] for model in rHMM_MVO_comparison_models],
        "simulators_tested": [simulator_name],
        "config": config,
    },
    open(
        f"data/rhmm_mvo_mc_metrics/mc_backtest_summary_rHMM_MVO_{simulator_name}.json",
        "w",
    ),
    indent=4,
)

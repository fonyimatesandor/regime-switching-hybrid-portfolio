import pandas as pd
import pickle as pkl
import json

from src.utils.config_loader import load_config, build_bounds_and_constraints

from src.models.EW import EqualWeightPortfolio
from src.models.IVP import InverseVariancePortfolio
from src.models.MVO import MeanVariancePortfolio
from src.models.HRP import HierarchicalRiskParityPortfolio
from src.models.RP import RiskParityPortfolio

from src.estimators.covariance import (
    HistoricalCovarianceEstimator,
    LedoitWolfCovarianceEstimator,
    FactorCovarianceEstimator,
    EWMACovarianceEstimator,
)

from src.estimators.returns import (
    HistoricalReturnEstimator,
    FactorReturnEstimator,
    EquilibriumReturnEstimator,
)

simulated_data = {
    "static_normal": "static_normal_simulated_data.pkl",
    "static_student_t": "static_student_t_simulated_data.pkl",
    "static_skewed_t": "static_skewed_t_simulated_data.pkl",
    "dynamic_skewed_t": "dynamic_skewed_t_simulated_data.pkl",
    "historical_bootstrap": "historical_bootstrap_simulated_data.pkl",
}


config = load_config("config.yaml")
bounds, constraints, dynamic_constraints = build_bounds_and_constraints(config)

asset_data = pd.read_csv("data/raw/stock_data_05_25.csv", index_col=0, parse_dates=True)
factor_data = pd.read_csv("data/raw/FF_factor_data.csv", index_col=0, parse_dates=True)

index_data = pd.read_csv("data/raw/index_data_05_25.csv", index_col=0, parse_dates=True)
market_cap_data = pd.read_csv(
    "data/raw/market_cap_05_25.csv", index_col=0, parse_dates=True
)

covariance_estimators = {
    "historical": HistoricalCovarianceEstimator(),
    "ledoit_wolf": LedoitWolfCovarianceEstimator(),
    "FF_3": FactorCovarianceEstimator(
        factor_data[["RF", "Mkt-RF", "SMB", "HML"]].values / 100.0
    ),
    "Carhart_4": FactorCovarianceEstimator(
        factor_data[["RF", "Mkt-RF", "SMB", "HML", "Mom"]].values / 100.0
    ),
    "FF_5": FactorCovarianceEstimator(
        factor_data[["RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values / 100.0
    ),
    "ewma": EWMACovarianceEstimator(),
}

return_estimators = {
    "historical": HistoricalReturnEstimator(),
    "FF_3": FactorReturnEstimator(
        factor_data[["RF", "Mkt-RF", "SMB", "HML"]].values / 100.0
    ),
    "Carhart_4": FactorReturnEstimator(
        factor_data[["RF", "Mkt-RF", "SMB", "HML", "Mom"]].values / 100.0
    ),
    "FF_5": FactorReturnEstimator(
        factor_data[["RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values / 100.0
    ),
    "equilibrium": EquilibriumReturnEstimator(
        market_index_prices=index_data["SPY"],
        risk_free_rates=factor_data["RF"] / 100.0,
        market_caps=market_cap_data.values,
    ),
}


models_to_test = []

EWModel_kwargs = {
    "initial_capital": config["backtest"]["initial_capital"],
    "rebalance_frequency": config["backtest"]["rebalance_period"],
    "integer_sizing": True,
    "use_costs": True,
    "commission_rate": config["frictions"]["commission_rate"],
    "slippage_rate": config["frictions"]["slippage"],
    "allocation_bounds": bounds,
    "static_constraints": constraints,
    "dynamic_constraints": dynamic_constraints,
}

models_to_test.append(
    {"name": "EW", "class": EqualWeightPortfolio, "kwargs": EWModel_kwargs}
)

IVPModel_kwargs = {
    "initial_capital": config["backtest"]["initial_capital"],
    "rebalance_frequency": config["backtest"]["rebalance_period"],
    "integer_sizing": True,
    "use_costs": True,
    "commission_rate": config["frictions"]["commission_rate"],
    "slippage_rate": config["frictions"]["slippage"],
    "allocation_bounds": bounds,
    "static_constraints": constraints,
    "dynamic_constraints": dynamic_constraints,
    "lookback_period": config["backtest"]["lookback_window"],
}

models_to_test.append(
    {"name": "IVP", "class": InverseVariancePortfolio, "kwargs": IVPModel_kwargs}
)

MVO_objectives = ["min_variance", "max_return", "max_sharpe"]
MVO_objectives = ["max_sharpe"]

for cov_name, cov_estimator in covariance_estimators.items():
    for ret_name, ret_estimator in return_estimators.items():
        for obj in MVO_objectives:
            model_name = f"MVO_{obj}_{cov_name}_cov_{ret_name}_ret"
            MVOModel_kwargs = {
                "initial_capital": config["backtest"]["initial_capital"],
                "rebalance_frequency": config["backtest"]["rebalance_period"],
                "integer_sizing": True,
                "use_costs": True,
                "commission_rate": config["frictions"]["commission_rate"],
                "slippage_rate": config["frictions"]["slippage"],
                "allocation_bounds": bounds,
                "static_constraints": constraints,
                "dynamic_constraints": dynamic_constraints,
                "lookback_period": config["backtest"]["lookback_window"],
                "covariance_estimator": cov_estimator,
                "return_estimator": ret_estimator,
                "objective": obj,
            }
            models_to_test.append(
                {
                    "name": model_name,
                    "class": MeanVariancePortfolio,
                    "kwargs": MVOModel_kwargs,
                }
            )

for cov_name, cov_estimator in covariance_estimators.items():
    model_name = f"HRP_{cov_name}_cov"
    HRPModel_kwargs = {
        "initial_capital": config["backtest"]["initial_capital"],
        "rebalance_frequency": config["backtest"]["rebalance_period"],
        "integer_sizing": True,
        "use_costs": True,
        "commission_rate": config["frictions"]["commission_rate"],
        "slippage_rate": config["frictions"]["slippage"],
        "allocation_bounds": bounds,
        "static_constraints": constraints,
        "dynamic_constraints": dynamic_constraints,
        "lookback_period": config["backtest"]["lookback_window"],
        "covariance_estimator": cov_estimator,
    }
    models_to_test.append(
        {
            "name": model_name,
            "class": HierarchicalRiskParityPortfolio,
            "kwargs": HRPModel_kwargs,
        }
    )

for cov_name, cov_estimator in covariance_estimators.items():
    model_name = f"RP_{cov_name}_cov"
    RPModel_kwargs = {
        "initial_capital": config["backtest"]["initial_capital"],
        "rebalance_frequency": config["backtest"]["rebalance_period"],
        "integer_sizing": True,
        "use_costs": True,
        "commission_rate": config["frictions"]["commission_rate"],
        "slippage_rate": config["frictions"]["slippage"],
        "allocation_bounds": bounds,
        "static_constraints": constraints,
        "dynamic_constraints": dynamic_constraints,
        "lookback_period": config["backtest"]["lookback_window"],
        "covariance_estimator": cov_estimator,
    }
    models_to_test.append(
        {"name": model_name, "class": RiskParityPortfolio, "kwargs": RPModel_kwargs}
    )

for simulator_name, output_file in simulated_data.items():

    with open(f"data/synthetic/{output_file}", "rb") as f:
        simulator, simulated_paths = pkl.load(f)

    simulated_prices = simulated_paths["simulated_prices"]
    simulated_factors = simulated_paths["simulated_factors"]

    for model in models_to_test:

        print(
            f"Running MC backtest for model {model['name']} on simulator {simulator_name}..."
        )

        testing_model = model["class"](assets=asset_data, **model["kwargs"])
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


json.dump(
    {
        "models_tested": [model["name"] for model in models_to_test],
        "simulators_tested": list(simulated_data.items()),
        "config": config,
    },
    open("data/processed/mc_backtest_summary.json", "w"),
    indent=4,
)

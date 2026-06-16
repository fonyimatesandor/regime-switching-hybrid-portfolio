import pandas as pd

from src.data_generation.static_normal_mc import StaticNormalSimulator
from src.data_generation.static_student_t_mc import StaticStudentTSimulator
from src.data_generation.static_skewed_t_mc import StaticSkewedTSimulator
from src.data_generation.dynamic_skewed_t_mc import DynamicSkewedTSimulator
from src.data_generation.historical_bootstrap_mc import HistoricalBootstrapSimulator
from src.data_generation.hmm_mc import HMMMonteCarloSimulator

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

from src.models.EW import EqualWeightPortfolio
from src.models.IVP import InverseVariancePortfolio
from src.models.MVO import MeanVariancePortfolio
from src.models.HRP import HierarchicalRiskParityPortfolio
from src.models.RP import RiskParityPortfolio


from src.utils.config_loader import load_config, build_bounds_and_constraints

config = load_config("./config.yaml")
bounds, constraints, dynamic_constraints = build_bounds_and_constraints(config)


simulators = {
    "static_normal": StaticNormalSimulator(),
    "static_student_t": StaticStudentTSimulator(),
    "static_skewed_t": StaticSkewedTSimulator(),
    "dynamic_skewed_t": DynamicSkewedTSimulator(),
    "historical_bootstrap": HistoricalBootstrapSimulator(),
}

for name, sim in list(simulators.items()):
    hmm_sim = HMMMonteCarloSimulator(
        n_components=config["hmm_model"]["n_components"],
        n_iter=config["hmm_model"]["n_iter"],
        df_bounds=config["hmm_model"]["df_bounds"],
        vol_window=config["hmm_model"]["vol_window"],
        corr_window=config["hmm_model"]["corr_window"],
        draw_window=config["hmm_model"]["draw_window"],
        simulator_type=type(sim),
    )
    simulators[name + "_hmm"] = hmm_sim


asset_prices = pd.read_csv(
    "./data/raw/stock_data_05_25.csv", index_col=0, parse_dates=True
)

factors = pd.read_csv("./data/raw/FF_factor_data.csv", index_col=0, parse_dates=True)

index_data = pd.read_csv(
    "./data/raw/index_data_05_25.csv", index_col=0, parse_dates=True
)
market_cap_data = pd.read_csv(
    "./data/raw/market_cap_05_25.csv", index_col=0, parse_dates=True
)

covariance_estimators = {
    "historical": HistoricalCovarianceEstimator(),
    "ledoit_wolf": LedoitWolfCovarianceEstimator(),
    "FF_3": FactorCovarianceEstimator(
        factors[["RF", "Mkt-RF", "SMB", "HML"]].values / 100.0
    ),
    "Carhart_4": FactorCovarianceEstimator(
        factors[["RF", "Mkt-RF", "SMB", "HML", "Mom"]].values / 100.0
    ),
    "FF_5": FactorCovarianceEstimator(
        factors[["RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values / 100.0
    ),
    "ewma": EWMACovarianceEstimator(),
}

return_estimators = {
    "historical": HistoricalReturnEstimator(),
    "FF_3": FactorReturnEstimator(
        factors[["RF", "Mkt-RF", "SMB", "HML"]].values / 100.0
    ),
    "Carhart_4": FactorReturnEstimator(
        factors[["RF", "Mkt-RF", "SMB", "HML", "Mom"]].values / 100.0
    ),
    "FF_5": FactorReturnEstimator(
        factors[["RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values / 100.0
    ),
    "equilibrium": EquilibriumReturnEstimator(
        market_index_prices=index_data["SPY"],
        risk_free_rates=factors["RF"] / 100.0,
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

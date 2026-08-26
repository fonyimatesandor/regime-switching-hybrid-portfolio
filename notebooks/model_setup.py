import pandas as pd
import pickle as pkl

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().resolve().parent))

from src.data_generation.static_normal_mc import StaticNormalSimulator
from src.data_generation.static_student_t_mc import StaticStudentTSimulator
from src.data_generation.static_skewed_t_mc import StaticSkewedTSimulator
from src.data_generation.dynamic_skewed_t_mc import DynamicSkewedTSimulator
from src.data_generation.historical_bootstrap_mc import HistoricalBootstrapSimulator

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
from src.models.rHMM_MVO_HRP import rHMM_MVO_HRP


from src.utils.config_loader import load_config, build_bounds_and_constraints

simulators = {
    "static_normal": StaticNormalSimulator(),
    "static_student_t": StaticStudentTSimulator(),
    "static_skewed_t": StaticSkewedTSimulator(),
    "dynamic_skewed_t": DynamicSkewedTSimulator(),
    "historical_bootstrap": HistoricalBootstrapSimulator(),
}

config = load_config("../config.yaml")
bounds, constraints, dynamic_constraints = build_bounds_and_constraints(config)

asset_prices_learning = pd.read_csv(
    "../data/raw/stock_data_learning.csv", index_col=0, parse_dates=True
)

factors_learning = pd.read_csv(
    "../data/raw/FF_factor_data_learning.csv", index_col=0, parse_dates=True
)

index_data_learning = pd.read_csv(
    "../data/raw/index_data_learning.csv", index_col=0, parse_dates=True
)
market_cap_data_learning = pd.read_csv(
    "../data/raw/market_cap_learning.csv", index_col=0, parse_dates=True
)

asset_prices_comparison = pd.read_csv(
    "../data/raw/stock_data_comparison.csv", index_col=0, parse_dates=True
)

factors_comparison = pd.read_csv(
    "../data/raw/FF_factor_data_comparison.csv", index_col=0, parse_dates=True
)

index_data_comparison = pd.read_csv(
    "../data/raw/index_data_comparison.csv", index_col=0, parse_dates=True
)
market_cap_data_comparison = pd.read_csv(
    "../data/raw/market_cap_comparison.csv", index_col=0, parse_dates=True
)

with open("../data/hmm_model/hmm_model.pkl", "rb") as f:
    hmm_model = pkl.load(f)

with open("../data/hmm_model/hmm_feature_extractor.pkl", "rb") as f:
    feature_extractor = pkl.load(f)

covariance_estimators = {
    "historical": HistoricalCovarianceEstimator(),
    "ledoit_wolf": LedoitWolfCovarianceEstimator(),
    "FF_3": FactorCovarianceEstimator(
        factors_comparison[["RF", "Mkt-RF", "SMB", "HML"]].values / 100.0
    ),
    "Carhart_4": FactorCovarianceEstimator(
        factors_comparison[["RF", "Mkt-RF", "SMB", "HML", "Mom"]].values / 100.0
    ),
    "FF_5": FactorCovarianceEstimator(
        factors_comparison[["RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values / 100.0
    ),
    "ewma": EWMACovarianceEstimator(),
}

return_estimators = {
    "historical": HistoricalReturnEstimator(),
    "FF_3": FactorReturnEstimator(
        factors_comparison[["RF", "Mkt-RF", "SMB", "HML"]].values / 100.0
    ),
    "Carhart_4": FactorReturnEstimator(
        factors_comparison[["RF", "Mkt-RF", "SMB", "HML", "Mom"]].values / 100.0
    ),
    "FF_5": FactorReturnEstimator(
        factors_comparison[["RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values / 100.0
    ),
    "equilibrium": EquilibriumReturnEstimator(
        market_index_prices=index_data_comparison["SPY"],
        risk_free_rates=factors_comparison["RF"] / 100.0,
        market_caps=market_cap_data_comparison.values,
    ),
}


models_to_test_frictionless = []

EWModel_kwargs = {
    "initial_capital": config["backtest"]["initial_capital"],
    "rebalance_frequency": config["backtest"]["rebalance_period"],
    "integer_sizing": False,
    "use_costs": False,
    "commission_rate": config["frictions"]["commission_rate"],
    "slippage_rate": config["frictions"]["slippage"],
    "allocation_bounds": bounds,
    "static_constraints": constraints,
}

models_to_test_frictionless.append(
    {"name": "EW", "class": EqualWeightPortfolio, "kwargs": EWModel_kwargs}
)

IVPModel_kwargs = {
    "initial_capital": config["backtest"]["initial_capital"],
    "rebalance_frequency": config["backtest"]["rebalance_period"],
    "integer_sizing": False,
    "use_costs": False,
    "commission_rate": config["frictions"]["commission_rate"],
    "slippage_rate": config["frictions"]["slippage"],
    "allocation_bounds": bounds,
    "static_constraints": constraints,
    "lookback_period": config["backtest"]["lookback_window"],
}

models_to_test_frictionless.append(
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
                "integer_sizing": False,
                "use_costs": False,
                "commission_rate": config["frictions"]["commission_rate"],
                "slippage_rate": config["frictions"]["slippage"],
                "allocation_bounds": bounds,
                "static_constraints": constraints,
                "lookback_period": config["backtest"]["lookback_window"],
                "covariance_estimator": cov_estimator,
                "return_estimator": ret_estimator,
                "objective": obj,
            }
            models_to_test_frictionless.append(
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
        "integer_sizing": False,
        "use_costs": False,
        "commission_rate": config["frictions"]["commission_rate"],
        "slippage_rate": config["frictions"]["slippage"],
        "allocation_bounds": bounds,
        "static_constraints": constraints,
        "lookback_period": config["backtest"]["lookback_window"],
        "covariance_estimator": cov_estimator,
    }
    models_to_test_frictionless.append(
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
        "integer_sizing": False,
        "use_costs": False,
        "commission_rate": config["frictions"]["commission_rate"],
        "slippage_rate": config["frictions"]["slippage"],
        "allocation_bounds": bounds,
        "static_constraints": constraints,
        "lookback_period": config["backtest"]["lookback_window"],
        "covariance_estimator": cov_estimator,
    }
    models_to_test_frictionless.append(
        {"name": model_name, "class": RiskParityPortfolio, "kwargs": RPModel_kwargs}
    )

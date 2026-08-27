import pandas as pd
from typing import Literal

from src.models.MVO import MeanVariancePortfolio
from src.models.HRP import HierarchicalRiskParityPortfolio

from src.models.rHMM import rHMM

from src.estimators.covariance import CovarianceEstimator
from src.estimators.returns import ReturnEstimator

from src.hmm.hmm_model import HMMModel
from src.hmm.feature_extraction import HMMFeatureExtractor


class rHMM_MVO_HRP(rHMM):
    def __init__(
        self,
        assets: pd.DataFrame,
        initial_capital: float = 1000000.0,
        rebalance_frequency: int = 20,
        integer_sizing: bool = False,
        use_costs: bool = False,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0002,
        allocation_bounds: list[tuple] | None = None,
        static_constraints: list[dict] | None = None,
        dynamic_constraints: dict | None = None,
        MVO_lookback_period: int = 252,
        MVO_covariance_estimator: CovarianceEstimator = None,
        MVO_return_estimator: ReturnEstimator = None,
        MVO_objective: Literal[
            "min_variance", "max_return", "max_sharpe", "risk_aversion"
        ] = "max_sharpe",
        MVO_risk_aversion_lambda: float = 1.0,
        HRP_lookback_period: int = 252,
        HRP_covariance_estimator: CovarianceEstimator = None,
        hmm_model: HMMModel | None = None,
        hmm_feature_extractor: HMMFeatureExtractor | None = None,
        regime_threshold: float = 0.6,
    ) -> None:

        MVO_model = MeanVariancePortfolio(
            assets=assets,
            initial_capital=initial_capital,
            rebalance_frequency=rebalance_frequency,
            integer_sizing=integer_sizing,
            use_costs=use_costs,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            allocation_bounds=allocation_bounds,
            static_constraints=static_constraints,
            dynamic_constraints=dynamic_constraints,
            lookback_period=MVO_lookback_period,
            covariance_estimator=MVO_covariance_estimator,
            return_estimator=MVO_return_estimator,
            objective=MVO_objective,
            risk_aversion_lambda=MVO_risk_aversion_lambda,
        )

        HRP_model = HierarchicalRiskParityPortfolio(
            assets=assets,
            initial_capital=initial_capital,
            rebalance_frequency=rebalance_frequency,
            integer_sizing=integer_sizing,
            use_costs=use_costs,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            allocation_bounds=allocation_bounds,
            static_constraints=static_constraints,
            dynamic_constraints=dynamic_constraints,
            lookback_period=HRP_lookback_period,
            covariance_estimator=HRP_covariance_estimator,
        )

        super().__init__(
            assets,
            initial_capital,
            rebalance_frequency,
            integer_sizing,
            use_costs,
            commission_rate,
            slippage_rate,
            allocation_bounds,
            static_constraints,
            dynamic_constraints,
            hmm_model,
            hmm_feature_extractor,
            HRP_model,
            MVO_model,
            regime_threshold,
        )

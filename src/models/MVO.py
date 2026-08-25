import numpy as np
import pandas as pd
from typing import Literal
import osqp
import scipy.sparse as sp

from src.backtest.engine import BaseStrategy

from src.estimators.covariance import CovarianceEstimator, HistoricalCovarianceEstimator
from src.estimators.returns import ReturnEstimator, HistoricalReturnEstimator


class MeanVariancePortfolio(BaseStrategy):
    """Implements a mean-variance optimization strategy using OSQP for quadratic programming. Supports various objectives and constraints."""

    def __init__(
        self,
        assets: pd.DataFrame,
        initial_capital: float = 1000000.0,
        rebalance_frequency: int = 20,
        integer_sizing: bool = False,
        use_costs: bool = False,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0002,
        allocation_bounds: list[tuple] = None,
        static_constraints: list[dict] = None,
        dynamic_constraints: dict = None,
        lookback_period: int = 252,
        covariance_estimator: CovarianceEstimator = None,
        return_estimator: ReturnEstimator = None,
        objective: Literal[
            "min_variance", "max_return", "max_sharpe", "risk_aversion"
        ] = "max_sharpe",
        risk_aversion_lambda: float = 1.0,
    ) -> None:
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
        )

        self.lookback_period = lookback_period

        if covariance_estimator is None:
            self.covariance_estimator = HistoricalCovarianceEstimator()
        else:
            self.covariance_estimator = covariance_estimator

        if return_estimator is None:
            self.return_estimator = HistoricalReturnEstimator()
        else:
            self.return_estimator = return_estimator

        self.objective = objective
        self.risk_aversion_lambda = risk_aversion_lambda

        if self.objective != "max_sharpe":
            In = sp.eye(self.num_assets, format="csc")
            ones = sp.csc_matrix(np.ones((1, self.num_assets)))

            A_rows = [In, ones]
            l_rows = [self._lb, np.array([1.0])]
            u_rows = [self._ub, np.array([1.0])]

            for c in self._osqp_static:
                A_rows.append(sp.csc_matrix(c["A"]))
                l_rows.append(
                    np.where(np.isfinite(c["l"]), c["l"], -osqp.constant("OSQP_INFTY"))
                )
                u_rows.append(
                    np.where(np.isfinite(c["u"]), c["u"], osqp.constant("OSQP_INFTY"))
                )

            self._A = sp.vstack(A_rows, format="csc")
            self._l = np.concatenate(l_rows)
            self._u = np.concatenate(u_rows)

            self._mvo_solver = None

        else:
            A_sum = sp.hstack(
                [sp.csc_matrix(np.ones((1, self.num_assets))), sp.csc_matrix([[-1.0]])],
                format="csc",
            )

            In = sp.eye(self.num_assets, format="csc")
            A_lb = sp.hstack([In, sp.csc_matrix(-self._lb).T], format="csc")
            A_ub = sp.hstack([In, sp.csc_matrix(-self._ub).T], format="csc")

            A_kappa = sp.hstack(
                [sp.csc_matrix(np.zeros((1, self.num_assets))), sp.csc_matrix([[1.0]])],
                format="csc",
            )

            A_rows = [A_sum, A_lb, A_ub, A_kappa]
            l_rows = [
                np.array([0.0]),
                np.zeros(self.num_assets),
                np.full(self.num_assets, -osqp.constant("OSQP_INFTY")),
                np.array([1e-8]),
            ]
            u_rows = [
                np.array([0.0]),
                np.full(self.num_assets, osqp.constant("OSQP_INFTY")),
                np.zeros(self.num_assets),
                np.array([osqp.constant("OSQP_INFTY")]),
            ]

            for c in self._osqp_static:
                Ac = sp.csc_matrix(c["A"])
                lc = c["l"]
                uc = c["u"]

                valid_l = np.isfinite(lc) & (lc != -osqp.constant("OSQP_INFTY"))
                if np.any(valid_l):
                    A_rows.append(
                        sp.hstack(
                            [Ac[valid_l, :], sp.csc_matrix(-lc[valid_l]).T],
                            format="csc",
                        )
                    )
                    l_rows.append(np.zeros(np.sum(valid_l)))
                    u_rows.append(np.full(np.sum(valid_l), osqp.constant("OSQP_INFTY")))

                valid_u = np.isfinite(uc) & (uc != osqp.constant("OSQP_INFTY"))
                if np.any(valid_u):
                    A_rows.append(
                        sp.hstack(
                            [Ac[valid_u, :], sp.csc_matrix(-uc[valid_u]).T],
                            format="csc",
                        )
                    )
                    l_rows.append(
                        np.full(np.sum(valid_u), -osqp.constant("OSQP_INFTY"))
                    )
                    u_rows.append(np.zeros(np.sum(valid_u)))

            self._A_static = sp.vstack(A_rows, format="csc")

            self._l = np.concatenate([np.array([1.0]), np.concatenate(l_rows)])
            self._u = np.concatenate([np.array([1.0]), np.concatenate(u_rows)])
            self._mvo_solver = None

    def _compute_target_weights(self, period: int) -> np.ndarray:

        period_start = max(0, period - self.lookback_period)
        price_window = self.prices[period_start:period]

        if price_window.shape[1] == 1:
            return np.array([1.0])

        cov_matrix = self.covariance_estimator.estimate(
            price_window, period, self.lookback_period
        )
        expected_returns = self.return_estimator.estimate(
            price_window, period, self.lookback_period, cov_matrix
        )

        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

        min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
        if min_eig < 1e-7:
            cov_matrix += (1e-7 - min_eig) * np.eye(self.num_assets)

        if self.objective == "max_sharpe":
            if np.max(expected_returns) <= 0:
                return np.ones(self.num_assets) / self.num_assets

            P_y = sp.triu(sp.csc_matrix(2.0 * cov_matrix), format="csc")
            P_upper = sp.bmat(
                [[P_y, None], [None, sp.csc_matrix([[1e-8]])]], format="csc"
            )
            q = np.zeros(self.num_assets + 1)

            A_ret = sp.hstack(
                [sp.csc_matrix(expected_returns), sp.csc_matrix([[0.0]])], format="csc"
            )
            A = sp.vstack([A_ret, self._A_static], format="csc")

            if self._mvo_solver is None:
                self._mvo_solver = osqp.OSQP()
                self._mvo_solver.setup(
                    P_upper, q, A, self._l, self._u, verbose=False, polish=True
                )
            else:
                try:
                    self._mvo_solver.update(Px=P_upper.data, Ax=A.data)
                except ValueError:
                    self._mvo_solver = osqp.OSQP()
                    self._mvo_solver.setup(
                        P_upper, q, A, self._l, self._u, verbose=False, polish=True
                    )

            res = self._mvo_solver.solve()

            if res.info.status in ["solved", "solved_inaccurate"]:
                y = res.x[: self.num_assets]
                kappa = res.x[-1]

                raw_weights = y / kappa
                return raw_weights / np.sum(raw_weights)
            else:
                return np.ones(self.num_assets) / self.num_assets

        else:
            if self.objective == "min_variance":
                P = sp.csc_matrix(2.0 * cov_matrix)
                q = np.zeros(self.num_assets)
            elif self.objective == "max_return":
                P = sp.diags(np.full(self.num_assets, 1e-6), format="csc")
                q = -expected_returns
            elif self.objective == "risk_aversion":
                P = sp.csc_matrix(2.0 * self.risk_aversion_lambda * cov_matrix)
                q = -expected_returns

            P_upper = sp.triu(P, format="csc")

            if self._mvo_solver is None:
                self._mvo_solver = osqp.OSQP()
                self._mvo_solver.setup(
                    P_upper, q, self._A, self._l, self._u, verbose=False, polish=True
                )
            else:
                try:
                    self._mvo_solver.update(Px=P_upper.data)
                except ValueError:
                    self._mvo_solver = osqp.OSQP()
                    self._mvo_solver.setup(
                        P_upper,
                        q,
                        self._A,
                        self._l,
                        self._u,
                        verbose=False,
                        polish=True,
                    )

            result = self._mvo_solver.solve()

            if result.info.status in ["solved", "solved_inaccurate"]:
                return result.x
            else:
                return np.ones(self.num_assets) / self.num_assets

    def compute_efficient_frontier(
        self, num_points: int = 20
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        lambdas = np.logspace(-3, 3, num_points)
        returns = np.zeros(num_points)
        risks = np.zeros(num_points)

        for i, lam in enumerate(lambdas):

            temp_model = MeanVariancePortfolio(
                assets=self.assets,
                initial_capital=self.initial_capital,
                rebalance_frequency=self.rebalance_frequency,
                integer_sizing=self.integer_sizing,
                use_costs=self.use_costs,
                commission_rate=self.commission_rate,
                slippage_rate=self.slippage_rate,
                allocation_bounds=self.allocation_bounds,
                static_constraints=self.static_constraints,
                dynamic_constraints=self.dynamic_constraints,
                lookback_period=self.lookback_period,
                covariance_estimator=self.covariance_estimator,
                return_estimator=self.return_estimator,
                objective="risk_aversion",
                risk_aversion_lambda=lam,
            )

            temp_model.run_backtest()
            returns[i] = temp_model.portfolio_value[-1] / self.initial_capital - 1.0
            risks[i] = np.std(
                temp_model.portfolio_value[1:] / temp_model.portfolio_value[:-1] - 1.0
            )

        return lambdas, returns, risks

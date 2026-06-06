from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import scipy.sparse as sp
import osqp
from joblib import Parallel, delayed
import inspect
import typing
from src.data_generation.base_monte_carlo import BaseMonteCarloSimulator
from src.utils.metrics import calculate_metrics

_OSQP_INF = osqp.constant("OSQP_INFTY")

_OSQP_SETTINGS = dict(
    warm_starting=True,
    verbose=False,
    polish=True,
    eps_abs=1e-8,
    eps_rel=1e-8,
    max_iter=4000,
)


class BaseStrategy(ABC):
    """Base class for backtesting portfolio strategies. Subclasses should implement the _compute_target_weights method to define their specific strategy logic."""

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
    ) -> None:

        self.assets = assets
        self.prices = assets.values
        self.dates = assets.index
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency

        self.integer_sizing = integer_sizing
        self.use_costs = use_costs
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.num_assets = self.prices.shape[1]

        if allocation_bounds is not None:
            self.allocation_bounds = allocation_bounds
        else:
            self.allocation_bounds = [(0.0, 1.0) for _ in range(self.prices.shape[1])]

        self._lb = np.array([b[0] for b in self.allocation_bounds], dtype=np.float64)
        self._ub = np.array([b[1] for b in self.allocation_bounds], dtype=np.float64)

        if static_constraints is not None:
            self.static_constraints = static_constraints
        else:
            self.static_constraints = []

        if dynamic_constraints is not None:
            self.dynamic_constraints = dynamic_constraints
        else:
            self.dynamic_constraints = {}

        self._osqp_static = self._scipy_constraints_to_osqp(
            self.static_constraints, self.num_assets
        )

        self._dc = self.dynamic_constraints
        self._has_turnover_limit = "max_turnover" in self._dc
        self._has_rebalance_threshold = "min_diff_to_rebalance" in self._dc

        self._has_dynamic_constraints = (
            self._has_turnover_limit or self._has_rebalance_threshold
        )

        if not self._has_dynamic_constraints:

            P, A, l, u = self._build_static_qp(
                self.num_assets, self._lb, self._ub, self._osqp_static
            )
            self._osqp_solver_static = osqp.OSQP()
            self._osqp_solver_static.setup(
                P, np.zeros(self.num_assets), A, l, u, **_OSQP_SETTINGS
            )
            self._q_s = np.zeros(self.num_assets, dtype=np.float64)

        else:
            P, A, l, u, row_t1, row_t2 = self._build_dynamic_qp(
                self.num_assets,
                self._lb,
                self._ub,
                self._osqp_static,
                self._has_turnover_limit,
                self._has_rebalance_threshold,
            )

            offset = 3 * self.num_assets + 1

            if self._has_turnover_limit:
                u[offset] = self._dc["max_turnover"]

            if self._has_rebalance_threshold:
                l[offset + int(self._has_turnover_limit)] = self._dc[
                    "min_diff_to_rebalance"
                ]

            self._osqp_solver_dynamic = osqp.OSQP()
            self._osqp_solver_dynamic.setup(
                P, np.zeros(2 * self.num_assets), A, l, u, **_OSQP_SETTINGS
            )

            self._l_d = l.copy()
            self._u_d = u.copy()

            self._row_t1 = row_t1
            self._row_t2 = row_t2

            self._q_d = np.zeros(2 * self.num_assets, dtype=np.float64)

    def run_backtest(self) -> None:
        """Runs the backtest for the strategy, populating the portfolio value and related attributes over time."""

        self.num_periods, self.num_assets = self.prices.shape
        self.portfolio_value = np.zeros(self.num_periods)
        self.asset_shares = np.zeros((self.num_periods, self.num_assets))
        self.asset_values = np.zeros((self.num_periods, self.num_assets))
        self.cash = np.zeros(self.num_periods)
        self.costs = np.zeros(self.num_periods)

        self.weights = np.zeros((self.num_periods, self.num_assets))

        self.cash[0] = self.initial_capital

        self._initialize_portfolio()

        for period in range(1, self.num_periods):
            if period % self.rebalance_frequency == 0:
                target_weights = self._compute_target_weights(period)
                self._rebalance_portfolio(period, target_weights)

            else:
                self.asset_shares[period] = self.asset_shares[period - 1]
                self.cash[period] = self.cash[period - 1]

                self.asset_values[period] = (
                    self.asset_shares[period] * self.prices[period]
                )
                self.portfolio_value[period] = self.cash[period] + np.sum(
                    self.asset_values[period]
                )
                self.weights[period] = np.divide(
                    self.asset_values[period],
                    self.portfolio_value[period],
                    out=np.zeros_like(self.asset_values[period]),
                    where=self.portfolio_value[period] != 0,
                )

    def _initialize_portfolio(self) -> None:
        """Initializes the portfolio with an equal-weighted allocation at the first period."""

        target_weights = np.ones(self.num_assets) / self.num_assets
        self._rebalance_portfolio(0, target_weights)

    def _rebalance_portfolio(self, period: int, target_weights: np.ndarray) -> None:
        """Rebalances the portfolio at the given period according to the target weights, accounting for transaction costs and constraints."""

        target_weights = self._validate_weights(period, target_weights)

        current_prices = self.prices[period]
        current_shares = (
            self.asset_shares[period - 1] if period > 0 else np.zeros(self.num_assets)
        )
        current_cash = self.cash[period - 1] if period > 0 else self.initial_capital

        current_values = current_shares * current_prices
        total_portfolio_value = current_cash + np.sum(current_values)
        fee_rate = self.commission_rate + self.slippage_rate

        if self.use_costs:
            naive_target_values = target_weights * total_portfolio_value
            trade_sign = np.sign(naive_target_values - current_values)

            numerator = total_portfolio_value + fee_rate * np.sum(
                trade_sign * current_values
            )
            denominator = 1 + fee_rate * np.sum(trade_sign * target_weights)

            net_portfolio_value = numerator / denominator
        else:
            net_portfolio_value = total_portfolio_value

        target_values = target_weights * net_portfolio_value
        target_shares = target_values / current_prices

        if self.integer_sizing:
            target_shares = np.floor(target_shares)

        trade_shares = target_shares - current_shares
        trade_values = trade_shares * current_prices

        if self.use_costs:
            trade_costs = np.abs(trade_values) * fee_rate
            self.costs[period] = np.sum(trade_costs)
            current_cash -= np.sum(trade_costs)

        self.asset_shares[period] = current_shares + trade_shares
        self.asset_values[period] = self.asset_shares[period] * current_prices

        self.cash[period] = current_cash - np.sum(trade_values)
        self.portfolio_value[period] = self.cash[period] + np.sum(
            self.asset_values[period]
        )

        self.weights[period] = np.divide(
            self.asset_values[period],
            self.portfolio_value[period],
            out=np.zeros_like(self.asset_values[period]),
            where=self.portfolio_value[period] != 0,
        )

    @abstractmethod
    def _compute_target_weights(self, period: int) -> np.ndarray:
        """Computes the target portfolio weights for the given period based on the strategy's logic. Must be implemented by subclasses."""

        pass

    def _validate_weights(self, period: int, target_weights: np.ndarray) -> np.ndarray:
        """Validates and adjusts the target weights to ensure they satisfy the allocation bounds and constraints using quadratic programming."""

        if not self._has_dynamic_constraints:

            q = self._q_s
            np.multiply(-2.0, target_weights, out=q)
            self._osqp_solver_static.update(q=q)
            res = self._osqp_solver_static.solve()

            if res.info.status in ["solved", "solved_inaccurate"]:
                return res.x
            else:
                return target_weights

        else:

            prew_w = (
                self.weights[period - 1] if period > 0 else np.zeros(self.num_assets)
            )

            q = self._q_d
            q[: self.num_assets] = -2.0 * target_weights

            u = self._u_d
            t1s = self._row_t1
            t2s = self._row_t2
            u[t1s : t1s + self.num_assets] = prew_w
            u[t2s : t2s + self.num_assets] = -prew_w

            self._osqp_solver_dynamic.update(q=q, u=u, l=self._l_d)
            res = self._osqp_solver_dynamic.solve()

            if res.info.status in ["solved", "solved_inaccurate"]:
                return res.x[: self.num_assets]
            else:
                return target_weights

    def run_MC_backtest(
        self,
        simulator: BaseMonteCarloSimulator = None,
        precomputed_prices: np.ndarray = None,
        num_simulations: int = 1000,
        workers: int = 1,
        batch_size: int = 100,
    ) -> np.ndarray:
        """Runs a Monte Carlo backtest using the provided simulator to generate future price paths and applying the strategy's logic to each simulated path. Returns an array of portfolio values over time for each simulation."""

        if precomputed_prices is not None:
            num_simulations = int(precomputed_prices.shape[0])
        elif simulator is not None:
            if not simulator.is_fitted:
                raise ValueError(
                    "Simulator must be fit to data before running Monte Carlo backtest."
                )
        else:
            raise ValueError(
                "Either a fitted simulator or precomputed prices must be provided to run a Monte Carlo backtest."
            )

        batch_size = int(batch_size)

        full_batches = num_simulations // batch_size
        remainder = num_simulations % batch_size

        batches = [batch_size] * full_batches
        if remainder > 0:
            batches.append(remainder)

        batch_indices = []
        current_index = 0

        for b_size in batches:
            batch_indices.append((current_index, current_index + b_size))
            current_index += b_size

        cls = self.__class__
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())[1:]

        init_kwargs = {}
        for p in params:
            if p == "assets":
                continue
            if hasattr(self, p):
                init_kwargs[p] = getattr(self, p)

        assets_columns = (
            list(self.assets.columns) if hasattr(self.assets, "columns") else None
        )
        dates = list(map(str, self.dates)) if hasattr(self, "dates") else None

        results = Parallel(n_jobs=workers)(
            delayed(_mc_batch_worker)(
                simulator if precomputed_prices is None else None,
                (
                    precomputed_prices[batch_start:batch_end]
                    if precomputed_prices is not None
                    else None
                ),
                cls,
                init_kwargs,
                assets_columns,
                dates,
                self.prices[0],
                self.num_periods + 1,
                b_size,
            )
            for (batch_start, batch_end), b_size in zip(batch_indices, batches)
        )

        results_list = list(results)
        return np.vstack(results_list)

    def _scipy_constraints_to_osqp(self, scipy_constraints: list[dict], n: int):
        """Converts constraints defined in the format used by scipy.optimize to the format required by OSQP."""

        w0 = np.zeros(n, dtype=np.float64)
        osqp_constraints = []

        for c in scipy_constraints:
            jac = np.asarray(c["jac"](w0), dtype=np.float64)
            f0 = float(c["fun"](w0))

            A_row = jac.reshape(-1, n)

            b = np.full(A_row.shape[0], -f0)

            l = np.zeros_like(b)
            u = np.zeros_like(b)

            if c["type"] == "ineq":
                l = b
                u = np.full(A_row.shape[0], _OSQP_INF)

            elif c["type"] == "eq":
                l = b
                u = b.copy()

            osqp_constraints.append({"A": A_row, "l": l, "u": u})

        return osqp_constraints

    def _build_static_qp(
        self,
        n: int,
        lb: np.ndarray,
        ub: np.ndarray,
        osqp_constraints: list[dict],
    ) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray, np.ndarray]:
        """Builds the matrices and vectors for a static quadratic program with the given allocation bounds and constraints."""

        In = sp.eye(n, format="csc")
        ones = sp.csc_matrix(np.ones((1, n)))

        A_rows = [In, ones]
        l_rows = [lb, np.array([1.0])]
        u_rows = [ub, np.array([1.0])]

        for c in osqp_constraints:
            A_rows.append(sp.csc_matrix(c["A"]))
            l_rows.append(np.where(np.isfinite(c["l"]), c["l"], 0.0 - float(_OSQP_INF)))
            u_rows.append(np.where(np.isfinite(c["u"]), c["u"], float(_OSQP_INF)))

        A = sp.csc_matrix(sp.vstack(A_rows, format="csc"))

        return (
            In * 2.0,
            A,
            np.concatenate(l_rows),
            np.concatenate(u_rows),
        )

    def _build_dynamic_qp(
        self,
        n: int,
        lb: np.ndarray,
        ub: np.ndarray,
        osqp_constraints: list[dict],
        has_turnover: bool,
        has_min_diff: bool,
    ) -> tuple:
        """Builds the matrices and vectors for a dynamic quadratic program that includes turnover and rebalance threshold constraints, in addition to the allocation bounds and static constraints."""

        In = sp.eye(n, format="csc")
        Zn = sp.csc_matrix((n, n))
        ones_n = sp.csc_matrix(np.ones((1, n)))
        zeros_n = sp.csc_matrix(np.zeros((1, n)))

        P = sp.block_diag([In * 2.0, sp.csc_matrix((n, n))], format="csc")

        A_rows, l_rows, u_rows = [], [], []

        A_rows.append(sp.hstack([In, Zn], format="csc"))
        l_rows.append(lb)
        u_rows.append(ub)

        A_rows.append(sp.hstack([ones_n, zeros_n], format="csc"))
        l_rows.append(np.array([1.0]))
        u_rows.append(np.array([1.0]))

        row_t1_start = n + 1
        A_rows.append(sp.hstack([In, -In], format="csc"))
        l_rows.append(np.full(n, 0.0 - float(_OSQP_INF)))
        u_rows.append(np.zeros(n))

        row_t2_start = 2 * n + 1
        A_rows.append(sp.hstack([-In, -In], format="csc"))
        l_rows.append(np.full(n, 0.0 - float(_OSQP_INF)))
        u_rows.append(np.zeros(n))

        if has_turnover:
            A_rows.append(sp.hstack([zeros_n, ones_n], format="csc"))
            l_rows.append(np.array([0.0 - float(_OSQP_INF)]))
            u_rows.append(np.array([0.0]))

        if has_min_diff:
            A_rows.append(sp.hstack([zeros_n, ones_n], format="csc"))
            l_rows.append(np.array([0.0]))
            u_rows.append(np.array([float(_OSQP_INF)]))

        for c in osqp_constraints:
            Ac = sp.csc_matrix(c["A"])
            m = Ac.shape[0]
            A_rows.append(sp.hstack([Ac, sp.csc_matrix((m, n))], format="csc"))
            l_rows.append(np.where(np.isfinite(c["l"]), c["l"], 0.0 - float(_OSQP_INF)))
            u_rows.append(np.where(np.isfinite(c["u"]), c["u"], float(_OSQP_INF)))

        A_rows.append(sp.hstack([Zn, In], format="csc"))
        l_rows.append(np.zeros(n))
        u_rows.append(np.full(n, float(_OSQP_INF)))

        return (
            P,
            sp.vstack(A_rows, format="csc"),
            np.concatenate(l_rows),
            np.concatenate(u_rows),
            row_t1_start,
            row_t2_start,
        )

    def calculate_performance_metrics(self) -> dict:
        """Calculates performance metrics based on the portfolio value over time."""

        if not hasattr(self, "portfolio_value"):
            raise ValueError(
                "Backtest must be run before calculating performance metrics."
            )

        return calculate_metrics(self.portfolio_value)


def _mc_batch_worker(
    simulator: BaseMonteCarloSimulator,
    precomputed_prices: np.ndarray,
    cls: typing.Type[BaseStrategy],
    init_kwargs: dict,
    assets_columns: typing.Optional[list],
    dates: typing.Optional[list],
    starting_prices: np.ndarray,
    num_steps: int,
    current_batch_size: int = 0,
) -> np.ndarray:
    """Worker function for running a batch of Monte Carlo simulations in parallel. Simulates future price paths and applies the strategy's logic to each path, returning the portfolio values over time for the batch."""

    if precomputed_prices is not None:
        simulated_prices = precomputed_prices
        num_steps = int(simulated_prices.shape[1])
    else:
        simulated_prices, _ = simulator.simulate(
            starting_prices=starting_prices,
            num_simulations=current_batch_size,
            num_steps=num_steps,
        )

    batch_results = np.zeros((current_batch_size, num_steps))

    index = pd.to_datetime(dates) if dates is not None else None
    for i in range(current_batch_size):
        sim_prices = simulated_prices[i]
        try:
            assets_df = pd.DataFrame(sim_prices, columns=assets_columns, index=index)
        except Exception:
            assets_df = pd.DataFrame(sim_prices)

        kwargs = init_kwargs.copy()
        kwargs["assets"] = assets_df
        strategy = cls(**kwargs)
        strategy.run_backtest()
        batch_results[i] = strategy.portfolio_value

    return batch_results

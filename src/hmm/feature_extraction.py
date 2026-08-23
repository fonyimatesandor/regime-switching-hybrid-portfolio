import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view


class HMMFeatureExtractor:
    """
    Extracts four regime-indicator features from a panel of daily asset prices.

    Parameters
    ----------
    vol_window : int, default 21
        Rolling window (days) for volatility estimation.
    corr_window : int, default 21
        Rolling window (days) for pairwise correlation estimation.
    draw_window : int, default 252
        Rolling window (days) for high-water mark used in drawdown.
    standardize : bool, default True
        If True, standardise each feature to zero mean and unit variance
        using the statistics of the training data.  S

    Attributes
    ----------
    scaler_ : StandardScaler or None
        Fitted scaler (set after calling ``extract_features``).
        Use ``scaler_.mean_`` and ``scaler_.scale_`` to inspect or
        invert the standardisation.

    Notes
    -----
    **Features computed**

    *return*
        Log return of the equal-price-weight portfolio:
        r_t = ln(P_bar_t / P_bar_{t-1}),  P_bar_t = mean of all prices at t.

    *volatility*
        Rolling standard deviation of portfolio log returns over
        ``vol_window`` trading days.

    *correlation*
        Rolling average pairwise log-return correlation across all assets::

            rho_bar(t) = (sum_{j,k} C_{jk}(t) - N) / (N*(N-1))

        where C(t) is the N x N rolling correlation matrix.
        The formula subtracts the N diagonal entries (each = 1) and
        divides by the N*(N-1) off-diagonal pairs.

    *drawdown*
        Current drawdown from the rolling high-water mark (always <= 0)::

            dd_t = (P_bar_t - max_{s in [t-draw_window, t]} P_bar_s)
                   / max_{s in [t-draw_window, t]} P_bar_s

    The first ``max(vol_window, corr_window) - 1`` rows are dropped because
    rolling windows are not yet full.  No backward-fill is applied, so there
    is no look-ahead contamination.
    """

    def __init__(
        self,
        vol_window: int = 21,
        corr_window: int = 21,
        draw_window: int = 252,
        standardize: bool = True,
    ):
        self.vol_window = vol_window
        self.corr_window = corr_window
        self.draw_window = draw_window
        self.standardize = standardize
        self.scaler_: StandardScaler | None = None

    def extract_features(self, price_df: pd.DataFrame) -> np.ndarray:
        """
        Compute features and fit the scaler (if ``standardize=True``).

        Call this on **training data only**.  For new / held-out data use
        ``transform`` so the same scaling is applied.

        Parameters
        ----------
        price_df : DataFrame of shape (T, N)
            Daily closing prices with a DatetimeIndex.

        Returns
        -------
        X : ndarray of shape (T', 4)
            Feature matrix with NaN rows dropped (T' <= T).
            Columns: [return, volatility, correlation, drawdown].
        """
        raw = self._compute_raw(price_df)

        if self.standardize:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(raw)

        return raw

    def transform(self, price_df: pd.DataFrame) -> np.ndarray:
        """
        Compute features and apply the **already-fitted** scaler.

        Use this for validation / test data after calling ``extract_features``
        on the training set.

        Parameters
        ----------
        price_df : DataFrame of shape (T, N)
            Daily closing prices with a DatetimeIndex.

        Returns
        -------
        X : ndarray of shape (T', 4)
        """
        raw = self._compute_raw(price_df)

        if self.standardize:
            if self.scaler_ is None:
                raise RuntimeError(
                    "Call extract_features() on training data before transform()."
                )
            return self.scaler_.transform(raw)

        return raw

    def _compute_raw(self, price_df: pd.DataFrame) -> np.ndarray:
        """Compute the four raw (un-standardised) features and drop NaN rows."""
        prices = price_df.values
        portfolio = prices.mean(axis=1)

        log_ret = np.full_like(portfolio, np.nan)
        log_ret[1:] = np.log(portfolio[1:] / portfolio[:-1])

        W_vol = self.vol_window
        if len(log_ret) >= W_vol:
            stride_vol = sliding_window_view(log_ret, window_shape=W_vol)
            vol_std = stride_vol.std(axis=1, ddof=1)
            volatility = np.concatenate([np.full(W_vol - 1, np.nan), vol_std])
        else:
            volatility = np.full_like(log_ret, np.nan)

        avg_corr = self._rolling_avg_pairwise_correlation(price_df).values

        W_draw = self.draw_window
        if len(portfolio) >= W_draw:
            stride_draw = sliding_window_view(portfolio, window_shape=W_draw)
            roll_max_full = stride_draw.max(axis=1)

            initial_max = np.maximum.accumulate(portfolio[: W_draw - 1])
            rolling_peak = np.concatenate([initial_max, roll_max_full])
        else:
            rolling_peak = np.maximum.accumulate(portfolio)

        drawdown = (portfolio - rolling_peak) / rolling_peak

        features = np.column_stack([log_ret, volatility, avg_corr, drawdown])

        valid_rows = ~np.isnan(features).any(axis=1)

        return features[valid_rows]

    def _rolling_avg_pairwise_correlation(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Rolling average pairwise correlation.

        For an N x N correlation matrix C(t), the average off-diagonal entry is:
            rho_bar(t) = (sum_{j,k} C_{jk}(t) - N) / (N*(N-1))
        because the diagonal contributes N*1 = N and there are N*(N-1)
        off-diagonal pairs.
        """

        N = price_df.shape[1]

        if N < 2:
            return pd.Series(0.0, index=price_df.index)

        W = self.corr_window
        T = len(price_df)

        if T <= W:
            return pd.Series(np.nan, index=price_df.index)

        log_rets = np.log(price_df / price_df.shift(1)).values[1:]

        windows = sliding_window_view(log_rets, window_shape=W, axis=0).transpose(
            0, 2, 1
        )

        means = windows.mean(axis=1, keepdims=True)
        centered = windows - means

        stds = centered.std(axis=1, ddof=1, keepdims=True)
        stds[stds == 0] = 1.0
        z = centered / stds

        z_sum_assets = z.sum(axis=2)
        date_total = (z_sum_assets**2).sum(axis=1) / (W - 1)

        avg_corr = (date_total - N) / (N * (N - 1))

        pad = np.full(W, np.nan)
        avg_corr_full = np.concatenate([pad, avg_corr])

        return pd.Series(avg_corr_full, index=price_df.index)

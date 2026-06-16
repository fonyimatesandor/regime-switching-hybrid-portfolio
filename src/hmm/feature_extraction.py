import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
        portfolio = price_df.mean(axis=1)
        log_ret = np.log(portfolio / portfolio.shift(1))

        volatility = log_ret.rolling(window=self.vol_window).std()

        avg_corr = self._rolling_avg_pairwise_correlation(price_df)

        rolling_peak = portfolio.rolling(window=self.draw_window, min_periods=1).max()
        drawdown = (portfolio - rolling_peak) / rolling_peak

        features = pd.DataFrame(
            {
                "return": log_ret,
                "volatility": volatility,
                "correlation": avg_corr,
                "drawdown": drawdown,
            }
        )

        return features.dropna().values

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

        log_rets = np.log(price_df / price_df.shift(1))
        corr_mat = log_rets.rolling(window=self.corr_window).corr()

        date_total = corr_mat.sum(axis=1).groupby(level=0).sum()
        avg_corr = (date_total - N) / (N * (N - 1))

        return avg_corr

from hmmlearn.base import BaseHMM
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import multivariate_t
from scipy.special import digamma, polygamma


class HMMModel(BaseHMM):
    """
    HMM with multivariate Student-t emissions, fitted by ECME.

    The emission for state i is  x_t | s_t=i ~ t_{nu_i}(mu_i, Sigma_i),
    where Sigma_i is scipy's scale (shape) matrix:
    Cov[x | s=i] = nu_i/(nu_i-2) * Sigma_i  (requires nu_i > 2).

    EM is implemented via the normal-variance mixture representation::

        x_t | u_t, s_t=i  ~  N(mu_i,  Sigma_i / u_t)
        u_t | s_t=i        ~  Gamma(nu_i/2,  nu_i/2)

    Parameters
    ----------
    n_components : int, default 2
        Number of hidden states.
    n_iter : int, default 100
        Maximum EM iterations.
    tol : float, default 1e-4
        Log-likelihood improvement convergence threshold.
    df_bounds : tuple of float, default (2.01, 50.0)
        Box constraints ``(lo, hi)`` on each state's degrees-of-freedom
        parameter nu.  ``lo`` must be >= 2.01 to guarantee finite covariance.

    Attributes
    ----------
    means_ : ndarray of shape (n_components, n_features)
        Location vectors.
    shapes_ : ndarray of shape (n_components, n_features, n_features)
        Scale (shape) matrices.  Related to covariance by
        ``Cov = nu/(nu-2) * shapes_[i]``.
    df_ : ndarray of shape (n_components,)
        Degrees-of-freedom parameter nu per state.

    Notes
    -----
    **E-step** — posterior of u_t given (x_t, s_t=i) is
    Gamma((nu+p)/2, (nu+delta^2)/2) where delta^2_t = (x_t-mu_i)^T
    Sigma_i^{-1} (x_t-mu_i), giving::

        E[u_t]     = (nu + p) / (nu + delta^2_t)
        E[ln u_t]  = psi((nu+p)/2) - ln((nu+delta^2_t)/2)

    Note that E[ln u] != ln(E[u]); the digamma formula is exact.

    **M-step** — exact Q-function maximisers::

        mu_hat  =  (sum_t w_t x_t) / (sum_t w_t),   w_t = gamma_t * E[u_t]
        Sigma_hat  =  S_i / N_i,   S_i = sum_t w_t (x_t-mu_hat)(x_t-mu_hat)^T
        nu_hat  solves  ln(nu/2) - psi(nu/2) + 1 + E_bar[ln u] - E_bar[u] = 0

    The Q-maximiser for Sigma_i is S_i/N_i *directly as the scale matrix*.
    The (nu-2)/nu rescaling only appears in ``_init`` when converting a sample
    covariance (which estimates Cov, not the scale matrix) to Sigma.

    Newton-Raphson for nu converges in <= 5 steps because the score g(nu) is
    strictly decreasing: g'(nu) = 1/nu - 0.5*psi^(1)(nu/2) < 0.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_iter: int = 100,
        tol: float = 1e-4,
        df_bounds: tuple = (2.01, 50.0),
    ):
        super().__init__(n_components=n_components, n_iter=n_iter, tol=tol)
        self.df_bounds = df_bounds

    def _init(self, X, lengths=None):
        p, K = X.shape[1], self.n_components
        self.n_features = p

        km = KMeans(n_clusters=K, n_init=10, random_state=0)
        km.fit(X)
        self.means_ = km.cluster_centers_.copy()

        self.df_ = np.full(K, 5.0)

        self.shapes_ = np.empty((K, p, p))
        for i in range(K):
            mask = km.labels_ == i
            cov = (
                np.atleast_2d(np.cov(X[mask], rowvar=False))
                if mask.sum() > 1
                else np.eye(p)
            )
            nu = self.df_[i]
            self.shapes_[i] = _ensure_pd(cov * (nu - 2.0) / nu)

        self.transmat_ = np.full((K, K), 1.0 / K)
        self.startprob_ = np.full(K, 1.0 / K)

    def _check(self):
        super()._check()
        if self.df_bounds is None:
            return
        if not (isinstance(self.df_bounds, (tuple, list)) and len(self.df_bounds) == 2):
            raise ValueError("df_bounds must be a 2-tuple or list (lo, hi)")
        lo, hi = self.df_bounds
        if lo < 2.01:
            raise ValueError(
                "df_bounds lo must be >= 2.01 to guarantee finite covariance"
            )
        if hi <= lo:
            raise ValueError("df_bounds hi must be strictly greater than lo")

    def _compute_log_likelihood(self, X):
        """Return array (T, K) of log p(x_t | s_t = i)."""
        ll = np.empty((X.shape[0], self.n_components))
        for i in range(self.n_components):
            ll[:, i] = multivariate_t.logpdf(
                X, loc=self.means_[i], shape=self.shapes_[i], df=self.df_[i]
            )
        return ll

    def _generate_sample_from_state(self, state, random_state=None):
        return multivariate_t.rvs(
            loc=self.means_[state],
            shape=self.shapes_[state],
            df=self.df_[state],
            random_state=random_state,
        )

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        K, p = self.n_components, self.n_features
        stats["N"] = np.zeros(K)
        stats["Nw"] = np.zeros(K)
        stats["xbar"] = np.zeros((K, p))
        stats["S"] = np.zeros((K, p, p))
        stats["Elu"] = np.zeros(K)
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice
    ):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice
        )
        T, p = obs.shape

        for i in range(self.n_components):
            nu = self.df_[i]
            Sinv = np.linalg.inv(self.shapes_[i])
            gamma_i = posteriors[:, i]

            diff = obs - self.means_[i]
            d2 = np.einsum("ti,ij,tj->t", diff, Sinv, diff)

            u = (nu + p) / (nu + d2)
            lnu = digamma((nu + p) / 2.0) - np.log((nu + d2) / 2.0)

            w = gamma_i * u

            stats["N"][i] += gamma_i.sum()
            stats["Nw"][i] += w.sum()
            stats["xbar"][i] += w @ obs
            stats["S"][i] += (obs * w[:, np.newaxis]).T @ obs
            stats["Elu"][i] += gamma_i @ lnu

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        lo, hi = self.df_bounds

        for i in range(self.n_components):
            N_i = max(stats["N"][i], 1e-10)
            Nw_i = max(stats["Nw"][i], 1e-10)

            mu = stats["xbar"][i] / Nw_i
            self.means_[i] = mu

            Elu = stats["Elu"][i] / N_i
            Eu = Nw_i / N_i

            nu = np.clip(self.df_[i], lo, hi)
            for _ in range(50):
                g = np.log(nu / 2.0) - digamma(nu / 2.0) + 1.0 + Elu - Eu
                gp = 1.0 / nu - 0.5 * polygamma(1, nu / 2.0)
                nu_new = np.clip(nu - g / gp, lo, hi)
                if abs(nu_new - nu) < 1e-10:
                    nu = nu_new
                    break
                nu = nu_new
            self.df_[i] = nu

            scatter = stats["S"][i] - Nw_i * np.outer(mu, mu)
            scatter = (scatter + scatter.T) / 2.0
            self.shapes_[i] = _ensure_pd(scatter / N_i)

    @staticmethod
    def _ensure_pd(M, floor: float = 1e-6):
        """Shift the minimum eigenvalue of M up to ``floor`` if it falls below."""
        deficit = floor - np.linalg.eigvalsh(M).min()
        if deficit > 0:
            M = M + np.eye(M.shape[0]) * deficit
        return M


def _ensure_pd(M, floor: float = 1e-6):
    """Module-level alias used in ``_init`` before class construction finishes."""
    deficit = floor - np.linalg.eigvalsh(M).min()
    if deficit > 0:
        M = M + np.eye(M.shape[0]) * deficit
    return M

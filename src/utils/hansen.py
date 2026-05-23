import numpy as np
from scipy.special import gamma, stdtr, stdtrit
from scipy.optimize import minimize, brentq
from scipy.integrate import quad


class HansenSkewedT:
    """
    Hansen (1994) skewed-t distribution.
    params: (eta, lam, loc, scale)
    eta  > 2       : tail thickness
    lam in (-1, 1) : skewness, negative = left skew
    """

    @staticmethod
    def _constants(eta, lam):
        c = gamma((eta + 1) / 2) / (np.sqrt(np.pi * (eta - 2)) * gamma(eta / 2))
        a = 4 * lam * c * (eta - 2) / (eta - 1)
        b = np.sqrt(1 + 3 * lam**2 - a**2)
        return a, b, c

    @classmethod
    def logpdf(cls, x, eta, lam, loc=0.0, scale=1.0):
        z = (x - loc) / scale
        a, b, c = cls._constants(eta, lam)

        left  = z < -a / b
        inner = np.where(
            left,
            (b * z + a) / (1 - lam),
            (b * z + a) / (1 + lam)
        )
        log_kernel = -((eta + 1) / 2) * np.log(1 + inner**2 / (eta - 2))
        return np.log(b) + np.log(c) + log_kernel - np.log(scale)

    @classmethod
    def cdf(cls, x, eta, lam, loc=0.0, scale=1.0):
        z = (x - loc) / scale
        a, b, c = cls._constants(eta, lam)

        bza = b * z + a
        left = z < -a / b

        cdf_vals = np.where(
            left,
            (1 - lam) * stdtr(eta, bza / (1 - lam) * np.sqrt(eta / (eta - 2))),
            (1 - lam) / 2 + (1 + lam) * (
                stdtr(eta, bza / (1 + lam) * np.sqrt(eta / (eta - 2))) - 0.5
            )
        )
        return np.clip(cdf_vals, 1e-12, 1 - 1e-12)

    @classmethod
    def ppf(cls, u, eta, lam, loc=0.0, scale=1.0):
        u = np.atleast_1d(u)
        result = np.zeros_like(u, dtype=float)
        for i, ui in enumerate(u):
            f = lambda x: cls.cdf(x, eta, lam, loc, scale) - ui
            lo = stdtrit(eta, 1e-7) * scale + loc
            hi = stdtrit(eta, 1 - 1e-7) * scale + loc
            result[i] = brentq(f, lo, hi, xtol=1e-10)
        return result

    @classmethod
    def fit(cls, data):
        loc0  = np.mean(data)
        scale0 = np.std(data)

        def nll(params):
            eta, lam, loc, scale = params
            if eta <= 2.01 or abs(lam) >= 0.999 or scale <= 0:
                return np.inf
            return -np.sum(cls.logpdf(data, eta, lam, loc, scale))

        res = minimize(
            nll,
            x0=[5.0, -0.1, loc0, scale0],
            bounds=[(2.1, 50), (-0.999, 0.999), (None, None), (1e-6, None)],
            method='L-BFGS-B'
        )
        return tuple(res.x)

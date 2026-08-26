"""Maximum-likelihood calibration of the goal model.

The likelihood is the joint probability of the *observed scoreline*, not of the
H/D/A outcome.  A 4-0 and a 1-0 are both "home win" but carry very different
information about scoring rates, and throwing that away costs a lot of
statistical power for free.

Performance note: fitting never materialises a scoreline grid.  For the
likelihood we only need P(h_i, a_i) at the one cell that actually occurred, so
the whole objective is a vectorised O(N) numpy expression over N matches
instead of N grid builds.  On 200k matches that is the difference between a
sub-second evaluation and a multi-minute one, and the optimiser calls the
objective hundreds of times.
"""
from __future__ import annotations

from wcq._compat import SLOTS

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from wcq.model.poisson import PoissonParams


@dataclass(frozen=True, **SLOTS)
class TrainingSet:
    """Columnar view of a training window -- built once, reused every
    likelihood evaluation."""
    elo_diff: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    neutral: np.ndarray

    def __len__(self) -> int:
        return int(self.elo_diff.size)


def build_training_set(rows: Sequence[tuple[float, int, int, bool]]) -> TrainingSet:
    if not rows:
        raise ValueError("empty training set")
    d, h, a, n = zip(*rows)
    return TrainingSet(
        np.asarray(d, dtype=float),
        np.asarray(h, dtype=np.int64),
        np.asarray(a, dtype=np.int64),
        np.asarray(n, dtype=bool),
    )


def _tau_at(h, a, lam, mu, rho):
    """Dixon-Coles correction evaluated only at the observed cells."""
    t = np.ones_like(lam)
    m00 = (h == 0) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m10 = (h == 1) & (a == 0)
    m11 = (h == 1) & (a == 1)
    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho
    return t


def negative_log_likelihood(theta: np.ndarray, ts: TrainingSet) -> float:
    mu, beta, gamma, rho = theta
    d = ts.elo_diff / 400.0
    g = np.where(ts.neutral, 0.0, gamma)
    lam_h = np.exp(mu + beta * d + g)
    lam_a = np.exp(mu - beta * d)

    ll = (
        -lam_h + ts.home_goals * np.log(lam_h) - gammaln(ts.home_goals + 1.0)
        - lam_a + ts.away_goals * np.log(lam_a) - gammaln(ts.away_goals + 1.0)
    )
    tau = _tau_at(ts.home_goals, ts.away_goals, lam_h, lam_a, rho)
    if np.any(tau <= 0):
        return 1e12                     # rho pushed a cell negative: infeasible
    ll = ll + np.log(tau)

    total = float(np.sum(ll))
    return 1e12 if not np.isfinite(total) else -total


def fit(ts: TrainingSet, start: PoissonParams | None = None) -> tuple[PoissonParams, dict]:
    """Fit by L-BFGS-B with box constraints.

    Bounds are not decoration: rho is only well defined in a narrow band before
    tau drives a probability negative, and an unbounded optimiser will happily
    walk there and return a nonsense fit that still 'converged'.
    """
    x0 = (start or PoissonParams()).as_vector()
    bounds = [(-2.0, 2.0), (0.0, 5.0), (-1.0, 1.0), (-0.25, 0.25)]
    res = minimize(
        negative_log_likelihood, x0, args=(ts,), method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 500, "ftol": 1e-10},
    )
    params = PoissonParams.from_vector(res.x)
    info = {
        "success": bool(res.success),
        "n": len(ts),
        "logL": float(-res.fun),
        "logL_per_match": float(-res.fun / len(ts)),
        "iterations": int(res.nit),
        "message": str(res.message),
    }
    return params, info


_CHI2_95_1DF = 3.841


def profile_ci(ts: TrainingSet, fitted: PoissonParams, index: int,
               step: float = 0.0125, max_steps: int = 400) -> tuple[float, float]:
    """Profile-likelihood 95% interval for one parameter.

    Walks outward from the MLE in each direction, re-optimising the other
    parameters at every step, until 2*(logL_max - logL) crosses 3.841
    (chi-square 95%, 1 df).  The walk goes as far as it needs to: a fixed
    scan range would silently clip a wide interval at the edge of the grid
    and report false precision, which is the one failure a CI must not have.
    Endpoints are linearly interpolated between the last point inside and the
    first outside; a bound not reached within `max_steps` is reported as
    +/-inf rather than as a made-up number.

    Slower than inverting the Hessian but does not assume the likelihood is
    locally quadratic, which near the rho bound it is not.
    """
    base = fitted.as_vector()
    ll_max = -negative_log_likelihood(base, ts)
    free = [i for i in range(4) if i != index]

    def profile_ll(v: float, start: np.ndarray) -> tuple[float, np.ndarray]:
        theta = base.copy()
        theta[index] = v

        def obj(sub):
            full = theta.copy()
            for j, i in enumerate(free):
                full[i] = sub[j]
            return negative_log_likelihood(full, ts)

        r = minimize(obj, start, method="Nelder-Mead",
                     options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4})
        return -r.fun, r.x

    def walk(direction: float) -> float:
        prev_v, prev_stat = base[index], 0.0
        start = base[free].copy()
        for k in range(1, max_steps + 1):
            v = base[index] + direction * k * step
            ll, start = profile_ll(v, start)          # warm-start the nuisances
            stat = 2.0 * (ll_max - ll)
            if stat > _CHI2_95_1DF:
                frac = (_CHI2_95_1DF - prev_stat) / (stat - prev_stat)
                return prev_v + frac * (v - prev_v)
            prev_v, prev_stat = v, stat
        return direction * float("inf")

    return walk(-1.0), walk(+1.0)

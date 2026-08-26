"""Elo rating gap -> goal rates -> full scoreline distribution -> 1X2.

Model
-----
    log lambda_home = mu + beta * d/400 + gamma
    log lambda_away = mu - beta * d/400

`d` is the Elo gap with home advantage folded in, so a single `beta` controls
how strongly a rating difference translates into goals.  The log-link keeps
rates positive without clipping and makes the parameters interpretable:
`mu` is the log of the league's baseline scoring rate, `gamma` the home
scoring bump in log-goals.

Independence is not assumed.  A plain product of two Poissons systematically
under-predicts draws, because the two teams' scores are correlated at low
scorelines (a 0-0 is a joint state, not two independent zeros).  Dixon-Coles
fixes this with a four-cell correction `tau` governed by one parameter `rho`,
applied to (0,0), (0,1), (1,0), (1,1).  Setting rho=0 recovers independence,
which is what the model reduces to if the data says the correction is not
needed.

Cost: the scoreline grid is (MAX_GOALS+1)^2 per match.  Built as a numpy outer
product so a whole season is vectorised rather than looped in Python.
"""
from __future__ import annotations

from wcq._compat import SLOTS

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAX_GOALS = 10
"""Floor for the scoreline grid, not a cap: `score_matrix` grows the grid
with the scoring rates so that the truncated tail stays below ~1e-9 even at
extreme Elo gaps.  A fixed grid of 10 is fine at league rates (lambda ~1.5)
but at lambda = 6 it silently discards ~4% of the probability mass, which
biases every derived number in one direction."""


@dataclass(frozen=True, **SLOTS)
class PoissonParams:
    mu: float = 0.10        # log baseline goals per team per match
    beta: float = 1.05      # sensitivity of log-goals to a 400-point Elo gap
    gamma: float = 0.22     # home scoring advantage, log-goals
    rho: float = -0.05      # Dixon-Coles low-score correlation; 0 = independent

    def as_vector(self) -> np.ndarray:
        return np.array([self.mu, self.beta, self.gamma, self.rho], dtype=float)

    @staticmethod
    def from_vector(v) -> "PoissonParams":
        return PoissonParams(float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def goal_rates(elo_diff, params: PoissonParams, neutral=False):
    """Vectorised: accepts a scalar or an array of Elo gaps.

    `elo_diff` must already include the venue effect from the Elo engine;
    `gamma` here is the separate *scoring-rate* home bump, which is not the
    same thing as the Elo-point home advantage and is fitted independently.
    """
    d = np.asarray(elo_diff, dtype=float) / 400.0
    g = np.where(np.asarray(neutral), 0.0, params.gamma)
    lam_h = np.exp(params.mu + params.beta * d + g)
    lam_a = np.exp(params.mu - params.beta * d)
    return lam_h, lam_a


def _poisson_pmf_grid(lam: float, n: int = MAX_GOALS) -> np.ndarray:
    """P(X=k) for k=0..n via the recurrence p_k = p_{k-1} * lam/k.

    Cheaper and far more numerically stable than exp(-lam)*lam**k/k!, which
    overflows k! for large k and loses precision for small lam.
    """
    p = np.empty(n + 1)
    p[0] = np.exp(-lam)
    for k in range(1, n + 1):
        p[k] = p[k - 1] * lam / k
    return p


def _tau(lam: float, mu: float, rho: float, n: int = MAX_GOALS) -> np.ndarray:
    """Dixon-Coles correction matrix; 1 everywhere except the four low cells."""
    t = np.ones((n + 1, n + 1))
    t[0, 0] = 1.0 - lam * mu * rho
    t[0, 1] = 1.0 + lam * rho
    t[1, 0] = 1.0 + mu * rho
    t[1, 1] = 1.0 - rho
    return t


def _grid_size(lam_h: float, lam_a: float) -> int:
    """Grid bound with a truncated tail below ~1e-9 at any rate: mean plus
    seven standard deviations, floored at MAX_GOALS."""
    m = max(lam_h, lam_a)
    return max(MAX_GOALS, int(math.ceil(m + 7.0 * math.sqrt(m) + 2.0)))


def score_matrix(lam_h: float, lam_a: float, rho: float = 0.0,
                 n: Optional[int] = None) -> np.ndarray:
    """Joint P(home=i, away=j) as an (n+1) x (n+1) matrix, renormalised.

    `n` defaults to an adaptive bound (see `_grid_size`); pass it explicitly
    only to reproduce a fixed-grid computation.  The renormalisation exists
    for the truncation (and the clip below): the Dixon-Coles tau itself
    preserves total mass exactly -- the four cell corrections cancel by
    construction -- so it needs no rescue here.
    """
    if n is None:
        n = _grid_size(lam_h, lam_a)
    ph = _poisson_pmf_grid(lam_h, n)
    pa = _poisson_pmf_grid(lam_a, n)
    m = np.outer(ph, pa)
    if rho != 0.0:
        m = m * _tau(lam_h, lam_a, rho, n)
        np.clip(m, 1e-15, None, out=m)     # tau can go negative at extreme rho
    return m / m.sum()


def outcome_probs(lam_h: float, lam_a: float, rho: float = 0.0,
                  n: Optional[int] = None) -> tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) by summing the scoreline grid."""
    m = score_matrix(lam_h, lam_a, rho, n)
    draw = float(np.trace(m))
    home = float(np.tril(m, -1).sum())     # i > j
    away = float(np.triu(m, 1).sum())      # i < j
    total = home + draw + away
    return home / total, draw / total, away / total


def match_probs(elo_diff: float, params: PoissonParams, neutral: bool = False) -> tuple[float, float, float]:
    lam_h, lam_a = goal_rates(elo_diff, params, neutral)
    return outcome_probs(float(lam_h), float(lam_a), params.rho)


def over_under(lam_h: float, lam_a: float, line: float = 2.5, rho: float = 0.0,
               n: Optional[int] = None) -> tuple[float, float]:
    """P(total goals > line), P(total < line).  Lines are half-goals so there
    is no push to handle."""
    m = score_matrix(lam_h, lam_a, rho, n)
    n = m.shape[0]
    idx = np.add.outer(np.arange(n), np.arange(n))
    over = float(m[idx > line].sum())
    return over, 1.0 - over

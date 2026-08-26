"""Knockout-bracket championship probabilities: exact and Monte Carlo.

There are two ways to answer "what is each team's chance of winning this
bracket", and knowing when each is the right tool is the point of this module.

**Exact dynamic programming.**  For a single-elimination bracket of n = 2^r
teams where every tie is an independent Bernoulli draw, the answer is a closed
recursion:

    P(i reaches round k+1) = P(i reaches round k)
                             * sum over possible opponents j
                               P(j reaches round k) * P(i beats j)

The set of possible opponents in round k is exactly the sibling sub-block of
size 2^(k-1), which the bracket structure hands you for free.  Total cost is
sum over rounds of n * 2^(k-1) = O(n^2) -- 1024 multiply-adds for a 32-team
bracket.  Exact, deterministic, microseconds.

**Monte Carlo.**  Sample the bracket many times and count.  Strictly worse
here: it returns an *estimate* of a quantity the DP gives exactly, with error
falling only as 1/sqrt(N).

So why does the C++ simulator exist at all?  Because the DP recursion depends
on every tie being independent and memoryless, and the moment the real world
intrudes that assumption dies:

  - group stages with goal-difference tiebreakers (the bracket is not fixed)
  - extra time and penalties as a distinct regime
  - fatigue, suspensions, and injuries that carry across rounds
  - correlated outcomes (a shared referee, weather, a squad-wide illness)
  - any question about a *joint* event, e.g. "P(both finalists from the same
    half AND total goals > 100)"

Each of those makes the state space path-dependent, and the DP blows up while
the simulator only needs one more line inside the loop.  The DP earns its keep
anyway: it is the ground truth the simulator is tested against.  `test_bracket`
asserts the MC estimate lands inside a few standard errors of the exact answer,
which catches RNG bugs, off-by-one bracket indexing, and biased seeding that no
amount of eyeballing the output would.
"""
from __future__ import annotations

from wcq._compat import SLOTS

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, **SLOTS)
class Team:
    name: str
    elo: float


def elo_win_prob(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def _check_bracket(teams: Sequence[Team]) -> int:
    n = len(teams)
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"bracket size must be a power of two, got {n}")
    return n


def win_matrix(teams: Sequence[Team]) -> np.ndarray:
    """P[i][j] = probability i beats j.  Built once, O(n^2)."""
    elo = np.array([t.elo for t in teams], dtype=float)
    return 1.0 / (1.0 + 10.0 ** ((elo[None, :] - elo[:, None]) / 400.0))


def exact_title_probs(teams: Sequence[Team]) -> np.ndarray:
    """Exact championship probability per team, O(n^2) time, O(n) space."""
    n = _check_bracket(teams)
    p_beats = win_matrix(teams)
    reach = np.ones(n)                       # P(alive entering round 1)

    block = 1
    while block < n:
        nxt = np.empty(n)
        for i in range(n):
            # In this round, i's block is [i - i%block, ...) of width `block`;
            # the opponent block is its sibling, flipped by one bit.
            base = (i // block) * block
            opp_base = base + block if (i // block) % 2 == 0 else base - block
            opp = slice(opp_base, opp_base + block)
            nxt[i] = reach[i] * float(np.dot(reach[opp], p_beats[i, opp]))
        reach = nxt
        block *= 2
    return reach


def monte_carlo_title_probs(teams: Sequence[Team], n_sims: int = 1_000_000,
                            seed: int = 12345) -> np.ndarray:
    """Vectorised NumPy reference simulator.

    Simulates all `n_sims` brackets in lockstep rather than one at a time: the
    survivor array is (n_sims, alive) and each round halves its width.  Same
    answer as a per-simulation loop, ~100x faster in Python, and it is the
    implementation the C++ version is cross-checked against.
    """
    n = _check_bracket(teams)
    p_beats = win_matrix(teams)
    rng = np.random.default_rng(seed)

    alive = np.tile(np.arange(n), (n_sims, 1))
    while alive.shape[1] > 1:
        a, b = alive[:, 0::2], alive[:, 1::2]
        p = p_beats[a, b]
        alive = np.where(rng.random(p.shape) < p, a, b)
    return np.bincount(alive.ravel(), minlength=n) / n_sims


def mc_standard_error(p: np.ndarray, n_sims: int) -> np.ndarray:
    """Binomial standard error of each MC estimate -- the yardstick for how
    much of a gap against the exact answer is acceptable."""
    return np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / n_sims)


def sims_for_precision(p: float, abs_error: float, z: float = 1.96) -> int:
    """How many simulations to pin a probability to +/- abs_error.

    n = z^2 p(1-p) / e^2.  Worth internalising: resolving a 15% favourite to
    +/-0.1 percentage points needs ~490,000 sims, and to +/-0.01pp needs ~49
    million.  Error falls as 1/sqrt(N), so each extra decimal digit costs 100x
    the compute -- which is precisely the argument for the exact DP whenever
    the model is simple enough to admit one.
    """
    return int(math.ceil(z * z * p * (1.0 - p) / (abs_error ** 2)))


def round_by_round(teams: Sequence[Team]) -> np.ndarray:
    """(n, rounds+1) matrix: P(team i is still alive entering each round).
    Column 0 is all ones; the last column is the title probability."""
    n = _check_bracket(teams)
    p_beats = win_matrix(teams)
    rounds = int(math.log2(n))
    out = np.ones((n, rounds + 1))
    reach = np.ones(n)
    block = 1
    for r in range(rounds):
        nxt = np.empty(n)
        for i in range(n):
            base = (i // block) * block
            opp_base = base + block if (i // block) % 2 == 0 else base - block
            opp = slice(opp_base, opp_base + block)
            nxt[i] = reach[i] * float(np.dot(reach[opp], p_beats[i, opp]))
        reach = nxt
        out[:, r + 1] = reach
        block *= 2
    return out

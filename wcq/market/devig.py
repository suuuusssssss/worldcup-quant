"""Turning quoted odds into a fair probability distribution.

The reciprocal of decimal odds is not a probability -- the three reciprocals
sum to roughly 1.06, and that 6% is the bookmaker's margin.  How you remove it
is not a detail.  Our signal is `model_prob - fair_prob`, a difference between
two numbers that are typically within a few points of each other, so a de-vig
method that is biased by 1-2% on longshots can invent or erase the entire edge.

Four methods, in increasing order of realism:

multiplicative  Scale all reciprocals by the same factor.  Implicitly assumes
                margin is spread proportionally.  Simple, and wrong in a known
                direction: it under-removes margin from favourites and
                over-removes it from longshots.
additive        Subtract the margin equally in probability space.  Errs the
                opposite way and can produce negative probabilities on heavy
                favourites.
power           Find k with sum(r_i^k) = 1.  A one-parameter family that
                handles the favourite-longshot skew far better.
shin            Solves for the fraction of informed money z that a
                market-maker prices against.  Has an actual economic story
                behind it rather than being a curve that happens to fit, and
                is the closest of the four to what books appear to do.

The favourite-longshot bias is real and well documented: longshots are
systematically overpriced relative to their true chance.  Since our strategy
mostly fires on underdogs (where model and market disagree most), the choice
between `multiplicative` and `shin` can flip the sign of the backtest.  The
harness therefore reports results under more than one method rather than
picking a flattering one.
"""
from __future__ import annotations

import math
from typing import Sequence

METHODS = ("multiplicative", "additive", "power", "shin")


def _reciprocals(odds: Sequence[float]) -> list[float]:
    if any(o is None or o <= 1.0 for o in odds):
        raise ValueError(f"decimal odds must exceed 1.0, got {odds}")
    return [1.0 / o for o in odds]


def multiplicative(odds: Sequence[float]) -> tuple[float, ...]:
    r = _reciprocals(odds)
    s = sum(r)
    return tuple(x / s for x in r)


def additive(odds: Sequence[float]) -> tuple[float, ...]:
    r = _reciprocals(odds)
    excess = (sum(r) - 1.0) / len(r)
    p = [x - excess for x in r]
    if min(p) <= 0:                       # degenerate on extreme favourites
        return multiplicative(odds)
    s = sum(p)
    return tuple(x / s for x in p)


def power(odds: Sequence[float], tol: float = 1e-12, max_iter: int = 200) -> tuple[float, ...]:
    """Solve sum(r_i ** k) = 1 for k by bisection.

    Monotone in k, so bisection is guaranteed to converge and cannot blow up
    the way Newton can when a reciprocal is near 1.  ~50 iterations to machine
    precision is nothing next to the cost of loading the data.
    """
    r = _reciprocals(odds)
    lo, hi = 0.5, 3.0
    f = lambda k: sum(x ** k for x in r) - 1.0
    if f(lo) * f(hi) > 0:
        return multiplicative(odds)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if abs(v) < tol:
            break
        if v > 0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    p = [x ** k for x in r]
    s = sum(p)
    return tuple(x / s for x in p)


def shin(odds: Sequence[float], tol: float = 1e-12, max_iter: int = 200) -> tuple[float, ...]:
    """Shin (1993): price-setting against a proportion z of informed traders.

    p_i = (sqrt(z^2 + 4(1-z) r_i^2 / S) - z) / (2(1-z)),  S = sum r_i

    Solve for z in [0, 1) such that the probabilities sum to 1.  Bisection
    again, for the same robustness reason as `power`.
    """
    r = _reciprocals(odds)
    S = sum(r)
    if S <= 1.0:
        # Sub-100% book: there is no margin to remove, so Shin's insider
        # parameter is undefined.  Normalising is the only sensible answer.
        # This is common for best-of-market lines and is why an edge computed
        # against aggregated best prices is not an edge you can trade.
        return multiplicative(odds)

    def probs(z: float) -> list[float]:
        if z <= 0:
            return [x / S for x in r]
        return [(math.sqrt(z * z + 4.0 * (1.0 - z) * x * x / S) - z) / (2.0 * (1.0 - z)) for x in r]

    lo, hi = 0.0, 0.9
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = sum(probs(mid))
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    p = probs(0.5 * (lo + hi))
    t = sum(p)
    return tuple(x / t for x in p)


_DISPATCH = {
    "multiplicative": multiplicative,
    "additive": additive,
    "power": power,
    "shin": shin,
}


def fair_probs(odds: Sequence[float], method: str = "shin") -> tuple[float, ...]:
    try:
        fn = _DISPATCH[method]
    except KeyError:
        raise ValueError(f"unknown de-vig method {method!r}; choose from {METHODS}") from None
    return fn(odds)


def margin(odds: Sequence[float]) -> float:
    return sum(_reciprocals(odds)) - 1.0

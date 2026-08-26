"""Position sizing.

Kelly maximises the expected log of terminal bankroll, which is the growth-
optimal choice for a repeated game with reinvestment.  For a single binary bet
at decimal odds `o` with true win probability `p`:

    f* = (p*(o-1) - (1-p)) / (o-1) = (p*o - 1) / (o - 1)

Everything painful about Kelly follows from one fact: it assumes `p` is known.
It is not -- `p` is an estimate with error, and the penalty is asymmetric.
Overestimating your edge makes you overbet, and the growth rate falls off far
faster above f* than below it.  In the continuous / Gaussian approximation the
growth rate hits zero at exactly 2*f*; for a discrete binary bet the zero
crossing sits very close to 2*f* but not exactly on it, and past it the growth
rate is negative even though every individual bet still has positive expected
value.  Meanwhile half-Kelly keeps 3/4 of the growth rate -- exactly, in the
same approximation -- for half the volatility.  Giving up 25% of growth to
halve your exposure to being wrong about p is the trade practitioners take,
and it is why nothing in this repo defaults to full Kelly.

This module therefore supports three separate brakes, because they defend
against different failures:

  fraction   Bet lambda * f*.  Half-Kelly gives up ~25% of the growth rate for
             roughly half the volatility and a much shallower drawdown.
  shrinkage  Pull the probability estimate toward the market before sizing.
             Defends against model error specifically, rather than against
             variance in general -- if the market is right and you are wrong,
             a smaller fraction still bets the wrong way.
  cap        Hard ceiling per position.  Defends against the tail where a
             single bad probability produces an enormous f*, which is a
             software failure mode as much as a statistical one.
"""
from __future__ import annotations

from dataclasses import dataclass


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll.  Zero if no edge."""
    b = decimal_odds - 1.0
    if b <= 0.0:
        return 0.0
    f = (p * decimal_odds - 1.0) / b
    return max(f, 0.0)


def expected_log_growth(p: float, decimal_odds: float, f: float) -> float:
    """E[log(bankroll multiple)] for staking fraction f.

    Used in tests to assert that f* really is the maximiser, and to locate the
    zero-crossing just past 2*f* that motivates fractional Kelly.
    """
    import math
    if f <= 0.0:
        return 0.0
    if f >= 1.0:
        return float("-inf")
    return p * math.log(1.0 + f * (decimal_odds - 1.0)) + (1.0 - p) * math.log(1.0 - f)


@dataclass(frozen=True, slots=True)
class SizingPolicy:
    fraction: float = 0.25      # quarter-Kelly by default; half is aggressive
    cap: float = 0.02           # never risk more than 2% of bankroll on one bet
    shrinkage: float = 0.0      # 0 = trust the model fully, 1 = defer to market
    min_edge: float = 0.02      # do not trade inside the noise

    def stake(self, model_prob: float, fair_prob: float, decimal_odds: float) -> float:
        """Return the bankroll fraction to risk; 0 means no trade."""
        p = (1.0 - self.shrinkage) * model_prob + self.shrinkage * fair_prob
        if p - fair_prob < self.min_edge:
            return 0.0
        f = kelly_fraction(p, decimal_odds)
        return min(self.fraction * f, self.cap)


def kelly_multi(probs, odds, fraction: float = 1.0) -> list[float]:
    """Simultaneous stakes across mutually exclusive outcomes of one event.

    Betting three outcomes of the same match independently is wrong: the
    outcomes are exclusive, so the positions hedge each other and the true
    joint-optimal stake is not the sum of three separate Kelly fractions.
    This solves the exclusive-outcomes case directly by the standard
    reserve-set algorithm: sort by expected value, admit outcomes while their
    revised edge stays positive, and size against the remaining reserve.
    """
    n = len(probs)
    order = sorted(range(n), key=lambda i: probs[i] * odds[i], reverse=True)
    admitted: list[int] = []
    reserve = 1.0
    p_sum = 0.0
    for i in order:
        if probs[i] * odds[i] <= 1.0:
            break
        cand = admitted + [i]
        p_tot = p_sum + probs[i]
        r_tot = sum(1.0 / odds[j] for j in cand)
        if r_tot >= 1.0:
            break
        new_reserve = (1.0 - p_tot) / (1.0 - r_tot)
        if new_reserve <= 0.0:
            break
        admitted, reserve, p_sum = cand, new_reserve, p_tot

    stakes = [0.0] * n
    for i in admitted:
        stakes[i] = max(0.0, fraction * (probs[i] - reserve / odds[i]))
    return stakes

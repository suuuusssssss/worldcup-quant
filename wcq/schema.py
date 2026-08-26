"""Core record types.

One `Match` type is shared by every loader so the model, backtest and
simulator never learn where the data came from.  Adding a new data source
means writing one function that yields `Match` objects; nothing downstream
changes.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

Outcome = str  # 'H' | 'D' | 'A'
OUTCOMES: tuple[str, str, str] = ("H", "D", "A")
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


@dataclass(frozen=True, slots=True)
class Odds:
    """Decimal odds for the 1X2 market, plus provenance.

    `taken` records *when* the price was observed.  The distinction between an
    opening line and a closing line matters enormously: the closing line is the
    market's most informed price and is the only honest benchmark for a
    strategy that would have traded before kickoff.  Anything that only beats
    the *opening* line is usually just slow to react, not right.
    """

    home: float
    draw: float
    away: float
    book: str = "unknown"
    taken: str = "prematch"  # 'open' | 'close' | 'prematch' | 'best'

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.home, self.draw, self.away)

    @property
    def overround(self) -> float:
        """Sum of implied probabilities.  1.05 means a 5% bookmaker margin."""
        return 1.0 / self.home + 1.0 / self.draw + 1.0 / self.away

    def is_valid(self) -> bool:
        """Data-integrity check only: finite decimal odds above 1.0.

        Deliberately does NOT require overround > 1.  A best-of-market line
        aggregated across ~17 books routinely sums to *under* 100% -- a
        theoretical cross-book arbitrage nobody can actually execute at size.
        An earlier version of this rejected those rows as malformed, which
        silently deleted precisely the matches where bookmakers disagreed
        most, i.e. the highest-variance and most interesting part of the
        sample.  That is a selection bias pointed in the worst possible
        direction, and it produced no error and no warning.

        Margin belongs to analysis, not parsing.  Use `has_positive_margin`
        to filter deliberately, and see `Odds.overround`.
        """
        return all(v is not None and v == v and v > 1.0 for v in self.as_tuple())

    @property
    def has_positive_margin(self) -> bool:
        """False for a sub-100% book -- normal for aggregated best prices,
        essentially impossible for a single book's own line."""
        return self.overround > 1.0


@dataclass(frozen=True, slots=True)
class Match:
    """A single completed fixture.

    Deliberately minimal.  Ratings are NOT stored here -- they are derived by
    the Elo engine in a causal pass, so a `Match` can never smuggle a
    post-match rating into a pre-match prediction.
    """

    date: dt.date
    home: str
    away: str
    competition: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    neutral: bool = False
    odds: Optional[Odds] = None
    source: str = ""

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def result(self) -> Optional[Outcome]:
        if not self.played:
            return None
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals == self.away_goals:
            return "D"
        return "A"

    def key(self) -> tuple:
        """Stable identity, used to deduplicate across overlapping sources."""
        return (self.date, self.home, self.away, self.competition)


@dataclass(slots=True)
class Prediction:
    """A model quote for one match, produced strictly from prior information."""

    match: Match
    probs: tuple[float, float, float]  # (H, D, A), sums to 1
    home_rating: float
    away_rating: float
    n_prior_home: int = 0  # matches the engine had seen for this team
    n_prior_away: int = 0

    def p(self, outcome: Outcome) -> float:
        return self.probs[OUTCOME_INDEX[outcome]]


@dataclass(slots=True)
class Bet:
    """A hypothetical position taken against a bookmaker price."""

    match: Match
    outcome: Outcome
    model_prob: float
    fair_prob: float       # vig-stripped market probability
    price: float           # decimal odds actually available
    edge: float            # model_prob - fair_prob
    stake: float           # fraction of bankroll
    won: Optional[bool] = None
    pnl: Optional[float] = None  # in bankroll fractions

    @property
    def unit_return(self) -> Optional[float]:
        """PnL per unit staked -- the quantity whose mean we test."""
        if self.pnl is None or self.stake <= 0:
            return None
        return self.pnl / self.stake


def sort_chronologically(matches: Sequence[Match]) -> list[Match]:
    """Total order by date, with a deterministic tiebreak.

    Sorting is the load-bearing precondition for every point-in-time guarantee
    in this codebase, so it lives in one place and is used everywhere.
    """
    return sorted(matches, key=lambda m: (m.date, m.competition, m.home, m.away))


def dedupe(matches: Iterable[Match]) -> list[Match]:
    seen: set[tuple] = set()
    out: list[Match] = []
    for m in matches:
        k = m.key()
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    return out

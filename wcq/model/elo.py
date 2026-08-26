"""Point-in-time Elo.

Why compute ratings instead of downloading them
-----------------------------------------------
A published rating table is a snapshot series, and joining a match to "the
nearest snapshot" is exactly where lookahead bias creeps in: a snapshot dated
the 15th already contains results from the 1st through the 14th, and if your
match is on the 10th you have just told the model the answer.  Computing the
ratings here means the guarantee is structural rather than a join condition
somebody has to remember to get right.

The engine exposes one primitive:

    snap = engine.observe(match)   # ratings BEFORE this match
    engine.update(match)           # now fold the result in

`stream()` fuses the two in the only legal order, so downstream code cannot
accidentally rate a match with its own outcome.  `test_elo.py` asserts this
with a property test: shuffling future results must not change any snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from wcq.schema import Match


@dataclass(frozen=True, slots=True)
class EloConfig:
    k: float = 20.0
    """Base learning rate.  Higher = faster adaptation, noisier ratings."""

    home_advantage: float = 65.0
    """Home edge in Elo points.  ~65 for club football, ~0 at a neutral venue."""

    initial: float = 1500.0
    """Rating for a team the engine has never seen."""

    goal_diff_scaling: bool = True
    """Scale K by margin of victory (World Football Elo convention).  Without
    it a 5-0 and a 1-0 move the rating identically, which throws away most of
    the signal in the scoreline."""

    season_regression: float = 0.0
    """Fraction pulled back toward `initial` at a year boundary, in [0,1].
    Squads turn over; a rating earned three years ago is stale.  0 disables."""


@dataclass(frozen=True, slots=True)
class RatingSnapshot:
    """Ratings as of immediately before a given match."""
    home: float
    away: float
    n_home: int
    n_away: int
    diff: float          # home - away, home advantage already applied

    @property
    def burned_in(self) -> bool:
        """True once both teams have enough history for the rating to mean
        anything.  Predictions below this are dropped from evaluation rather
        than silently degrading the metrics."""
        return self.n_home >= 5 and self.n_away >= 5


def _margin_multiplier(goal_diff: int) -> float:
    """World Football Elo margin-of-victory multiplier."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


class EloEngine:
    """Streaming Elo. O(1) per match, O(#teams) memory."""

    __slots__ = ("cfg", "_r", "_n", "_last_year", "_last_date")

    def __init__(self, cfg: EloConfig | None = None):
        self.cfg = cfg or EloConfig()
        self._r: dict[str, float] = {}
        self._n: dict[str, int] = {}
        self._last_year: dict[str, int] = {}
        self._last_date = None

    # ---- read side -------------------------------------------------------
    def rating(self, team: str) -> float:
        return self._r.get(team, self.cfg.initial)

    def games(self, team: str) -> int:
        return self._n.get(team, 0)

    def table(self) -> dict[str, float]:
        return dict(self._r)

    def observe(self, match: Match) -> RatingSnapshot:
        """Pre-match ratings.  Read-only: calling this twice is identical."""
        rh = self._regressed(match.home, match.date.year)
        ra = self._regressed(match.away, match.date.year)
        adv = 0.0 if match.neutral else self.cfg.home_advantage
        return RatingSnapshot(rh, ra, self.games(match.home), self.games(match.away), rh - ra + adv)

    def _regressed(self, team: str, year: int) -> float:
        """Apply between-season mean reversion lazily, on first sight in a new
        year, so an inactive team is not repeatedly regressed."""
        r = self._r.get(team, self.cfg.initial)
        f = self.cfg.season_regression
        if f <= 0.0 or team not in self._last_year:
            return r
        gap = year - self._last_year[team]
        if gap <= 0:
            return r
        pull = 1.0 - (1.0 - f) ** gap
        return r + pull * (self.cfg.initial - r)

    # ---- write side ------------------------------------------------------
    def update(self, match: Match) -> None:
        """Fold a completed result into the ratings.  Must be called in
        chronological order; `stream()` guarantees that."""
        if not match.played:
            return
        if self._last_date is not None and match.date < self._last_date:
            raise ValueError(
                f"out-of-order match {match.date} after {self._last_date}: "
                "point-in-time ratings require a chronological stream"
            )
        self._last_date = match.date

        snap = self.observe(match)
        exp_h = expected_score(snap.diff)
        gd = match.home_goals - match.away_goals
        actual = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k = self.cfg.k * (_margin_multiplier(gd) if self.cfg.goal_diff_scaling else 1.0)
        delta = k * (actual - exp_h)

        self._r[match.home] = snap.home + delta
        self._r[match.away] = snap.away - delta       # zero-sum
        self._n[match.home] = self._n.get(match.home, 0) + 1
        self._n[match.away] = self._n.get(match.away, 0) + 1
        self._last_year[match.home] = self._last_year[match.away] = match.date.year

    # ---- the only sanctioned traversal ----------------------------------
    def stream(self, matches: Iterable[Match]) -> Iterator[tuple[Match, RatingSnapshot]]:
        """Yield (match, pre-match ratings) then absorb the result.

        Every consumer in this repo goes through here.  It is the single point
        where the observe-before-update ordering is enforced.
        """
        for m in matches:
            snap = self.observe(m)
            yield m, snap
            self.update(m)


def expected_score(elo_diff: float) -> float:
    """Standard logistic Elo expectation for the home side, in [0,1]."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

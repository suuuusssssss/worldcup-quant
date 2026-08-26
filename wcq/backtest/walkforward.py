"""Walk-forward evaluation and betting backtest.

The rule the whole harness exists to enforce: a prediction for a match on date
t may only use information available strictly before t.

Two moving parts, both causal:

* Elo ratings update continuously as results arrive.  Causality is structural
  here -- `EloEngine.stream` hands out the snapshot before folding in the
  result, so a rating can never contain its own match.
* Poisson parameters are refit on a schedule (default: annually) using only
  matches already seen, then *frozen* for the whole next period.  Refitting
  per-match would be both prohibitively slow and unrealistic; nobody
  re-estimates a model between two Saturday fixtures.

The single most common way to fake a good backtest is to fit on everything and
then "test" on a slice of it.  The structure below makes that awkward to do by
accident: training rows are only appended after the period that used them has
already been scored.
"""
from __future__ import annotations

from wcq._compat import SLOTS

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from wcq.market.devig import fair_probs
from wcq.market.kelly import SizingPolicy
from wcq.model.calibrate import TrainingSet, build_training_set, fit
from wcq.model.elo import EloConfig, EloEngine
from wcq.model.poisson import PoissonParams, match_probs
from wcq.schema import OUTCOMES, OUTCOME_INDEX, Bet, Match, Prediction


@dataclass(frozen=True, **SLOTS)
class WalkForwardConfig:
    refit_every_days: int = 365
    min_train_matches: int = 2_000
    rolling_window_matches: int = 0
    """0 = expanding window (use all history).  A positive value keeps only the
    most recent N matches, which trades statistical power for adaptivity when
    the data-generating process drifts -- scoring rates and home advantage in
    football have both moved measurably over 25 years."""

    burn_in_matches: int = 5
    """Skip predictions for teams with fewer than this many prior matches; an
    Elo rating with two games behind it is a prior, not an estimate."""

    elo: EloConfig = field(default_factory=EloConfig)


@dataclass
class WalkForwardResult:
    predictions: list[Prediction]
    param_history: list[tuple[dt.date, PoissonParams, dict]]
    skipped: int

    @property
    def probs(self) -> np.ndarray:
        return np.array([p.probs for p in self.predictions], dtype=float)

    @property
    def actual_idx(self) -> np.ndarray:
        return np.array([OUTCOME_INDEX[p.match.result] for p in self.predictions], dtype=np.int64)

    def with_odds(self) -> list[Prediction]:
        return [p for p in self.predictions if p.match.odds is not None]


def run_walk_forward(matches: Sequence[Match], cfg: WalkForwardConfig | None = None,
                     progress: Optional[Callable[[int, int], None]] = None) -> WalkForwardResult:
    cfg = cfg or WalkForwardConfig()
    engine = EloEngine(cfg.elo)

    train_rows: list[tuple[float, int, int, bool]] = []
    params: PoissonParams | None = None
    param_history: list[tuple[dt.date, PoissonParams, dict]] = []
    next_refit: dt.date | None = None
    pending: list[tuple[dt.date, tuple[float, int, int, bool]]] = []

    predictions: list[Prediction] = []
    skipped = 0
    total = len(matches)

    for i, (m, snap) in enumerate(engine.stream(matches)):
        if progress and i % 20_000 == 0:
            progress(i, total)

        # Fixtures without a result cannot train or be scored.  Guarding here
        # keeps a scheduled-but-unplayed row from planting a None goal count
        # in the training set, where it would blow up (or worse, not) three
        # calls away from the cause.
        if not m.played:
            skipped += 1
            continue

        # -- refit boundary: fold in everything strictly older than today,
        #    then re-estimate on data that is now firmly in the past.  Rows
        #    from *this* date stay pending -- within a date the sort order is
        #    alphabetical, not temporal, so "earlier today" is not "earlier".
        if next_refit is None or m.date >= next_refit:
            still_pending = [(d, r) for d, r in pending if d >= m.date]
            train_rows.extend(r for d, r in pending if d < m.date)
            pending = still_pending
            if len(train_rows) >= cfg.min_train_matches:
                window = train_rows
                if cfg.rolling_window_matches:
                    # The window never undercuts the training floor: a rolling
                    # window smaller than min_train_matches would silently
                    # re-enable exactly the small-sample fits the floor exists
                    # to prevent.
                    keep = max(cfg.rolling_window_matches, cfg.min_train_matches)
                    window = train_rows[-keep:]
                params, info = fit(build_training_set(window), start=params)
                param_history.append((m.date, params, info))
            next_refit = m.date + dt.timedelta(days=cfg.refit_every_days)

        row = (snap.diff, m.home_goals, m.away_goals, m.neutral)

        if params is None or snap.n_home < cfg.burn_in_matches or snap.n_away < cfg.burn_in_matches:
            skipped += 1
            pending.append((m.date, row))
            continue

        probs = match_probs(snap.diff, params, m.neutral)
        predictions.append(Prediction(
            match=m, probs=probs, home_rating=snap.home, away_rating=snap.away,
            n_prior_home=snap.n_home, n_prior_away=snap.n_away,
        ))
        pending.append((m.date, row))

    return WalkForwardResult(predictions, param_history, skipped)


def generate_bets(predictions: Iterable[Prediction], policy: SizingPolicy | None = None,
                  devig: str = "shin", one_bet_per_match: bool = True) -> list[Bet]:
    """Turn model quotes into settled hypothetical positions.

    `one_bet_per_match` keeps at most the single best edge per fixture.  Taking
    two of three exclusive outcomes on the same match is not two independent
    positions -- they partially hedge, and counting them as two observations
    inflates the apparent sample size, which is exactly the quantity every
    significance test depends on.

    With `one_bet_per_match=False` each qualifying outcome is sized by the
    independent single-outcome Kelly formula, which overstates the joint
    stake on exclusive outcomes; `wcq.market.kelly.kelly_multi` solves the
    joint problem properly if simultaneous legs are actually wanted.  The
    clustered bootstrap treats the match as the resampling unit either way.
    """
    policy = policy or SizingPolicy()
    bets: list[Bet] = []

    for pred in predictions:
        m = pred.match
        if m.odds is None or not m.odds.is_valid() or m.result is None:
            continue
        prices = m.odds.as_tuple()
        fair = fair_probs(prices, devig)

        candidates: list[Bet] = []
        for outcome in OUTCOMES:
            k = OUTCOME_INDEX[outcome]
            edge = pred.probs[k] - fair[k]
            stake = policy.stake(pred.probs[k], fair[k], prices[k])
            if stake <= 0.0:
                continue
            won = (m.result == outcome)
            pnl = stake * (prices[k] - 1.0) if won else -stake
            candidates.append(Bet(
                match=m, outcome=outcome, model_prob=pred.probs[k], fair_prob=fair[k],
                price=prices[k], edge=edge, stake=stake, won=won, pnl=pnl,
            ))

        if not candidates:
            continue
        if one_bet_per_match:
            bets.append(max(candidates, key=lambda b: b.edge))
        else:
            bets.extend(candidates)

    return bets

"""End-to-end causality tests for the backtest harness."""
import datetime as dt

import pytest

from conftest import make_match
from wcq.backtest.walkforward import WalkForwardConfig, generate_bets, run_walk_forward
from wcq.market.kelly import SizingPolicy
from wcq.model.elo import EloConfig
from wcq.schema import Odds


CFG = WalkForwardConfig(refit_every_days=120, min_train_matches=100,
                        burn_in_matches=3, elo=EloConfig(k=20, home_advantage=50))


def test_produces_valid_probabilities(synthetic_season):
    res = run_walk_forward(synthetic_season, CFG)
    assert res.predictions
    for p in res.predictions:
        assert sum(p.probs) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 < x < 1.0 for x in p.probs)


def test_beats_a_uniform_forecast(synthetic_season):
    import numpy as np
    from wcq.backtest import metrics
    res = run_walk_forward(synthetic_season, CFG)
    assert metrics.log_loss(res.probs, res.actual_idx) < 1.0986


def test_burn_in_matches_are_skipped(synthetic_season):
    res = run_walk_forward(synthetic_season, CFG)
    assert res.skipped > 0
    assert all(p.n_prior_home >= CFG.burn_in_matches for p in res.predictions)
    assert all(p.n_prior_away >= CFG.burn_in_matches for p in res.predictions)


def test_predictions_do_not_change_when_the_future_is_rewritten(synthetic_season):
    """The harness-level leakage test.

    Rewrite every result after a cut point and re-run.  Predictions at or
    before the cut must be identical to the last bit.  This covers the whole
    pipeline -- Elo state, the refit schedule, the training-row accumulation --
    not just the rating engine.
    """
    import random
    cut = len(synthetic_season) // 2
    base = run_walk_forward(synthetic_season, CFG).predictions

    rng = random.Random(7)
    tampered = list(synthetic_season[:cut])
    for m in synthetic_season[cut:]:
        tampered.append(make_match((m.date - dt.date(2020, 1, 1)).days, m.home, m.away,
                                   rng.randint(0, 5), rng.randint(0, 5)))
    after = run_walk_forward(tampered, CFG).predictions

    keys = {m.key() for m in synthetic_season[:cut]}
    a = [(p.match.key(), p.probs) for p in base if p.match.key() in keys]
    b = [(p.match.key(), p.probs) for p in after if p.match.key() in keys]
    assert a == b, "rewriting future results changed a past prediction"


def test_unplayed_fixtures_are_skipped_not_trained_on():
    """A scheduled-but-unplayed fixture has no goals.  It must be counted and
    stepped over -- not scored, and above all not appended to the training
    rows, where a None goal count would detonate inside the MLE three calls
    away from the cause."""
    from wcq.schema import Match
    ms = []
    for d in range(300):
        ms.append(make_match(d, f"T{d % 8}", f"T{(d + 3) % 8}", d % 4, (d + 1) % 3))
    ms.append(Match(date=dt.date(2020, 1, 1) + dt.timedelta(days=300),
                    home="T0", away="T1", competition="T"))     # unplayed
    for d in range(301, 340):
        ms.append(make_match(d, f"T{d % 8}", f"T{(d + 3) % 8}", d % 4, (d + 1) % 3))

    with_fixture = run_walk_forward(ms, CFG)
    without = run_walk_forward([m for m in ms if m.played], CFG)
    assert all(p.match.played for p in with_fixture.predictions)
    assert with_fixture.skipped == without.skipped + 1
    a = [(p.match.key(), p.probs) for p in with_fixture.predictions]
    b = [(p.match.key(), p.probs) for p in without.predictions]
    assert a == b, "an unplayed fixture changed the played matches' predictions"


def test_refits_happen_and_parameters_move(synthetic_season):
    res = run_walk_forward(synthetic_season, CFG)
    assert len(res.param_history) >= 2
    assert all(info["success"] for _, _, info in res.param_history)


def test_no_bets_without_prices(synthetic_season):
    res = run_walk_forward(synthetic_season, CFG)
    assert generate_bets(res.predictions) == []      # fixture carries no odds


def test_bets_settle_consistently_with_results():
    from wcq.schema import Match
    ms = []
    for d in range(400):
        h, a = f"T{d % 9}", f"T{(d + 4) % 9}"
        ms.append(make_match(d, h, a, d % 4, (d + 1) % 3,
                             odds=(2.30, 3.40, 3.10)))
    res = run_walk_forward(ms, CFG)
    bets = generate_bets(res.with_odds(), SizingPolicy(min_edge=0.0, cap=0.05))
    assert bets
    for b in bets:
        assert b.won == (b.match.result == b.outcome)
        assert b.pnl == pytest.approx(b.stake * (b.price - 1) if b.won else -b.stake)
        assert b.unit_return == pytest.approx((b.price - 1) if b.won else -1.0)


def test_one_bet_per_match_is_enforced():
    ms = [make_match(d, f"A{d%7}", f"B{d%5}", d % 3, (d + 2) % 3, odds=(2.5, 3.3, 2.9))
          for d in range(400)]
    res = run_walk_forward(ms, CFG)
    bets = generate_bets(res.with_odds(), SizingPolicy(min_edge=0.0, cap=0.05),
                         one_bet_per_match=True)
    keys = [b.match.key() for b in bets]
    assert len(keys) == len(set(keys))

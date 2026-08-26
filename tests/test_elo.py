"""The Elo engine's job is to be causal.  These tests attack that directly."""
import datetime as dt
import random

import pytest

from conftest import make_match
from wcq.model.elo import EloConfig, EloEngine, expected_score, _margin_multiplier


def test_expected_score_symmetry():
    assert expected_score(0.0) == pytest.approx(0.5)
    assert expected_score(400.0) == pytest.approx(10 / 11)
    assert expected_score(-400.0) == pytest.approx(1 / 11)
    for d in (0, 50, 200, 800):
        assert expected_score(d) + expected_score(-d) == pytest.approx(1.0)


def test_rating_is_zero_sum():
    e = EloEngine(EloConfig(k=20, home_advantage=0))
    e.update(make_match(0, "A", "B", 3, 0))
    assert e.rating("A") + e.rating("B") == pytest.approx(3000.0)


def test_margin_multiplier_monotone():
    vals = [_margin_multiplier(g) for g in range(0, 8)]
    assert vals[0] == vals[1] == 1.0
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_bigger_win_moves_rating_further():
    def after(hg, ag):
        e = EloEngine(EloConfig(k=20, home_advantage=0))
        e.update(make_match(0, "A", "B", hg, ag))
        return e.rating("A")
    assert after(5, 0) > after(2, 0) > after(1, 0)


def test_out_of_order_stream_is_rejected():
    """Silent acceptance of an out-of-order match would break every
    point-in-time guarantee downstream, so it must be loud."""
    e = EloEngine()
    e.update(make_match(10, "A", "B", 1, 0))
    with pytest.raises(ValueError, match="out-of-order"):
        e.update(make_match(2, "C", "D", 1, 0))


def test_observe_is_pure():
    e = EloEngine()
    m = make_match(0, "A", "B", 2, 1)
    assert e.observe(m) == e.observe(m)
    assert e.rating("A") == 1500.0          # observing must not mutate


# -- the important one ------------------------------------------------------

def test_no_lookahead_under_future_permutation():
    """THE leakage test.

    Take a fixture list, record the pre-match snapshot for every match, then
    rebuild the history with all matches after index i replaced by different
    results.  Every snapshot up to and including i must be bit-identical.  If
    any future information reached a past rating -- through a join, a global
    fit, a mutable default, a sort that is not stable -- this test fails.
    """
    rng = random.Random(99)
    teams = [f"T{i}" for i in range(8)]
    base = []
    for d in range(200):
        h, a = rng.sample(teams, 2)
        base.append(make_match(d, h, a, rng.randint(0, 4), rng.randint(0, 4)))

    def snapshots(ms):
        e = EloEngine(EloConfig(k=25, home_advantage=60))
        return [(m.key(), s.home, s.away) for m, s in e.stream(ms)]

    truth = snapshots(base)

    for cut in (0, 1, 37, 120, 199):
        rng2 = random.Random(cut + 5)
        tampered = list(base[: cut + 1])
        for m in base[cut + 1:]:
            tampered.append(make_match(
                (m.date - dt.date(2020, 1, 1)).days, m.home, m.away,
                rng2.randint(0, 6), rng2.randint(0, 6)))
        assert snapshots(tampered)[: cut + 1] == truth[: cut + 1], (
            f"future results leaked into snapshots at or before index {cut}")


def test_season_regression_pulls_toward_initial():
    cfg = EloConfig(k=20, home_advantage=0, season_regression=0.5, initial=1500.0)
    e = EloEngine(cfg)
    for d in range(10):
        e.update(make_match(d, "A", "B", 4, 0))
    hot = e.rating("A")
    assert hot > 1500.0
    later = e.observe(make_match(0, "A", "B", 0, 0).__class__(
        date=dt.date(2022, 6, 1), home="A", away="B", competition="T",
        home_goals=0, away_goals=0))
    assert 1500.0 < later.home < hot

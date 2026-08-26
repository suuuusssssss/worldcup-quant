"""The Elo engine's job is to be causal.  These tests attack that directly."""
import datetime as dt
import random

import pytest

from conftest import make_match
from wcq.model.elo import (EloConfig, EloEngine, expected_score, international_k,
                           _margin_multiplier)


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


def test_out_of_order_unplayed_fixture_is_also_rejected():
    """An unplayed fixture arriving out of order is the same data-integrity
    failure as a played one; skipping the guard for it would let a corrupted
    stream pass silently."""
    from wcq.schema import Match
    e = EloEngine()
    e.update(make_match(10, "A", "B", 1, 0))
    future_fixture = Match(date=dt.date(2020, 1, 3), home="C", away="D",
                           competition="T")          # no goals: not played
    with pytest.raises(ValueError, match="out-of-order"):
        e.update(future_fixture)


def test_stream_never_yields_the_out_of_order_match():
    """The guard has to fire before the offending match's snapshot escapes:
    yielding first and raising afterwards hands the consumer one contaminated
    prediction per corruption, which is how a 'guarded' leak still leaks."""
    ms = [make_match(0, "A", "B", 1, 0), make_match(5, "C", "D", 2, 2),
          make_match(3, "E", "F", 0, 1)]            # the offender
    e = EloEngine()
    seen = []
    with pytest.raises(ValueError, match="out-of-order"):
        for m, _ in e.stream(ms):
            seen.append(m.key())
    assert make_match(3, "E", "F", 0, 1).key() not in seen


def test_same_date_snapshots_are_day_start():
    """Kickoff times are not in the data, so within a date the sort order is
    alphabetical, not temporal.  A snapshot must therefore never contain a
    result from the match's own date -- otherwise a team's second listing on
    a date is predicted with its first result already folded in, which is
    lookahead in a thinner disguise."""
    e = EloEngine(EloConfig(k=20, home_advantage=0))
    warmup = make_match(0, "A", "X", 1, 1)
    first = make_match(5, "A", "B", 5, 0)     # big same-day win for A
    second = make_match(5, "C", "A", 0, 0)    # A appears again, same date
    snaps = {m.key(): s for m, s in e.stream([warmup, first, second])}
    assert snaps[second.key()].away == snaps[first.key()].home, (
        "the second same-day appearance saw a rating containing that day's result")
    # ...while the *stored* rating after the day reflects both results.
    assert e.rating("A") != snaps[first.key()].home


def test_importance_weighting_moves_big_matches_more():
    ks = {c: international_k(c) for c in
          ("FIFA World Cup", "FIFA World Cup qualification", "UEFA Euro",
           "Copa América", "UEFA Nations League", "Friendly", "Gold Cup")}
    assert ks["FIFA World Cup"] == 60.0
    assert ks["FIFA World Cup qualification"] == 40.0
    assert ks["UEFA Euro"] == ks["Copa América"] == 50.0
    assert ks["Friendly"] == 20.0

    def swing(comp):
        e = EloEngine(EloConfig(k=30, home_advantage=0, k_fn=international_k))
        e.update(make_match(0, "A", "B", 2, 0, comp=comp))
        return e.rating("A") - 1500.0

    assert swing("FIFA World Cup") == pytest.approx(3.0 * swing("Friendly"))


def test_table_with_year_applies_the_same_regression_as_observe():
    cfg = EloConfig(k=20, home_advantage=0, season_regression=0.3)
    e = EloEngine(cfg)
    for d in range(8):
        e.update(make_match(d, "A", "B", 3, 0))
    snap = e.observe(make_match(0, "A", "B", 0, 0).__class__(
        date=dt.date(2023, 6, 1), home="A", away="B", competition="T",
        home_goals=0, away_goals=0))
    assert e.table(year=2023)["A"] == pytest.approx(snap.home)
    assert e.rating("A", year=2023) == pytest.approx(snap.home)
    assert e.table()["A"] != pytest.approx(snap.home)    # unregressed raw view


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

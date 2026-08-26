import datetime as dt

import numpy as np
import pytest

from wcq.backtest.bootstrap import (cluster_bootstrap, deflated_p_value,
                                    stationary_block_bootstrap)
from wcq.schema import Bet, Match


def synth_bets(n, win_prob, price=2.60, seed=0, per_match=1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = Match(date=dt.date(2020, 1, 1) + dt.timedelta(days=i // 5),
                  home=f"H{i}", away=f"A{i}", competition="X",
                  home_goals=1, away_goals=0)
        for _ in range(per_match):
            won = bool(rng.random() < win_prob)
            stake = 0.01
            out.append(Bet(m, "H", 0.4, 0.38, price, 0.02, stake,
                           won, stake * (price - 1) if won else -stake))
    return out


def test_null_strategy_is_not_significant():
    """A break-even strategy (p * odds == 1) must not be flagged as edge."""
    bets = synth_bets(6000, win_prob=1 / 2.60, seed=11)
    r = cluster_bootstrap(bets, n_resamples=2000)
    assert not r.significant()
    assert r.roi_ci[0] < 0 < r.roi_ci[1]


def test_real_edge_is_detected():
    bets = synth_bets(6000, win_prob=0.45, price=2.60, seed=12)   # ~17% ROI
    r = cluster_bootstrap(bets, n_resamples=2000)
    assert r.significant()
    assert r.observed_roi > 0


def test_p_value_is_uniform_under_the_null():
    """The load-bearing property.  Under H0 a correctly built p-value is
    uniform on [0,1], so it should fall below 0.05 about 5% of the time.  The
    un-recentred version of this test -- counting resamples below zero --
    fails it badly, which is exactly why the recentring is there.
    """
    hits = 0
    trials = 60
    for s in range(trials):
        bets = synth_bets(1500, win_prob=1 / 2.60, seed=1000 + s)
        if cluster_bootstrap(bets, n_resamples=600, seed=s).p_value < 0.05:
            hits += 1
    assert hits <= 0.18 * trials, f"null rejected {hits}/{trials} times -- test is not calibrated"


def test_clustering_widens_intervals_when_bets_share_a_match():
    """Two bets on one fixture are not two observations.  Treating them as
    independent would tighten the interval spuriously; the clustered version
    must be at least as wide."""
    single = cluster_bootstrap(synth_bets(2000, 0.42, seed=5, per_match=1), 1500)
    paired = cluster_bootstrap(synth_bets(1000, 0.42, seed=5, per_match=2), 1500)
    w = lambda r: r.roi_ci[1] - r.roi_ci[0]
    assert w(paired) >= w(single) * 0.9


def test_block_bootstrap_agrees_with_cluster_on_iid_data():
    """With no serial dependence the two estimators must land in the same
    place; a large disagreement here means the block index construction is
    broken, which is a very easy bug to write and a very hard one to see."""
    bets = synth_bets(4000, 0.42, seed=21)
    a = cluster_bootstrap(bets, 2000)
    b = stationary_block_bootstrap(bets, 1500, mean_block=40)
    wa = a.roi_ci[1] - a.roi_ci[0]
    wb = b.roi_ci[1] - b.roi_ci[0]
    assert 0.6 < wb / wa < 1.7


def test_deflated_p_value_punishes_searching():
    assert deflated_p_value(0.03, 1) == pytest.approx(0.03)
    assert deflated_p_value(0.03, 40) > 0.5
    assert deflated_p_value(0.0, 100) == pytest.approx(0.0)


def test_empty_input_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        cluster_bootstrap([], 100)

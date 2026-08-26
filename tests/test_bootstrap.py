import datetime as dt

import numpy as np
import pytest

from wcq.backtest.bootstrap import (cluster_bootstrap, deflated_p_value,
                                    stationary_block_bootstrap)
from wcq.schema import Bet, Match


def synth_bets(n, win_prob, price=2.60, seed=0, per_match=1, shared_outcome=False,
               unique_matches=False):
    """`shared_outcome` draws one result per *match* so same-match bets are
    perfectly dependent -- the structure a clustered bootstrap exists for.
    `unique_matches` gives every bet its own match key, which degrades the
    clustered bootstrap to a bet-level one on purpose (for comparison)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = Match(date=dt.date(2020, 1, 1) + dt.timedelta(days=i // 5),
                  home=f"H{i}", away=f"A{i}", competition="X",
                  home_goals=1, away_goals=0)
        match_won = bool(rng.random() < win_prob)
        for j in range(per_match):
            won = match_won if shared_outcome else bool(rng.random() < win_prob)
            stake = 0.01
            mm = Match(date=m.date, home=m.home, away=m.away, competition=f"X{j}",
                       home_goals=1, away_goals=0) if unique_matches else m
            out.append(Bet(mm, "H", 0.4, 0.38, price, 0.02, stake,
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
    uniform on [0,1].  Checking only "not too many rejections" would also
    pass a p-value that is always 1.0 -- a broken test statistic can be
    conservative -- so this asserts a band on the rejection rate from both
    sides of the distribution and on the mean.
    """
    trials = 60
    ps = []
    for s in range(trials):
        bets = synth_bets(1500, win_prob=1 / 2.60, seed=1000 + s)
        ps.append(cluster_bootstrap(bets, n_resamples=600, seed=s).p_value)
    ps = np.asarray(ps)
    assert (ps < 0.05).mean() <= 0.18, f"null rejected {(ps < 0.05).sum()}/{trials} times"
    below_half = (ps < 0.5).mean()          # binomial(60, 0.5): 3 sigma ~ [0.31, 0.69]
    assert 0.28 <= below_half <= 0.72, f"p-values are not centred: P(p<0.5) = {below_half:.2f}"
    assert 0.38 <= ps.mean() <= 0.62, f"p-value mean {ps.mean():.3f} is far from uniform's 0.5"


def test_clustering_widens_intervals_when_bets_share_a_match():
    """Two bets on one fixture with the same outcome are ONE observation.

    The pairs of bets here are perfectly dependent (they settle on the same
    result), so the honest interval treats 2000 bets as 1000 observations:
    the clustered CI must be ~sqrt(2) wider than a bet-level resample of the
    identical P&L stream.  The bet-level comparison is built by giving every
    bet a unique match key, which reduces the same estimator to the naive
    one -- if the clustering ever silently broke, the two widths would agree
    and this test fails."""
    clustered = cluster_bootstrap(
        synth_bets(1000, 0.42, seed=5, per_match=2, shared_outcome=True), 2000)
    naive = cluster_bootstrap(
        synth_bets(1000, 0.42, seed=5, per_match=2, shared_outcome=True,
                   unique_matches=True), 2000)
    w = lambda r: r.roi_ci[1] - r.roi_ci[0]
    assert clustered.n_clusters == 1000 and naive.n_clusters == 2000
    assert w(clustered) > 1.25 * w(naive), (
        f"clustered width {w(clustered):.4f} vs naive {w(naive):.4f}: "
        "duplicated bets are being counted as independent information")


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

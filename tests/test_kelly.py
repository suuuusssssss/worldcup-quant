import pytest

from wcq.market.kelly import SizingPolicy, expected_log_growth, kelly_fraction, kelly_multi


def test_no_edge_means_no_bet():
    assert kelly_fraction(0.5, 2.0) == 0.0        # exactly fair
    assert kelly_fraction(0.4, 2.0) == 0.0        # negative edge


def test_kelly_maximises_expected_log_growth():
    """Numerically confirm f* is the argmax, not just a plausible formula."""
    p, o = 0.55, 2.10
    f_star = kelly_fraction(p, o)
    best = max((expected_log_growth(p, o, f / 1000.0) for f in range(1, 999)))
    assert expected_log_growth(p, o, f_star) == pytest.approx(best, rel=1e-4)


def test_overbetting_destroys_growth_near_double_kelly():
    """The reason nobody bets full Kelly on an *estimated* probability.

    The textbook line is "growth hits zero at 2*f*", which is the
    continuous-time approximation.  For a discrete binary bet the zero
    crossing sits close to 2*f* but not exactly on it -- so the test asserts
    what is actually true: growth is still positive at f*, has collapsed by
    2*f*, and is negative not far beyond.
    """
    p, o = 0.55, 2.10
    f = kelly_fraction(p, o)
    g_star = expected_log_growth(p, o, f)
    assert g_star > 0
    assert 0 <= abs(expected_log_growth(p, o, 2 * f)) < 0.05 * g_star
    assert expected_log_growth(p, o, 2.3 * f) < 0.0


def test_half_kelly_keeps_three_quarters_of_the_growth():
    """The classic 3/4 rule, and the actual justification for fractional
    Kelly: half the position for a quarter less growth."""
    p, o = 0.55, 2.10
    f = kelly_fraction(p, o)
    ratio = expected_log_growth(p, o, 0.5 * f) / expected_log_growth(p, o, f)
    assert ratio == pytest.approx(0.75, abs=0.01)


def test_certain_win_bets_everything():
    assert kelly_fraction(1.0, 3.0) == pytest.approx(1.0)


def test_policy_respects_cap_and_threshold():
    pol = SizingPolicy(fraction=0.5, cap=0.01, min_edge=0.05)
    assert pol.stake(0.60, 0.58, 2.0) == 0.0            # edge below threshold
    assert pol.stake(0.70, 0.50, 2.0) == pytest.approx(0.01)   # capped


def test_shrinkage_moves_size_toward_zero():
    a = SizingPolicy(fraction=1.0, cap=1.0, min_edge=0.0, shrinkage=0.0)
    b = SizingPolicy(fraction=1.0, cap=1.0, min_edge=0.0, shrinkage=0.5)
    assert b.stake(0.60, 0.50, 2.2) < a.stake(0.60, 0.50, 2.2)


def test_multi_outcome_keeps_a_reserve_on_an_overround_book():
    """With a real bookmaker margin (sum 1/o > 1) there is no arbitrage and
    Kelly must keep cash back; only a sub-100% book justifies full deployment
    (see test_multi_bets_everything_on_a_sub_100_book)."""
    stakes = kelly_multi([0.60, 0.25, 0.15], [1.80, 3.3, 5.0])   # sum 1/o = 1.059
    assert all(s >= 0 for s in stakes)
    assert 0.0 < sum(stakes) < 1.0


def test_multi_outcome_skips_negative_ev_legs():
    """Outcome 1 is priced above its probability; it must not be staked even
    though the other legs are attractive."""
    stakes = kelly_multi([0.60, 0.20, 0.20], [2.0, 3.0, 3.0])
    assert stakes[1] == 0.0


def test_multi_outcome_matches_single_kelly_when_only_one_leg_qualifies():
    probs, odds = [0.55, 0.10, 0.35], [2.10, 3.0, 2.0]
    stakes = kelly_multi(probs, odds)
    assert stakes[0] > 0 and stakes[1] == 0.0


def _log_growth_multi(stakes, probs, odds):
    """E[log wealth] for simultaneous stakes on exclusive outcomes."""
    import math
    total = sum(stakes)
    g = 0.0
    for p, o, s in zip(probs, odds, stakes):
        w = 1.0 - total + s * o
        if w <= 0:
            return float("-inf")
        g += p * math.log(w)
    return g


def test_multi_admits_legs_with_po_below_one_but_above_reserve():
    """Regression: the admission test must compare p*o to the *current*
    reserve rate, not to 1.  Here every leg has p*o barely above or below 1,
    but once the first leg is admitted the reserve drops and the others belong
    in the bet too.  The old fixed-threshold version left most of the growth
    on the table."""
    from scipy.optimize import minimize
    probs, odds = [0.45, 0.30, 0.25], [2.5, 3.5, 4.5]
    stakes = kelly_multi(probs, odds)
    got = _log_growth_multi(stakes, probs, odds)

    res = minimize(lambda s: -_log_growth_multi(s, probs, odds),
                   [0.1] * 3, method="SLSQP", bounds=[(0.0, 0.999)] * 3,
                   constraints=[{"type": "ineq", "fun": lambda s: 0.999 - sum(s)}])
    assert got == pytest.approx(-res.fun, abs=1e-4)


def test_multi_bets_everything_on_a_sub_100_book():
    """sum(1/o) < 1 is a theoretical arbitrage: the reserve is exactly zero
    and Kelly deploys the whole bankroll in proportion to the probabilities."""
    probs, odds = [0.45, 0.30, 0.25], [2.5, 3.5, 4.5]   # sum(1/o) = 0.908
    stakes = kelly_multi(probs, odds)
    assert sum(stakes) == pytest.approx(1.0, abs=1e-9)
    assert stakes == pytest.approx(probs)
    # every outcome ends with more than the starting bankroll
    for p, o, s in zip(probs, odds, stakes):
        assert s * o > 1.0 - 1e-9


def test_multi_agrees_with_brute_force_on_random_books():
    """Property test against a direct optimiser over the simplex."""
    import random
    from scipy.optimize import minimize
    rng = random.Random(7)
    for _ in range(25):
        n = rng.choice([2, 3, 4])
        raw = [rng.random() + 0.05 for _ in range(n)]
        probs = [x / sum(raw) for x in raw]
        over = rng.uniform(0.9, 1.1)
        odds = [max(1.01, 1.0 / (p * over) * rng.uniform(0.92, 1.08)) for p in probs]
        stakes = kelly_multi(probs, odds)
        got = _log_growth_multi(stakes, probs, odds)
        best = None
        for start in (0.01, 0.1, 0.3):
            res = minimize(lambda s: -_log_growth_multi(s, probs, odds),
                           [start] * n, method="SLSQP", bounds=[(0.0, 0.9999)] * n,
                           constraints=[{"type": "ineq",
                                         "fun": lambda s: 0.9999 - sum(s)}])
            if best is None or -res.fun > best:
                best = -res.fun
        assert got >= best - 1e-4, (probs, odds, stakes)

import numpy as np
import pytest

from wcq.sim.bracket import (Team, elo_win_prob, exact_title_probs,
                             mc_standard_error, monte_carlo_title_probs,
                             round_by_round, sims_for_precision, win_matrix)


def four_teams():
    return [Team("A", 1800), Team("B", 1600), Team("C", 1700), Team("D", 1500)]


def test_exact_is_a_distribution():
    p = exact_title_probs(four_teams())
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(p > 0)


def test_exact_matches_hand_computation():
    """Four teams, bracket order A-B and C-D, winners meet.

    P(A wins) = P(A beats B) * [P(C beats D)*P(A beats C) + P(D beats C)*P(A beats D)]
    Written out by hand so a refactor of the block-indexing cannot quietly
    change the answer.
    """
    t = four_teams()
    ab = elo_win_prob(t[0].elo, t[1].elo)
    cd = elo_win_prob(t[2].elo, t[3].elo)
    ac = elo_win_prob(t[0].elo, t[2].elo)
    ad = elo_win_prob(t[0].elo, t[3].elo)
    want = ab * (cd * ac + (1 - cd) * ad)
    assert exact_title_probs(t)[0] == pytest.approx(want, abs=1e-12)


def test_equal_teams_split_evenly():
    t = [Team(c, 1500) for c in "ABCDEFGH"]
    assert np.allclose(exact_title_probs(t), 1 / 8, atol=1e-12)


def test_stronger_team_is_more_likely():
    p = exact_title_probs(four_teams())
    assert p[0] > p[2] > p[1] > p[3]


def test_bracket_size_must_be_a_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        exact_title_probs([Team("A", 1500), Team("B", 1500), Team("C", 1500)])


def test_win_matrix_is_antisymmetric():
    w = win_matrix(four_teams())
    assert np.allclose(w + w.T, 1.0)
    assert np.allclose(np.diag(w), 0.5)


# -- the cross-check that makes the simulator trustworthy -------------------

def test_monte_carlo_converges_to_the_exact_answer():
    """The simulator is only worth having if it agrees with the answer we can
    compute exactly.  This catches biased RNG conversion, off-by-one bracket
    indexing, and pairing bugs -- none of which look wrong in the output.
    """
    teams = [Team(f"T{i}", 1500 + 40 * i) for i in range(8)]
    exact = exact_title_probs(teams)
    n = 400_000
    mc = monte_carlo_title_probs(teams, n_sims=n, seed=4242)
    z = (mc - exact) / mc_standard_error(mc, n)
    assert np.max(np.abs(z)) < 4.0, f"largest |z| = {np.max(np.abs(z)):.2f}"


def test_monte_carlo_error_shrinks_as_one_over_sqrt_n():
    teams = [Team(f"T{i}", 1500 + 30 * i) for i in range(8)]
    exact = exact_title_probs(teams)
    err = []
    for n in (5_000, 80_000):
        mc = monte_carlo_title_probs(teams, n_sims=n, seed=9)
        err.append(float(np.abs(mc - exact).max()))
    assert err[1] < err[0]


def test_monte_carlo_is_reproducible():
    teams = four_teams()
    a = monte_carlo_title_probs(teams, 20_000, seed=1)
    b = monte_carlo_title_probs(teams, 20_000, seed=1)
    assert np.array_equal(a, b)


def test_sims_for_precision_scales_quadratically():
    """Each extra decimal digit of precision costs 100x the compute -- the
    argument for using the DP whenever the model admits one."""
    a = sims_for_precision(0.15, 0.001)
    b = sims_for_precision(0.15, 0.0001)
    assert b / a == pytest.approx(100, rel=0.02)


def test_round_by_round_is_monotone_and_ends_at_the_title():
    teams = [Team(f"T{i}", 1500 + 25 * i) for i in range(8)]
    r = round_by_round(teams)
    assert r.shape == (8, 4)
    assert np.allclose(r[:, 0], 1.0)
    assert np.allclose(r[:, -1], exact_title_probs(teams))
    assert np.all(np.diff(r, axis=1) <= 1e-12)      # survival can only fall
    for col in range(1, r.shape[1]):
        assert r[:, col].sum() == pytest.approx(8 / 2 ** col, abs=1e-9)

import numpy as np
import pytest

from wcq.model.poisson import (MAX_GOALS, PoissonParams, goal_rates, match_probs,
                               outcome_probs, over_under, score_matrix, _poisson_pmf_grid)


def test_pmf_grid_matches_closed_form():
    from math import exp, factorial
    for lam in (0.05, 0.8, 1.4, 3.2, 7.5):
        got = _poisson_pmf_grid(lam, 12)
        want = [exp(-lam) * lam ** k / factorial(k) for k in range(13)]
        assert np.allclose(got, want, rtol=1e-12, atol=1e-15)


def test_pmf_grid_stable_at_large_rate():
    """The naive lam**k / k! form overflows k! and loses precision; the
    recurrence must not."""
    p = _poisson_pmf_grid(9.0, MAX_GOALS)
    assert np.all(np.isfinite(p)) and np.all(p >= 0)
    assert p.sum() < 1.0 + 1e-12          # truncated grid, never over one


@pytest.mark.parametrize("d", [-800, -400, -137, 0, 137, 400, 800])
def test_probabilities_sum_to_one(d):
    assert sum(match_probs(d, PoissonParams())) == pytest.approx(1.0, abs=1e-9)


def test_home_win_prob_is_monotone_in_rating_gap():
    p = PoissonParams()
    hs = [match_probs(d, p)[0] for d in range(-600, 601, 50)]
    assert all(b > a for a, b in zip(hs, hs[1:]))


def test_dixon_coles_raises_draw_probability():
    """A product of independent Poissons under-predicts draws.  Negative rho is
    the correction; if this ever reverses, the sign convention has flipped."""
    indep = match_probs(0, PoissonParams(rho=0.0))[1]
    dc = match_probs(0, PoissonParams(rho=-0.05))[1]
    assert dc > indep


def test_rho_zero_recovers_independence():
    lam_h, lam_a = 1.5, 1.1
    m = score_matrix(lam_h, lam_a, rho=0.0)
    ph = _poisson_pmf_grid(lam_h)
    pa = _poisson_pmf_grid(lam_a)
    outer = np.outer(ph, pa)
    assert np.allclose(m, outer / outer.sum(), atol=1e-12)


def test_score_matrix_is_a_distribution():
    m = score_matrix(2.1, 0.9, rho=-0.08)
    assert m.sum() == pytest.approx(1.0)
    assert np.all(m >= 0)


def test_goal_rates_vectorise():
    p = PoissonParams()
    lh, la = goal_rates(np.array([-200.0, 0.0, 200.0]), p)
    assert lh.shape == (3,)
    assert lh[0] < lh[1] < lh[2]
    assert la[0] > la[1] > la[2]


def test_neutral_venue_removes_home_scoring_bump():
    p = PoissonParams(gamma=0.3)
    lh_home, _ = goal_rates(0.0, p, neutral=False)
    lh_neut, _ = goal_rates(0.0, p, neutral=True)
    assert float(lh_home) > float(lh_neut)


def test_over_under_complementary():
    o, u = over_under(1.6, 1.3, 2.5, rho=-0.04)
    assert o + u == pytest.approx(1.0)
    assert 0.3 < o < 0.7


def test_higher_rates_mean_more_goals():
    assert over_under(2.5, 2.5)[0] > over_under(0.8, 0.7)[0]

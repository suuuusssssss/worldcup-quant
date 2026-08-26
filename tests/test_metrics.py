import numpy as np
import pytest

from wcq.backtest import metrics


def uniform(n):
    return np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))


def test_known_uniform_values():
    a = np.array([0, 1, 2, 0, 1, 2])
    p = uniform(6)
    assert metrics.log_loss(p, a) == pytest.approx(np.log(3), abs=1e-9)
    assert metrics.brier_score(p, a) == pytest.approx(2 / 3, abs=1e-9)
    assert metrics.ranked_probability_score(p, a) == pytest.approx(2 / 9, abs=1e-9)


def test_perfect_forecast_scores_zero():
    a = np.array([0, 1, 2])
    p = np.eye(3)
    assert metrics.log_loss(p, a) == pytest.approx(0.0, abs=1e-9)
    assert metrics.brier_score(p, a) == pytest.approx(0.0, abs=1e-9)
    assert metrics.ranked_probability_score(p, a) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("scorer", [metrics.log_loss, metrics.brier_score,
                                    metrics.ranked_probability_score])
def test_propriety_truth_is_the_optimal_report(scorer):
    """A proper scoring rule is minimised in expectation by reporting the true
    distribution.  If any of these could be gamed by shading the forecast, a
    model tuned on it would learn to lie -- and a model that lies about its
    probabilities cannot be Kelly-sized.
    """
    rng = np.random.default_rng(0)
    truth = np.array([0.45, 0.28, 0.27])
    n = 200_000
    actual = rng.choice(3, size=n, p=truth)

    honest = scorer(np.tile(truth, (n, 1)), actual)
    for shade in ([0.55, 0.24, 0.21], [0.35, 0.32, 0.33], [0.45, 0.35, 0.20]):
        assert scorer(np.tile(np.array(shade), (n, 1)), actual) > honest


def test_rps_is_sensitive_to_outcome_ordering():
    """H < D < A is an ordered scale.  Predicting the draw when the home side
    wins should hurt less than predicting the away win.  Brier cannot see this
    difference; RPS must."""
    a = np.array([0])
    near = np.array([[0.0, 1.0, 0.0]])
    far = np.array([[0.0, 0.0, 1.0]])
    assert metrics.ranked_probability_score(near, a) < metrics.ranked_probability_score(far, a)
    assert metrics.brier_score(near, a) == pytest.approx(metrics.brier_score(far, a))


def test_log_loss_clipping_prevents_infinity():
    assert np.isfinite(metrics.log_loss(np.array([[1.0, 0.0, 0.0]]), np.array([2])))


def test_calibration_of_a_calibrated_forecaster():
    rng = np.random.default_rng(3)
    n = 400_000
    p = rng.dirichlet([4, 3, 3], size=n)
    actual = np.array([rng.choice(3, p=row) for row in p[:20_000]])
    bins = metrics.calibration(p[:20_000], actual, bins=10)
    assert metrics.expected_calibration_error(bins) < 0.02


def test_pnl_stats_on_a_hand_computed_case():
    st = metrics.pnl_stats([1.0, -1.0, 1.0, -1.0], [0.1] * 4, [True, False, True, False])
    assert st.n_bets == 4
    assert st.hit_rate == pytest.approx(0.5)
    assert st.total_staked == pytest.approx(0.4)
    assert st.total_pnl == pytest.approx(0.0)
    assert st.roi == pytest.approx(0.0)


def test_max_drawdown_is_peak_to_trough():
    st = metrics.pnl_stats([1.0, 1.0, -1.0, -1.0, -1.0], [1.0] * 5,
                           [True, True, False, False, False])
    assert st.max_drawdown == pytest.approx(3.0)


def test_closing_line_value_sign():
    assert metrics.closing_line_value(2.20, 0.50) > 0     # took 2.20, fair 2.00
    assert metrics.closing_line_value(1.80, 0.50) < 0

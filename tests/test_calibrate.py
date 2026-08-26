"""The MLE has one job: recover the parameters that generated the data."""
import numpy as np
import pytest

from wcq.model.calibrate import (TrainingSet, build_training_set, fit,
                                 negative_log_likelihood, profile_ci)
from wcq.model.poisson import PoissonParams, goal_rates


def synth_training_set(true: PoissonParams, n=4000, seed=3) -> TrainingSet:
    rng = np.random.default_rng(seed)
    diffs = rng.uniform(-350, 350, size=n)
    neutral = rng.random(n) < 0.2
    lam_h, lam_a = goal_rates(diffs, true, neutral)
    rows = [(float(d), int(h), int(a), bool(ne)) for d, h, a, ne in
            zip(diffs, rng.poisson(lam_h), rng.poisson(lam_a), neutral)]
    return build_training_set(rows)


TRUE = PoissonParams(mu=0.18, beta=0.9, gamma=0.25, rho=0.0)


def test_fit_recovers_the_generating_parameters():
    ts = synth_training_set(TRUE, n=8000)
    params, info = fit(ts)
    assert info["success"]
    assert params.mu == pytest.approx(TRUE.mu, abs=0.05)
    assert params.beta == pytest.approx(TRUE.beta, abs=0.10)
    assert params.gamma == pytest.approx(TRUE.gamma, abs=0.06)
    assert abs(params.rho - TRUE.rho) < 0.06


def test_likelihood_prefers_truth_over_perturbation():
    ts = synth_training_set(TRUE, n=4000)
    at_truth = negative_log_likelihood(TRUE.as_vector(), ts)
    off = PoissonParams(TRUE.mu + 0.3, TRUE.beta, TRUE.gamma, TRUE.rho)
    assert at_truth < negative_log_likelihood(off.as_vector(), ts)


def test_infeasible_rho_is_rejected_not_evaluated():
    ts = synth_training_set(TRUE, n=500)
    bad = np.array([TRUE.mu, TRUE.beta, TRUE.gamma, 30.0])   # tau goes negative
    assert negative_log_likelihood(bad, ts) == pytest.approx(1e12)


def test_profile_ci_brackets_the_mle_and_tightens_with_data():
    """The interval must contain the point estimate, be finite when the data
    actually pin the parameter down, and shrink as the sample grows -- and
    because the walk expands until the likelihood ratio crosses the
    threshold, it can never return an interval clipped at an arbitrary scan
    edge (the failure mode of a fixed grid)."""
    small = synth_training_set(TRUE, n=1200, seed=5)
    big = synth_training_set(TRUE, n=6000, seed=6)

    p_small, _ = fit(small)
    p_big, _ = fit(big)

    lo_s, hi_s = profile_ci(small, p_small, index=0)     # mu
    lo_b, hi_b = profile_ci(big, p_big, index=0)

    assert lo_s < p_small.mu < hi_s
    assert lo_b < p_big.mu < hi_b
    assert np.isfinite([lo_s, hi_s, lo_b, hi_b]).all()
    assert (hi_b - lo_b) < (hi_s - lo_s)

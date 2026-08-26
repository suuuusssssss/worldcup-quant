import pytest

from wcq.market.devig import METHODS, fair_probs, margin, multiplicative, power, shin


ODDS_BALANCED = (2.40, 3.30, 3.00)
ODDS_LOPSIDED = (1.20, 7.00, 15.00)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("odds", [ODDS_BALANCED, ODDS_LOPSIDED, (1.9, 3.6, 4.2)])
def test_every_method_returns_a_distribution(method, odds):
    p = fair_probs(odds, method)
    assert sum(p) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < x < 1.0 for x in p)


@pytest.mark.parametrize("method", METHODS)
def test_ordering_is_preserved(method):
    p = fair_probs(ODDS_LOPSIDED, method)
    assert p[0] > p[1] > p[2]


def test_margin_is_positive_and_removed():
    assert margin(ODDS_BALANCED) > 0
    assert sum(fair_probs(ODDS_BALANCED, "shin")) == pytest.approx(1.0)


def test_methods_agree_on_a_balanced_book():
    """With little skew there is nothing for the methods to disagree about;
    divergence here would mean one of them is simply wrong."""
    ps = [fair_probs(ODDS_BALANCED, m) for m in METHODS]
    for p in ps[1:]:
        assert all(abs(a - b) < 0.005 for a, b in zip(p, ps[0]))


def test_methods_diverge_on_a_lopsided_book():
    """This divergence is the point: on a longshot the choice of de-vig moves
    the fair probability by more than a typical edge threshold, so a strategy
    can be manufactured or destroyed by this one line."""
    m = multiplicative(ODDS_LOPSIDED)[2]
    s = shin(ODDS_LOPSIDED)[2]
    assert m > s                       # multiplicative over-assigns to longshots
    assert m - s > 0.002


def test_power_solves_its_defining_equation():
    p = power(ODDS_LOPSIDED)
    assert sum(p) == pytest.approx(1.0, abs=1e-9)


def test_zero_margin_book_is_a_fixed_point():
    fair = (1 / 0.5, 1 / 0.3, 1 / 0.2)
    for m in METHODS:
        assert fair_probs(fair, m) == pytest.approx((0.5, 0.3, 0.2), abs=1e-6)


def test_rejects_impossible_odds():
    with pytest.raises(ValueError):
        fair_probs((1.0, 3.0, 4.0), "shin")
    with pytest.raises(ValueError):
        fair_probs(ODDS_BALANCED, "not-a-method")

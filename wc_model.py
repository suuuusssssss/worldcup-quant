"""
World Cup match-probability model + betting-edge backtest.

Quant framing:
  bookmaker decimal odds  ->  implied probability (the "market price")
  our model               ->  fair probability (our "signal")
  edge = model_prob - implied_prob   ->  trade only when edge > threshold
  Kelly criterion          ->  position sizing
  backtest                 ->  ROI, hit rate, and Sharpe of per-bet PnL

The model is a bivariate-Poisson goal model whose team scoring rates are
derived from Elo ratings (a la Dixon-Coles, simplified). All data here is
illustrative; swap `SAMPLE_MATCHES` for a real historical odds file to
reproduce a genuine backtest.

Run:  python3 wc_model.py
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from math import exp, factorial, log, sqrt
from statistics import mean, pstdev

# ---------------------------------------------------------------------------
# 1. Match model: Elo -> expected goals -> scoreline -> 1X2 probabilities
# ---------------------------------------------------------------------------

LEAGUE_AVG_GOALS = 1.35          # avg goals per team per match (WC-ish)
HOME_ADV = 0.20                  # neutral venue at a WC -> small
MAX_GOALS = 8                    # truncate the Poisson scoreline grid


def elo_expected_score(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected result for A in [0,1]."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def expected_goals(elo_a: float, elo_b: float, scale: float = 1.0) -> tuple[float, float]:
    """Map an Elo gap to a pair of Poisson scoring rates (lambda_a, lambda_b).

    `scale` multiplies the league scoring level and is the free parameter we
    fit by maximum likelihood in `fit_scale_mle`.
    """
    p_a = elo_expected_score(elo_a, elo_b)          # A's win-ish expectation
    base = LEAGUE_AVG_GOALS * scale
    lam_a = base * (2.0 * p_a) ** 0.9 + HOME_ADV
    lam_b = base * (2.0 * (1 - p_a)) ** 0.9
    return max(lam_a, 0.05), max(lam_b, 0.05)


def _pois(k: int, lam: float) -> float:
    return exp(-lam) * lam ** k / factorial(k)


def match_probs(elo_a: float, elo_b: float, scale: float = 1.0) -> tuple[float, float, float]:
    """Return (P(A win), P(draw), P(B win)) by summing the scoreline grid."""
    lam_a, lam_b = expected_goals(elo_a, elo_b, scale)
    pw = pd = pl = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = _pois(i, lam_a) * _pois(j, lam_b)
            if i > j:
                pw += p
            elif i == j:
                pd += p
            else:
                pl += p
    s = pw + pd + pl
    return pw / s, pd / s, pl / s


# ---------------------------------------------------------------------------
# 2. Odds utilities
# ---------------------------------------------------------------------------

def implied_probs(odds: tuple[float, float, float]) -> tuple[float, float, float]:
    """Decimal odds -> vig-free implied probabilities (normalise out the overround)."""
    raw = [1.0 / o for o in odds]
    s = sum(raw)
    return tuple(r / s for r in raw)  # type: ignore


def kelly_fraction(p: float, dec_odds: float) -> float:
    """Kelly stake as a fraction of bankroll for a single outcome bet."""
    b = dec_odds - 1.0            # net odds
    q = 1.0 - p
    f = (b * p - q) / b
    return max(f, 0.0)


# ---------------------------------------------------------------------------
# 3. Backtest
# ---------------------------------------------------------------------------

@dataclass
class Match:
    home: str
    away: str
    elo_home: float
    elo_away: float
    odds: tuple[float, float, float]   # decimal odds (home, draw, away)
    result: str                        # 'H', 'D', or 'A'


# Illustrative sample (Elo ratings approximate; odds & results are synthetic).
SAMPLE_MATCHES = [
    Match("Brazil", "Serbia", 2030, 1780, (1.35, 5.0, 9.5), "H"),
    Match("Argentina", "Mexico", 2010, 1820, (1.55, 4.0, 7.0), "H"),
    Match("France", "Denmark", 1990, 1850, (1.70, 3.8, 5.2), "H"),
    Match("Germany", "Japan", 1960, 1830, (1.65, 4.1, 5.6), "A"),
    Match("Spain", "Morocco", 1980, 1810, (1.60, 3.9, 6.2), "D"),
    Match("Portugal", "Uruguay", 1975, 1890, (1.85, 3.6, 4.4), "H"),
    Match("England", "Senegal", 1955, 1800, (1.50, 4.2, 7.5), "H"),
    Match("Netherlands", "USA", 1945, 1790, (1.62, 3.9, 6.0), "H"),
    Match("Croatia", "Canada", 1900, 1770, (1.72, 3.7, 5.4), "A"),
    Match("Belgium", "Poland", 1930, 1810, (1.58, 4.0, 6.4), "D"),
    Match("Switzerland", "Cameroon", 1870, 1760, (1.80, 3.6, 4.8), "H"),
    Match("Korea Rep", "Ghana", 1790, 1750, (2.30, 3.4, 3.1), "A"),
]

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
EDGE_THRESHOLD = 0.03      # only bet when our edge exceeds 3%
KELLY_SCALE = 0.5          # half-Kelly (standard risk control)


def backtest(matches: list[Match]) -> dict:
    bankroll_returns: list[float] = []   # per-bet PnL as fraction of stake
    n_bets = wins = 0
    staked = pnl = 0.0

    for m in matches:
        model = match_probs(m.elo_home, m.elo_away)          # (H, D, A)
        market = implied_probs(m.odds)
        for outcome, idx in OUTCOME_INDEX.items():
            edge = model[idx] - market[idx]
            if edge <= EDGE_THRESHOLD:
                continue
            dec = m.odds[idx]
            stake = KELLY_SCALE * kelly_fraction(model[idx], dec)
            if stake <= 0:
                continue
            n_bets += 1
            staked += stake
            if m.result == outcome:
                profit = stake * (dec - 1.0)
                wins += 1
            else:
                profit = -stake
            pnl += profit
            bankroll_returns.append(profit / stake)          # unit return

    roi = pnl / staked if staked else 0.0
    hit = wins / n_bets if n_bets else 0.0
    sharpe = (mean(bankroll_returns) / pstdev(bankroll_returns) * sqrt(len(bankroll_returns))
              if len(bankroll_returns) > 1 and pstdev(bankroll_returns) > 0 else 0.0)
    return {
        "bets": n_bets, "hit_rate": hit, "total_return_units": pnl,
        "roi_on_stake": roi, "sharpe": sharpe, "returns": bankroll_returns,
    }


# ---------------------------------------------------------------------------
# 4. Statistical inference: MLE calibration, bootstrap CIs, hypothesis test
# ---------------------------------------------------------------------------

def fit_scale_mle(matches: list[Match], grid=None) -> tuple[float, float]:
    """Maximum-likelihood estimate of the scoring-scale parameter.

    We choose the `scale` that maximizes the log-likelihood of the observed
    match results (H/D/A) under the Poisson model. Returns (scale_hat, logL).
    """
    if grid is None:
        grid = [0.5 + 0.02 * i for i in range(101)]     # 0.50 .. 2.50
    best_s, best_ll = 1.0, -1e18
    for s in grid:
        ll = 0.0
        for m in matches:
            p = match_probs(m.elo_home, m.elo_away, s)
            ll += log(max(p[OUTCOME_INDEX[m.result]], 1e-12))
        if ll > best_ll:
            best_ll, best_s = ll, s
    return best_s, best_ll


def _percentile(sorted_x: list[float], q: float) -> float:
    if not sorted_x:
        return float("nan")
    i = min(len(sorted_x) - 1, max(0, int(q * len(sorted_x))))
    return sorted_x[i]


def bootstrap_ci(matches: list[Match], b: int = 3000, seed: int = 7) -> dict:
    """Nonparametric bootstrap: resample matches with replacement and
    recompute the strategy metrics to get 95% confidence intervals, plus a
    one-sided bootstrap p-value for H0: mean per-bet return <= 0."""
    rng = random.Random(seed)
    n = len(matches)
    rois, sharpes, mean_rets = [], [], []
    for _ in range(b):
        sample = [matches[rng.randrange(n)] for _ in range(n)]
        r = backtest(sample)
        if r["bets"] == 0:
            continue
        rois.append(r["roi_on_stake"])
        sharpes.append(r["sharpe"])
        if r["returns"]:
            mean_rets.append(mean(r["returns"]))
    rois.sort(); sharpes.sort()
    p_value = sum(1 for x in mean_rets if x <= 0) / len(mean_rets) if mean_rets else float("nan")
    return {
        "roi_ci": (_percentile(rois, 0.025), _percentile(rois, 0.975)),
        "sharpe_ci": (_percentile(sharpes, 0.025), _percentile(sharpes, 0.975)),
        "edge_p_value": p_value,
    }


def main() -> None:
    # (1) Calibrate the model's scoring level by maximum likelihood.
    scale_hat, logL = fit_scale_mle(SAMPLE_MATCHES)
    print(f"=== MLE calibration ===\nscale_hat = {scale_hat:.2f}   logL = {logL:.2f}\n")

    print("=== Sample match probabilities (model vs market) ===")
    for m in SAMPLE_MATCHES[:5]:
        model = match_probs(m.elo_home, m.elo_away, scale_hat)
        market = implied_probs(m.odds)
        print(f"{m.home:>12} vs {m.away:<10} "
              f"model H/D/A = {model[0]:.2f}/{model[1]:.2f}/{model[2]:.2f}  "
              f"market = {market[0]:.2f}/{market[1]:.2f}/{market[2]:.2f}")

    print("\n=== Backtest (edge>{:.0%}, half-Kelly) ===".format(EDGE_THRESHOLD))
    r = backtest(SAMPLE_MATCHES)
    print(f"bets placed      : {r['bets']}")
    print(f"hit rate         : {r['hit_rate']:.1%}")
    print(f"total return     : {r['total_return_units']:+.3f} units")
    print(f"ROI on stake     : {r['roi_on_stake']:+.1%}")
    print(f"Sharpe (per bet) : {r['sharpe']:.2f}")

    # (2) Quantify uncertainty by bootstrap, and test H0: no edge.
    ci = bootstrap_ci(SAMPLE_MATCHES)
    print("\n=== Bootstrap inference (3000 resamples) ===")
    print(f"ROI 95% CI       : [{ci['roi_ci'][0]:+.1%}, {ci['roi_ci'][1]:+.1%}]")
    print(f"Sharpe 95% CI    : [{ci['sharpe_ci'][0]:.2f}, {ci['sharpe_ci'][1]:.2f}]")
    print(f"H0 (no edge) p   : {ci['edge_p_value']:.3f}")


if __name__ == "__main__":
    main()

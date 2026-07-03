# World Cup Edge — Probabilistic Match Model & Betting-Strategy Backtest

A quantitative-research-style project that treats sports betting as a toy
market: **bookmaker odds are the price, a statistical model is the signal, and
mispricings are the trade.**

## Idea

1. **Model** each match with a bivariate-Poisson goal model whose scoring rates
   come from team Elo ratings (a simplified Dixon–Coles). Summing the scoreline
   grid gives calibrated win/draw/loss probabilities. The league scoring level
   is fit by **maximum-likelihood estimation** on observed results.
2. **Price** the market by converting decimal odds to implied probabilities and
   stripping the bookmaker's overround (vig).
3. **Find edge**: `edge = model_prob − implied_prob`. Bet only when the edge
   clears a threshold.
4. **Size** each bet with the **Kelly criterion** (half-Kelly for risk control).
5. **Backtest & infer**: track ROI, hit rate, and Sharpe, then quantify
   uncertainty with a **nonparametric bootstrap** (95% confidence intervals)
   and a **hypothesis test** of H0: no edge (bootstrap p-value).
6. **Simulate** the whole knockout bracket in **C++** via Monte Carlo to get
   each team's championship probability — the performance-critical piece.

## Files

| File | What it does |
|------|--------------|
| `wc_model.py` | Poisson/Elo match model, vig removal, Kelly sizing, backtest |
| `mc_tournament.cpp` | Fast Monte Carlo bracket simulator (xorshift128+ RNG) |

## Run

```bash
python3 wc_model.py

g++ -O2 -std=c++17 mc_tournament.cpp -o mc_tournament
./mc_tournament 10000000        # 10M sims in ~1s
```

## Sample output

```
=== MLE calibration ===
scale_hat = 1.20   logL = -11.06

=== Backtest (edge>3%, half-Kelly) ===
bets placed      : 10
hit rate         : 60.0%
ROI on stake     : -0.9%
Sharpe (per bet) : -0.05

=== Bootstrap inference (3000 resamples) ===
ROI 95% CI       : [-56.5%, +46.9%]
Sharpe 95% CI    : [-2.26, 2.81]
H0 (no edge) p   : 0.510

=== Championship probabilities over 10,000,000 simulations ===
 1. Brazil        18.22%
 2. Argentina     14.17%
 3. Spain         12.14%
 ...
```

**Reading the result (this is the point):** on a 12-match toy sample the
strategy shows no statistically significant edge — the 95% ROI interval
straddles zero and we fail to reject H0 (p ≈ 0.51). A positive point-estimate
ROI on a handful of bets is *not* a signal; only with a real dataset of
hundreds of matches would the confidence interval tighten enough to conclude
anything. Reporting that honestly — with intervals and a p-value rather than a
single flattering number — is the whole exercise.

## Notes / honesty

The Elo ratings, odds, and results bundled here are **illustrative** so the
code runs out of the box. To produce a real backtest, replace `SAMPLE_MATCHES`
with a historical file of matches + closing odds (e.g. from football-data.co.uk)
and real Elo ratings (e.g. eloratings.net). The methodology — calibrated
probabilities, vig-free pricing, edge detection, Kelly sizing, Monte Carlo —
is exactly what carries over to a real dataset.

## Quant concepts demonstrated

Probability modeling · Poisson processes · **maximum-likelihood estimation** ·
statistical calibration · **hypothesis testing (H0: no edge)** ·
**bootstrap confidence intervals** · expected value · Kelly criterion ·
Monte Carlo simulation · variance/Sharpe · Python + performance-oriented C++.

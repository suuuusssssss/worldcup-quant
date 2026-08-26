# World Cup Quant — point-in-time football model, walk-forward backtest, bracket Monte Carlo

Treats sports betting as a toy market: **bookmaker odds are the price, a
statistical model is the signal, and the disagreement between them is the
trade.** The interesting part is not the model — it is the machinery that
tells you honestly whether the model has anything.

**Headline result: it does not, and the project can prove it on 219,271
matches.** That is the finding, and it is reported here rather than buried.

---

## What was measured

Everything below is out-of-sample under a walk-forward protocol on real data.
`make backtest` reproduces it end to end in about 90 seconds.

### The model is genuinely good in absolute terms

Club football, 221,391 scored matches, 2000–2025:

| scoring rule | model | base rate | uniform |
|---|---|---|---|
| log loss | **1.0258** | 1.0711 | 1.0986 |
| Brier | **0.6156** | 0.6476 | 0.6667 |
| RPS | **0.2110** | 0.2264 | 0.2222 |

Expected calibration error **0.0068** — when it says 30%, it happens 30% of
the time. That is the property Kelly sizing actually requires, and it holds.

### And the market is still better

On the 219,271 matches that carry a real bookmaker price:

| scoring rule | model | market (vig-free) | verdict |
|---|---|---|---|
| log loss | 1.0257 | **1.0053** | market |
| Brier | 0.6155 | **0.6017** | market |
| RPS | 0.2109 | **0.2046** | market |
| calibration error | 0.0068 | **0.0020** | market |

The model closes about 69% of the gap between a naive base rate and the
closing market. Impressive, and completely insufficient: the remaining 31% is
where all the money is.

### So the strategy loses, decisively

de-vig = Shin, edge > 3%, quarter-Kelly, 2% cap:

```
bets placed          : 151,187
hit rate             : 30.19%
ROI on stake         : -9.93%
t-stat               : -24.88
clustered bootstrap  : 95% CI [-10.79%, -9.07%]   p = 0.0000
stationary block     : 95% CI [-10.74%, -9.13%]   p = 0.0000
```

This is not an underpowered null. It is a confident, tightly-bounded negative.

### Raising the bar does not help — which is the real diagnostic

45 configurations (3 de-vig methods × 5 edge thresholds × 3 Kelly fractions),
top-5 European leagues:

| de-vig | edge > | bets | ROI | raw p | **deflated p** |
|---|---|---|---|---|---|
| multiplicative | 2% | 28,553 | −8.35% | 0.0000 | 0.0000 |
| multiplicative | 8% | 7,471 | −7.75% | 0.0000 | 0.0000 |
| multiplicative | 12% | 2,229 | −7.37% | 0.0080 | **0.3033** |
| shin | 6% | 12,768 | −9.53% | 0.0000 | 0.0000 |
| shin | 12% | 2,182 | −8.35% | 0.0127 | **0.4365** |

ROI sits between −7% and −10% no matter what you turn. If the model had a
genuine-but-small edge, a higher threshold would select better bets and ROI
would climb. It does not move. **The "edge" is noise with a bookmaker margin
attached**, and the deflated p-value column is there because the best of 45
configurations is not a discovery.

### International matches, where the model does better

49,520 matches since 1872, 46,475 scored: log loss **0.9346** vs base rate
1.0462. But calibration degrades at the top — on fixtures where it says 94%,
the favourite wins 90%. The model is **overconfident on heavy favourites**,
which is exactly the population a betting strategy concentrates in.

---

## Design

```
wcq/
  schema.py              Match / Odds / Prediction / Bet -- one vocabulary
  data/sources.py        content-addressed download cache, atomic writes
  data/loaders.py        streaming CSV parsers -> Match
  model/elo.py           point-in-time Elo (the causality guarantee lives here)
  model/poisson.py       Elo gap -> goal rates -> Dixon-Coles scoreline grid
  model/calibrate.py     vectorised MLE, profile-likelihood intervals
  market/devig.py        multiplicative / additive / power / Shin
  market/kelly.py        Kelly, fractional Kelly, exclusive-outcome Kelly
  backtest/walkforward.py  the harness that forbids lookahead
  backtest/metrics.py    log loss, Brier, RPS, calibration, P&L, CLV
  backtest/bootstrap.py  clustered + stationary-block, recentred p-values
  sim/bracket.py         exact O(n^2) DP and a vectorised Monte Carlo
  execution/kalshi.py    rate-limited, idempotent REST client (dry-run default)
cpp/mc_tournament.cpp    multithreaded simulator, deterministic, self-checking
```

### Data sources

| what | source | rows |
|---|---|---|
| international results | [martj42/international_results](https://github.com/martj42/international_results) | 49,520 (1872–2026) |
| club results + Bet365 1X2 prices | [football-data.co.uk](https://www.football-data.co.uk/) via [xgabora mirror](https://github.com/xgabora/Club-Football-Match-Data-2000-2025) | 230,554, of which 227,515 priced |

`wcq/data/loaders.py` also carries a parser for raw football-data.co.uk season
files, which additionally expose an opening **and** closing Pinnacle line
(`PSH/PSD/PSA` vs `PSCH/PSCD/PSCA`). Closing prices are preferred wherever
available — a strategy that only beats the opening line is slow, not right.

**Where the data does not stretch:** there is no free archive of historical
closing odds for international fixtures. The betting backtest therefore runs on
club data, and the tournament simulator on international data. That split is
stated rather than papered over.

### The causality guarantee

Ratings are computed here rather than downloaded, on purpose. A published
rating table is a snapshot series, and joining a match to "the nearest
snapshot" is exactly where lookahead leaks in: a snapshot dated the 15th
already contains results from the 1st through the 14th.

`EloEngine.stream()` is the only sanctioned traversal. It yields the pre-match
snapshot, *then* folds in the result, so a rating cannot contain its own match.
Out-of-order input raises rather than silently corrupting state.

Two property tests enforce it:

- `test_no_lookahead_under_future_permutation` — rewrite every result after
  index *i* and assert that every snapshot up to *i* is bit-identical.
- `test_predictions_do_not_change_when_the_future_is_rewritten` — the same
  attack against the whole harness, covering the refit schedule and the
  training-row accumulation, not just the rating engine.

### Statistics, done properly

- **Scoreline likelihood, not outcome likelihood.** A 4–0 and a 1–0 are both
  "home win" but say different things about scoring rates.
- **Dixon–Coles.** Independent Poissons under-predict draws; the τ correction
  on the four low-score cells fixes it. Fitted ρ ≈ −0.045 on 226k matches.
- **The MLE never builds a grid.** Only P(observed scoreline) is needed, so the
  objective is one vectorised O(N) numpy expression. 226k matches fit in 1.6s.
- **Clustered bootstrap.** The independent unit is the *match*; two bets on one
  fixture are not two observations.
- **Recentred p-values.** A bootstrap distribution is centred on the observed
  statistic, not the null. Counting resamples below zero is not a hypothesis
  test. `test_p_value_is_uniform_under_the_null` verifies the corrected version
  is actually calibrated.
- **Deflated p-values.** Sidak correction on every configuration search.
- **Four de-vig methods, reported side by side.** On a lopsided book the choice
  moves the fair probability by more than a typical edge threshold, so picking
  the flattering one is a way to manufacture a strategy.

### Exact vs Monte Carlo, and why both exist

For a bracket of independent ties, championship probabilities have a closed
recursion in O(n²) — 0.27 ms for 16 teams, exact. Monte Carlo returns an
*estimate* of that same number with error falling only as 1/√N: resolving the
favourite to ±0.1pp needs ~494,000 simulations, ±0.01pp needs ~49 million.

So the simulator is not there for this model. It is there for the models where
no recursion exists — group stages with goal-difference tiebreakers, extra time
as a separate regime, fatigue and suspensions carrying across rounds,
correlated outcomes, joint queries. Each makes the state space path-dependent;
the DP blows up, the loop gains one line.

The DP earns its keep anyway as ground truth. `--check` runs both and prints a
z-score per team; CI fails the build if any |z| ≥ 4. That catches biased RNG
conversion, off-by-one bracket indexing and bad seeding — none of which look
wrong in the output.

```
$ ./cpp/mc_tournament --sims 20000000 --threads 2 --check
=== 20000000 simulations, 2 threads, 1.31s (15.3M sims/s) ===
  1 Spain           22.4883%  22.4909%    -0.28
  2 Argentina       21.8341%  21.8423%    -0.89
  ...
largest |z| vs exact: 1.91  (consistent)
```

### C++ concurrency choices

- **Deterministic across thread counts.** Work splits into numbered chunks;
  chunk *c* is seeded `splitmix64(seed, c)`, so `--threads 1` and `--threads 8`
  produce bit-identical counts. Seeding per *thread* instead would make results
  depend on the scheduler and turn every Monte Carlo bug into a heisenbug. CI
  asserts the md5 of the output table matches across 1 and 8 threads.
- **splitmix64 seeding.** Adjacent raw seeds can give correlated streams;
  an avalanche mixer is the standard xoshiro seeding procedure, not optional.
- **xoshiro256++ over xorshift128+.** The latter has weak low bits, and
  `u < p` is sensitive to the whole mantissa.
- **No false sharing.** Thread-local counters merged once at the end; a shared
  array puts several threads' counters on one cache line and can make the
  parallel build slower than serial.
- **Precomputed win matrix.** One `pow()` per team pair up front instead of one
  per simulated tie — at 100M sims that removes ~400M transcendental calls.

### Execution layer (Kalshi)

Complete and unit-tested against a fake transport; **never pointed at a funded
account**, and `dry_run=True` is the default you have to switch off.

Three details that cost real money, each with a test:

1. **`yes_ask = 100 − best_no_bid`.** There is no ask array. Reading the top of
   the `yes` array as an ask fabricates edge on every market.
2. **Fees are `ceil(0.07·C·P·(1−P))`**, quadratic and maximal at 50c — exactly
   where a probabilistic model is least certain. A 3c gross edge at 45c is
   about 1.7c of fee. `expected_value_after_fees` is the only EV function
   exposed; there is no fee-free variant to call by accident.
3. **Deterministic idempotency keys.** A POST that times out may still have
   filled; a random UUID on retry doubles the position.

Auth sits behind a protocol with two implementations, because Kalshi migrated
from email/password JWT to RSA-PSS request signing.

---

## Running it

```bash
pip install -e ".[dev]"
make data          # ~50 MB, cached and content-addressed
make test          # 116 tests, no network, ~3s
make backtest      # the headline numbers, ~90s
make sweep         # 45 configurations with deflated p-values
make cpp bench     # determinism + throughput + exact cross-check
make tournament    # bracket probabilities, exact DP vs Monte Carlo
```

---

## What I would do next, in order

1. **Beat the closing line before touching P&L.** CLV is measurable on a far
   smaller sample than returns are. Until the model beats the close, realised
   P&L is noise around a known negative.
2. **Give the model information the market lacks.** Elo compresses a match into
   one number. Shot-based expected goals, lineups, rest days, and travel are in
   the price already; the only route to edge is a feature the price is slow to
   absorb.
3. **Fix the overconfidence at the top of the range.** Isotonic or Platt
   recalibration on a rolling window, since heavy favourites are where the
   errors concentrate and where a Kelly-sized position is largest.
4. **Move to markets that are actually softer.** The premise for a venue like
   Kalshi is retail-dominated flow on prices that lag a competent model. Top-5
   European league 1X2 is the most efficient market in the sport; testing there
   was the right *test* and the wrong *target*.

## Licence

MIT. Datasets belong to their respective authors under their own licences.

#!/usr/bin/env python3
"""End-to-end walk-forward evaluation and betting backtest on real data.

    python3 scripts/run_backtest.py --dataset club --devig shin --min-edge 0.03
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wcq.backtest import metrics
from wcq.backtest.bootstrap import cluster_bootstrap, deflated_p_value, stationary_block_bootstrap
from wcq.backtest.walkforward import WalkForwardConfig, generate_bets, run_walk_forward
from wcq.data import loaders, sources
from wcq.market.devig import fair_probs
from wcq.market.kelly import SizingPolicy
from wcq.model.elo import EloConfig, international_k
from wcq.schema import OUTCOME_INDEX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("club", "international"), default="club")
    ap.add_argument("--divisions", default="", help="comma-separated filter, e.g. E0,SP1,D1")
    ap.add_argument("--price", choices=("b365", "best"), default="b365")
    ap.add_argument("--devig", choices=("multiplicative", "additive", "power", "shin"), default="shin")
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--cap", type=float, default=0.02)
    ap.add_argument("--elo-k", type=float, default=20.0)
    ap.add_argument("--home-adv", type=float, default=65.0)
    ap.add_argument("--refit-days", type=int, default=365)
    ap.add_argument("--resamples", type=int, default=5000)
    ap.add_argument("--configs-tried", type=int, default=1,
                    help="how many strategy variants were searched, for the deflated p-value")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    t0 = time.time()
    if args.dataset == "club":
        divs = {d.strip() for d in args.divisions.split(",") if d.strip()} or None
        matches = loaders.load_club(sources.fetch("club_matches"), price=args.price, divisions=divs)
    else:
        matches = loaders.load_international(sources.fetch("international"))
    print(f"loaded {len(matches):,} matches "
          f"({matches[0].date} .. {matches[-1].date}) in {time.time()-t0:.1f}s")

    cfg = WalkForwardConfig(
        refit_every_days=args.refit_days,
        elo=EloConfig(
            k=args.elo_k, home_advantage=args.home_adv, season_regression=0.15,
            # International matches carry the eloratings.net importance tiers
            # (a World Cup final is not a friendly); club league fixtures all
            # carry comparable weight, so the flat K applies there.
            k_fn=international_k if args.dataset == "international" else None,
        ),
    )
    t0 = time.time()
    res = run_walk_forward(matches, cfg,
                           progress=lambda i, n: print(f"  ..{i:,}/{n:,}", flush=True))
    print(f"walk-forward: {len(res.predictions):,} scored, {res.skipped:,} skipped "
          f"(burn-in / no params) in {time.time()-t0:.1f}s")
    print(f"refits: {len(res.param_history)}")
    if res.param_history:
        d, p, info = res.param_history[-1]
        print(f"  final params @ {d}: mu={p.mu:.3f} beta={p.beta:.3f} "
              f"gamma={p.gamma:.3f} rho={p.rho:.4f}  logL/match={info['logL_per_match']:.4f}")

    probs, actual = res.probs, res.actual_idx
    base = np.tile(np.bincount(actual, minlength=3) / len(actual), (len(actual), 1))

    # The benchmark that actually decides whether a betting strategy can work.
    # If the vig-free market probabilities score better than the model on the
    # same fixtures, there is no edge to find, and that verdict rests on
    # hundreds of thousands of matches rather than on noisy realised P&L.
    with_odds = res.with_odds()
    m_model, m_market, m_actual = [], [], []
    for pr in with_odds:
        m_model.append(pr.probs)
        m_market.append(fair_probs(pr.match.odds.as_tuple(), args.devig))
        m_actual.append(OUTCOME_INDEX[pr.match.result])
    have_prices = len(m_model) > 0
    if have_prices:
        m_model = np.array(m_model); m_market = np.array(m_market)
        m_actual = np.array(m_actual, dtype=np.int64)
        m_base = np.tile(np.bincount(m_actual, minlength=3) / len(m_actual), (len(m_actual), 1))

    print("\n=== Forecast quality (out-of-sample, walk-forward) ===")
    print(f"{'':<22}{'model':>10}{'base rate':>12}{'uniform':>10}")
    for name, fn, uni in (("log loss", metrics.log_loss, 1.0986),
                          ("Brier", metrics.brier_score, 0.6667),
                          ("RPS", metrics.ranked_probability_score, 0.2222)):
        print(f"{name:<22}{fn(probs, actual):>10.4f}{fn(base, actual):>12.4f}{uni:>10.4f}")

    if have_prices:
        print(f"\n=== Model vs market on the {len(with_odds):,} priced matches ===")
        print(f"{'':<22}{'model':>10}{'market':>10}{'base':>10}{'verdict':>26}")
        for name, fn in (("log loss", metrics.log_loss),
                         ("Brier", metrics.brier_score),
                         ("RPS", metrics.ranked_probability_score)):
            a, b, c = fn(m_model, m_actual), fn(m_market, m_actual), fn(m_base, m_actual)
            v = "model better" if a < b else "MARKET BETTER"
            print(f"{name:<22}{a:>10.4f}{b:>10.4f}{c:>10.4f}{v:>26}")
        mkt_bins = metrics.calibration(m_market, m_actual, bins=10)
        print(f"market ECE {metrics.expected_calibration_error(mkt_bins):.4f}  "
              f"vs model ECE "
              f"{metrics.expected_calibration_error(metrics.calibration(m_model, m_actual)):.4f}")
    else:
        print("\n=== Model vs market ===")
        print("no bookmaker prices attached to this dataset -- forecast quality only.")
        print("Historical closing odds for international fixtures are not in any of the")
        print("free archives this project uses, which is why the betting backtest runs on")
        print("club data. Stating that plainly is better than quietly backtesting against")
        print("prices that were never real.")

    bins = metrics.calibration(probs, actual, bins=10)
    print(f"\nexpected calibration error : {metrics.expected_calibration_error(bins):.4f}")
    print(f"{'bucket':>14}{'n':>10}{'predicted':>12}{'observed':>11}{'gap':>9}")
    for b in bins:
        print(f"  [{b.lo:.1f},{b.hi:.1f}){b.n:>12,}{b.mean_pred:>12.3f}"
              f"{b.observed:>11.3f}{b.gap:>+9.3f}")

    if not have_prices:
        print("\nno prices -> no betting backtest. Done.")
        return 0

    policy = SizingPolicy(fraction=args.kelly, cap=args.cap, min_edge=args.min_edge)
    bets = generate_bets(res.with_odds(), policy, devig=args.devig)
    print(f"\n=== Betting backtest (de-vig={args.devig}, min edge={args.min_edge:.0%}, "
          f"{args.kelly:g}-Kelly, cap={args.cap:.0%}) ===")
    print(f"matches with a price : {len(res.with_odds()):,}")
    print(f"bets placed          : {len(bets):,}")
    if not bets:
        print("no bets cleared the threshold")
        return 0

    st = metrics.pnl_stats([b.unit_return for b in bets], [b.stake for b in bets],
                           [bool(b.won) for b in bets])
    print(f"hit rate             : {st.hit_rate:.2%}")
    print(f"total staked         : {st.total_staked:.2f} bankroll units")
    print(f"total P&L            : {st.total_pnl:+.3f} units")
    print(f"ROI on stake         : {st.roi:+.2%}")
    print(f"mean unit return     : {st.mean_unit_return:+.4f}")
    print(f"t-stat (NOT Sharpe)  : {st.t_stat:+.2f}")
    print(f"max drawdown         : {st.max_drawdown:.3f} units")

    n_block = max(args.resamples // 2, 1000)
    print(f"\n=== Inference ({args.resamples:,} clustered / {n_block:,} block resamples) ===")
    cb = cluster_bootstrap(bets, n_resamples=args.resamples)
    print("clustered by match   :", cb.summary())
    sb = stationary_block_bootstrap(bets, n_resamples=n_block)
    print("stationary block     :", sb.summary())
    if args.configs_tried > 1:
        print(f"deflated p ({args.configs_tried} configs): "
              f"{deflated_p_value(cb.p_value, args.configs_tried):.4f}")

    if args.json_out:
        def json_safe(x):
            """NaN and inf are not JSON; a file that json.load cannot read
            back is not a record of anything."""
            if isinstance(x, dict):
                return {k: json_safe(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [json_safe(v) for v in x]
            if isinstance(x, float) and not np.isfinite(x):
                return None
            return x

        Path(args.json_out).write_text(json.dumps(json_safe({
            "n_matches": len(matches), "n_scored": len(res.predictions),
            "log_loss": metrics.log_loss(probs, actual),
            "log_loss_base": metrics.log_loss(base, actual),
            "brier": metrics.brier_score(probs, actual),
            "rps": metrics.ranked_probability_score(probs, actual),
            "ece": metrics.expected_calibration_error(bins),
            "pnl": st.as_dict(),
            "bootstrap": {"roi_ci": list(cb.roi_ci), "p_value": cb.p_value},
        }), indent=2, default=float, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

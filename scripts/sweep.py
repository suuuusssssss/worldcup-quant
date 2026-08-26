#!/usr/bin/env python3
"""Strategy configuration sweep, with an honest multiple-testing correction.

The walk-forward pass runs ONCE and its predictions are reused across every
configuration.  That is both an order-of-magnitude speedup and the correct
experiment: the model is fixed, and only the trading rule varies.

The output deliberately reports the deflated p-value next to the raw one.
Searching 30 configurations and quoting the best one's raw p-value is the most
common way a backtest lies, and the deflated column is what stops that.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wcq.backtest import metrics
from wcq.backtest.bootstrap import cluster_bootstrap, deflated_p_value
from wcq.backtest.walkforward import WalkForwardConfig, generate_bets, run_walk_forward
from wcq.data import loaders, sources
from wcq.market.kelly import SizingPolicy
from wcq.model.elo import EloConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divisions", default="", help="comma-separated, e.g. E0,SP1,D1,I1,F1")
    ap.add_argument("--price", choices=("b365", "best"), default="b365")
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()

    divs = {d.strip() for d in args.divisions.split(",") if d.strip()} or None
    matches = loaders.load_club(sources.fetch("club_matches"), price=args.price, divisions=divs)
    print(f"{len(matches):,} matches, {matches[0].date} .. {matches[-1].date}")

    t0 = time.time()
    res = run_walk_forward(matches, WalkForwardConfig(
        elo=EloConfig(k=20, home_advantage=65, season_regression=0.15)))
    priced = res.with_odds()
    print(f"walk-forward once: {len(res.predictions):,} predictions "
          f"({len(priced):,} priced) in {time.time()-t0:.1f}s\n")

    devigs = ("multiplicative", "power", "shin")
    edges = (0.02, 0.04, 0.06, 0.08, 0.12)
    kellys = (0.10, 0.25, 0.50)
    grid = list(itertools.product(devigs, edges, kellys))

    print(f"{'de-vig':<15}{'edge':>6}{'kelly':>7}{'bets':>9}{'hit':>8}"
          f"{'ROI':>9}{'t':>8}{'raw p':>9}{'deflated p':>12}")
    print("-" * 83)

    rows = []
    for devig, edge, kelly in grid:
        bets = generate_bets(priced, SizingPolicy(fraction=kelly, cap=0.02, min_edge=edge),
                             devig=devig)
        if len(bets) < 200:
            print(f"{devig:<15}{edge:>6.0%}{kelly:>7.2f}{len(bets):>9,}   (too few bets)")
            continue
        st = metrics.pnl_stats([b.unit_return for b in bets], [b.stake for b in bets],
                               [bool(b.won) for b in bets])
        bs = cluster_bootstrap(bets, n_resamples=args.resamples)
        dp = deflated_p_value(bs.p_value, len(grid))
        rows.append((devig, edge, kelly, st, bs, dp))
        print(f"{devig:<15}{edge:>6.0%}{kelly:>7.2f}{st.n_bets:>9,}{st.hit_rate:>8.1%}"
              f"{st.roi:>+9.2%}{st.t_stat:>+8.2f}{bs.p_value:>9.4f}{dp:>12.4f}")

    print("-" * 83)
    if rows:
        best = max(rows, key=lambda r: r[3].roi)
        print(f"\nbest ROI configuration: de-vig={best[0]}, edge>{best[1]:.0%}, "
              f"{best[2]:g}-Kelly -> ROI {best[3].roi:+.2%}")
        print(f"  raw p = {best[4].p_value:.4f}; after correcting for the "
              f"{len(grid)} configurations searched, p = {best[5]:.4f}")
        print("  A configuration search is not a discovery. The deflated p-value "
              "is the one that counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

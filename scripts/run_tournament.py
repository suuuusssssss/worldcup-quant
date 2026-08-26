#!/usr/bin/env python3
"""Championship probabilities for a knockout bracket, from point-in-time Elo.

    python3 scripts/run_tournament.py --as-of 2026-06-01 --sims 2000000

Ratings are built by streaming every international result up to `--as-of` and
stopping there, so a bracket dated during a tournament is never rated with
matches from later in that tournament.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wcq.data import loaders, sources
from wcq.model.elo import EloConfig, EloEngine
from wcq.sim.bracket import (Team, exact_title_probs, mc_standard_error,
                             monte_carlo_title_probs, round_by_round, sims_for_precision)

WC_KEYWORDS = ("FIFA World Cup", "UEFA Euro", "Copa Am")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="", help="YYYY-MM-DD; ratings use matches strictly before this")
    ap.add_argument("--size", type=int, default=16, help="bracket size, a power of two")
    ap.add_argument("--sims", type=int, default=1_000_000)
    ap.add_argument("--min-games", type=int, default=50)
    ap.add_argument("--export-cpp", default="", help="write a C++ bracket literal here")
    args = ap.parse_args()

    cutoff = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.max
    matches = loaders.load_international(sources.fetch("international"))
    used = [m for m in matches if m.date < cutoff]
    print(f"{len(used):,} international matches through {used[-1].date} "
          f"(cutoff {cutoff if cutoff != dt.date.max else 'none'})")

    eng = EloEngine(EloConfig(k=20, home_advantage=65, season_regression=0.05))
    recent: dict[str, dt.date] = {}
    for m, _ in eng.stream(used):
        recent[m.home] = recent[m.away] = m.date

    active = {t: r for t, r in eng.table().items()
              if eng.games(t) >= args.min_games
              and (used[-1].date - recent[t]).days < 900}
    top = sorted(active.items(), key=lambda kv: -kv[1])[: args.size]
    if len(top) < args.size:
        raise SystemExit(f"only {len(top)} qualifying teams")

    # Seed 1 v 16, 2 v 15, ... laid out in bracket order so the top two seeds
    # can only meet in the final -- the standard serpentine seeding.
    seeds = list(range(args.size))
    order: list[int] = []
    def build(block):
        if len(block) == 2:
            order.extend(block)
            return
        half = len(block) // 2
        build([block[i] for i in range(half)])
        build([block[i] for i in range(half, len(block))])
    pairs = []
    lo, hi = 0, args.size - 1
    while lo < hi:
        pairs.append((lo, hi)); lo += 1; hi -= 1
    bracket_idx = [i for pair in pairs for i in pair]
    teams = [Team(top[i][0], top[i][1]) for i in bracket_idx]

    t0 = time.time(); exact = exact_title_probs(teams); t_exact = time.time() - t0
    t0 = time.time(); mc = monte_carlo_title_probs(teams, args.sims, seed=20260614); t_mc = time.time() - t0
    se = mc_standard_error(mc, args.sims)
    rbr = round_by_round(teams)
    rounds = rbr.shape[1] - 1
    labels = {1: "final", 2: "semi", 3: "quarter", 4: "R16", 5: "R32"}

    print(f"\nexact DP: {t_exact*1000:.2f} ms   "
          f"Monte Carlo ({args.sims:,} sims): {t_mc:.2f} s   "
          f"speed-up {t_mc/max(t_exact,1e-9):,.0f}x\n")

    hdr = f"{'#':>3} {'team':<16}{'elo':>8}{'title':>9}{'mc':>9}{'z':>7}"
    for r in range(1, rounds):
        hdr += f"{labels.get(rounds-r, f'R{2**(rounds-r)}'):>9}"
    print(hdr)
    for rank, i in enumerate(np.argsort(-exact), start=1):
        row = (f"{rank:>3} {teams[i].name:<16}{teams[i].elo:>8.0f}"
               f"{exact[i]:>9.4f}{mc[i]:>9.4f}{(mc[i]-exact[i])/se[i] if se[i]>0 else 0:>7.2f}")
        for r in range(1, rounds):
            row += f"{rbr[i, r]:>9.3f}"
        print(row)

    print(f"\nlargest |z| vs exact: {np.max(np.abs((mc-exact)/np.where(se>0,se,1))):.2f}")
    print(f"to resolve the favourite to +/-0.1pp you would need "
          f"{sims_for_precision(float(exact.max()), 0.001):,} simulations; "
          f"the DP is exact in milliseconds.")

    if args.export_cpp:
        lines = ["    return {"]
        for i in range(0, len(teams), 2):
            a, b = teams[i], teams[i + 1]
            lines.append(f'        {{"{a.name}", {a.elo:.1f}}},'.ljust(40) +
                         f'{{"{b.name}", {b.elo:.1f}}},')
        lines.append("    };")
        Path(args.export_cpp).write_text("\n".join(lines) + "\n")
        print(f"\nwrote C++ bracket literal to {args.export_cpp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

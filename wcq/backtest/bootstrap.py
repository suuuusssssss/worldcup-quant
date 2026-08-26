"""Resampling inference on strategy P&L.

Three things this module is careful about, each of which is a way the naive
version gets the answer wrong:

1. **What gets resampled.**  The independent unit is the *match*, not the bet.
   Bets on the same fixture share an outcome and are strongly dependent;
   resampling bets individually pretends there is more information than there
   is and produces intervals that are too tight.  So this is a clustered
   bootstrap, resampling match blocks with replacement.

2. **How the p-value is computed.**  A bootstrap distribution is centred on the
   *observed* statistic, not on the null.  Counting the fraction of resamples
   below zero therefore answers "how confident am I in the sign of what I saw",
   which is not a hypothesis test.  To test H0: mean unit return = 0 you have
   to shift the resample distribution so it is centred at the null, then ask
   how often it exceeds the observed value.  That recentring is the difference
   between a p-value and a number that merely looks like one.

3. **Serial structure.**  Bets arrive in time order and a model's edge can
   decay as the market improves.  A stationary block bootstrap preserves local
   ordering so a strategy whose edge only existed in 2003 does not get credit
   for it in a shuffled resample.

Implementation
--------------
The naive loop -- materialise a resampled list of Bet objects, recompute the
statistic -- is O(resamples * n) in *Python objects* and takes minutes at
n = 150k.  The fix is to notice that every statistic we need is a ratio of two
sums over clusters, so each cluster collapses to four floats up front and a
resample becomes a numpy gather.  Resamples are generated in chunks because
the full index matrix (resamples x clusters) would be several GB.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from wcq.schema import Bet

_CHUNK = 128        # resamples per batch; bounds peak memory at ~chunk*k int64


@dataclass
class BootstrapResult:
    observed_roi: float
    observed_mean_return: float
    roi_ci: tuple[float, float]
    mean_return_ci: tuple[float, float]
    p_value: float
    n_resamples: int
    n_clusters: int

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def summary(self) -> str:
        lo, hi = self.roi_ci
        verdict = "REJECT H0" if self.significant() else "fail to reject H0"
        return (f"ROI {self.observed_roi:+.2%}  95% CI [{lo:+.2%}, {hi:+.2%}]  "
                f"p={self.p_value:.4f}  ({verdict}, {self.n_clusters:,} clusters)")


@dataclass(frozen=True)
class _Collapsed:
    """Cluster-level sufficient statistics.  Everything downstream is a ratio
    of sums of these, so a resample never touches a Bet again."""
    pnl: np.ndarray
    stake: np.ndarray
    ret: np.ndarray      # sum of per-bet unit returns in the cluster
    cnt: np.ndarray      # bets in the cluster

    @property
    def k(self) -> int:
        return int(self.pnl.size)


def _collapse(bets: Sequence[Bet]) -> _Collapsed:
    """Collapse bets to per-match clusters, ordered chronologically.

    The explicit sort matters for the *stationary block* bootstrap, whose
    entire premise is that adjacent clusters are adjacent in time; feeding it
    clusters in arbitrary insertion order would silently degrade it to an
    i.i.d. resample.  Match.key() starts with the date, so sorting keys is a
    chronological sort with a deterministic tiebreak."""
    groups: dict[tuple, list[Bet]] = defaultdict(list)
    for b in bets:
        groups[b.match.key()].append(b)
    pnl, stake, ret, cnt = [], [], [], []
    for k in sorted(groups):
        g = groups[k]
        pnl.append(sum(b.pnl for b in g))
        stake.append(sum(b.stake for b in g))
        ret.append(sum(b.unit_return for b in g))
        cnt.append(len(g))
    return _Collapsed(np.array(pnl), np.array(stake), np.array(ret), np.array(cnt, dtype=float))


def _observed(c: _Collapsed) -> tuple[float, float]:
    return float(c.pnl.sum() / c.stake.sum()), float(c.ret.sum() / c.cnt.sum())


def _resample_stats(c: _Collapsed, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """idx: (chunk, k) matrix of cluster indices -> (roi, mean_return) per row."""
    pnl = c.pnl[idx].sum(axis=1)
    stake = c.stake[idx].sum(axis=1)
    ret = c.ret[idx].sum(axis=1)
    cnt = c.cnt[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(stake > 0, pnl / stake, 0.0), np.where(cnt > 0, ret / cnt, 0.0)


def _finish(obs_roi, obs_mean, rois, means, alpha, n_resamples, k) -> BootstrapResult:
    null = means - means.mean()          # recentre under H0: mean return = 0
    # Add-one smoothing: with B resamples the smallest honest p-value is
    # 1/(B+1), not 0.  A p of exactly zero would also pass through the Sidak
    # deflation unchanged, laundering "smaller than our resolution" into
    # "impossible under the null".
    exceed = int((np.abs(null) >= abs(obs_mean)).sum())
    return BootstrapResult(
        observed_roi=obs_roi,
        observed_mean_return=obs_mean,
        roi_ci=(float(np.quantile(rois, alpha / 2)), float(np.quantile(rois, 1 - alpha / 2))),
        mean_return_ci=(float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))),
        p_value=(exceed + 1) / (len(means) + 1),
        n_resamples=n_resamples,
        n_clusters=k,
    )


def cluster_bootstrap(bets: Sequence[Bet], n_resamples: int = 10_000,
                      seed: int = 7, alpha: float = 0.05) -> BootstrapResult:
    if not bets:
        raise ValueError("no bets to resample")
    c = _collapse(bets)
    rng = np.random.default_rng(seed)
    obs_roi, obs_mean = _observed(c)

    rois = np.empty(n_resamples)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, _CHUNK):
        size = min(_CHUNK, n_resamples - start)
        idx = rng.integers(0, c.k, size=(size, c.k))
        rois[start:start + size], means[start:start + size] = _resample_stats(c, idx)
    return _finish(obs_roi, obs_mean, rois, means, alpha, n_resamples, c.k)


def stationary_block_bootstrap(bets: Sequence[Bet], n_resamples: int = 5_000,
                               mean_block: int = 50, seed: int = 7,
                               alpha: float = 0.05) -> BootstrapResult:
    """Politis-Romano stationary bootstrap with geometric block lengths.

    Preserves short-range time dependence -- fixture clustering, a model
    drifting out of calibration, a market getting more efficient over the
    sample -- which an i.i.d. resample destroys.  Blocks are geometric rather
    than fixed length so the resampled series is stationary.

    Vectorised trick: a stationary-bootstrap index path is a cumulative sum
    where each step is +1 with probability 1-p and a fresh uniform start with
    probability p.  That is expressible as two random draws and one cumsum per
    row, with no Python-level while loop.
    """
    if not bets:
        raise ValueError("no bets to resample")
    c = _collapse(bets)
    n = c.k
    rng = np.random.default_rng(seed)
    p_new = 1.0 / max(mean_block, 1)
    obs_roi, obs_mean = _observed(c)

    rois = np.empty(n_resamples)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, _CHUNK):
        size = min(_CHUNK, n_resamples - start)
        new_block = rng.random((size, n)) < p_new
        new_block[:, 0] = True
        starts = rng.integers(0, n, size=(size, n))
        pos = np.arange(n)[None, :]
        # Position of the most recent block start at or before each column.
        # `pos` is increasing, so a running max over (pos where new_block else
        # -1) is exactly "the latest True index so far" -- the one place a
        # running max coincides with a running *last*, which is why this works
        # here and would be a silent bug if applied to `starts` directly.
        block_start_pos = np.maximum.accumulate(np.where(new_block, pos, -1), axis=1)
        anchor = np.take_along_axis(starts, block_start_pos, axis=1)
        idx = (anchor + (pos - block_start_pos)) % n
        rois[start:start + size], means[start:start + size] = _resample_stats(c, idx)
    return _finish(obs_roi, obs_mean, rois, means, alpha, n_resamples, n)


def deflated_p_value(p: float, n_strategies_tried: int) -> float:
    """Sidak correction for multiple testing.

    If you try 40 threshold / de-vig / fraction combinations and report the
    best one, its p-value is not its p-value:
    P(at least one of n independent tests hits p) = 1 - (1-p)^n.
    Every configuration sweep in this repo reports both the raw and the
    deflated number, because the raw one stops meaning anything the moment you
    have looked at the data more than once.
    """
    n = max(int(n_strategies_tried), 1)
    return 1.0 - (1.0 - p) ** n

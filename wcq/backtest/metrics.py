"""Scoring rules and P&L statistics.

Two families here, and they answer different questions:

* Forecast quality (log loss, Brier, RPS, calibration).  "Is the model's
  probability any good?"  This can be measured on every match ever played and
  needs no odds at all, so it is where the sample size lives.
* Strategy quality (ROI, t-stat, drawdown, CLV).  "Would betting it have made
  money?"  This can only be measured where prices exist, and the effective
  sample is the number of *bets*, not matches -- usually orders of magnitude
  smaller, which is why it takes so much data to say anything.

A model can be excellent on the first and worthless on the second.  Reporting
only the second is how people fool themselves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from wcq.schema import OUTCOME_INDEX

EPS = 1e-15


def log_loss(probs: np.ndarray, actual_idx: np.ndarray) -> float:
    """Mean negative log probability of the realised outcome.

    The strictly proper scoring rule.  Unboundedly punishing when you assign
    near-zero probability to something that happens, which is the correct
    incentive for a model that will size positions off these numbers.
    Benchmark: log(3) = 1.0986 for a uniform guess.
    """
    p = np.clip(probs[np.arange(len(actual_idx)), actual_idx], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(probs: np.ndarray, actual_idx: np.ndarray) -> float:
    """Mean squared error against the one-hot outcome, summed over classes.

    Bounded, unlike log loss, so a single confident miss cannot dominate the
    average.  Uniform guess = 0.6667.
    """
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual_idx)), actual_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ranked_probability_score(probs: np.ndarray, actual_idx: np.ndarray) -> float:
    """RPS over the ordered outcome scale H < D < A.

    Football results are ordinal, not categorical: predicting a draw when the
    home side wins is a smaller error than predicting an away win.  Brier and
    log loss are both blind to that; RPS is not, which is why it is the
    standard scoring rule in the football-forecasting literature.
    Uniform guess = 0.2222.
    """
    n, k = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), actual_idx] = 1.0
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    return float(np.sum((cum_p - cum_o) ** 2) / (n * (k - 1)))


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    observed: float

    @property
    def gap(self) -> float:
        return self.observed - self.mean_pred


def calibration(probs: np.ndarray, actual_idx: np.ndarray, bins: int = 10) -> list[CalibrationBin]:
    """Reliability curve, pooled across all three outcomes.

    Flattens every (match, outcome) pair into one (predicted, happened) point.
    If the model says 30% and those events happen 30% of the time, the model is
    calibrated -- which is the property Kelly sizing actually requires.  A model
    can rank matches perfectly and still be badly calibrated, and Kelly on
    miscalibrated probabilities overbets systematically.
    """
    n = len(actual_idx)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), actual_idx] = 1.0
    p = probs.ravel()
    y = onehot.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[CalibrationBin] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        out.append(CalibrationBin(lo, hi, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return out


def expected_calibration_error(bins: Sequence[CalibrationBin]) -> float:
    total = sum(b.n for b in bins)
    return sum(b.n * abs(b.gap) for b in bins) / total if total else float("nan")


# --------------------------------------------------------------------------
# Strategy statistics
# --------------------------------------------------------------------------

@dataclass
class PnLStats:
    n_bets: int
    hit_rate: float
    total_staked: float
    total_pnl: float
    roi: float
    mean_unit_return: float
    std_unit_return: float
    t_stat: float
    max_drawdown: float
    mean_clv: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def pnl_stats(unit_returns: Sequence[float], stakes: Sequence[float],
              wins: Sequence[bool], clv: Sequence[float] | None = None) -> PnLStats:
    """Summarise a sequence of settled bets.

    `t_stat` is mean/stderr of the per-bet unit return -- a t-statistic, NOT a
    Sharpe ratio.  Calling it Sharpe would imply a per-period risk-adjusted
    return with a defined horizon; bets do not arrive on a fixed clock and
    have no annualisation, so the honest statistic is a t-stat on the mean.
    It is also serially dependent whenever two bets settle on the same match,
    which is why the confidence intervals come from a clustered bootstrap
    rather than from this number.
    """
    r = np.asarray(unit_returns, dtype=float)
    s = np.asarray(stakes, dtype=float)
    w = np.asarray(wins, dtype=bool)
    n = int(r.size)
    if n == 0:
        return PnLStats(0, *[float("nan")] * 9)

    pnl_series = r * s
    staked = float(s.sum())
    total = float(pnl_series.sum())
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    t = float(r.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else float("nan")

    equity = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    dd = float(np.max(peak - np.concatenate([[0.0], equity])))

    return PnLStats(
        n_bets=n,
        hit_rate=float(w.mean()),
        total_staked=staked,
        total_pnl=total,
        roi=total / staked if staked else float("nan"),
        mean_unit_return=float(r.mean()),
        std_unit_return=sd,
        t_stat=t,
        max_drawdown=dd,
        mean_clv=float(np.mean(clv)) if clv is not None and len(clv) else float("nan"),
    )


def closing_line_value(price_taken: float, closing_fair_prob: float) -> float:
    """CLV: how much better than the closing line the fill was.

    The single most predictive diagnostic in sports betting, because the
    closing line is the most efficient price the market ever produces.  Beating
    it consistently is evidence of edge on a far smaller sample than P&L is,
    since it strips out the variance of the actual result.
    """
    return price_taken * closing_fair_prob - 1.0


def outcome_indices(results: Sequence[str]) -> np.ndarray:
    return np.array([OUTCOME_INDEX[r] for r in results], dtype=np.int64)

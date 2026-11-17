"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of Backtest Overfitting".

Given a T x N matrix of per-period returns for N candidate strategies (e.g. a GA cohort):
1. Split the T rows into S contiguous blocks (S even).
2. For every way of choosing S/2 blocks as in-sample (IS), the rest out-of-sample (OOS):
   pick the strategy with the best IS Sharpe and find its relative rank OOS.
3. PBO = share of splits where the IS winner lands in the *bottom half* OOS.

PBO near 0: in-sample winners keep winning (the selection process has skill).
PBO near 0.5 or more: picking the best backtest is no better than picking at random, i.e.
overfitting.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_splits: int
    logits: list[float]
    degradation_slope: float  # OOS Sharpe vs IS Sharpe of the IS winners (<1 = decay)
    prob_oos_loss: float  # share of splits where the IS winner loses money OOS


def _sharpe_cols(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mu / sd, 0.0)


def pbo_cscv(
    returns: npt.NDArray[np.float64],
    n_blocks: int = 10,
    max_splits: int = 2000,
    seed: int = 0,
) -> PBOResult:
    m = np.asarray(returns, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("need a T x N matrix with at least 2 strategies")
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be an even number >= 2")
    t = m.shape[0] - m.shape[0] % n_blocks
    if t < n_blocks * 2:
        raise ValueError("not enough rows for the requested number of blocks")
    blocks = np.split(np.nan_to_num(m[:t]), n_blocks)

    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]

    n = m.shape[1]
    logits, is_best, oos_best, oos_ret = [], [], [], []
    for is_idx in combos:
        oos_idx = [b for b in range(n_blocks) if b not in is_idx]
        is_ret = np.vstack([blocks[b] for b in is_idx])
        oos_r = np.vstack([blocks[b] for b in oos_idx])
        sr_is = _sharpe_cols(is_ret)
        sr_oos = _sharpe_cols(oos_r)
        best = int(np.argmax(sr_is))
        # relative rank of the IS winner among OOS performances, in (0, 1)
        rank = (np.argsort(np.argsort(sr_oos))[best] + 1) / (n + 1)
        logits.append(math.log(rank / (1 - rank)))
        is_best.append(sr_is[best])
        oos_best.append(sr_oos[best])
        oos_ret.append(oos_r[:, best].sum())

    x, y = np.asarray(is_best), np.asarray(oos_best)
    slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else 0.0
    return PBOResult(
        pbo=float(np.mean(np.asarray(logits) <= 0)),
        n_splits=len(combos),
        logits=[round(v, 4) for v in logits],
        degradation_slope=slope,
        prob_oos_loss=float(np.mean(np.asarray(oos_ret) < 0)),
    )

"""Fitness of a genome on the GA's training window.

Objectives (all maximized, used by NSGA-II):
  1. ``edge``: t-statistic of trade R-multiples (significance, not just a high average on a
     handful of trades), minus a small penalty per condition (Occam's razor).
  2. ``drawdown``: negative max drawdown of the equity curve.
  3. ``consistency``: share of K equal sub-periods with positive expectancy.

Genomes with fewer than ``min_trades`` trades get the floor on every objective, so they
lose to anything that trades. The scalar ``fitness`` (for ranking, the leaderboard and the
hall of fame) is edge x consistency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from engine.backtest.runner import BacktestRun
from engine.strategy.genome import Genome, Group, walk

FLOOR = -1e6


@dataclass(frozen=True)
class FitnessConfig:
    min_trades: int = 30
    sub_periods: int = 4
    complexity_penalty: float = 0.05  # per condition beyond the first two
    max_t: float = 10.0  # cap so one freakish genome can't dominate the scale


@dataclass(frozen=True)
class Fitness:
    objectives: tuple[float, float, float]
    fitness: float
    metrics: dict[str, Any]

    @property
    def valid(self) -> bool:
        return self.objectives[0] > FLOOR


def n_conditions(genome: Genome) -> int:
    roots = [genome.entry_long or genome.entry_short, genome.filter]
    return sum(
        1 for r in roots if r is not None for node, _ in walk(r) if not isinstance(node, Group)
    )


def score(genome: Genome, run: BacktestRun, cfg: FitnessConfig) -> Fitness:
    trades = run.trades
    m = run.metrics
    summary = {
        k: m.get(k)
        for k in (
            "n_trades",
            "expectancy_r",
            "win_rate",
            "profit_factor",
            "sharpe",
            "max_drawdown",
            "sqn",
        )
    }
    n = len(trades)
    if n < cfg.min_trades:
        return Fitness((FLOOR, FLOOR, FLOOR), FLOOR, summary)
    r = trades["r"].to_numpy()
    sd = r.std(ddof=1)
    t = float(np.clip(r.mean() / (sd / math.sqrt(n)) if sd > 0 else 0.0, -cfg.max_t, cfg.max_t))
    edge = t - cfg.complexity_penalty * max(0, n_conditions(genome) - 2)

    start, end = run.period_start, run.period_end
    edges = [start + (end - start) * i / cfg.sub_periods for i in range(cfg.sub_periods + 1)]
    bucket = np.searchsorted(
        np.asarray([e.value for e in edges[1:-1]]),
        pd.DatetimeIndex(trades["entry_ts"]).asi8,
        side="right",
    )
    pos = sum(1 for b in range(cfg.sub_periods) if (sel := r[bucket == b]).size and sel.mean() > 0)
    consistency = pos / cfg.sub_periods

    dd = float(m.get("max_drawdown") or 0.0)
    fit = edge * consistency if edge > 0 else edge
    summary.update(t_stat=t, consistency=consistency)
    return Fitness((edge, -dd, consistency), float(fit), summary)

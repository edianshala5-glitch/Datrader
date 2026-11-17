"""Batch genome evaluation on the GA training window, serial or across processes.

* De-duplicates by genome hash. A genome is backtested once per run, and ``evaluations``
  counts *unique* strategies tried. That count becomes ``n_trials`` for the Deflated Sharpe.
* Results come back in input order whatever the parallelism, so runs are reproducible.
* Workers use the ``spawn`` start method (safe next to the API's threads). Each builds its
  own FeatureProvider once and reuses its indicator cache across all genomes it scores.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any

import pandas as pd

from app.models.enums import Timeframe
from engine.backtest.runner import CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.evolution.fitness import FLOOR, Fitness, FitnessConfig, score
from engine.models_util import genome_key
from engine.strategy.compiler import FeatureProvider
from engine.strategy.genome import Genome

logger = logging.getLogger(__name__)


@dataclass
class EvalContext:
    bars: pd.DataFrame  # GA window plus warm-up history before ``start``
    timeframe: Timeframe
    start: pd.Timestamp
    end: pd.Timestamp
    calendar: MarketCalendar | None
    sessions: dict[str, Any] | None
    costs: CostConfig
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    htf_bars: dict[Timeframe, pd.DataFrame] = field(default_factory=dict)


def _evaluate(ctx: EvalContext, fp: FeatureProvider, genome: Genome) -> Fitness:
    try:
        run = run_backtest(
            genome,
            ctx.bars,
            ctx.timeframe,
            calendar=ctx.calendar,
            sessions=ctx.sessions,
            costs=ctx.costs,
            htf_bars=ctx.htf_bars,
            start=ctx.start,
            end=ctx.end,
            with_context=False,
            provider=fp,
        )
    except (KeyError, ValueError) as exc:  # e.g. needs an HTF the context lacks
        return Fitness((FLOOR, FLOOR, FLOOR), FLOOR, {"error": str(exc)})
    return score(genome, run, ctx.fitness)


# ---- worker process state
_W_CTX: EvalContext | None = None
_W_FP: FeatureProvider | None = None


def _init_worker(ctx: EvalContext) -> None:
    global _W_CTX, _W_FP
    _W_CTX = ctx
    _W_FP = FeatureProvider(ctx.bars, ctx.timeframe, ctx.calendar, ctx.sessions, ctx.htf_bars)


def _worker_eval(genome_json: str) -> tuple[tuple[float, float, float], float, dict[str, Any]]:
    assert _W_CTX is not None and _W_FP is not None
    f = _evaluate(_W_CTX, _W_FP, Genome.model_validate_json(genome_json))
    return f.objectives, f.fitness, f.metrics


class Evaluator:
    def __init__(self, ctx: EvalContext, workers: int = 1) -> None:
        self.ctx = ctx
        self.cache: dict[str, Fitness] = {}
        self.evaluations = 0
        self._pool: ProcessPoolExecutor | None = None
        self._fp: FeatureProvider | None = None
        if workers > 1:
            self._pool = ProcessPoolExecutor(
                workers, mp_context=get_context("spawn"), initializer=_init_worker, initargs=(ctx,)
            )
            self._workers = workers
        else:
            self._fp = FeatureProvider(
                ctx.bars, ctx.timeframe, ctx.calendar, ctx.sessions, ctx.htf_bars
            )

    def evaluate(self, genomes: list[Genome]) -> list[Fitness]:
        keys = [genome_key(g) for g in genomes]
        todo: dict[str, Genome] = {}
        for k, g in zip(keys, genomes, strict=True):
            if k not in self.cache and k not in todo:
                todo[k] = g
        if todo:
            if self._pool is not None:
                chunk = max(1, len(todo) // (self._workers * 4))
                results = list(
                    self._pool.map(
                        _worker_eval, [g.model_dump_json() for g in todo.values()], chunksize=chunk
                    )
                )
                for k, (obj, fit, metrics) in zip(todo, results, strict=True):
                    self.cache[k] = Fitness(obj, fit, metrics)
            else:
                assert self._fp is not None
                for k, g in todo.items():
                    self.cache[k] = _evaluate(self.ctx, self._fp, g)
            self.evaluations += len(todo)
        return [self.cache[k] for k in keys]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

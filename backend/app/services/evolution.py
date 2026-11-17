"""Evolution runs end to end: create → evolve (persist + stream every generation) →
validate the hall of fame → done. Checkpointed after every generation, so a stopped or
crashed run resumes exactly where it was.

Data layers (no layer ever sees the next one's data during selection):
    GA training window        research_start → first fold's train_end   (fitness)
    walk-forward test windows  the folds' test periods                   (validation)
    locked holdout             holdout_start → end                       (promotion, Step 8)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.events import Channel, Event, EventBus
from app.models import EvolutionRun, Generation, Instrument, Job, Strategy, strategy_lineage
from app.models.enums import DataSource, JobKind, JobStatus, RunStatus, Timeframe
from app.services.validation import apply_validation, prior_oos_validations
from engine.backtest.runner import CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.data.store import ParquetStore
from engine.evolution.engine import EvolutionConfig, EvolutionEngine, GenerationReport, Individual
from engine.evolution.evaluator import EvalContext, Evaluator
from engine.strategy.compiler import FeatureProvider
from engine.strategy.explain import explain
from engine.validation.pbo import pbo_cscv
from engine.validation.splits import DataSplits, make_splits
from engine.validation.validator import ValidationConfig, returns_matrix, validate_genome

logger = logging.getLogger(__name__)

WARMUP_BARS = 3000


async def create_run(
    session: AsyncSession,
    *,
    name: str,
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    config: EvolutionConfig,
    source: DataSource = DataSource.DUKASCOPY,
    holdout_fraction: float = 0.2,
    n_folds: int = 5,
    validate_top: int = 10,
) -> EvolutionRun:
    instrument = await session.scalar(select(Instrument).where(Instrument.symbol == symbol.upper()))
    if instrument is None:
        raise ValueError(f"unknown instrument {symbol}")
    splits = make_splits(start, end, holdout_fraction=holdout_fraction, n_folds=n_folds)
    job = Job(kind=JobKind.EVOLUTION, status=JobStatus.QUEUED, params={"name": name})
    run = EvolutionRun(
        name=name,
        instrument=instrument,
        timeframe=timeframe,
        status=RunStatus.PENDING,
        algorithm="nsga2-islands",
        config={**config.to_dict(), "source": source.value, "validate_top": validate_top},
        data_splits={**splits.to_dict(), "ga_window_end": splits.folds[0].train_end.isoformat()},
        seed=config.seed,
    )
    session.add_all([job, run])
    await session.flush()
    run.job_id = job.id
    job.params = {**job.params, "run_id": str(run.id)}
    return run


def _checkpoint_path(root: Path, run_id: Any) -> Path:
    return root / f"{run_id}.json"


def _write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, default=str))
    tmp.replace(path)


class RunContext:
    """Everything loaded once per execution: instrument settings, bars, splits."""

    def __init__(self, run: EvolutionRun, instrument: Instrument, store: ParquetStore) -> None:
        self.cfg_raw = dict(run.config)
        self.source = DataSource(self.cfg_raw.pop("source", DataSource.DUKASCOPY.value))
        self.validate_top = int(self.cfg_raw.pop("validate_top", 10))
        self.config = EvolutionConfig.from_dict(self.cfg_raw)
        self.splits = DataSplits.from_dict(run.data_splits)
        self.ga_end = pd.Timestamp(run.data_splits["ga_window_end"])
        self.timeframe = run.timeframe
        self.calendar = MarketCalendar.from_instrument_config(
            instrument.session_timezone, instrument.trading_hours
        )
        self.sessions = instrument.sessions
        self.costs = CostConfig.from_instrument(instrument)
        warm = self.splits.research_start - pd.Timedelta(
            seconds=self.timeframe.seconds * WARMUP_BARS * 1.6
        )
        self.bars = store.read_bars(
            instrument.symbol,
            self.timeframe,
            source=self.source,
            start=warm,
            end=self.splits.holdout_start,
        )
        if self.bars.empty or self.bars["ts"].iloc[-1] < self.ga_end:
            raise ValueError(
                f"not enough {self.timeframe} bars ({self.source}) for this run; ingest data first"
            )
        self.htf = {
            tf: store.read_bars(
                instrument.symbol, tf, source=self.source, start=warm, end=self.splits.holdout_start
            )
            for tf in self.config.htf_timeframes
        }

    def eval_context(self) -> EvalContext:
        cut = self.ga_end
        return EvalContext(
            bars=self.bars[self.bars["ts"] < cut].reset_index(drop=True),
            timeframe=self.timeframe,
            start=self.splits.research_start,
            end=cut,
            calendar=self.calendar,
            sessions=self.sessions,
            costs=self.costs,
            fitness=self.config.fitness,
            htf_bars={tf: b[b["ts"] < cut].reset_index(drop=True) for tf, b in self.htf.items()},
        )


async def _persist_generation(
    session: AsyncSession, run: EvolutionRun, report: GenerationReport, ids: dict[str, Any]
) -> None:
    for isl, stats in enumerate(report.islands):
        session.add(
            Generation(
                run_id=run.id,
                island=isl,
                number=report.number,
                population_size=stats["size"],
                evaluated=sum(1 for i in report.new if i.island == isl),
                stats=stats,
                pareto_size=stats["pareto_size"],
                diversity=stats["diversity"],
                duration_ms=report.duration_ms,
            )
        )
    fresh = [i for i in report.new if i.fitness is not None and i.fitness.valid]
    if fresh:
        existing = {
            h: sid
            for h, sid in (
                await session.execute(
                    select(Strategy.genome_hash, Strategy.id).where(
                        Strategy.genome_hash.in_([i.key for i in fresh]),
                        Strategy.instrument_id == run.instrument_id,
                        Strategy.timeframe == run.timeframe,
                    )
                )
            ).all()
        }
        created: list[tuple[Individual, Strategy]] = []
        for ind in fresh:
            if ind.key in existing:  # found by an earlier run: keep its history
                ids[ind.key] = existing[ind.key]
                continue
            assert ind.fitness is not None
            s = Strategy.from_genome(
                ind.genome.canonical(),
                instrument_id=run.instrument_id,
                timeframe=run.timeframe,
                run_id=run.id,
                generation_number=ind.born,
                island=ind.island,
                origin=ind.origin,
                fitness=ind.fitness.fitness,
                is_sharpe=ind.fitness.metrics.get("sharpe"),
                description=explain(ind.genome),
            )
            session.add(s)
            created.append((ind, s))
        await session.flush()
        links = []
        for ind, s in created:
            ids[ind.key] = s.id
            links += [
                {"child_id": s.id, "parent_id": ids[p]}
                for p in ind.parents
                if p in ids and ids[p] != s.id
            ]
        if links:
            await session.execute(insert(strategy_lineage), links)
    run.current_generation = report.number
    run.evaluations = report.evaluations
    if report.best is not None and report.best.fitness is not None:
        run.best_metrics = {
            "key": report.best.key,
            "fitness": report.best.score,
            **report.best.fitness.metrics,
        }


def _lab_payload(run: EvolutionRun, report: GenerationReport) -> dict[str, Any]:
    best = report.best
    return {
        "run_id": str(run.id),
        "generation": report.number,
        "evaluations": report.evaluations,
        "duration_ms": report.duration_ms,
        "migrated": report.migrated,
        "islands": [
            {
                "island": i,
                "fitness": s["fitness"],
                "diversity": s["diversity"],
                "pareto_size": s["pareto_size"],
                "valid": s["valid"],
            }
            for i, s in enumerate(report.islands)
        ],
        "best": None
        if best is None or best.fitness is None
        else {
            "key": best.key,
            "fitness": best.score,
            "metrics": best.fitness.metrics,
            "explanation": explain(best.genome),
        },
    }


async def execute_run(
    sessionmaker: async_sessionmaker[AsyncSession],
    store: ParquetStore,
    bus: EventBus,
    run_id: Any,
    *,
    checkpoint_dir: Path,
    workers: int = 1,
    stop: threading.Event | None = None,
    validation: ValidationConfig | None = None,
) -> RunStatus:
    stop = stop or threading.Event()
    loop = asyncio.get_running_loop()
    async with sessionmaker() as session:
        run = await session.get(EvolutionRun, run_id)
        if run is None:
            raise ValueError(f"run {run_id} not found")
        instrument = await session.get(Instrument, run.instrument_id)
        assert instrument is not None
        job = await session.get(Job, run.job_id) if run.job_id else None
        try:
            ctx = await loop.run_in_executor(None, RunContext, run, instrument, store)
        except ValueError as exc:
            run.status, run.notes = RunStatus.FAILED, str(exc)
            if job:
                job.status, job.error = JobStatus.FAILED, str(exc)
            await session.commit()
            await bus.publish(
                Event(
                    channel=Channel.LAB,
                    type="run_failed",
                    payload={"run_id": str(run_id), "error": str(exc)},
                )
            )
            return RunStatus.FAILED
        run.status = RunStatus.RUNNING
        run.started_at = run.started_at or datetime.now(UTC)
        if job:
            job.status, job.started_at = JobStatus.RUNNING, job.started_at or datetime.now(UTC)
        await session.commit()
        ids: dict[str, Any] = {
            h: sid
            for h, sid in (
                await session.execute(
                    select(Strategy.genome_hash, Strategy.id).where(Strategy.run_id == run.id)
                )
            ).all()
        }

    ckpt = _checkpoint_path(checkpoint_dir, run_id)
    evaluator = await loop.run_in_executor(None, Evaluator, ctx.eval_context(), workers)
    try:
        if ckpt.exists():
            state = json.loads(ckpt.read_text())
            engine = EvolutionEngine.from_state(state, evaluator)
        else:
            engine = EvolutionEngine(ctx.config, evaluator)
        await bus.publish(
            Event(
                channel=Channel.LAB,
                type="run_started",
                payload={
                    "run_id": str(run_id),
                    "generation": engine.generation,
                    "config": ctx.config.to_dict(),
                },
            )
        )

        while engine.generation < ctx.config.generations and not stop.is_set():
            report = await loop.run_in_executor(None, engine.step)
            async with sessionmaker() as session:
                run = await session.get(EvolutionRun, run_id)
                assert run is not None
                await _persist_generation(session, run, report, ids)
                if run.job_id and (job := await session.get(Job, run.job_id)):
                    job.progress = 0.9 * (report.number + 1) / (ctx.config.generations + 1)
                    job.message, job.heartbeat_at = f"generation {report.number}", datetime.now(UTC)
                await session.commit()
                payload = _lab_payload(run, report)
            state = engine.state_dict()
            await loop.run_in_executor(None, _write_checkpoint, ckpt, state)
            await bus.publish(Event(channel=Channel.LAB, type="generation", payload=payload))

        if stop.is_set() and engine.generation < ctx.config.generations:
            async with sessionmaker() as session:
                run = await session.get(EvolutionRun, run_id)
                assert run is not None
                run.status = RunStatus.PAUSED
                await session.commit()
            await bus.publish(
                Event(
                    channel=Channel.LAB,
                    type="run_paused",
                    payload={"run_id": str(run_id), "generation": engine.generation},
                )
            )
            return RunStatus.PAUSED

        await _validate_hall_of_fame(sessionmaker, bus, run_id, ctx, engine, ids, validation, loop)

        async with sessionmaker() as session:
            run = await session.get(EvolutionRun, run_id)
            assert run is not None
            run.status, run.finished_at = RunStatus.COMPLETED, datetime.now(UTC)
            if run.job_id and (job := await session.get(Job, run.job_id)):
                job.status, job.progress, job.finished_at = (
                    JobStatus.SUCCEEDED,
                    1.0,
                    datetime.now(UTC),
                )
                job.result = {
                    "evaluations": engine.evaluator.evaluations,
                    "generations": engine.generation,
                }
            await session.commit()
        await bus.publish(
            Event(
                channel=Channel.LAB,
                type="run_completed",
                payload={"run_id": str(run_id), "evaluations": engine.evaluator.evaluations},
            )
        )
        return RunStatus.COMPLETED
    except Exception as exc:
        logger.exception("evolution run failed", extra={"run_id": str(run_id)})
        async with sessionmaker() as session:
            run = await session.get(EvolutionRun, run_id)
            if run is not None:
                run.status, run.notes = (
                    RunStatus.FAILED,
                    "".join(traceback.format_exception_only(type(exc), exc)),
                )
                if run.job_id and (job := await session.get(Job, run.job_id)):
                    job.status, job.error = JobStatus.FAILED, run.notes
                await session.commit()
        await bus.publish(
            Event(
                channel=Channel.LAB,
                type="run_failed",
                payload={"run_id": str(run_id), "error": str(exc)},
            )
        )
        raise
    finally:
        await loop.run_in_executor(None, evaluator.close)


async def _validate_hall_of_fame(
    sessionmaker: async_sessionmaker[AsyncSession],
    bus: EventBus,
    run_id: Any,
    ctx: RunContext,
    engine: EvolutionEngine,
    ids: dict[str, Any],
    validation: ValidationConfig | None,
    loop: asyncio.AbstractEventLoop,
) -> None:
    top = [i for i in engine.best(ctx.validate_top) if i.key in ids]
    if not top:
        return
    # The GA selected these on its training window only. The walk-forward test windows they're
    # judged on were never seen by the search, so the selection bias the Deflated Sharpe
    # corrects comes from (a) choosing among the K candidates validated now and (b) every
    # earlier strategy judged on the same OOS data. The raw GA evaluation count is recorded,
    # not charged. Charging it would wrongly reject edges the search found honestly.
    search_evaluations = engine.evaluator.evaluations
    async with sessionmaker() as session:
        run_row = await session.get(EvolutionRun, run_id)
        assert run_row is not None
        prior = await prior_oos_validations(
            session, run_row.instrument_id, run_row.timeframe, exclude_run=run_id
        )
    n_trials = len(top) + prior
    await bus.publish(
        Event(
            channel=Channel.LAB,
            type="validating",
            payload={
                "run_id": str(run_id),
                "candidates": len(top),
                "n_trials": n_trials,
                "prior_oos_validations": prior,
                "search_evaluations": search_evaluations,
            },
        )
    )

    def cohort_pbo() -> float | None:
        if len(top) < 2:
            return None
        research = ctx.bars[ctx.bars["ts"] < ctx.splits.holdout_start].reset_index(drop=True)
        fp = FeatureProvider(research, ctx.timeframe, ctx.calendar, ctx.sessions, ctx.htf)
        runs = {
            i.key: run_backtest(
                i.genome,
                research,
                ctx.timeframe,
                calendar=ctx.calendar,
                sessions=ctx.sessions,
                costs=ctx.costs,
                htf_bars=ctx.htf,
                start=ctx.splits.research_start,
                end=ctx.splits.holdout_start,
                with_context=False,
                provider=fp,
            )
            for i in top
        }
        try:
            return pbo_cscv(returns_matrix(runs, ctx.calendar).to_numpy(), n_blocks=10).pbo
        except ValueError:
            return None

    pbo = await loop.run_in_executor(None, cohort_pbo)
    vcfg = validation or ValidationConfig()
    for ind in top:
        report = await loop.run_in_executor(
            None,
            lambda g=ind.genome: validate_genome(  # type: ignore[misc]
                g,
                ctx.bars,
                ctx.timeframe,
                ctx.splits,
                calendar=ctx.calendar,
                sessions=ctx.sessions,
                costs=ctx.costs,
                htf_bars=ctx.htf,
                n_trials=n_trials,
                pbo=pbo,
                config=vcfg,
            ),
        )
        async with sessionmaker() as session:
            strategy = await session.get(Strategy, ids[ind.key])
            assert strategy is not None
            await apply_validation(
                session,
                strategy,
                report,
                context={
                    "search_evaluations": search_evaluations,
                    "prior_oos_validations": prior,
                    "candidates": len(top),
                    "selection": "GA training window only",
                },
            )
            await session.commit()
            sid, name = str(strategy.id), strategy.name
        await bus.publish(
            Event(
                channel=Channel.LAB,
                type="validated",
                payload={
                    "run_id": str(run_id),
                    "strategy_id": sid,
                    "name": name,
                    "passed": report.passed,
                    "overfit_risk": report.overfit_risk,
                    "dsr": report.dsr,
                    "reasons": report.reasons,
                },
            )
        )

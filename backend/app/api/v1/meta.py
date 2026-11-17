"""Meta-Analyzer endpoints: context rules for stored strategies and ad-hoc genomes."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import ContextRule, EvolutionRun, Instrument, Strategy
from app.models.enums import DataSource, RuleStatus, Timeframe
from app.services.meta import persist_meta, record_meta_job
from engine.backtest.runner import CostConfig
from engine.data.calendar import MarketCalendar
from engine.data.store import ParquetStore
from engine.meta.analyzer import analyze
from engine.strategy.genome import Genome
from engine.validation.splits import DataSplits, make_splits

router = APIRouter(tags=["meta"])
WARMUP_BARS = 3000


class RuleRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    strategy_id: uuid.UUID
    name: str
    description: str
    conditions: list[Any]
    miner: str
    status: RuleStatus
    is_trades: int | None
    is_expectancy_r: float | None
    oos_trades: int | None
    oos_win_rate: float | None
    oos_profit_factor: float | None
    oos_expectancy_r: float | None
    coverage: float | None
    lift: float | None
    p_value: float | None
    p_value_adjusted: float | None
    details: dict[str, Any]
    created_at: datetime


class MetaRequest(BaseModel):
    """Window for strategies without a run (and for ad-hoc genomes)."""

    source: DataSource = DataSource.DUKASCOPY
    start: datetime | None = None
    end: datetime | None = None
    holdout_fraction: float = Field(0.2, ge=0.05, lt=0.9)


class AdHocMetaRequest(MetaRequest):
    genome: Genome
    symbol: str = "XAUUSD"
    timeframe: Timeframe = Timeframe.M15
    start: datetime
    end: datetime


def _load(
    store: ParquetStore,
    symbol: str,
    tf: Timeframe,
    source: DataSource,
    splits: DataSplits,
    genome: Genome,
) -> tuple[pd.DataFrame, dict[Timeframe, pd.DataFrame]]:
    warm = splits.research_start - pd.Timedelta(seconds=tf.seconds * WARMUP_BARS * 1.6)
    bars = store.read_bars(symbol, tf, source=source, start=warm, end=splits.holdout_start)
    htf = {
        f.timeframe: store.read_bars(
            symbol, f.timeframe, source=source, start=warm, end=splits.holdout_start
        )
        for f in genome.features()
        if f.timeframe
    }
    return bars, htf


@router.get("/strategies/{strategy_id}/rules", response_model=list[RuleRead])
async def list_rules(
    strategy_id: uuid.UUID, session: SessionDep, status_: RuleStatus | None = None
) -> list[ContextRule]:
    stmt = select(ContextRule).where(ContextRule.strategy_id == strategy_id)
    if status_:
        stmt = stmt.where(ContextRule.status == status_)
    rows = await session.scalars(stmt.order_by(ContextRule.p_value_adjusted.asc().nulls_last()))
    return list(rows)


@router.post("/strategies/{strategy_id}/meta")
async def analyze_strategy(
    strategy_id: uuid.UUID, body: MetaRequest, request: Request, session: SessionDep
) -> dict[str, Any]:
    """Mine and test context rules for a stored strategy; persist rules and refined strategies."""
    strategy = await session.get(Strategy, strategy_id)
    if strategy is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "strategy not found")
    instrument = await session.get(Instrument, strategy.instrument_id)
    assert instrument is not None
    run = await session.get(EvolutionRun, strategy.run_id) if strategy.run_id else None
    source = DataSource(run.config.get("source", body.source)) if run else body.source
    if run is not None:
        splits = DataSplits.from_dict(run.data_splits)
    elif body.start and body.end:
        splits = make_splits(body.start, body.end, holdout_fraction=body.holdout_fraction)
    else:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "strategy has no run: pass start and end"
        )
    genome = Genome.model_validate(strategy.genome)
    store = ParquetStore(request.app.state.settings.data_dir / "parquet")
    calendar = MarketCalendar.from_instrument_config(
        instrument.session_timezone, instrument.trading_hours
    )

    def work() -> Any:
        bars, htf = _load(store, instrument.symbol, strategy.timeframe, source, splits, genome)
        if bars.empty:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no bars for this strategy's window")
        return analyze(
            genome,
            bars,
            strategy.timeframe,
            splits,
            calendar=calendar,
            sessions=instrument.sessions,
            costs=CostConfig.from_instrument(instrument),
            htf_bars=htf,
        )

    report = await run_in_threadpool(work)
    rules = await persist_meta(session, strategy, report)
    await record_meta_job(session, strategy, report)
    await session.commit()
    return {**report.to_dict(), "rule_ids": [str(r.id) for r in rules]}


@router.post("/meta")
async def analyze_genome(
    body: AdHocMetaRequest, request: Request, session: SessionDep
) -> dict[str, Any]:
    """Context analysis for any genome over a window (nothing persisted)."""
    instrument = await session.scalar(
        select(Instrument).where(Instrument.symbol == body.symbol.upper())
    )
    if instrument is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown instrument {body.symbol}")
    try:
        splits = make_splits(body.start, body.end, holdout_fraction=body.holdout_fraction)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    store = ParquetStore(request.app.state.settings.data_dir / "parquet")
    calendar = MarketCalendar.from_instrument_config(
        instrument.session_timezone, instrument.trading_hours
    )

    def work() -> dict[str, Any]:
        bars, htf = _load(
            store, instrument.symbol, body.timeframe, body.source, splits, body.genome
        )
        if bars.empty:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no bars in the requested window")
        return analyze(
            body.genome,
            bars,
            body.timeframe,
            splits,
            calendar=calendar,
            sessions=instrument.sessions,
            costs=CostConfig.from_instrument(instrument),
            htf_bars=htf,
        ).to_dict()

    return await run_in_threadpool(work)

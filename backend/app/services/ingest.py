"""Ingestion orchestration: source → Parquet lake → higher timeframes → quality → catalogue.

Flow for one run:
1. A ``Job`` row is created (kind=ingest) and progress is streamed on the JOBS channel.
2. Monthly chunks from the source are written to the lake (ticks if present, plus M1).
3. Every higher timeframe is rebuilt from M1 for each touched year (with a day of
   padding on both sides, so bars straddling New Year are complete).
4. A quality report is computed on the touched M1 range.
5. ``Dataset`` rows (one per source and timeframe) are upserted with coverage and quality.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.events import Channel, Event, EventBus
from app.models import Dataset, Instrument, Job
from app.models.enums import DataSource, JobKind, JobStatus, Timeframe
from engine.data.calendar import MarketCalendar
from engine.data.quality import m1_quality_report
from engine.data.resample import resample_bars
from engine.data.sources.base import HistoricalSource
from engine.data.store import ParquetStore

logger = logging.getLogger(__name__)

HIGHER_TIMEFRAMES = (
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1,
)


@dataclass
class IngestResult:
    job_id: str
    status: JobStatus
    m1_rows_written: int
    tick_rows_written: int
    years: list[int]
    quality: dict[str, Any]


def calendar_for(instrument: Instrument) -> MarketCalendar:
    return MarketCalendar.from_instrument_config(
        instrument.session_timezone, instrument.trading_hours
    )


def rebuild_higher_timeframes(
    store: ParquetStore,
    symbol: str,
    source: DataSource,
    calendar: MarketCalendar,
    years: list[int],
    timeframes: tuple[Timeframe, ...] = HIGHER_TIMEFRAMES,
) -> dict[Timeframe, int]:
    counts: dict[Timeframe, int] = {tf: 0 for tf in timeframes}
    for year in sorted(set(years)):
        y0 = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        y1 = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        pad = pd.Timedelta(days=2)
        m1 = store.read_bars(symbol, Timeframe.M1, source=source, start=y0 - pad, end=y1 + pad)
        for tf in timeframes:
            bars = resample_bars(m1, tf, calendar)
            bars = bars[(bars["ts"] >= y0) & (bars["ts"] < y1)]
            store.write_bars(symbol, source, tf, bars, replace_years=True)
            counts[tf] += len(bars)
    return counts


async def _upsert_datasets(
    session: AsyncSession,
    store: ParquetStore,
    instrument: Instrument,
    source: DataSource,
    quality: dict[str, Any],
    has_ticks: bool,
) -> None:
    timeframes = [Timeframe.M1, *HIGHER_TIMEFRAMES]
    if has_ticks:
        timeframes.insert(0, Timeframe.TICK)
    now = datetime.now(UTC)
    for tf in timeframes:
        if tf is Timeframe.TICK:
            cov = store.tick_coverage(instrument.symbol, source)
            uri = str(store.root / "ticks" / f"symbol={instrument.symbol}" / f"source={source}")
        else:
            cov = store.bar_coverage(instrument.symbol, tf, source)
            uri = str(
                store.root
                / "bars"
                / f"symbol={instrument.symbol}"
                / f"source={source}"
                / f"timeframe={tf}"
            )
        if cov.rows == 0:
            continue
        ds = await session.scalar(
            select(Dataset).where(
                Dataset.instrument_id == instrument.id,
                Dataset.source == source,
                Dataset.timeframe == tf,
            )
        )
        if ds is None:
            ds = Dataset(instrument_id=instrument.id, source=source, timeframe=tf, storage_uri=uri)
            session.add(ds)
        ds.storage_uri = uri
        ds.start_ts = cov.start.to_pydatetime() if cov.start is not None else None
        ds.end_ts = cov.end.to_pydatetime() if cov.end is not None else None
        ds.row_count = cov.rows
        ds.last_ingested_at = now
        if tf is Timeframe.M1 and quality:
            ds.quality = quality
        elif tf is not Timeframe.M1:
            ds.quality = {"derived_from": "1m"}


async def run_ingest(
    sessionmaker: async_sessionmaker[AsyncSession],
    store: ParquetStore,
    bus: EventBus,
    source: HistoricalSource,
    *,
    start: datetime,
    end: datetime,
    params: dict[str, Any] | None = None,
) -> IngestResult:
    async with sessionmaker() as session:
        instrument = await session.scalar(
            select(Instrument).where(Instrument.symbol == source.symbol)
        )
        if instrument is None:
            raise ValueError(f"unknown instrument {source.symbol}; run `swarm seed` first")
        job = Job(
            kind=JobKind.INGEST,
            status=JobStatus.RUNNING,
            params={
                "symbol": source.symbol,
                "source": source.source.value,
                "start": start.isoformat(),
                "end": end.isoformat(),
                **(params or {}),
            },
            started_at=datetime.now(UTC),
            heartbeat_at=datetime.now(UTC),
        )
        session.add(job)
        await session.commit()
        calendar = calendar_for(instrument)
        symbol = instrument.symbol

    async def report(progress: float, message: str, **extra: Any) -> None:
        async with sessionmaker() as s:
            j = await s.get(Job, job.id)
            assert j is not None
            j.progress, j.message, j.heartbeat_at = progress, message, datetime.now(UTC)
            await s.commit()
        await bus.publish(
            Event(
                channel=Channel.JOBS,
                type="progress",
                payload={"job_id": str(job.id), "progress": progress, "message": message, **extra},
            )
        )

    m1_rows = tick_rows = 0
    years: set[int] = set()
    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    try:
        async for chunk in source.chunks(start, end):
            if chunk.ticks is not None and not chunk.ticks.empty:
                store.write_ticks(symbol, source.source, chunk.ticks)
                tick_rows += len(chunk.ticks)
            if not chunk.m1.empty:
                store.write_bars(symbol, source.source, Timeframe.M1, chunk.m1)
                m1_rows += len(chunk.m1)
                years.update(int(y) for y in chunk.m1["ts"].dt.year.unique())
                first_ts = chunk.m1["ts"].iloc[0] if first_ts is None else first_ts
                last_ts = chunk.m1["ts"].iloc[-1]
            await report(0.85 * chunk.progress, f"downloaded {chunk.label}", rows=len(chunk.m1))

        await report(0.88, "building 5m/15m/30m/1h/4h/1d bars")
        rebuild_higher_timeframes(store, symbol, source.source, calendar, sorted(years))

        quality: dict[str, Any] = {}
        if first_ts is not None and last_ts is not None:
            await report(0.95, "checking data quality")
            m1 = store.read_bars(
                symbol,
                Timeframe.M1,
                source=source.source,
                start=first_ts,
                end=last_ts + pd.Timedelta(minutes=1),
            )
            quality = m1_quality_report(m1, calendar)

        async with sessionmaker() as session:
            instrument = await session.get(Instrument, instrument.id)
            assert instrument is not None
            await _upsert_datasets(
                session, store, instrument, source.source, quality, has_ticks=tick_rows > 0
            )
            j = await session.get(Job, job.id)
            assert j is not None
            j.status, j.progress, j.finished_at = JobStatus.SUCCEEDED, 1.0, datetime.now(UTC)
            j.message = f"{m1_rows} M1 bars ingested"
            j.result = {
                "m1_rows": m1_rows,
                "tick_rows": tick_rows,
                "years": sorted(years),
                "quality_score": quality.get("score"),
                "client_stats": getattr(getattr(source, "client", None), "stats", None),
            }
            await session.commit()
        await bus.publish(
            Event(channel=Channel.JOBS, type="completed", payload={"job_id": str(job.id)})
        )
        return IngestResult(
            str(job.id), JobStatus.SUCCEEDED, m1_rows, tick_rows, sorted(years), quality
        )
    except BaseException as exc:
        logger.exception("ingest failed", extra={"job_id": str(job.id)})
        async with sessionmaker() as session:
            j = await session.get(Job, job.id)
            if j is not None:
                j.status = (
                    JobStatus.CANCELLED if isinstance(exc, KeyboardInterrupt) else JobStatus.FAILED
                )
                j.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                j.finished_at = datetime.now(UTC)
                await session.commit()
        await bus.publish(
            Event(
                channel=Channel.JOBS,
                type="failed",
                payload={"job_id": str(job.id), "error": str(exc)},
            )
        )
        raise

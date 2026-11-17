"""Data catalogue, chart bars and job progress."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import Dataset, Instrument, Job
from app.models.enums import DataSource, JobKind, JobStatus, Timeframe
from engine.data.store import ParquetStore

router = APIRouter(tags=["data"])


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source: DataSource
    timeframe: Timeframe
    start_ts: datetime | None
    end_ts: datetime | None
    row_count: int
    quality: dict[str, Any]
    last_ingested_at: datetime | None


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: JobKind
    status: JobStatus
    progress: float
    message: str | None
    params: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class PriceSide(StrEnum):
    BID = "bid"
    ASK = "ask"
    MID = "mid"


class BarsResponse(BaseModel):
    symbol: str
    timeframe: Timeframe
    source: DataSource
    side: PriceSide
    # Column arrays, which are compact and map straight onto Lightweight Charts series.
    time: list[int]  # bar open, unix seconds UTC
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    spread: list[float]  # ask_close - bid_close


def _store(request: Request) -> ParquetStore:
    return ParquetStore(request.app.state.settings.data_dir / "parquet")


@router.get("/data/{symbol}/datasets", response_model=list[DatasetRead])
async def list_datasets(symbol: str, session: SessionDep) -> Sequence[Dataset]:
    stmt = (
        select(Dataset)
        .join(Instrument)
        .where(Instrument.symbol == symbol.upper())
        .order_by(Dataset.source, Dataset.timeframe)
    )
    return (await session.scalars(stmt)).all()


@router.get("/data/{symbol}/bars", response_model=BarsResponse)
async def get_bars(
    request: Request,
    symbol: str,
    timeframe: Timeframe = Timeframe.M15,
    source: DataSource = DataSource.DUKASCOPY,
    side: PriceSide = PriceSide.MID,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(2000, ge=1, le=20000),
) -> BarsResponse:
    """Bars for charting. Without ``start`` returns the latest ``limit`` bars."""
    if timeframe is Timeframe.TICK:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "use a bar timeframe")
    store = _store(request)
    bars = await run_in_threadpool(
        store.read_bars,
        symbol.upper(),
        timeframe,
        source=source,
        start=start,
        end=end,
        limit=limit,
        tail=start is None,
    )

    def field(name: str) -> list[float]:
        if side is PriceSide.MID:
            values = (bars[f"bid_{name}"] + bars[f"ask_{name}"]) / 2
        else:
            values = bars[f"{side.value}_{name}"]
        return [round(float(v), 3) for v in values]

    return BarsResponse(
        symbol=symbol.upper(),
        timeframe=timeframe,
        source=source,
        side=side,
        time=(bars["ts"].astype("int64") // 1_000_000_000).tolist(),
        open=field("open"),
        high=field("high"),
        low=field("low"),
        close=field("close"),
        volume=np.round(bars["volume"].to_numpy(), 4).tolist(),
        spread=np.round((bars["ask_close"] - bars["bid_close"]).to_numpy(), 3).tolist(),
    )


@router.get("/jobs", response_model=list[JobRead])
async def list_jobs(
    session: SessionDep,
    kind: JobKind | None = None,
    status_: JobStatus | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=500),
) -> Sequence[Job]:
    stmt = select(Job)
    if kind:
        stmt = stmt.where(Job.kind == kind)
    if status_:
        stmt = stmt.where(Job.status == status_)
    return (await session.scalars(stmt.order_by(Job.created_at.desc()).limit(limit))).all()


@router.get("/jobs/{job_id}", response_model=JobRead)
async def get_job(job_id: uuid.UUID, session: SessionDep) -> Job:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.api.deps import SessionDep
from app.models import EvolutionRun, Generation
from app.models.enums import DataSource, RunStatus, Timeframe
from app.schemas.common import Page
from app.schemas.evolution import GenerationRead, RunRead
from app.services.evolution import create_run, execute_run
from engine.data.store import ParquetStore
from engine.evolution.engine import EvolutionConfig
from engine.evolution.fitness import FitnessConfig

router = APIRouter(prefix="/runs", tags=["lab"])


class StartRunRequest(BaseModel):
    name: str = Field("xau-swarm", max_length=128)
    symbol: str = "XAUUSD"
    timeframe: Timeframe = Timeframe.M15
    source: DataSource = DataSource.DUKASCOPY
    start: datetime
    end: datetime
    islands: int = Field(4, ge=1, le=32)
    population: int = Field(40, ge=6, le=1000)
    generations: int = Field(25, ge=1, le=1000)
    htf: list[Timeframe] = Field(default_factory=list)
    seed: int = 0
    holdout_fraction: float = Field(0.2, ge=0.05, lt=0.9)
    n_folds: int = Field(5, ge=2, le=20)
    validate_top: int = Field(10, ge=0, le=200)
    meta_top: int = Field(5, ge=0, le=100)
    workers: int = Field(1, ge=1, le=64)
    min_trades: int = Field(30, ge=5, le=10_000)


def _registry(request: Request) -> dict[uuid.UUID, tuple[asyncio.Task[Any], threading.Event]]:
    reg: dict[uuid.UUID, tuple[asyncio.Task[Any], threading.Event]] = request.app.state.runs
    return reg


def _launch(request: Request, run_id: uuid.UUID, workers: int) -> None:
    settings = request.app.state.settings
    stop = threading.Event()
    task = asyncio.create_task(
        execute_run(
            request.app.state.sessionmaker,
            ParquetStore(settings.data_dir / "parquet"),
            request.app.state.event_bus,
            run_id,
            checkpoint_dir=settings.data_dir / "runs",
            workers=workers,
            stop=stop,
        ),
        name=f"evolution-{run_id}",
    )
    reg = _registry(request)
    reg[run_id] = (task, stop)
    task.add_done_callback(lambda _t: reg.pop(run_id, None))


@router.post("", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED)
async def start_run(body: StartRunRequest, request: Request, session: SessionDep) -> EvolutionRun:
    """Create an evolution run and start it in the background (progress on ``ws/lab``)."""
    if body.end <= body.start:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "end must be after start")
    cfg = EvolutionConfig(
        islands=body.islands,
        population=body.population,
        generations=body.generations,
        htf=tuple(t.value for t in body.htf),
        seed=body.seed,
        fitness=FitnessConfig(min_trades=body.min_trades),
    )
    try:
        run = await create_run(
            session,
            name=body.name,
            symbol=body.symbol,
            timeframe=body.timeframe,
            start=body.start,
            end=body.end,
            config=cfg,
            source=body.source,
            holdout_fraction=body.holdout_fraction,
            n_folds=body.n_folds,
            validate_top=body.validate_top,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    await session.commit()
    _launch(request, run.id, body.workers)
    return run


@router.post("/{run_id}/stop", response_model=RunRead)
async def stop_run(run_id: uuid.UUID, request: Request, session: SessionDep) -> EvolutionRun:
    """Pause after the current generation (checkpointed; resume with /resume)."""
    run = await session.get(EvolutionRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    entry = _registry(request).get(run_id)
    if entry is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "run is not executing in this process")
    entry[1].set()
    return run


@router.post("/{run_id}/resume", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED)
async def resume_run(
    run_id: uuid.UUID, request: Request, session: SessionDep, workers: int = 1
) -> EvolutionRun:
    run = await session.get(EvolutionRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    if run_id in _registry(request):
        raise HTTPException(status.HTTP_409_CONFLICT, "run is already executing")
    if run.status not in (RunStatus.PAUSED, RunStatus.FAILED, RunStatus.PENDING):
        raise HTTPException(status.HTTP_409_CONFLICT, f"cannot resume a {run.status.value} run")
    _launch(request, run_id, workers)
    return run


@router.get("", response_model=Page[RunRead])
async def list_runs(
    session: SessionDep,
    status_: RunStatus | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Page[RunRead]:
    stmt = select(EvolutionRun)
    if status_ is not None:
        stmt = stmt.where(EvolutionRun.status == status_)
    total = await session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = await session.scalars(
        stmt.order_by(EvolutionRun.created_at.desc()).limit(limit).offset(offset)
    )
    return Page(
        items=[RunRead.model_validate(r) for r in rows], total=total, limit=limit, offset=offset
    )


@router.get("/{run_id}", response_model=RunRead)
async def get_run(run_id: uuid.UUID, session: SessionDep) -> EvolutionRun:
    run = await session.get(EvolutionRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    return run


@router.get("/{run_id}/generations", response_model=list[GenerationRead])
async def list_generations(
    run_id: uuid.UUID,
    session: SessionDep,
    island: int | None = None,
    since: int = Query(0, ge=0, description="Only generations with number >= since"),
) -> Sequence[Generation]:
    if await session.get(EvolutionRun, run_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    stmt = select(Generation).where(Generation.run_id == run_id, Generation.number >= since)
    if island is not None:
        stmt = stmt.where(Generation.island == island)
    return (await session.scalars(stmt.order_by(Generation.number, Generation.island))).all()

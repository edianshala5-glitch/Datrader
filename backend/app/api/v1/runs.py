from __future__ import annotations

import uuid
from collections.abc import Sequence

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.api.deps import SessionDep
from app.models import EvolutionRun, Generation
from app.models.enums import RunStatus
from app.schemas.common import Page
from app.schemas.evolution import GenerationRead, RunRead

router = APIRouter(prefix="/runs", tags=["lab"])


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

from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import Instrument
from app.schemas.instrument import InstrumentRead

router = APIRouter(prefix="/instruments", tags=["instruments"])


@router.get("", response_model=list[InstrumentRead])
async def list_instruments(session: SessionDep, active_only: bool = True) -> Sequence[Instrument]:
    stmt = select(Instrument).order_by(Instrument.symbol)
    if active_only:
        stmt = stmt.where(Instrument.is_active.is_(True))
    return (await session.scalars(stmt)).all()


@router.get("/{symbol}", response_model=InstrumentRead)
async def get_instrument(symbol: str, session: SessionDep) -> Instrument:
    instrument = await session.scalar(select(Instrument).where(Instrument.symbol == symbol.upper()))
    if instrument is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"instrument {symbol!r} not found")
    return instrument

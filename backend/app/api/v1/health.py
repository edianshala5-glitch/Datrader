from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import __version__
from app.api.deps import EventBusDep, SessionDep
from app.schemas.common import Health, HealthComponent

router = APIRouter(tags=["system"])


@router.get("/health", response_model=Health)
async def health(request: Request, session: SessionDep, bus: EventBusDep) -> Health:
    try:
        await session.execute(text("SELECT 1"))
        db = HealthComponent(ok=True)
    except SQLAlchemyError as exc:
        db = HealthComponent(ok=False, detail=type(exc).__name__)

    bus_ok = await bus.ping()
    bus_component = HealthComponent(ok=bus_ok, detail=type(bus).__name__)

    settings = request.app.state.settings
    return Health(
        status="ok" if db.ok and bus_ok else "degraded",
        version=__version__,
        environment=settings.environment,
        database=db,
        event_bus=bus_component,
    )

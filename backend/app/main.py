"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1 import api_router
from app.core.config import Settings, get_settings
from app.core.events import EventBus, create_event_bus
from app.core.logging import configure_logging
from app.db.session import create_engine, create_sessionmaker

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, event_bus: EventBus | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings.database_url, echo=settings.database_echo)
        bus = event_bus or create_event_bus(settings.redis_url, settings.event_channel_prefix)
        await bus.start()
        app.state.settings = settings
        app.state.engine = engine
        app.state.sessionmaker = create_sessionmaker(engine)
        app.state.event_bus = bus
        app.state.runs = {}  # run_id -> (asyncio.Task, stop Event) for background evolution runs
        logger.info(
            "api started",
            extra={"environment": settings.environment, "bus": type(bus).__name__},
        )
        try:
            yield
        finally:
            # Ask running evolutions to pause (they checkpoint), then wait for them.
            for _task, stop in list(app.state.runs.values()):
                stop.set()
            tasks = [t for t, _ in app.state.runs.values()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await bus.close()
            await engine.dispose()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app

"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__
from app.api.auth import require_auth
from app.api.v1 import api_router
from app.core import metrics
from app.core.config import Settings, get_settings
from app.core.events import EventBus, create_event_bus
from app.core.logging import configure_logging
from app.core.observability import ObservabilityMiddleware
from app.core.security import LoginThrottle, TokenSigner, check_hash_format
from app.db.session import create_engine, create_sessionmaker

logger = logging.getLogger(__name__)
SHUTDOWN_GRACE_S = 15.0


def _make_signer(settings: Settings) -> TokenSigner | None:
    if not settings.auth_enabled:
        if settings.environment == "prod":
            logger.warning("authentication is DISABLED (SWARM_AUTH_DISABLED=true)")
        return None
    assert settings.auth_password_hash is not None
    password_hash = settings.auth_password_hash.get_secret_value()
    check_hash_format(password_hash)
    if settings.secret_key and settings.secret_key.get_secret_value():
        key = settings.secret_key.get_secret_value().encode()
    else:
        logger.warning("SWARM_SECRET_KEY not set: sessions end when the API restarts")
        key = secrets.token_bytes(32)
    return TokenSigner(key, password_hash, settings.token_ttl_hours * 3600)


def _refresh_gauges(app: FastAPI) -> None:
    runs = getattr(app.state, "runs", {})
    metrics.RUNS_ACTIVE.set(sum(1 for task, _ in runs.values() if not task.done()))
    live = getattr(app.state, "live", None)
    running = bool(live and not live["task"].done())
    metrics.LIVE_RUNNING.set(1 if running else 0)
    status = live["service"].status if live else {}
    metrics.LIVE_BARS.set(status.get("bars", 0))
    last = status.get("last_bar")
    if running and last:
        age = (datetime.now(UTC) - pd.Timestamp(last).to_pydatetime()).total_seconds()
        metrics.LIVE_LAST_BAR_AGE.set(max(0.0, age))
    else:
        metrics.LIVE_LAST_BAR_AGE.set(float("nan"))


def create_app(settings: Settings | None = None, event_bus: EventBus | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    signer = _make_signer(settings)

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
            extra={
                "environment": settings.environment,
                "bus": type(bus).__name__,
                "auth": signer is not None,
            },
        )
        if settings.live_autostart:
            from app.api.v1.live import LiveStartRequest, start_live
            from app.models.enums import DataSource

            try:
                start_live(
                    app.state,
                    LiveStartRequest(
                        feed="oanda", history_source=DataSource(settings.live_history_source)
                    ),
                )
                logger.info("live engine autostarted (oanda, supervised)")
            except (HTTPException, ValueError) as exc:
                logger.error("live autostart failed", extra={"error": repr(exc)})
        try:
            yield
        finally:
            # Ask running evolutions to pause (they checkpoint) and the live engine to stop.
            for _task, stop in list(app.state.runs.values()):
                stop.set()
            tasks = [t for t, _ in app.state.runs.values()]
            live = getattr(app.state, "live", None)
            if live and not live["task"].done():
                live["stop"].set()
                tasks.append(live["task"])
            if tasks:
                # The live engine only notices `stop` on its next bar (none arrive at weekends).
                _done, pending = await asyncio.wait(tasks, timeout=SHUTDOWN_GRACE_S)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await bus.close()
            await engine.dispose()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs" if settings.docs_enabled else None,
        openapi_url="/api/openapi.json" if settings.docs_enabled else None,
    )
    app.state.signer = signer
    app.state.login_throttle = LoginThrottle(
        settings.login_max_failures, settings.login_lockout_seconds
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,  # bearer tokens, no cookies
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    app.add_middleware(ObservabilityMiddleware)  # outermost: sees every request
    app.include_router(api_router)

    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False, dependencies=[Depends(require_auth)])
        async def prometheus_metrics(request: Request) -> Response:
            _refresh_gauges(request.app)
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app

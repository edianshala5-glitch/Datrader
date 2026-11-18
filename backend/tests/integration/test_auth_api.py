"""Auth, observability and live-supervisor behaviour of the API."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.v1.live import supervise_live
from app.cli import resolve_ingest_start
from app.core.config import Settings
from app.core.events import InMemoryEventBus
from app.core.security import hash_password
from app.db.base import Base
from app.db.session import create_engine, create_sessionmaker
from app.main import create_app
from app.models.enums import DataSource
from app.services.seed import seed_reference_data
from engine.data.store import Coverage

PW = "a long dashboard password"


async def _prepare_db(url: str) -> None:
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with create_sessionmaker(engine)() as session:
        await seed_reference_data(session)
        await session.commit()
    await engine.dispose()


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    url = f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}"
    asyncio.run(_prepare_db(url))
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, database_url=url, data_dir=tmp_path, environment="test", **overrides
    )
    app = create_app(settings, event_bus=InMemoryEventBus())

    @app.get("/api/v1/_boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def pw_hash() -> str:
    return hash_password(PW)


@pytest.fixture
def secured(tmp_path: Path, pw_hash: str) -> Iterator[TestClient]:
    with _client(
        tmp_path, auth_password_hash=pw_hash, secret_key="k" * 40, login_max_failures=3
    ) as c:
        yield c


def _login(c: TestClient) -> str:
    res = c.post("/api/v1/auth/login", json={"password": PW})
    assert res.status_code == 200, res.text
    return str(res.json()["token"])


def test_routes_require_token(secured: TestClient) -> None:
    assert secured.get("/api/v1/health").status_code == 200  # public
    res = secured.get("/api/v1/instruments")
    assert res.status_code == 401 and res.headers["www-authenticate"] == "Bearer"
    assert secured.get("/metrics").status_code == 401
    assert secured.get("/api/v1/auth/session").json() == {
        "auth_enabled": True,
        "authenticated": False,
    }

    token = _login(secured)
    auth = {"Authorization": f"Bearer {token}"}
    assert secured.get("/api/v1/instruments", headers=auth).status_code == 200
    assert secured.get("/api/v1/auth/session", headers=auth).json()["authenticated"]
    assert secured.get("/api/v1/live/status", headers=auth).json() == {"running": False}
    # Query-string tokens are for WebSockets only.
    assert secured.get(f"/api/v1/instruments?token={token}").status_code == 401
    bad = {"Authorization": f"Bearer {token[:-3]}xyz"}
    assert secured.get("/api/v1/instruments", headers=bad).status_code == 401


def test_websocket_auth(secured: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as exc, secured.websocket_connect("/api/v1/ws/lab"):
        pass
    assert exc.value.code == 1008
    token = _login(secured)
    with secured.websocket_connect(f"/api/v1/ws/lab?token={token}") as ws:
        ws.close()


def test_login_lockout(secured: TestClient) -> None:
    for _ in range(3):
        assert secured.post("/api/v1/auth/login", json={"password": "nope"}).status_code == 401
    locked = secured.post("/api/v1/auth/login", json={"password": PW})
    assert locked.status_code == 429 and int(locked.headers["retry-after"]) > 0


def test_metrics_and_request_ids(secured: TestClient) -> None:
    token = _login(secured)
    auth = {"Authorization": f"Bearer {token}"}
    res = secured.get("/api/v1/instruments/XAUUSD", headers={**auth, "X-Request-ID": "abc-123"})
    assert res.headers["x-request-id"] == "abc-123"
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["cache-control"] == "no-store"
    generated = secured.get("/api/v1/health", headers={"X-Request-ID": "bad id with spaces"})
    assert generated.headers["x-request-id"] != "bad id with spaces"

    body = secured.get("/metrics", headers=auth).text
    assert 'route="/api/v1/instruments/{symbol}"' in body  # templated, never raw paths
    assert "swarm_live_engine_running 0.0" in body
    assert 'swarm_login_attempts_total{outcome="success"}' in body


def test_unhandled_error_is_json_with_request_id(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        res = c.get("/api/v1/_boom", headers={"X-Request-ID": "trace-me"})
    assert res.status_code == 500
    assert res.json() == {"detail": "internal server error", "request_id": "trace-me"}


def test_auth_disabled(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        assert c.get("/api/v1/instruments").status_code == 200
        assert c.get("/metrics").status_code == 200
        assert c.get("/api/v1/auth/session").json()["auth_enabled"] is False
        assert c.post("/api/v1/auth/login", json={"password": "x"}).status_code == 409
        res = c.post("/api/v1/live/start", json={"feed": "capital"})
        assert res.status_code == 422 and "SWARM_CAPITAL_API_KEY" in res.text
        assert c.get("/api/v1/live/status").json() == {"running": False}
        with c.websocket_connect("/api/v1/ws/signals") as ws:
            ws.close()


async def test_live_supervisor_restarts_after_crash(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)  # type: ignore[call-arg]
    state = SimpleNamespace(
        settings=settings, event_bus=InMemoryEventBus(), sessionmaker=None, live={}
    )
    stop = threading.Event()
    calls: list[int] = []

    class CrashingFeed:
        name = "crashing"

        async def backfill(self, n: int) -> pd.DataFrame:
            calls.append(n)
            if len(calls) == 3:
                stop.set()
            raise ConnectionError("stream down")

    await asyncio.wait_for(
        supervise_live(
            state,
            stop,
            CrashingFeed,  # type: ignore[arg-type]
            symbol="XAUUSD",
            history_source=DataSource.SYNTHETIC,
            min_backoff=0.01,
        ),
        timeout=10,
    )
    assert len(calls) == 3
    status = state.live["service"].status
    assert status["supervised"] and "stream down" in status["error"]


def test_ingest_start_last() -> None:
    class Store:
        def __init__(self, cov: Coverage) -> None:
            self.cov = cov

        def bar_coverage(self, *_: Any) -> Coverage:
            return self.cov

    last = pd.Timestamp("2024-03-05 14:37", tz="UTC")
    start = resolve_ingest_start("last", Store(Coverage(None, last, 10)), "XAUUSD", "dukascopy")
    assert start == datetime(2024, 3, 5, tzinfo=UTC)  # whole newest day is re-fetched
    explicit = resolve_ingest_start("2020-01-02", Store(Coverage(None, None, 0)), "X", "dukascopy")
    assert explicit == datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(SystemExit, match="no stored"):
        resolve_ingest_start("last", Store(Coverage(None, None, 0)), "XAUUSD", "dukascopy")


def test_tokens_are_redacted_from_logs() -> None:
    import logging

    from app.core.observability import RedactTokensFilter

    record = logging.LogRecord(
        "uvicorn.error",
        logging.INFO,
        "",
        0,
        '%s - "WebSocket %s" [accepted]',
        ("1.2.3.4:5", "/api/v1/ws/lab?token=abc.def&x=1"),
        None,
    )
    assert RedactTokensFilter().filter(record)
    assert record.getMessage() == '1.2.3.4:5 - "WebSocket /api/v1/ws/lab?token=***&x=1" [accepted]'


def test_cli_initialises_a_fresh_sqlite_database(tmp_path: Path) -> None:
    import sqlite3

    from app.cli import ensure_sqlite_database

    url = f"sqlite+aiosqlite:///{tmp_path / 'new' / 'swarm.db'}"  # folder doesn't exist yet
    assert ensure_sqlite_database(url) is True
    with sqlite3.connect(tmp_path / "new" / "swarm.db") as conn:
        assert conn.execute("SELECT symbol FROM instruments").fetchall() == [("XAUUSD",)]
    assert ensure_sqlite_database(url) is False  # already initialised: no-op
    assert ensure_sqlite_database("postgresql+asyncpg://u:p@db/x_test") is False

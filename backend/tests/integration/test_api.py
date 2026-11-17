from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.events import Channel, Event, InMemoryEventBus
from app.db.base import Base
from app.db.session import create_engine, create_sessionmaker
from app.main import create_app
from app.models import EvolutionRun, Generation, Strategy
from app.models.enums import StrategyOrigin, StrategyStatus, Timeframe
from app.services.seed import seed_reference_data


async def _prepare_db(url: str) -> dict[str, uuid.UUID]:
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    ids: dict[str, uuid.UUID] = {}
    async with create_sessionmaker(engine)() as session:
        [xau] = await seed_reference_data(session)
        run = EvolutionRun(name="run-1", instrument=xau, timeframe=Timeframe.M5, seed=1)
        run.generations += [
            Generation(island=i, number=n, population_size=50, stats={"fitness": {"max": n}})
            for n in range(3)
            for i in range(2)
        ]
        rows = [
            # (genome, fitness, trades, win rate, overfit risk, status)
            ({"k": 1}, 2.1, 340, 0.58, 0.20, StrategyStatus.VALIDATED),
            ({"k": 2}, 3.5, 25, 0.71, 0.85, StrategyStatus.CANDIDATE),  # too few trades
            ({"k": 3}, 1.2, 510, 0.49, 0.30, StrategyStatus.VALIDATED),
            ({"k": 4}, None, None, None, None, StrategyStatus.CANDIDATE),
        ]
        strategies = []
        for genome, fit, n, wr, risk, st in rows:
            s = Strategy.from_genome(
                genome,
                instrument=xau,
                timeframe=Timeframe.M5,
                run=run,
                origin=StrategyOrigin.RANDOM,
                status=st,
                fitness=fit,
                oos_trades=n,
                oos_win_rate=wr,
                overfit_risk=risk,
            )
            strategies.append(s)
        strategies[2].parents.append(strategies[0])
        session.add_all([run, *strategies])
        await session.commit()
        ids["run"] = run.id
        ids["child"] = strategies[2].id
        ids["parent"] = strategies[0].id
    await engine.dispose()
    return ids


@pytest.fixture
def ids_and_client(tmp_path: Path) -> Iterator[tuple[dict[str, uuid.UUID], TestClient]]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'api.db'}"
    ids = asyncio.run(_prepare_db(url))
    settings = Settings(_env_file=None, database_url=url, environment="test")  # type: ignore[call-arg]
    app = create_app(settings, event_bus=InMemoryEventBus())
    with TestClient(app) as client:
        yield ids, client


@pytest.fixture
def client(ids_and_client: tuple[dict[str, uuid.UUID], TestClient]) -> TestClient:
    return ids_and_client[1]


def test_health(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["database"]["ok"] and body["event_bus"]["ok"]
    assert body["environment"] == "test"


def test_instruments(client: TestClient) -> None:
    items = client.get("/api/v1/instruments").json()
    assert [i["symbol"] for i in items] == ["XAUUSD"]
    xau = client.get("/api/v1/instruments/xauusd").json()  # case-insensitive
    assert xau["trading_hours"]["day_roll"] == "17:00:00"
    assert xau["tick_size"] == "0.01000000"
    assert client.get("/api/v1/instruments/EURUSD").status_code == 404


def test_strategy_leaderboard_filters_and_sort(client: TestClient) -> None:
    page = client.get("/api/v1/strategies", params={"symbol": "XAUUSD"}).json()
    assert page["total"] == 4
    fitness = [s["fitness"] for s in page["items"]]
    assert fitness == [3.5, 2.1, 1.2, None]  # desc, nulls last

    robust = client.get(
        "/api/v1/strategies",
        params={"min_trades": 100, "max_overfit_risk": 0.5, "sort": "oos_win_rate"},
    ).json()
    assert [s["oos_win_rate"] for s in robust["items"]] == [0.58, 0.49]

    validated = client.get(
        "/api/v1/strategies", params=[("status", "validated"), ("timeframe", "5m")]
    ).json()
    assert validated["total"] == 2

    assert client.get("/api/v1/strategies", params={"sort": "DROP TABLE"}).status_code == 422
    paged = client.get("/api/v1/strategies", params={"limit": 1, "offset": 1}).json()
    assert len(paged["items"]) == 1 and paged["total"] == 4


def test_strategy_detail(ids_and_client: tuple[dict[str, uuid.UUID], TestClient]) -> None:
    ids, client = ids_and_client
    body = client.get(f"/api/v1/strategies/{ids['child']}").json()
    assert body["parent_ids"] == [str(ids["parent"])]
    assert body["genome"] == {"k": 3}
    assert body["symbol"] == "XAUUSD"
    assert client.get(f"/api/v1/strategies/{uuid.uuid4()}").status_code == 404


def test_runs_and_generations(ids_and_client: tuple[dict[str, uuid.UUID], TestClient]) -> None:
    ids, client = ids_and_client
    runs = client.get("/api/v1/runs").json()
    assert runs["total"] == 1 and runs["items"][0]["symbol"] == "XAUUSD"
    gens = client.get(f"/api/v1/runs/{ids['run']}/generations", params={"since": 1}).json()
    assert [(g["number"], g["island"]) for g in gens] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    island1 = client.get(f"/api/v1/runs/{ids['run']}/generations", params={"island": 1}).json()
    assert len(island1) == 3
    assert client.get(f"/api/v1/runs/{uuid.uuid4()}/generations").status_code == 404


def test_websocket_streams_bus_events(client: TestClient) -> None:
    bus = client.app.state.event_bus  # type: ignore[attr-defined]
    with client.websocket_connect("/api/v1/ws/lab") as ws:
        client.portal.call(  # type: ignore[union-attr]
            bus.publish, Event(channel=Channel.LAB, type="generation", payload={"number": 7})
        )
        msg = ws.receive_json()
    assert msg["type"] == "generation" and msg["payload"] == {"number": 7}


def test_websocket_rejects_unknown_channel(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/v1/ws/bogus") as ws:
        ws.receive_text()

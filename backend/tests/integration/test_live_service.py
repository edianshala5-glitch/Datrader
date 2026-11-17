"""Champions lifecycle and the live service against a real DB and data lake."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.core.config import Settings
from app.core.events import Channel, InMemoryEventBus
from app.db.base import Base
from app.db.session import create_engine, create_sessionmaker
from app.main import create_app
from app.models import Champion, Instrument, Signal, Strategy
from app.models.enums import (
    ChampionStatus,
    DataSource,
    Side,
    SignalKind,
    SignalStatus,
    StrategyOrigin,
    StrategyStatus,
    Timeframe,
)
from app.services.champions import ChampionPolicy, PromotionError, evaluate_champions, promote
from app.services.ingest import run_ingest
from app.services.live import LiveService
from app.services.seed import seed_reference_data
from app.services.validation import HoldoutAlreadyEvaluatedError
from engine.backtest.runner import CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.data.sources.base import Chunk
from engine.data.sources.synthetic import generate_m1
from engine.data.store import ParquetStore
from engine.live.feeds import ReplayFeed
from engine.live.notify import WebhookNotifier
from engine.strategy.genome import Genome

CAL = MarketCalendar.xauusd()
START, END = datetime(2022, 1, 2, tzinfo=UTC), datetime(2023, 7, 2, tzinfo=UTC)
MORNING = Genome.model_validate(
    {
        "name": "NY morning",
        "entry_long": {
            "type": "group",
            "conditions": [
                {
                    "type": "range",
                    "feature": {"indicator": "time", "output": "minute_ny"},
                    "lo": 480,
                    "hi": 480,
                }
            ],
        },
        "exit": {
            "stop": {"type": "atr", "mult": 4.0},
            "target": None,
            "max_bars": 8,
            "flat_at": "daily",
        },
    }
)


def drift(m: pd.DatetimeIndex) -> np.ndarray:  # type: ignore[type-arg]
    h = m.tz_convert("America/New_York").hour
    return np.where((h >= 8) & (h < 11), 6e-6, 0.0)


class Planted:
    source = DataSource.SYNTHETIC
    symbol = "XAUUSD"

    def __init__(self, mu_on: bool = True) -> None:
        self.mu_on = mu_on

    async def chunks(self, start: datetime, end: datetime):  # type: ignore[no-untyped-def]
        yield Chunk(
            "all",
            generate_m1(start, end, CAL, seed=9, drift=drift if self.mu_on else None),
            progress=1.0,
        )


@pytest.fixture
async def lake(engine: AsyncEngine, xauusd: Instrument, tmp_path: Path) -> ParquetStore:
    store = ParquetStore(tmp_path / "parquet")
    await run_ingest(
        create_sessionmaker(engine), store, InMemoryEventBus(), Planted(), start=START, end=END
    )
    return store


def _strategy(xau: Instrument, genome: Genome = MORNING) -> Strategy:
    return Strategy.from_genome(
        genome.canonical(),
        instrument_id=xau.id,
        timeframe=Timeframe.M15,
        origin=StrategyOrigin.MANUAL,
        status=StrategyStatus.VALIDATED,
        oos_expectancy_r=0.1,
        oos_trades=300,
        oos_win_rate=0.55,
    )


# --------------------------------------------------------------------------- promotion
async def test_promote_runs_holdout_once(
    session: AsyncSession, xauusd: Instrument, lake: ParquetStore
) -> None:
    s = _strategy(xauusd)
    session.add(s)
    await session.commit()
    res = await promote(session, lake, s, source=DataSource.SYNTHETIC, start=START, end=END)
    await session.commit()
    assert res.passed, res.reason
    assert res.holdout["n_trades"] >= 20 and res.holdout["expectancy_r"] > 0
    assert s.status is StrategyStatus.INCUBATING and res.champion is not None
    assert res.champion.status is ChampionStatus.INCUBATING
    assert res.champion.promotion_snapshot["holdout_expectancy_r"] == res.holdout["expectancy_r"]
    with pytest.raises(PromotionError, match="only validated"):
        await promote(session, lake, s, source=DataSource.SYNTHETIC, start=START, end=END)
    s.status = StrategyStatus.VALIDATED  # even if someone resets the status...
    with pytest.raises(HoldoutAlreadyEvaluatedError):  # ...the holdout can't be looked at twice
        await promote(session, lake, s, source=DataSource.SYNTHETIC, start=START, end=END)


async def test_failed_holdout_rejects(
    session: AsyncSession, xauusd: Instrument, lake: ParquetStore
) -> None:
    evening = Genome.model_validate(
        {
            **MORNING.canonical(),
            "entry_long": {
                "type": "group",
                "conditions": [
                    {
                        "type": "range",
                        "feature": {"indicator": "time", "output": "minute_ny"},
                        "lo": 1320,
                        "hi": 1320,
                    }
                ],
            },
            "entry_short": None,
        }
    )
    s = _strategy(xauusd, evening)
    session.add(s)
    await session.commit()
    res = await promote(session, lake, s, source=DataSource.SYNTHETIC, start=START, end=END)
    assert not res.passed and s.status is StrategyStatus.REJECTED and res.champion is None


# --------------------------------------------------------------------------- lifecycle
async def _champion_with_trades(
    session: AsyncSession, xau: Instrument, rs: list[float], expected: float = 0.1
) -> Champion:
    s = Strategy.from_genome(
        {**MORNING.canonical(), "name": f"c{len(rs)}{rs[:2]}{expected}"},
        instrument=xau,
        timeframe=Timeframe.M15,
        origin=StrategyOrigin.MANUAL,
        status=StrategyStatus.INCUBATING,
    )
    session.add(s)
    await session.flush()
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    c = Champion(
        strategy_id=s.id,
        status=ChampionStatus.INCUBATING,
        promoted_at=t0,
        promotion_snapshot={"oos_expectancy_r": expected},
        live_stats={},
    )
    session.add(c)
    await session.flush()
    for i, r in enumerate(rs):
        ts = t0 + timedelta(hours=i)
        session.add(
            Signal(
                champion_id=c.id,
                strategy_id=s.id,
                instrument_id=xau.id,
                timeframe=Timeframe.M15,
                kind=SignalKind.ENTRY,
                side=Side.LONG,
                status=SignalStatus.CLOSED,
                bar_ts=ts,
                emitted_at=ts,
                closed_at=ts + timedelta(minutes=30),
                pnl_r=r,
            )
        )
    await session.flush()
    return c


async def test_activation_ranking_and_demotion(session: AsyncSession, xauusd: Instrument) -> None:
    good = [await _champion_with_trades(session, xauusd, [0.5, -0.2] * (6 + k)) for k in range(4)]
    young = await _champion_with_trades(session, xauusd, [0.5] * 3)  # too few trades
    crash = await _champion_with_trades(session, xauusd, [-1.0] * 12)  # drawdown 12R > 10R
    # mean -0.1R vs +0.3R expected, shallow drawdown (3R): only the degradation test can catch it
    decay = await _champion_with_trades(session, xauusd, [0.1, -0.3] * 15, expected=0.3)
    await session.commit()
    changes = await evaluate_champions(
        session, ChampionPolicy(), now=datetime(2024, 2, 1, tzinfo=UTC)
    )
    await session.commit()
    active = [c for c in good if c.status is ChampionStatus.ACTIVE]
    assert len(active) == 3 and sorted(c.rank for c in active) == [1, 2, 3]  # type: ignore[type-var]
    assert good[3].rank == 1  # most live trades at the same edge → most significant
    assert young.status is ChampionStatus.INCUBATING
    assert crash.status is ChampionStatus.DEMOTED and "drawdown" in (crash.demotion_reason or "")
    assert decay.status is ChampionStatus.DEMOTED and "vs backtest" in (decay.demotion_reason or "")
    assert {c["change"] for c in changes} >= {"activated", "demoted"}
    strat = await session.get(Strategy, crash.strategy_id)
    assert strat is not None and strat.status is StrategyStatus.RETIRED


# --------------------------------------------------------------------------- live service
async def test_live_service_matches_backtest_and_is_idempotent(
    engine: AsyncEngine, xauusd: Instrument, lake: ParquetStore
) -> None:
    factory = create_sessionmaker(engine)
    async with factory() as s:
        st = _strategy(xauusd)
        s.add(st)
        await s.flush()
        s.add(
            Champion(
                strategy_id=st.id,
                status=ChampionStatus.ACTIVE,
                rank=1,
                promoted_at=datetime.now(UTC),
                promotion_snapshot={"oos_expectancy_r": 0.1, "oos_win_rate": 0.55},
                live_stats={},
            )
        )
        await s.commit()
        sid = st.id
    replay_start = pd.Timestamp("2023-05-07 22:00", tz="UTC")  # Sunday open
    replay_end = pd.Timestamp("2023-06-11", tz="UTC")  # five weeks
    m1 = lake.read_bars("XAUUSD", Timeframe.M1, source=DataSource.SYNTHETIC, end=replay_end)
    bus = InMemoryEventBus()
    sent: list[str] = []

    def hook(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content)["content"])
        return httpx.Response(200)

    published: list[str] = []

    async def listen() -> None:
        async with bus.subscribe(Channel.SIGNALS) as events:
            async for e in events:
                published.append(e.type)
                if e.type == "live_stopped":
                    return

    listener = asyncio.create_task(listen())
    await asyncio.sleep(0)
    svc = LiveService(
        factory,
        lake,
        bus,
        history_source=DataSource.SYNTHETIC,
        window=3000,
        notifiers=[WebhookNotifier("https://hooks.example/x", transport=httpx.MockTransport(hook))],
        policy=ChampionPolicy(max_drawdown_r=1e9, degradation_min_trades=10**6),
    )
    await svc.run(ReplayFeed(m1, replay_start), backfill=10**7)
    await asyncio.wait_for(listener, 5)

    async with factory() as s:
        entries = (
            await s.scalars(
                select(Signal)
                .where(Signal.strategy_id == sid, Signal.kind == SignalKind.ENTRY)
                .order_by(Signal.bar_ts)
            )
        ).all()
        exits = await s.scalar(
            select(func.count()).select_from(Signal).where(Signal.kind == SignalKind.EXIT)
        )
    bars = lake.read_bars("XAUUSD", Timeframe.M15, source=DataSource.SYNTHETIC)
    bt = run_backtest(
        MORNING,
        bars,
        Timeframe.M15,
        calendar=CAL,
        sessions=xauusd.sessions,
        costs=CostConfig.from_instrument(xauusd),
        start=replay_start,
        end=replay_end,
        with_context=False,
    ).trades
    bt = bt[(bt["entry_ts"] > replay_start) & (bt["exit_reason"] != "end_of_data")]
    closed = [e for e in entries if e.pnl_r is not None]
    assert len(bt) >= 20 and len(closed) == len(bt)
    np.testing.assert_allclose([e.pnl_r for e in closed], bt["r"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(
        [e.entry_price for e in closed], bt["entry_price"].to_numpy(), atol=1e-9
    )
    assert exits == len(closed)
    assert all(e.rationale_text.startswith("LONG XAUUSD 15m") for e in closed)
    reasons = dict(zip(bt["entry_price"].round(9), bt["exit_reason"], strict=False))
    expected = {
        "stop": SignalStatus.STOP_HIT,
        "trailing_stop": SignalStatus.STOP_HIT,
        "target": SignalStatus.TARGET_HIT,
    }
    for e in closed:
        assert e.status is expected.get(reasons[round(e.entry_price, 9)], SignalStatus.CLOSED)
    assert {"entry", "fill", "exit", "live_started", "live_stopped"} <= set(published)
    assert len(sent) >= 2 * len(closed)  # active champion → entry + exit notifications

    # Restart and replay the same bars: no duplicate signals.
    svc2 = LiveService(
        factory,
        lake,
        InMemoryEventBus(),
        history_source=DataSource.SYNTHETIC,
        window=3000,
        policy=ChampionPolicy(max_drawdown_r=1e9, degradation_min_trades=10**6),
    )
    await svc2.run(ReplayFeed(m1, replay_start), backfill=10**7)
    async with factory() as s:
        again = await s.scalar(
            select(func.count()).select_from(Signal).where(Signal.kind == SignalKind.ENTRY)
        )
    assert again == len(entries)


async def test_incubating_champion_does_not_notify(
    engine: AsyncEngine, xauusd: Instrument, lake: ParquetStore
) -> None:
    factory = create_sessionmaker(engine)
    async with factory() as s:
        st = _strategy(xauusd)
        s.add(st)
        await s.flush()
        s.add(
            Champion(
                strategy_id=st.id,
                status=ChampionStatus.INCUBATING,
                promoted_at=datetime.now(UTC),
                promotion_snapshot={},
                live_stats={},
            )
        )
        await s.commit()
    sent: list[int] = []
    m1 = lake.read_bars("XAUUSD", Timeframe.M1, source=DataSource.SYNTHETIC)
    svc = LiveService(
        factory,
        lake,
        InMemoryEventBus(),
        history_source=DataSource.SYNTHETIC,
        window=3000,
        notifiers=[
            WebhookNotifier(
                "https://h/x",
                transport=httpx.MockTransport(lambda r: sent.append(1) or httpx.Response(200)),
            )
        ],
    )  # type: ignore[func-returns-value]
    await svc.run(
        ReplayFeed(m1[m1["ts"] < "2023-05-20"], pd.Timestamp("2023-05-07 22:00", tz="UTC")),
        backfill=10**7,
    )
    async with factory() as s:
        assert (await s.scalar(select(func.count()).select_from(Signal))) > 0  # paper-traded
    assert sent == []  # but silently


# --------------------------------------------------------------------------- API
def test_champions_and_live_api(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'api.db'}"

    async def prepare() -> str:
        eng = create_engine(url)
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = create_sessionmaker(eng)
        async with factory() as s:
            [xau] = await seed_reference_data(s)
            st = _strategy(xau)
            s.add(st)
            await s.commit()
            sid = str(st.id)
        await run_ingest(
            factory,
            ParquetStore(tmp_path / "parquet"),
            InMemoryEventBus(),
            Planted(),
            start=START,
            end=END,
        )
        await eng.dispose()
        return sid

    sid = asyncio.run(prepare())
    settings = Settings(_env_file=None, database_url=url, data_dir=tmp_path)  # type: ignore[call-arg]
    with TestClient(create_app(settings, event_bus=InMemoryEventBus())) as client:
        body = {"source": "synthetic", "start": START.isoformat(), "end": END.isoformat()}
        res = client.post(f"/api/v1/champions/promote/{sid}", json=body)
        assert res.status_code == 201, res.text
        assert res.json()["status"] == "incubating" and res.json()["holdout"]["n_trades"] >= 20
        assert (
            client.post(f"/api/v1/champions/promote/{sid}", json=body).status_code == 422
        )  # not validated now
        champs = client.get("/api/v1/champions").json()
        assert len(champs) == 1 and champs[0]["strategy_id"] == sid

        assert client.post("/api/v1/live/start", json={"feed": "replay"}).status_code == 422
        start = client.post(
            "/api/v1/live/start",
            json={
                "feed": "replay",
                "history_source": "synthetic",
                "replay_start": "2023-06-04T22:00:00Z",
            },
        )
        assert start.status_code == 202
        deadline = time.time() + 120
        while time.time() < deadline:
            st = client.get("/api/v1/live/status").json()
            if not st.get("running") and st.get("bars", 0) > 0:
                break
            time.sleep(0.5)
        assert st["champions"] == 1 and st["bars"] > 0 and "error" not in st
        sigs = client.get("/api/v1/signals", params={"kind": "entry"}).json()
        assert sigs and sigs[0]["rationale_text"].startswith("LONG")
        assert client.get(f"/api/v1/signals/{sigs[0]['id']}").json()["id"] == sigs[0]["id"]
        demoted = client.post(
            f"/api/v1/champions/{champs[0]['id']}/demote", json={"reason": "test"}
        ).json()
        assert demoted["status"] == "demoted"
        assert client.post("/api/v1/live/stop").status_code == 409  # replay already finished

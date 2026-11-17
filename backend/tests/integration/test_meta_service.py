"""Meta-Analyzer persistence and API against a real DB."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.events import InMemoryEventBus
from app.db.base import Base
from app.db.session import create_engine, create_sessionmaker
from app.main import create_app
from app.models import BacktestResult, Instrument, Strategy, strategy_lineage
from app.models.enums import (
    DataSource,
    RuleStatus,
    Segment,
    StrategyOrigin,
    StrategyStatus,
    Timeframe,
)
from app.services.ingest import run_ingest
from app.services.meta import persist_meta, record_meta_job
from app.services.seed import XAUUSD as SPEC
from app.services.seed import seed_reference_data
from engine.backtest.runner import CostConfig
from engine.data.calendar import MarketCalendar
from engine.data.resample import resample_bars
from engine.data.sources.base import Chunk
from engine.data.sources.synthetic import generate_m1
from engine.data.store import ParquetStore
from engine.meta.analyzer import analyze
from engine.strategy.genome import Genome
from engine.validation.splits import make_splits

CAL = MarketCalendar.xauusd()
ALWAYS_IN = Genome.model_validate(
    {
        "name": "always-in",
        "entry_long": {
            "type": "group",
            "conditions": [
                {
                    "type": "range",
                    "feature": {"indicator": "time", "output": "minute_ny"},
                    "lo": 0,
                    "hi": 1439,
                }
            ],
        },
        "exit": {
            "stop": {"type": "atr", "mult": 3.0},
            "target": None,
            "max_bars": 4,
            "flat_at": "daily",
        },
    }
)
START, END = datetime(2021, 1, 3, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)


def tuesday_drift(m: pd.DatetimeIndex) -> np.ndarray:  # type: ignore[type-arg]
    ny = m.tz_convert("America/New_York")
    return np.where((ny.day_of_week == 1) & (ny.hour >= 9) & (ny.hour < 11), 5e-5, 0.0)


async def test_persist_meta(session: AsyncSession, xauusd: Instrument) -> None:
    m15 = resample_bars(
        generate_m1(START, END, CAL, seed=9, drift=tuesday_drift), Timeframe.M15, CAL
    )
    splits = make_splits(START, END)
    report = analyze(
        ALWAYS_IN,
        m15,
        Timeframe.M15,
        splits,
        calendar=CAL,
        sessions=SPEC.model_dump(mode="json")["sessions"],
        costs=CostConfig.from_instrument(xauusd),
    )
    base = Strategy.from_genome(
        ALWAYS_IN.canonical(),
        instrument=xauusd,
        timeframe=Timeframe.M15,
        origin=StrategyOrigin.MANUAL,
    )
    session.add(base)
    await session.flush()
    rules = await persist_meta(session, base, report)
    job = await record_meta_job(session, base, report)
    await session.commit()

    assert len(rules) == len(report.rules) == 20
    kept = [r for r in rules if r.status is RuleStatus.VALIDATED]
    assert 1 <= len(kept) <= 3 and len(kept) == len(report.selected)
    top = kept[0]
    assert top.conditions[0]["feature"] in ("weekday", "hour_ny", "session_comex") or top.conditions
    assert top.p_value_adjusted is not None and top.p_value_adjusted < 0.05
    refined_id = top.details["refined_strategy_id"]
    refined = await session.get(Strategy, __import__("uuid").UUID(refined_id))
    assert refined is not None and refined.origin is StrategyOrigin.META
    assert (
        refined.status is StrategyStatus.VALIDATED
        and refined.oos_trades
        and refined.overfit_risk is not None
    )
    assert refined.overfit_risk < 0.5
    link = await session.scalar(
        select(func.count())
        .select_from(strategy_lineage)
        .where(strategy_lineage.c.child_id == refined.id, strategy_lineage.c.parent_id == base.id)
    )
    assert link == 1
    res = await session.scalar(
        select(BacktestResult).where(BacktestResult.context_rule_id == top.id)
    )
    assert res is not None and res.segment is Segment.VALIDATION and res.strategy_id == refined.id
    rejected = [r for r in rules if r.status is RuleStatus.REJECTED]
    assert all(r.details["reasons"] for r in rejected)
    assert job.result is not None and job.result["selected"] == len(kept)


def test_meta_api(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'api.db'}"

    class PlantedSource:
        source = DataSource.SYNTHETIC
        symbol = "XAUUSD"

        async def chunks(self, start: datetime, end: datetime):  # type: ignore[no-untyped-def]
            bars = generate_m1(start, end, CAL, seed=9, drift=tuesday_drift)
            yield Chunk(label="all", m1=bars, progress=1.0)

    async def prepare() -> str:
        eng = create_engine(url)
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = create_sessionmaker(eng)
        async with factory() as s:
            [xau] = await seed_reference_data(s)
            base = Strategy.from_genome(
                ALWAYS_IN.canonical(),
                instrument=xau,
                timeframe=Timeframe.M15,
                origin=StrategyOrigin.MANUAL,
            )
            s.add(base)
            await s.commit()
            sid = str(base.id)
        await run_ingest(
            factory,
            ParquetStore(tmp_path / "parquet"),
            InMemoryEventBus(),
            PlantedSource(),
            start=START,
            end=END,
        )
        await eng.dispose()
        return sid

    sid = asyncio.run(prepare())
    settings = Settings(_env_file=None, database_url=url, data_dir=tmp_path)  # type: ignore[call-arg]
    with TestClient(create_app(settings, event_bus=InMemoryEventBus())) as client:
        assert (
            client.post(f"/api/v1/strategies/{sid}/meta", json={"source": "synthetic"}).status_code
            == 422
        )
        res = client.post(
            f"/api/v1/strategies/{sid}/meta",
            json={"source": "synthetic", "start": START.isoformat(), "end": END.isoformat()},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert (
            body["candidates_tested"] == 20
            and body["selected"] >= 1
            and len(body["rule_ids"]) == 20
        )
        assert body["importance"] and body["importance"][0]["feature"]
        rules = client.get(
            f"/api/v1/strategies/{sid}/rules", params={"status_": "validated"}
        ).json()
        assert len(rules) == body["selected"]
        assert "Tue" in rules[0]["description"] and rules[0]["details"]["refined_strategy_id"]
        meta_strats = client.get(
            "/api/v1/strategies", params={"symbol": "XAUUSD", "status": "validated"}
        ).json()
        assert meta_strats["total"] >= 1

        adhoc = client.post(
            "/api/v1/meta",
            json={
                "genome": ALWAYS_IN.canonical(),
                "source": "synthetic",
                "start": START.isoformat(),
                "end": END.isoformat(),
            },
        )
        assert adhoc.status_code == 200 and adhoc.json()["selected"] >= 1

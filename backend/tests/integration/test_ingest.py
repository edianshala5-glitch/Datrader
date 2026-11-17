from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings
from app.core.events import Channel, InMemoryEventBus
from app.db.base import Base
from app.db.session import create_engine, create_sessionmaker
from app.main import create_app
from app.models import Dataset, Instrument, Job
from app.models.enums import DataSource, JobStatus, Timeframe
from app.services.ingest import run_ingest
from app.services.seed import seed_reference_data
from engine.data.calendar import MarketCalendar
from engine.data.sources.base import Chunk
from engine.data.sources.synthetic import SyntheticSource
from engine.data.store import ParquetStore

START = datetime(2023, 12, 20, tzinfo=UTC)  # spans New Year → two yearly partitions
END = datetime(2024, 1, 20, tzinfo=UTC)


async def test_synthetic_ingest_end_to_end(
    engine: AsyncEngine, xauusd: Instrument, tmp_path: Path
) -> None:
    store = ParquetStore(tmp_path)
    bus = InMemoryEventBus()
    factory = create_sessionmaker(engine)
    events = []

    async def listen() -> None:
        async with bus.subscribe(Channel.JOBS) as stream:
            async for e in stream:
                events.append(e)
                if e.type == "completed":
                    return

    listener = asyncio.create_task(listen())
    await asyncio.sleep(0)
    result = await run_ingest(
        factory,
        store,
        bus,
        SyntheticSource("XAUUSD", MarketCalendar.xauusd()),
        start=START,
        end=END,
    )
    await asyncio.wait_for(listener, 5)

    assert result.status is JobStatus.SUCCEEDED and result.years == [2023, 2024]
    assert result.quality["coverage"] > 0.99
    assert events[-1].type == "completed" and any(e.type == "progress" for e in events)

    async with factory() as s:
        job = await s.scalar(select(Job))
        assert job is not None and job.status is JobStatus.SUCCEEDED and job.progress == 1.0
        datasets = {ds.timeframe: ds for ds in await s.scalars(select(Dataset))}
    assert set(datasets) == {
        Timeframe.M1,
        Timeframe.M5,
        Timeframe.M15,
        Timeframe.M30,
        Timeframe.H1,
        Timeframe.H4,
        Timeframe.D1,
    }
    m1 = datasets[Timeframe.M1]
    assert m1.row_count == result.m1_rows_written and m1.quality["score"] > 0.9
    assert m1.start_ts is not None and m1.start_ts.tzinfo is not None
    assert datasets[Timeframe.M5].row_count == pytest.approx(m1.row_count / 5, rel=0.01)

    # D1 bar straddling New Year must be complete (built from padded M1).
    d1 = store.read_bars("XAUUSD", Timeframe.D1, source=DataSource.SYNTHETIC)
    m1_bars = store.read_bars("XAUUSD", Timeframe.M1, source=DataSource.SYNTHETIC)
    assert d1["volume"].sum() == pytest.approx(m1_bars["volume"].sum())

    # Re-running is idempotent.
    await run_ingest(
        factory,
        store,
        bus,
        SyntheticSource("XAUUSD", MarketCalendar.xauusd()),
        start=START,
        end=END,
    )
    assert store.bar_coverage("XAUUSD", Timeframe.M1, DataSource.SYNTHETIC).rows == m1.row_count


class _FailingSource:
    source = DataSource.SYNTHETIC
    symbol = "XAUUSD"

    async def chunks(self, start: datetime, end: datetime) -> AsyncIterator[Chunk]:
        raise RuntimeError("feed exploded")
        yield  # pragma: no cover


async def test_failed_ingest_marks_job(
    engine: AsyncEngine, xauusd: Instrument, tmp_path: Path
) -> None:
    factory = create_sessionmaker(engine)
    with pytest.raises(RuntimeError, match="exploded"):
        await run_ingest(
            factory,
            ParquetStore(tmp_path),
            InMemoryEventBus(),
            _FailingSource(),
            start=START,
            end=END,
        )
    async with factory() as s:
        job = await s.scalar(select(Job))
    assert job is not None and job.status is JobStatus.FAILED and "exploded" in (job.error or "")


async def test_unknown_instrument(engine: AsyncEngine, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown instrument"):
        await run_ingest(
            create_sessionmaker(engine),
            ParquetStore(tmp_path),
            InMemoryEventBus(),
            SyntheticSource("XAUUSD", MarketCalendar.xauusd()),
            start=START,
            end=END,
        )


# --------------------------------------------------------------------------- API
async def _prepare(url: str, data_dir: Path) -> None:
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_sessionmaker(engine)
    async with factory() as s:
        await seed_reference_data(s)
        await s.commit()
    await run_ingest(
        factory,
        ParquetStore(data_dir / "parquet"),
        InMemoryEventBus(),
        SyntheticSource("XAUUSD", MarketCalendar.xauusd()),
        start=datetime(2024, 3, 3, tzinfo=UTC),
        end=datetime(2024, 3, 9, tzinfo=UTC),
    )
    await engine.dispose()


def test_data_api(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'api.db'}"
    asyncio.run(_prepare(url, tmp_path))
    settings = Settings(_env_file=None, database_url=url, data_dir=tmp_path)  # type: ignore[call-arg]
    with TestClient(create_app(settings, event_bus=InMemoryEventBus())) as client:
        datasets = client.get("/api/v1/data/xauusd/datasets").json()
        assert {d["timeframe"] for d in datasets} >= {"1m", "5m", "15m", "30m"}
        assert all(d["source"] == "synthetic" for d in datasets)

        bars = client.get(
            "/api/v1/data/XAUUSD/bars",
            params={"timeframe": "15m", "source": "synthetic", "limit": 10},
        ).json()
        assert len(bars["time"]) == 10 and bars["side"] == "mid"
        assert bars["time"] == sorted(bars["time"])  # latest 10, ascending
        assert all(h >= lo for h, lo in zip(bars["high"], bars["low"], strict=True))
        assert all(s > 0 for s in bars["spread"])

        window = client.get(
            "/api/v1/data/XAUUSD/bars",
            params={
                "timeframe": "1h",
                "source": "synthetic",
                "side": "bid",
                "start": "2024-03-05T14:00:00Z",
                "end": "2024-03-05T18:00:00Z",
            },
        ).json()
        assert window["time"][0] == int(pd.Timestamp("2024-03-05 14:00", tz="UTC").timestamp())
        assert len(window["time"]) == 4

        assert (
            client.get("/api/v1/data/XAUUSD/bars", params={"timeframe": "tick"}).status_code == 422
        )
        empty = client.get("/api/v1/data/XAUUSD/bars", params={"source": "csv"}).json()
        assert empty["time"] == []

        jobs = client.get("/api/v1/jobs", params={"kind": "ingest"}).json()
        assert len(jobs) == 1 and jobs[0]["status"] == "succeeded"
        assert client.get(f"/api/v1/jobs/{jobs[0]['id']}").json()["result"]["m1_rows"] > 0


def test_indicator_catalog_and_overlays(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'api.db'}"
    asyncio.run(_prepare(url, tmp_path))
    settings = Settings(_env_file=None, database_url=url, data_dir=tmp_path)  # type: ignore[call-arg]
    with TestClient(create_app(settings, event_bus=InMemoryEventBus())) as client:
        catalog = {i["name"]: i for i in client.get("/api/v1/indicators").json()}
        assert {"rsi", "structure", "fvg", "order_blocks", "divergence", "time", "sessions"} <= set(
            catalog
        )
        assert catalog["rsi"]["params"]["length"] == {
            "kind": "int",
            "default": 14,
            "low": 2,
            "high": 100,
            "step": 1,
            "choices": None,
        }
        assert catalog["structure"]["outputs"]["choch_bull"]["kind"] == "event"
        assert catalog["structure"]["outputs"]["trend"]["values"] == [-1, 0, 1]

        ov = client.get(
            "/api/v1/data/XAUUSD/overlays",
            params={"timeframe": "5m", "source": "synthetic", "limit": 600},
        ).json()
        kinds = {m["kind"] for m in ov["markers"]}
        assert {"swing_high", "swing_low"} <= kinds and kinds & {
            "bos_bull",
            "bos_bear",
            "choch_bull",
            "choch_bear",
        }
        # Swing markers sit on the pivot bar but are only knowable 5 *bars* later: 25 minutes,
        # or more when the daily 17:00-18:00 NY break falls in between.
        bars = client.get(
            "/api/v1/data/XAUUSD/bars",
            params={"timeframe": "5m", "source": "synthetic", "limit": 5000},
        ).json()["time"]
        pos = {t: i for i, t in enumerate(bars)}
        for m in ov["markers"]:
            if m["kind"].startswith("swing"):
                assert pos[m["confirmed_time"]] - pos[m["time"]] == 5
            else:
                assert m["confirmed_time"] == m["time"]
        assert ov["zones"] and all(z["top"] > z["bottom"] for z in ov["zones"])
        empty = client.get("/api/v1/data/XAUUSD/overlays", params={"source": "csv"}).json()
        assert empty["markers"] == [] and empty["zones"] == []


def test_backtest_api(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'api.db'}"
    asyncio.run(_prepare(url, tmp_path))
    settings = Settings(_env_file=None, database_url=url, data_dir=tmp_path)  # type: ignore[call-arg]
    with TestClient(create_app(settings, event_bus=InMemoryEventBus())) as client:
        templates = {t["key"]: t for t in client.get("/api/v1/strategy-templates").json()}
        assert {"smc_choch_fvg", "sweep_reversal", "ema_pullback", "divergence_bbands"} <= set(
            templates
        )
        genome = templates["ema_pullback"]["genome"]
        body = {
            "genome": genome,
            "timeframe": "5m",
            "source": "synthetic",
            "start": "2024-03-04T00:00:00Z",
            "end": "2024-03-09T00:00:00Z",
            "max_trades": 3,
        }
        res = client.post("/api/v1/backtest", json=body)
        assert res.status_code == 200, res.text
        out = res.json()
        assert out["explanation"].startswith("LONG when")
        assert out["metrics"]["n_trades"] >= len(out["trades"])
        assert len(out["trades"]) <= 3 and out["equity"]["time"] == sorted(out["equity"]["time"])
        assert out["costs"]["commission_per_lot_rt"] == 7.0

        bad = {
            **body,
            "genome": {
                **genome,
                "entry_long": {
                    "type": "group",
                    "conditions": [
                        {
                            "type": "compare",
                            "left": {
                                "kind": "feature",
                                "feature": {"indicator": "rsi", "output": "value"},
                            },
                            "op": ">",
                            "right": {"kind": "const", "value": 250},
                        }
                    ],
                },
            },
        }
        assert client.post("/api/v1/backtest", json=bad).status_code == 422
        assert (
            client.post(
                "/api/v1/backtest", json={**body, "end": "2024-03-01T00:00:00Z"}
            ).status_code
            == 422
        )
        empty = client.post(
            "/api/v1/backtest",
            json={**body, "start": "2030-01-01T00:00:00Z", "end": "2030-02-01T00:00:00Z"},
        )
        assert empty.status_code == 404

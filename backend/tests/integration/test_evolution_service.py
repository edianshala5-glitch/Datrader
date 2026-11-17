"""Evolution runs through the service layer and the API, against a real DB and data lake."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings
from app.core.events import Channel, InMemoryEventBus
from app.db.base import Base
from app.db.session import create_engine, create_sessionmaker
from app.main import create_app
from app.models import (
    BacktestResult,
    EvolutionRun,
    Generation,
    Instrument,
    Strategy,
    strategy_lineage,
)
from app.models.enums import DataSource, RunStatus, Segment, StrategyStatus, Timeframe
from app.services.evolution import create_run, execute_run
from app.services.ingest import run_ingest
from app.services.seed import seed_reference_data
from engine.data.calendar import MarketCalendar
from engine.data.sources.synthetic import SyntheticSource
from engine.data.store import ParquetStore
from engine.evolution.engine import EvolutionConfig
from engine.validation.validator import ValidationConfig

START, END = datetime(2022, 1, 2, tzinfo=UTC), datetime(2023, 1, 1, tzinfo=UTC)
CFG = EvolutionConfig(islands=2, population=10, generations=3, migration_interval=2, seed=5)
FAST_VALIDATION = ValidationConfig(n_perturbations=2, min_oos_trades=20)


@pytest.fixture
async def lake(engine: AsyncEngine, xauusd: Instrument, tmp_path: Path) -> ParquetStore:
    store = ParquetStore(tmp_path / "parquet")
    await run_ingest(
        create_sessionmaker(engine),
        store,
        InMemoryEventBus(),
        SyntheticSource("XAUUSD", MarketCalendar.xauusd(), seed=4),
        start=START,
        end=END,
    )
    return store


async def _new_run(engine: AsyncEngine, generations: int = 3) -> EvolutionRun:
    async with create_sessionmaker(engine)() as s:
        run = await create_run(
            s,
            name="t",
            symbol="XAUUSD",
            timeframe=Timeframe.M15,
            start=START,
            end=END,
            config=EvolutionConfig(
                **{**CFG.to_dict(), "generations": generations, "fitness": CFG.fitness, "htf": ()}
            ),
            source=DataSource.SYNTHETIC,
            n_folds=3,
            validate_top=3,
        )
        await s.commit()
        return run


async def test_run_end_to_end(engine: AsyncEngine, lake: ParquetStore, tmp_path: Path) -> None:
    run = await _new_run(engine)
    bus = InMemoryEventBus()
    events = []

    async def listen() -> None:
        async with bus.subscribe(Channel.LAB) as stream:
            async for e in stream:
                events.append(e)
                if e.type in ("run_completed", "run_failed"):
                    return

    listener = asyncio.create_task(listen())
    await asyncio.sleep(0)
    status = await execute_run(
        create_sessionmaker(engine),
        lake,
        bus,
        run.id,
        checkpoint_dir=tmp_path / "runs",
        validation=FAST_VALIDATION,
    )
    await asyncio.wait_for(listener, 10)
    assert status is RunStatus.COMPLETED

    types = [e.type for e in events]
    assert types[0] == "run_started" and types[-1] == "run_completed"
    assert types.count("generation") == 4 and "validating" in types
    gen = next(e for e in events if e.type == "generation")
    assert {"islands", "best", "evaluations"} <= set(gen.payload)

    async with create_sessionmaker(engine)() as s:
        r = await s.get(EvolutionRun, run.id)
        assert r is not None and r.status is RunStatus.COMPLETED and r.current_generation == 3
        assert r.evaluations > 0 and r.best_metrics["fitness"] is not None
        assert (
            await s.scalar(
                select(func.count()).select_from(Generation).where(Generation.run_id == run.id)
            )
            == 8
        )
        n_strat = await s.scalar(
            select(func.count()).select_from(Strategy).where(Strategy.run_id == run.id)
        )
        assert 0 < n_strat <= r.evaluations  # only genomes with enough trades are stored
        assert await s.scalar(select(func.count()).select_from(strategy_lineage)) > 0
        judged = (await s.scalars(select(Strategy).where(Strategy.overfit_risk.is_not(None)))).all()
        assert 1 <= len(judged) <= 3
        assert all(
            st.status in (StrategyStatus.VALIDATED, StrategyStatus.REJECTED) for st in judged
        )
        summary = await s.scalar(
            select(BacktestResult).where(BacktestResult.segment == Segment.VALIDATION)
        )
        rep = summary.extra_metrics["validation_report"]  # type: ignore[union-attr]
        assert rep["selection"] == "GA training window only"
        assert rep["search_evaluations"] == r.evaluations
        assert rep["n_trials"] == rep["candidates"] + rep["prior_oos_validations"]
    assert (tmp_path / "runs" / f"{run.id}.json").exists()


async def test_pause_and_resume(engine: AsyncEngine, lake: ParquetStore, tmp_path: Path) -> None:
    run = await _new_run(engine, generations=4)
    factory = create_sessionmaker(engine)
    stop = threading.Event()
    bus = InMemoryEventBus()

    async def stop_after_first_generation() -> None:
        async with bus.subscribe(Channel.LAB) as stream:
            async for e in stream:
                if e.type == "generation" and e.payload["generation"] >= 1:
                    stop.set()
                    return

    watcher = asyncio.create_task(stop_after_first_generation())
    await asyncio.sleep(0)
    assert (
        await execute_run(
            factory,
            lake,
            bus,
            run.id,
            checkpoint_dir=tmp_path,
            stop=stop,
            validation=FAST_VALIDATION,
        )
        is RunStatus.PAUSED
    )
    await watcher
    async with factory() as s:
        paused = await s.get(EvolutionRun, run.id)
        assert (
            paused is not None
            and paused.status is RunStatus.PAUSED
            and paused.current_generation < 4
        )
    assert (
        await execute_run(
            factory,
            lake,
            InMemoryEventBus(),
            run.id,
            checkpoint_dir=tmp_path,
            validation=FAST_VALIDATION,
        )
        is RunStatus.COMPLETED
    )
    async with factory() as s:
        done = await s.get(EvolutionRun, run.id)
        assert done is not None and done.current_generation == 4
        numbers = (
            await s.scalars(
                select(Generation.number).where(Generation.run_id == run.id, Generation.island == 0)
            )
        ).all()
        assert sorted(numbers) == [
            0,
            1,
            2,
            3,
            4,
        ]  # no generation lost or duplicated across the pause


async def test_run_without_data_fails_cleanly(
    engine: AsyncEngine, xauusd: Instrument, tmp_path: Path
) -> None:
    run = await _new_run(engine)
    status = await execute_run(
        create_sessionmaker(engine),
        ParquetStore(tmp_path),
        InMemoryEventBus(),
        run.id,
        checkpoint_dir=tmp_path,
    )
    assert status is RunStatus.FAILED
    async with create_sessionmaker(engine)() as s:
        r = await s.get(EvolutionRun, run.id)
        assert r is not None and "ingest data first" in (r.notes or "")


def test_runs_api(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'api.db'}"

    async def prepare() -> None:
        eng = create_engine(url)
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = create_sessionmaker(eng)
        async with factory() as s:
            await seed_reference_data(s)
            await s.commit()
        await run_ingest(
            factory,
            ParquetStore(tmp_path / "parquet"),
            InMemoryEventBus(),
            SyntheticSource("XAUUSD", MarketCalendar.xauusd(), seed=4),
            start=START,
            end=END,
        )
        await eng.dispose()

    asyncio.run(prepare())
    settings = Settings(_env_file=None, database_url=url, data_dir=tmp_path)  # type: ignore[call-arg]
    with TestClient(create_app(settings, event_bus=InMemoryEventBus())) as client:
        body = {
            "name": "api-run",
            "timeframe": "15m",
            "source": "synthetic",
            "start": START.isoformat(),
            "end": END.isoformat(),
            "islands": 2,
            "population": 8,
            "generations": 2,
            "n_folds": 3,
            "validate_top": 0,
        }
        res = client.post("/api/v1/runs", json=body)
        assert res.status_code == 202, res.text
        run_id = res.json()["id"]
        deadline = time.time() + 120
        while time.time() < deadline:
            status = client.get(f"/api/v1/runs/{run_id}").json()["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(0.5)
        assert status == "completed"
        gens = client.get(f"/api/v1/runs/{run_id}/generations").json()
        assert len(gens) == 6 and gens[0]["stats"]["size"] == 8
        strategies = client.get("/api/v1/strategies", params={"run_id": run_id}).json()
        assert strategies["total"] > 0
        assert (
            client.post(f"/api/v1/runs/{run_id}/stop").status_code == 409
        )  # not executing anymore
        assert (
            client.post(f"/api/v1/runs/{run_id}/resume").status_code == 409
        )  # completed runs don't resume
        bad = client.post("/api/v1/runs", json={**body, "end": "2021-01-01T00:00:00Z"})
        assert bad.status_code == 422

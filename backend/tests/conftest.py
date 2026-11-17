"""Shared fixtures.

Database tests run on in-memory SQLite by default and also on PostgreSQL when
``SWARM_TEST_POSTGRES_URL`` is set (e.g. postgresql+asyncpg://swarm:swarm@localhost/swarm_test).
Redis bus tests run when ``SWARM_TEST_REDIS_URL`` is set.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

import app.models  # noqa: F401
from app.db.base import Base
from app.db.session import create_engine, create_sessionmaker
from app.models import Instrument
from app.services.seed import seed_reference_data

PG_URL = os.getenv("SWARM_TEST_POSTGRES_URL")
if PG_URL and not make_url(PG_URL).database.endswith("_test"):  # type: ignore[union-attr]
    # Tests drop every table (and the migration test drops the whole schema).
    raise RuntimeError("SWARM_TEST_POSTGRES_URL must point at a database whose name ends in _test")
REDIS_URL = os.getenv("SWARM_TEST_REDIS_URL")

requires_postgres = pytest.mark.skipif(not PG_URL, reason="SWARM_TEST_POSTGRES_URL not set")
requires_redis = pytest.mark.skipif(not REDIS_URL, reason="SWARM_TEST_REDIS_URL not set")


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=requires_postgres)])
def database_url(request: pytest.FixtureRequest) -> str:
    if request.param == "sqlite":
        return "sqlite+aiosqlite://"
    assert PG_URL
    return PG_URL


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with create_sessionmaker(engine)() as s:
        yield s


@pytest.fixture
async def xauusd(session: AsyncSession) -> Instrument:
    [instrument] = await seed_reference_data(session)
    await session.commit()
    return instrument

"""Async engine and session factories."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")  # SQLite ignores FKs unless asked
    cursor.execute("PRAGMA journal_mode=WAL")  # API reads while workers write
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def create_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    url = make_url(database_url)
    kwargs: dict[str, Any] = {"echo": echo}

    if url.get_backend_name() == "sqlite":
        database = url.database
        if not database or database == ":memory:":
            # One shared connection, or every session would see a different empty DB.
            kwargs["poolclass"] = StaticPool
        else:
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(url, **kwargs)
        event.listen(engine.sync_engine, "connect", _sqlite_pragmas)
        return engine

    kwargs.update(pool_pre_ping=True, pool_size=10, max_overflow=20)
    return create_async_engine(url, **kwargs)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Commit on success, roll back on error. Used by the FastAPI dependency and CLI."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise

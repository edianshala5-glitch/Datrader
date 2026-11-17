"""Migrations must build exactly the schema the models describe, on both dialects."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from app.cli import alembic_config
from app.db.base import Base
from tests.conftest import PG_URL, requires_postgres


def _sync_url(async_url: str) -> str:
    return async_url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg2")


def _assert_no_drift(async_url: str) -> None:
    cfg = alembic_config(async_url)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")

    engine = create_engine(_sync_url(async_url))
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn, opts={"compare_type": True})
            diff = compare_metadata(ctx, Base.metadata)
            assert diff == [], f"models and migrations disagree: {diff}"
            tables = set(inspect(conn).get_table_names())
            assert set(Base.metadata.tables) <= tables
    finally:
        engine.dispose()

    command.downgrade(cfg, "base")


def test_sqlite_migrations_match_models(tmp_path: Path) -> None:
    _assert_no_drift(f"sqlite+aiosqlite:///{tmp_path / 'm.db'}")


@requires_postgres
def test_postgres_migrations_match_models() -> None:
    assert PG_URL
    pytest.importorskip("psycopg2", reason="psycopg2 needed for sync inspection")
    engine = create_engine(_sync_url(PG_URL))
    with engine.begin() as conn:  # start from an empty schema
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    engine.dispose()
    _assert_no_drift(PG_URL)

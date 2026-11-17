"""Alembic environment: async engine, dialect-portable rendering, SQLite batch mode."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import JSON, BigInteger, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

import app.models  # noqa: F401  (registers all tables)
from app.core.config import get_settings
from app.db.base import Base
from app.db.types import UTCDateTime

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    # Emit the portable base type so migrations don't import application code.
    if type_ != "type":
        return False
    if isinstance(obj, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    variants = getattr(obj, "_variant_mapping", {})
    if isinstance(obj, JSON) and "postgresql" in variants:
        autogen_context.imports.add("from sqlalchemy.dialects import postgresql")
        return "sa.JSON().with_variant(postgresql.JSONB(), 'postgresql')"
    if isinstance(obj, BigInteger) and "sqlite" in variants:
        return "sa.BigInteger().with_variant(sa.Integer(), 'sqlite')"
    return False


def _configure(**kwargs: Any) -> None:
    url = config.get_main_option("sqlalchemy.url") or ""
    context.configure(
        target_metadata=target_metadata,
        render_item=render_item,
        render_as_batch=url.startswith("sqlite"),  # SQLite needs table rebuilds for ALTERs
        compare_type=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    _configure(url=config.get_main_option("sqlalchemy.url"), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())

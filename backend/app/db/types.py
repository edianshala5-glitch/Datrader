"""Portable column types that behave identically on SQLite and PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Dialect, Enum, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

# JSONB on Postgres (indexable, compact); plain JSON text elsewhere.
JSONType = JSON().with_variant(JSONB(), "postgresql")

# Auto-incrementing big integer PK. SQLite only autoincrements INTEGER PRIMARY KEY.
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC timestamps everywhere.

    Rejects naive datetimes on write. A naive timestamp in a trading system usually
    means a timezone bug (session filters and day-of-week context depend on this).
    SQLite returns naive values on read, which we tag as UTC.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"naive datetime not allowed, attach a timezone: {value!r}")
        return value.astimezone(UTC)

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        parsed: datetime = (
            datetime.fromisoformat(value)  # SQLite server_default CURRENT_TIMESTAMP
            if isinstance(value, str)
            else value
        )
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)


def str_enum(enum_cls: type[StrEnum], length: int = 32) -> Enum:
    """Store enum *values* as VARCHAR, not a native DB enum.

    Adding a member then needs no ALTER TYPE migration, and the column reads the same
    on SQLite and Postgres. Pydantic/SQLAlchemy still validate values on the way in.
    """
    return Enum(
        enum_cls,
        name=enum_cls.__name__.lower(),
        native_enum=False,
        create_constraint=False,
        length=length,
        values_callable=lambda e: [member.value for member in e],
        validate_strings=True,
    )

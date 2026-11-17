from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Float, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import DataSource, JobKind, JobStatus, Timeframe
from app.models.instrument import Instrument


class Dataset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Catalogue entry for market data stored in the Parquet lake.

    The bars and ticks themselves live in Parquet under ``data_dir``. This row records
    what exists, where it is, and how clean it is.
    """

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("instrument_id", "source", "timeframe"),)

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), index=True
    )
    source: Mapped[DataSource] = mapped_column(str_enum(DataSource))
    timeframe: Mapped[Timeframe] = mapped_column(str_enum(Timeframe))
    start_ts: Mapped[datetime | None]
    end_ts: Mapped[datetime | None]
    row_count: Mapped[int] = mapped_column(BigInteger, default=0)
    storage_uri: Mapped[str] = mapped_column(String(512))
    # Gap counts, duplicate counts, spread stats, outliers. See engine.data.quality (Step 2).
    quality: Mapped[dict[str, Any]] = mapped_column(default=dict)
    last_ingested_at: Mapped[datetime | None]

    instrument: Mapped[Instrument] = relationship(lazy="joined")


class Job(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Long-running background work, so the UI can show progress and failures."""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_kind_status", "kind", "status"),)

    kind: Mapped[JobKind] = mapped_column(str_enum(JobKind))
    status: Mapped[JobStatus] = mapped_column(str_enum(JobStatus), default=JobStatus.QUEUED)
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    message: Mapped[str | None] = mapped_column(String(512))
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    result: Mapped[dict[str, Any] | None]
    error: Mapped[str | None] = mapped_column(Text)
    worker_id: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime | None]
    heartbeat_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import RunStatus, Timeframe
from app.models.instrument import Instrument

if TYPE_CHECKING:
    from app.models.strategy import Strategy


class EvolutionRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One optimization campaign: a population evolving on one instrument and timeframe."""

    __tablename__ = "evolution_runs"

    name: Mapped[str] = mapped_column(String(128))
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="RESTRICT"), index=True
    )
    timeframe: Mapped[Timeframe] = mapped_column(str_enum(Timeframe))
    status: Mapped[RunStatus] = mapped_column(
        str_enum(RunStatus), default=RunStatus.PENDING, index=True
    )
    algorithm: Mapped[str] = mapped_column(String(32), default="nsga2")
    # Population size, generations, islands, migration, mutation/crossover rates, objectives.
    config: Mapped[dict[str, Any]] = mapped_column(default=dict)
    # Train / walk-forward folds / validation / locked holdout boundaries (ISO timestamps).
    data_splits: Mapped[dict[str, Any]] = mapped_column(default=dict)
    seed: Mapped[int] = mapped_column(BigInteger, default=0)

    current_generation: Mapped[int] = mapped_column(default=0)
    # Total strategies ever evaluated. Feeds the Deflated Sharpe multiple-testing correction.
    evaluations: Mapped[int] = mapped_column(BigInteger, default=0)
    best_metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)

    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"))
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    notes: Mapped[str | None] = mapped_column(Text)

    instrument: Mapped[Instrument] = relationship(lazy="joined")
    generations: Mapped[list[Generation]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Generation.number"
    )
    strategies: Mapped[list[Strategy]] = relationship(back_populates="run")

    @property
    def symbol(self) -> str:
        return self.instrument.symbol


class Generation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Per-island, per-generation statistics. This is what the Lab view animates."""

    __tablename__ = "generations"
    __table_args__ = (UniqueConstraint("run_id", "island", "number"),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evolution_runs.id", ondelete="CASCADE"), index=True
    )
    island: Mapped[int] = mapped_column(default=0)
    number: Mapped[int]
    population_size: Mapped[int]
    evaluated: Mapped[int] = mapped_column(default=0)
    # {"fitness": {"min":..,"mean":..,"max":..}, "sortino": {...}, "best_strategy_id": ...}
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict)
    pareto_size: Mapped[int] = mapped_column(default=0)
    diversity: Mapped[float | None] = mapped_column(Float)  # genotype diversity (0..1)
    duration_ms: Mapped[int | None]

    run: Mapped[EvolutionRun] = relationship(back_populates="generations")

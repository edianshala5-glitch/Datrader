from __future__ import annotations

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import StrategyOrigin, StrategyStatus, Timeframe
from app.models.instrument import Instrument

if TYPE_CHECKING:
    from app.models.backtest import BacktestResult
    from app.models.context_rule import ContextRule
    from app.models.evolution import EvolutionRun

strategy_lineage = Table(
    "strategy_lineage",
    Base.metadata,
    Column("child_id", ForeignKey("strategies.id", ondelete="CASCADE"), primary_key=True),
    Column("parent_id", ForeignKey("strategies.id", ondelete="CASCADE"), primary_key=True),
)


def genome_hash(genome: dict[str, Any]) -> str:
    """Hash of the canonical genome JSON, so the swarm never evaluates the same
    strategy twice and lineage stays a DAG rather than duplicate copies."""
    canonical = json.dumps(genome, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def strategy_codename(hash_hex: str) -> str:
    """Short readable id, e.g. ``AU-7F3A9C``."""
    return f"AU-{hash_hex[:6].upper()}"


class Strategy(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A candidate trading strategy, defined by its genome (JSON DSL, see Step 4).

    The ``oos_*`` columns are denormalized aggregates of its out-of-sample backtests,
    kept on this row so the leaderboard can filter and sort 100k+ strategies with plain
    indexed queries. ``BacktestResult`` holds the full per-fold numbers.
    """

    __tablename__ = "strategies"
    __table_args__ = (
        UniqueConstraint("genome_hash", "instrument_id", "timeframe"),
        Index("ix_strategies_leaderboard", "instrument_id", "timeframe", "status", "fitness"),
    )

    name: Mapped[str] = mapped_column(String(64))
    genome: Mapped[dict[str, Any]]
    genome_hash: Mapped[str] = mapped_column(String(64), index=True)
    genome_version: Mapped[int] = mapped_column(default=1)
    description: Mapped[str | None] = mapped_column(Text)  # human-readable rule summary

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="RESTRICT"), index=True
    )
    timeframe: Mapped[Timeframe] = mapped_column(str_enum(Timeframe))
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("evolution_runs.id", ondelete="SET NULL"), index=True
    )
    generation_number: Mapped[int | None]
    island: Mapped[int | None]
    origin: Mapped[StrategyOrigin] = mapped_column(str_enum(StrategyOrigin))
    status: Mapped[StrategyStatus] = mapped_column(
        str_enum(StrategyStatus), default=StrategyStatus.CANDIDATE, index=True
    )

    # Composite multi-objective score used for ranking (definition: engine.evolution.fitness)
    fitness: Mapped[float | None] = mapped_column(Float, index=True)
    is_sharpe: Mapped[float | None] = mapped_column(Float)
    oos_sharpe: Mapped[float | None] = mapped_column(Float)
    oos_sortino: Mapped[float | None] = mapped_column(Float, index=True)
    oos_profit_factor: Mapped[float | None] = mapped_column(Float)
    oos_win_rate: Mapped[float | None] = mapped_column(Float)
    oos_expectancy_r: Mapped[float | None] = mapped_column(Float)
    oos_max_drawdown: Mapped[float | None] = mapped_column(Float)
    oos_trades: Mapped[int | None] = mapped_column(Integer)
    # Overfitting diagnostics (Step 5)
    deflated_sharpe: Mapped[float | None] = mapped_column(Float)  # P(true SR > 0), 0..1
    pbo: Mapped[float | None] = mapped_column(Float)  # probability of backtest overfitting
    overfit_risk: Mapped[float | None] = mapped_column(Float)  # 0 (robust) .. 1 (likely noise)

    instrument: Mapped[Instrument] = relationship(lazy="joined")
    run: Mapped[EvolutionRun | None] = relationship(back_populates="strategies")
    parents: Mapped[list[Strategy]] = relationship(
        secondary=strategy_lineage,
        primaryjoin=lambda: Strategy.id == strategy_lineage.c.child_id,
        secondaryjoin=lambda: Strategy.id == strategy_lineage.c.parent_id,
        back_populates="children",
    )
    children: Mapped[list[Strategy]] = relationship(
        secondary=strategy_lineage,
        primaryjoin=lambda: Strategy.id == strategy_lineage.c.parent_id,
        secondaryjoin=lambda: Strategy.id == strategy_lineage.c.child_id,
        back_populates="parents",
    )
    backtests: Mapped[list[BacktestResult]] = relationship(
        back_populates="strategy", cascade="all, delete-orphan", passive_deletes=True
    )
    context_rules: Mapped[list[ContextRule]] = relationship(
        back_populates="strategy", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def symbol(self) -> str:
        return self.instrument.symbol

    @classmethod
    def from_genome(cls, genome: dict[str, Any], **kwargs: Any) -> Strategy:
        h = genome_hash(genome)
        kwargs.setdefault("name", strategy_codename(h))
        return cls(genome=genome, genome_hash=h, **kwargs)

    def __repr__(self) -> str:
        return f"<Strategy {self.name} {self.timeframe} {self.status}>"

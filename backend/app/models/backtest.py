from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import BigIntPK, str_enum
from app.models.enums import ExitReason, Segment, Side

if TYPE_CHECKING:
    from app.models.strategy import Strategy


class BacktestResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Metrics for one strategy over one data segment (fold), optionally with a
    context rule applied. A strategy has many: walk-forward folds, validation, holdout,
    plus a filtered re-run for each context rule the Meta-Analyzer proposes."""

    __tablename__ = "backtest_results"
    __table_args__ = (
        Index("ix_backtest_results_strategy_segment", "strategy_id", "segment"),
        # The locked holdout may be evaluated once per strategy. Enforced by the database,
        # not only by application code, so concurrent requests can't both look.
        Index(
            "uq_backtest_results_one_holdout",
            "strategy_id",
            unique=True,
            postgresql_where=text("segment = 'holdout'"),
            sqlite_where=text("segment = 'holdout'"),
        ),
    )

    strategy_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("strategies.id", ondelete="CASCADE"))
    context_rule_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("context_rules.id", ondelete="CASCADE"), index=True
    )
    segment: Mapped[Segment] = mapped_column(str_enum(Segment))
    fold_index: Mapped[int] = mapped_column(default=0)
    period_start: Mapped[datetime]
    period_end: Mapped[datetime]

    n_trades: Mapped[int] = mapped_column(default=0)
    total_return: Mapped[float | None] = mapped_column(Float)  # fraction of starting equity
    sharpe: Mapped[float | None] = mapped_column(Float)  # annualized
    sortino: Mapped[float | None] = mapped_column(Float)
    calmar: Mapped[float | None] = mapped_column(Float)
    max_drawdown: Mapped[float | None] = mapped_column(Float)  # positive fraction
    profit_factor: Mapped[float | None] = mapped_column(Float)
    win_rate: Mapped[float | None] = mapped_column(Float)
    expectancy_r: Mapped[float | None] = mapped_column(Float)  # mean R-multiple per trade
    avg_bars_held: Mapped[float | None] = mapped_column(Float)
    exposure: Mapped[float | None] = mapped_column(Float)  # fraction of bars in market
    extra_metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)

    cost_model: Mapped[dict[str, Any]] = mapped_column(default=dict)  # costs used for this run
    equity_uri: Mapped[str | None] = mapped_column(String(512))  # Parquet equity curve
    engine_version: Mapped[str] = mapped_column(String(16))

    strategy: Mapped[Strategy] = relationship(back_populates="backtests")
    trades: Mapped[list[Trade]] = relationship(
        back_populates="backtest", cascade="all, delete-orphan", passive_deletes=True
    )


class Trade(Base):
    """A simulated trade with its entry-time context. This is the Meta-Analyzer's input.

    High volume, so it uses a bigint PK and no timestamps mixin. Only trades of strategies
    that pass the in-sample gate are persisted here. The rest stay in the
    workers' Parquet scratch space.
    """

    __tablename__ = "trades"
    __table_args__ = (Index("ix_trades_strategy_entry", "strategy_id", "entry_ts"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    strategy_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("strategies.id", ondelete="CASCADE"))
    backtest_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("backtest_results.id", ondelete="CASCADE"), index=True
    )
    side: Mapped[Side] = mapped_column(str_enum(Side, length=8))
    entry_ts: Mapped[datetime]
    exit_ts: Mapped[datetime]
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    stop_price: Mapped[float | None] = mapped_column(Float)
    target_price: Mapped[float | None] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)  # lots
    pnl: Mapped[float] = mapped_column(Float)  # quote currency, net of costs
    pnl_r: Mapped[float | None] = mapped_column(Float)  # R-multiple
    mae_r: Mapped[float | None] = mapped_column(Float)  # max adverse excursion in R
    mfe_r: Mapped[float | None] = mapped_column(Float)  # max favourable excursion in R
    bars_held: Mapped[int]
    exit_reason: Mapped[ExitReason] = mapped_column(str_enum(ExitReason))
    # Feature snapshot at entry: trading-day weekday, hour (NY), session, ATR regime,
    # trend state, bars since last BOS/CHoCH, active divergences, distance to liquidity...
    context: Mapped[dict[str, Any]] = mapped_column(default=dict)

    backtest: Mapped[BacktestResult] = relationship(back_populates="trades")

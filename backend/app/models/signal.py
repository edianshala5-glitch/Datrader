from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Float, ForeignKey, Index, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import Side, SignalKind, SignalStatus, Timeframe


class Signal(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A live alert from a champion.

    ``bar_ts`` is the *close* time of the bar that triggered the signal. The unique
    constraint makes emission idempotent: a feed reconnect that replays bars can never
    send the same alert twice.
    """

    __tablename__ = "signals"
    __table_args__ = (
        UniqueConstraint("strategy_id", "timeframe", "bar_ts", "kind"),
        Index("ix_signals_instrument_emitted", "instrument_id", "emitted_at"),
    )

    champion_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("champions.id", ondelete="SET NULL"), index=True
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("strategies.id", ondelete="CASCADE"))
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE")
    )
    parent_signal_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("signals.id", ondelete="SET NULL")  # exit -> its entry
    )
    timeframe: Mapped[Timeframe] = mapped_column(str_enum(Timeframe))
    kind: Mapped[SignalKind] = mapped_column(str_enum(SignalKind))
    side: Mapped[Side] = mapped_column(str_enum(Side, length=8))
    status: Mapped[SignalStatus] = mapped_column(
        str_enum(SignalStatus), default=SignalStatus.PENDING, index=True
    )

    bar_ts: Mapped[datetime]
    emitted_at: Mapped[datetime]
    entry_price: Mapped[float | None] = mapped_column(Float)
    stop_price: Mapped[float | None] = mapped_column(Float)
    target_price: Mapped[float | None] = mapped_column(Float)
    risk_reward: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)  # 0..1, calibrated in Step 8

    # {"conditions": [{"label": "Bullish CHoCH (15m)", "value": ..., "met": true}, ...],
    #  "context_rule": {...}, "champion_stats": {...}}
    rationale: Mapped[dict[str, Any]] = mapped_column(default=dict)
    rationale_text: Mapped[str] = mapped_column(Text, default="")

    closed_at: Mapped[datetime | None]
    exit_price: Mapped[float | None] = mapped_column(Float)
    pnl_r: Mapped[float | None] = mapped_column(Float)

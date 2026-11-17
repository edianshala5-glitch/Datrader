from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import ChampionStatus

if TYPE_CHECKING:
    from app.models.context_rule import ContextRule
    from app.models.strategy import Strategy

MAX_ACTIVE_CHAMPIONS = 3


class Champion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One promotion episode of a strategy (plus optional context rule) to live signaling.

    Lifecycle: ``incubating`` (paper-trades live data, no alerts) -> ``active`` (ranked
    1-3, emits alerts) -> ``demoted`` (live drift / drawdown breach / replaced). A new
    promotion of the same strategy creates a new row, so history is kept.
    """

    __tablename__ = "champions"
    __table_args__ = (
        CheckConstraint(
            f"rank IS NULL OR (rank >= 1 AND rank <= {MAX_ACTIVE_CHAMPIONS})", name="rank_range"
        ),
    )

    strategy_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("strategies.id", ondelete="CASCADE"), index=True
    )
    context_rule_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("context_rules.id", ondelete="SET NULL")
    )
    status: Mapped[ChampionStatus] = mapped_column(
        str_enum(ChampionStatus), default=ChampionStatus.INCUBATING, index=True
    )
    rank: Mapped[int | None]

    promoted_at: Mapped[datetime]
    activated_at: Mapped[datetime | None]
    demoted_at: Mapped[datetime | None]
    demotion_reason: Mapped[str | None] = mapped_column(Text)

    promotion_snapshot: Mapped[dict[str, Any]] = mapped_column(default=dict)  # metrics then
    live_stats: Mapped[dict[str, Any]] = mapped_column(default=dict)  # rolling forward stats

    strategy: Mapped[Strategy] = relationship(lazy="joined")
    context_rule: Mapped[ContextRule | None] = relationship(lazy="joined")

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import RuleStatus

if TYPE_CHECKING:
    from app.models.strategy import Strategy


class ContextRule(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A contextual filter mined by the Meta-Analyzer for one strategy, e.g.
    "trading-day = Tue AND 09:00 <= hour_ny < 11:00 AND bullish CHoCH within 5 bars".

    ``conditions`` is machine-readable (``[{"feature": "weekday", "op": "in",
    "value": [1]}, ...]``) so the live engine can evaluate it. ``description`` is the
    plain-English version shown in the UI.

    A rule is only ``validated`` if it improves results out of sample, keeps enough
    trades, and survives a multiple-testing correction (``p_value_adjusted``).
    """

    __tablename__ = "context_rules"

    strategy_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("strategies.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128))
    conditions: Mapped[list[Any]]
    description: Mapped[str] = mapped_column(Text)
    miner: Mapped[str] = mapped_column(String(32))  # decision_tree | gbm_shap | manual
    status: Mapped[RuleStatus] = mapped_column(
        str_enum(RuleStatus), default=RuleStatus.PROPOSED, index=True
    )

    is_trades: Mapped[int | None]
    is_win_rate: Mapped[float | None] = mapped_column(Float)
    is_profit_factor: Mapped[float | None] = mapped_column(Float)
    is_expectancy_r: Mapped[float | None] = mapped_column(Float)
    oos_trades: Mapped[int | None]
    oos_win_rate: Mapped[float | None] = mapped_column(Float)
    oos_profit_factor: Mapped[float | None] = mapped_column(Float)
    oos_expectancy_r: Mapped[float | None] = mapped_column(Float)

    coverage: Mapped[float | None] = mapped_column(Float)  # share of base trades kept
    lift: Mapped[float | None] = mapped_column(Float)  # OOS expectancy gain vs unfiltered
    p_value: Mapped[float | None] = mapped_column(Float)
    p_value_adjusted: Mapped[float | None] = mapped_column(Float)  # Holm / BH corrected
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)

    strategy: Mapped[Strategy] = relationship(back_populates="context_rules")

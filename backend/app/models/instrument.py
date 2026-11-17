from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import AssetClass


class Instrument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A tradable symbol and the market conventions the engine needs.

    The JSON columns are validated by ``app.schemas.instrument`` (TradingHours,
    SessionWindow, CostModel) before they are written.
    """

    __tablename__ = "instruments"

    symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(64))
    asset_class: Mapped[AssetClass] = mapped_column(str_enum(AssetClass))
    base_currency: Mapped[str] = mapped_column(String(8))
    quote_currency: Mapped[str] = mapped_column(String(8))

    tick_size: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    price_precision: Mapped[int]
    # Units per 1.0 lot (XAUUSD: 100 troy oz -> $1 move = $100 per lot).
    contract_size: Mapped[Decimal] = mapped_column(Numeric(18, 4))

    # IANA zone that defines the trading day and day-of-week context. XAUUSD rolls at
    # 17:00 America/New_York, so "Tuesday" means the session that opens Monday 18:00 NY.
    session_timezone: Mapped[str] = mapped_column(String(64))
    trading_hours: Mapped[dict[str, Any]]
    sessions: Mapped[dict[str, Any]]  # named intraday windows (Asia, London, NY, kill zones)
    cost_model: Mapped[dict[str, Any]]  # fallback costs when data has no bid/ask spread

    is_active: Mapped[bool] = mapped_column(default=True)

    def __repr__(self) -> str:
        return f"<Instrument {self.symbol}>"

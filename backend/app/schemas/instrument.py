"""Schemas for instruments, including strict validation of the JSON config columns."""

from __future__ import annotations

import uuid
from datetime import time
from decimal import Decimal
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import AssetClass

Weekday = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _validate_tz(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone: {value}") from exc
    return value


class SessionWindow(BaseModel):
    """An intraday window in its *own* market's timezone, so DST shifts are handled
    correctly (London and New York change clocks on different dates)."""

    model_config = ConfigDict(extra="forbid")

    label: str
    tz: str
    start: time
    end: time  # may be < start for windows that cross midnight

    @field_validator("tz")
    @classmethod
    def _check_tz(cls, value: str) -> str:
        return _validate_tz(value)


class WeeklyBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day: Weekday
    at: time


class TradingHours(BaseModel):
    """Weekly schedule in the instrument's ``session_timezone``."""

    model_config = ConfigDict(extra="forbid")

    week_open: WeeklyBoundary
    week_close: WeeklyBoundary
    daily_break_start: time | None = None
    daily_break_end: time | None = None
    # Wall-clock time at which a new trading day begins (bars after it belong to the next day).
    day_roll: time


class CostModel(BaseModel):
    """Fallback execution costs, in price units. Used when data lacks bid/ask quotes and
    as a floor when it has them."""

    model_config = ConfigDict(extra="forbid")

    spread: float = Field(ge=0, description="Typical bid/ask spread in price units")
    commission_per_lot_round_turn: float = Field(ge=0, description="Quote currency per lot")
    slippage_ticks: float = Field(ge=0, description="Adverse slippage per fill, in ticks")
    swap_long_per_lot_day: float = 0.0
    swap_short_per_lot_day: float = 0.0


class InstrumentBase(BaseModel):
    symbol: str = Field(min_length=1, max_length=32, pattern=r"^[A-Z0-9._-]+$")
    display_name: str
    asset_class: AssetClass
    base_currency: str = Field(max_length=8)
    quote_currency: str = Field(max_length=8)
    tick_size: Decimal = Field(gt=0)
    price_precision: int = Field(ge=0, le=10)
    contract_size: Decimal = Field(gt=0)
    session_timezone: str
    trading_hours: TradingHours
    sessions: dict[str, SessionWindow]
    cost_model: CostModel
    is_active: bool = True

    @field_validator("session_timezone")
    @classmethod
    def _check_tz(cls, value: str) -> str:
        return _validate_tz(value)


class InstrumentCreate(InstrumentBase):
    pass


class InstrumentRead(InstrumentBase):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID

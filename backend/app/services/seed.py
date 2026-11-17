"""Reference data: instrument definitions. Idempotent upsert keyed by symbol."""

from __future__ import annotations

import logging
from datetime import time
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Instrument
from app.models.enums import AssetClass
from app.schemas.instrument import (
    CostModel,
    InstrumentCreate,
    SessionWindow,
    TradingHours,
    WeeklyBoundary,
)

logger = logging.getLogger(__name__)

NY = "America/New_York"

XAUUSD = InstrumentCreate(
    symbol="XAUUSD",
    display_name="Gold Spot / US Dollar",
    asset_class=AssetClass.METAL,
    base_currency="XAU",
    quote_currency="USD",
    tick_size=Decimal("0.01"),
    price_precision=2,
    contract_size=Decimal("100"),  # 1 lot = 100 oz
    session_timezone=NY,
    # Spot gold trades Sun 18:00 to Fri 17:00 New York time, with a daily 17:00-18:00 break.
    # Anchoring to NY time keeps the schedule correct through US/EU DST changes.
    trading_hours=TradingHours(
        week_open=WeeklyBoundary(day="sunday", at=time(18, 0)),
        week_close=WeeklyBoundary(day="friday", at=time(17, 0)),
        daily_break_start=time(17, 0),
        daily_break_end=time(18, 0),
        day_roll=time(17, 0),
    ),
    sessions={
        "asia": SessionWindow(label="Asia (Tokyo)", tz="Asia/Tokyo", start=time(9), end=time(18)),
        "london": SessionWindow(label="London", tz="Europe/London", start=time(8), end=time(17)),
        "new_york": SessionWindow(label="New York", tz=NY, start=time(8), end=time(17)),
        "comex": SessionWindow(
            label="COMEX gold pit hours", tz=NY, start=time(8, 20), end=time(13, 30)
        ),
        # ICT-style kill zones, common in SMC trading on gold.
        "asia_kz": SessionWindow(label="Asian kill zone", tz=NY, start=time(20), end=time(0)),
        "london_kz": SessionWindow(label="London kill zone", tz=NY, start=time(2), end=time(5)),
        "ny_am_kz": SessionWindow(label="NY AM kill zone", tz=NY, start=time(7), end=time(10)),
        "ny_pm_kz": SessionWindow(label="NY PM kill zone", tz=NY, start=time(13, 30), end=time(16)),
    },
    # Conservative retail-ECN defaults: ~20c spread, $7/lot round turn, 2 ticks slippage.
    # Dukascopy history carries real bid/ask, so these act as a floor, not the main cost source.
    cost_model=CostModel(spread=0.20, commission_per_lot_round_turn=7.0, slippage_ticks=2),
)

DEFAULT_INSTRUMENTS: tuple[InstrumentCreate, ...] = (XAUUSD,)


async def upsert_instrument(session: AsyncSession, spec: InstrumentCreate) -> Instrument:
    data = spec.model_dump(mode="json")
    # Numeric columns take Decimals, not the JSON-mode strings.
    data["tick_size"] = spec.tick_size
    data["contract_size"] = spec.contract_size

    existing = await session.scalar(select(Instrument).where(Instrument.symbol == spec.symbol))
    if existing is None:
        instrument = Instrument(**data)
        session.add(instrument)
        logger.info("seeded instrument", extra={"symbol": spec.symbol})
    else:
        instrument = existing
        for key, value in data.items():
            setattr(instrument, key, value)
        logger.info("updated instrument", extra={"symbol": spec.symbol})
    await session.flush()
    return instrument


async def seed_reference_data(session: AsyncSession) -> list[Instrument]:
    return [await upsert_instrument(session, spec) for spec in DEFAULT_INSTRUMENTS]

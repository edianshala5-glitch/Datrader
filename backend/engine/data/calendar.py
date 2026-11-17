"""Market calendar: when the market is open, which trading day a bar belongs to,
and how higher-timeframe bars are aligned.

All maths is vectorized and works on UTC timestamps. Local wall-clock logic happens in
the instrument's ``session_timezone`` (America/New_York for XAUUSD), so DST is handled
by ``zoneinfo``, not by hand-maintained offsets.

Trading-day convention: a new day starts at ``day_roll`` local time. With a 17:00 NY
roll, Monday 18:30 NY belongs to *Tuesday's* session. This matches how gold desks and
"NY-close" broker charts label days.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.models.enums import Timeframe

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MINUTES_PER_WEEK = 7 * 1440


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _as_index(ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        raise ValueError("timestamps must be timezone-aware (UTC)")
    return idx


@dataclass(frozen=True)
class MarketCalendar:
    tz: ZoneInfo
    week_open_mow: int  # minute-of-week (Mon 00:00 = 0) in local time
    week_close_mow: int
    break_start: int | None  # minute-of-day, local
    break_end: int | None
    day_roll: int  # minute-of-day, local

    @classmethod
    def from_instrument_config(
        cls, session_timezone: str, trading_hours: dict[str, Any]
    ) -> MarketCalendar:
        """Build from the validated ``Instrument.trading_hours`` JSON."""
        from app.schemas.instrument import TradingHours

        th = TradingHours.model_validate(trading_hours)
        open_mow = _WEEKDAYS.index(th.week_open.day) * 1440 + _minutes(th.week_open.at)
        close_mow = _WEEKDAYS.index(th.week_close.day) * 1440 + _minutes(th.week_close.at)
        return cls(
            tz=ZoneInfo(session_timezone),
            week_open_mow=open_mow,
            week_close_mow=close_mow,
            break_start=_minutes(th.daily_break_start) if th.daily_break_start else None,
            break_end=_minutes(th.daily_break_end) if th.daily_break_end else None,
            day_roll=_minutes(th.day_roll),
        )

    @classmethod
    def xauusd(cls) -> MarketCalendar:
        from app.services.seed import XAUUSD

        return cls.from_instrument_config(
            XAUUSD.session_timezone, XAUUSD.trading_hours.model_dump(mode="json")
        )

    # ------------------------------------------------------------------ helpers
    def _local_minutes(
        self, ts: pd.Series | pd.DatetimeIndex
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        local = _as_index(ts).tz_convert(self.tz)
        mod = np.asarray(local.hour * 60 + local.minute, dtype=np.int64)
        mow = np.asarray(local.day_of_week, dtype=np.int64) * 1440 + mod
        return mow, mod

    # ------------------------------------------------------------------ public API
    def is_open(self, ts: pd.Series | pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        """True where the market is open at that instant (minute resolution)."""
        mow, mod = self._local_minutes(ts)
        if self.week_open_mow > self.week_close_mow:  # week wraps over Monday 00:00
            in_week = (mow >= self.week_open_mow) | (mow < self.week_close_mow)
        else:
            in_week = (mow >= self.week_open_mow) & (mow < self.week_close_mow)
        if self.break_start is not None and self.break_end is not None:
            if self.break_start <= self.break_end:
                in_break = (mod >= self.break_start) & (mod < self.break_end)
            else:
                in_break = (mod >= self.break_start) | (mod < self.break_end)
            return np.asarray(in_week & ~in_break)
        return np.asarray(in_week)

    def trading_clock(self, ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Naive local wall clock shifted so each trading day starts at 00:00."""
        local = _as_index(ts).tz_convert(self.tz).tz_localize(None)
        shift = pd.Timedelta(minutes=(1440 - self.day_roll) % 1440)
        return local + shift

    def trading_day(self, ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Session date each timestamp belongs to (naive midnight dates)."""
        return self.trading_clock(ts).normalize()

    def bucket_start(self, ts: pd.Series | pd.DatetimeIndex, timeframe: Timeframe) -> pd.Series:
        """UTC start of the ``timeframe`` bar containing each timestamp.

        Buckets are aligned to the trading-day start, so 4h bars open at
        17:00/21:00/01:00/... NY and daily bars span 17:00→17:00 NY. For timeframes
        that divide an hour this is the same as plain clock flooring. The offset is
        computed and subtracted in UTC, so DST transitions need no ambiguous localization.
        """
        idx = _as_index(ts).tz_convert("UTC")
        if timeframe is Timeframe.TICK:
            raise ValueError("tick data has no bar buckets")
        clock = self.trading_clock(idx)
        floored = clock.floor(f"{timeframe.seconds}s")
        return pd.Series(idx - (clock - floored), index=getattr(ts, "index", None))

    def open_minutes(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Every minute in [start, end) during which the market should be trading."""
        minutes = pd.date_range(start.floor("min"), end, freq="1min", inclusive="left", tz="UTC")
        return minutes[self.is_open(minutes)]

    def hour_has_trading(self, hours: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        """For hour-start timestamps, True if any minute in that hour is open."""
        if len(hours) == 0:
            return np.zeros(0, dtype=bool)
        # Check every 15 minutes: no market boundary falls between those points.
        offsets = [pd.Timedelta(minutes=m) for m in (0, 15, 30, 45, 59)]
        result = np.zeros(len(hours), dtype=bool)
        for off in offsets:
            result |= self.is_open(hours + off)
        return result

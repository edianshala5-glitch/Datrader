from __future__ import annotations

import pandas as pd
import pytest

from app.models.enums import Timeframe
from engine.data.calendar import MarketCalendar

CAL = MarketCalendar.xauusd()


def ny(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        ("2024-03-02 12:00", False),  # Saturday
        ("2024-03-03 17:59", False),  # Sunday before open
        ("2024-03-03 18:00", True),  # Sunday open
        ("2024-03-05 09:30", True),  # Tuesday NY morning
        ("2024-03-05 17:00", False),  # daily break
        ("2024-03-05 17:59", False),
        ("2024-03-05 18:00", True),  # reopen
        ("2024-03-08 16:59", True),  # Friday before close
        ("2024-03-08 17:00", False),  # weekly close
        ("2024-07-10 17:30", False),  # break also in summer (EDT)
    ],
)
def test_is_open(when: str, expected: bool) -> None:
    assert bool(CAL.is_open(pd.DatetimeIndex([ny(when)]))[0]) is expected


def test_trading_day_rolls_at_17_ny() -> None:
    idx = pd.DatetimeIndex([ny("2024-03-04 18:30"), ny("2024-03-05 16:59"), ny("2024-03-05 18:01")])
    days = CAL.trading_day(idx)
    assert [d.day_name() for d in days] == ["Tuesday", "Tuesday", "Wednesday"]
    # Sunday evening belongs to Monday's session.
    assert CAL.trading_day(pd.DatetimeIndex([ny("2024-03-03 19:00")]))[0].day_name() == "Monday"


@pytest.mark.parametrize(
    ("ts_utc", "tf", "bucket_utc"),
    [
        ("2024-01-15 14:37", Timeframe.M5, "2024-01-15 14:35"),
        ("2024-01-15 14:37", Timeframe.M30, "2024-01-15 14:30"),
        # Winter: 17:00 NY = 22:00 UTC, so D1 opens at 22:00 UTC.
        ("2024-01-15 23:10", Timeframe.D1, "2024-01-15 22:00"),
        ("2024-01-15 21:59", Timeframe.D1, "2024-01-14 22:00"),
        # Summer: 17:00 NY = 21:00 UTC.
        ("2024-07-15 21:30", Timeframe.D1, "2024-07-15 21:00"),
        # 4h buckets open 17/21/01/05/09/13 NY. 14:37 UTC = 09:37 EST → 09:00 EST.
        ("2024-01-15 14:37", Timeframe.H4, "2024-01-15 14:00"),
        ("2024-07-15 14:37", Timeframe.H4, "2024-07-15 13:00"),
    ],
)
def test_bucket_start(ts_utc: str, tf: Timeframe, bucket_utc: str) -> None:
    out = CAL.bucket_start(pd.DatetimeIndex([pd.Timestamp(ts_utc, tz="UTC")]), tf)
    assert out.iloc[0] == pd.Timestamp(bucket_utc, tz="UTC")


def test_open_minutes_one_week() -> None:
    mins = CAL.open_minutes(ny("2024-03-03 00:00"), ny("2024-03-10 00:00"))
    # Sun 18:00 → Fri 17:00 = 5 days x 23h of trading (1h daily break).
    assert len(mins) == 5 * 23 * 60


def test_hour_has_trading_handles_boundaries() -> None:
    hours = pd.DatetimeIndex(
        [ny("2024-03-03 17:00"), ny("2024-03-03 18:00"), ny("2024-03-08 16:00")]
    )
    assert CAL.hour_has_trading(hours).tolist() == [False, True, True]


def test_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        CAL.is_open(pd.DatetimeIndex([pd.Timestamp("2024-01-01 12:00")]))

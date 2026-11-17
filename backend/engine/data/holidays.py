"""US market holidays relevant to COMEX/OTC gold.

On these New York dates gold either does not trade or closes early (typically around
13:30-14:30 NY), then reopens at 18:00 for the next session. Examples: real Dukascopy data
for MLK Day 2024 stops at 14:30 NY. The quality report uses this list to classify such gaps
as expected. The context features expose it so the Meta-Analyzer can learn that holiday
sessions behave differently.

Columbus Day and Veterans Day are federal holidays but gold trades normally, so they are
deliberately not included.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

CLOSED = "closed"
EARLY_CLOSE = "early_close"


class _MetalsHolidayCalendar(AbstractHolidayCalendar):
    rules = [  # noqa: RUF012 - pandas API
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_FULL_CLOSURE = {"NewYearsDay", "Good Friday", "GoodFriday", "Christmas"}


@lru_cache(maxsize=8)
def us_market_holidays(start_year: int = 2000, end_year: int = 2040) -> dict[date, str]:
    """Map of NY calendar date → ``closed`` | ``early_close``."""
    cal = _MetalsHolidayCalendar()
    series = cal.holidays(
        start=pd.Timestamp(start_year, 1, 1), end=pd.Timestamp(end_year, 12, 31), return_name=True
    )
    dates = pd.DatetimeIndex(series.index)
    return {
        d.date(): CLOSED if name in _FULL_CLOSURE else EARLY_CLOSE
        for d, name in zip(dates, series.to_numpy(), strict=True)
    }


def is_us_holiday(day: date) -> bool:
    return day in us_market_holidays()

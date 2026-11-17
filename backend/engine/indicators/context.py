"""Time, session and regime context: the dimensions the Meta-Analyzer slices trades by.

All time features use the instrument calendar. ``weekday`` is the *trading-day* weekday
(Sunday 18:00 NY onwards counts as Monday). ``hour_ny``/``minute_ny`` are New York wall
clock at the bar's open. Session flags evaluate each window in its own timezone.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from engine.data.holidays import us_market_holidays
from engine.indicators.base import (
    Category,
    Inputs,
    Output,
    OutputKind,
    Outputs,
    Param,
    indicator,
)
from engine.indicators.standard import _adx as adx_fn
from engine.indicators.standard import atr_values, ema

FLAG = Output(OutputKind.FLAG)


def _minutes(t: time | str) -> int:
    if isinstance(t, str):
        t = time.fromisoformat(t)
    return t.hour * 60 + t.minute


def session_mask(ts: pd.DatetimeIndex, tz: str, start: time | str, end: time | str) -> np.ndarray:
    local = ts.tz_convert(tz)
    mod = np.asarray(local.hour * 60 + local.minute)
    s, e = _minutes(start), _minutes(end)
    return (mod >= s) & (mod < e) if s <= e else (mod >= s) | (mod < e)


@indicator(
    "time",
    category=Category.CONTEXT,
    outputs={
        "weekday": Output(
            OutputKind.STATE, values=(0, 1, 2, 3, 4), description="trading-day Mon=0..Fri=4"
        ),
        "hour_ny": Output(OutputKind.STATE, values=tuple(range(24)), description="New York hour"),
        "minute_ny": Output(OutputKind.OSCILLATOR, 0, 1439, description="New York minute of day"),
        "minutes_into_day": Output(
            OutputKind.OSCILLATOR, 0, 1439, description="since 18:00 NY open"
        ),
        "us_holiday": Output(
            OutputKind.FLAG, description="US market holiday (closed or early close)"
        ),
        "pre_weekend": Output(OutputKind.FLAG, description="Friday session after 12:00 NY"),
    },
    description="Calendar context on the gold trading day (17:00 NY roll).",
    needs_calendar=True,
)
def _time(inp: Inputs) -> Outputs:
    assert inp.calendar is not None
    cal = inp.calendar
    ny = inp.ts.tz_convert(cal.tz)
    day = cal.trading_day(inp.ts)
    weekday = np.asarray(day.dayofweek, dtype=np.int8)
    minute = np.asarray(ny.hour * 60 + ny.minute, dtype=np.int32)
    clock = cal.trading_clock(inp.ts)
    into_day = np.asarray(clock.hour * 60 + clock.minute, dtype=np.int32) - 60  # day opens 18:00
    holidays = us_market_holidays()
    ny_dates = ny.date
    hol = np.fromiter((d in holidays for d in ny_dates), dtype=bool, count=len(ny_dates))
    return {
        "weekday": weekday,
        "hour_ny": np.asarray(ny.hour, dtype=np.int8),
        "minute_ny": minute,
        "minutes_into_day": np.clip(into_day, 0, 1439),
        "us_holiday": hol,
        "pre_weekend": (weekday == 4) & (minute >= 12 * 60),
    }


@indicator(
    "sessions",
    category=Category.CONTEXT,
    outputs={
        "asia": FLAG,
        "london": FLAG,
        "new_york": FLAG,
        "london_ny_overlap": FLAG,
        "comex": FLAG,
        "asia_kz": FLAG,
        "london_kz": FLAG,
        "ny_am_kz": FLAG,
        "ny_pm_kz": FLAG,
    },
    description="Session and ICT kill-zone membership, each in its own timezone (DST-correct).",
    needs_calendar=True,
)
def _sessions(inp: Inputs) -> Outputs:
    if not inp.sessions:
        # Fail loudly: silently returning "never in session" would make every
        # session-filtered strategy look like it simply never trades.
        raise ValueError("sessions indicator needs the instrument's session definitions")
    out: Outputs = {}
    for key in (
        "asia",
        "london",
        "new_york",
        "comex",
        "asia_kz",
        "london_kz",
        "ny_am_kz",
        "ny_pm_kz",
    ):
        w = inp.sessions.get(key)
        out[key] = (
            session_mask(inp.ts, w["tz"], w["start"], w["end"])
            if w
            else np.zeros(len(inp), dtype=bool)
        )
    out["london_ny_overlap"] = out["london"] & out["new_york"]
    return out


@indicator(
    "regime",
    category=Category.REGIME,
    params={
        "window": Param(2000, 200, 20000, 100),
        "trend_fast": Param(50, 5, 200, 1),
        "trend_slow": Param(200, 20, 400, 1),
        "adx_min": Param(20.0, 10.0, 40.0, 1.0),
    },
    outputs={
        "vol_rank": Output(
            OutputKind.RATIO,
            0,
            1,
            description="ATR percentile over `window` bars",
            uses=("window",),
        ),
        "vol_regime": Output(
            OutputKind.STATE,
            values=(0, 1, 2),
            description="0 low, 1 normal, 2 high",
            uses=("window",),
        ),
        "trend_regime": Output(
            OutputKind.STATE,
            values=(-1, 0, 1),
            description="EMA alignment + ADX",
            uses=("trend_fast", "trend_slow", "adx_min"),
        ),
        "efficiency": Output(
            OutputKind.RATIO, 0, 1, description="Kaufman efficiency ratio (20)", uses=()
        ),
    },
    description="Volatility and trend regime (trailing percentile ranks, no lookahead).",
)
def _regime(inp: Inputs, window: int, trend_fast: int, trend_slow: int, adx_min: float) -> Outputs:
    if trend_fast >= trend_slow:
        raise ValueError("regime: trend_fast must be < trend_slow")
    atr = pd.Series(atr_values(inp, 14))
    rank = atr.rolling(window, min_periods=max(50, window // 10)).rank(pct=True).to_numpy()
    vol_regime = np.where(rank < 1 / 3, 0, np.where(rank > 2 / 3, 2, 1)).astype(np.int8)
    vol_regime = np.where(np.isnan(rank), 1, vol_regime).astype(np.int8)

    fast, slow = ema(inp.close, trend_fast), ema(inp.close, trend_slow)
    adx = adx_fn(inp, 14)["adx"]
    strong = np.nan_to_num(adx) >= adx_min
    trend = np.where(strong & (fast > slow), 1, np.where(strong & (fast < slow), -1, 0))

    n = 20
    c = inp.close
    change = np.abs(c - np.concatenate([np.full(n, np.nan), c[:-n]]))
    path = pd.Series(np.abs(np.diff(c, prepend=np.nan))).rolling(n, min_periods=n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(path > 0, change / path, 0.0)
    er = np.where(np.isnan(path), np.nan, er)
    return {
        "vol_rank": rank,
        "vol_regime": vol_regime,
        "trend_regime": trend.astype(np.int8),
        "efficiency": er,
    }

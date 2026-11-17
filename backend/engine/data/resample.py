"""Ticks → M1 bars → higher timeframes.

Bar timestamps are the bar's *open* time (UTC). Higher timeframes are aligned to the
trading-day start via :meth:`MarketCalendar.bucket_start`, so XAUUSD D1 bars run
17:00→17:00 New York and 4h bars open at 17/21/01/05/09/13 NY all year.
"""

from __future__ import annotations

import pandas as pd

from app.models.enums import Timeframe
from engine.data.calendar import MarketCalendar
from engine.data.schema import BAR_COLUMNS, empty_bars, normalize_bars

_OHLC_AGG = {
    "bid_open": "first",
    "bid_high": "max",
    "bid_low": "min",
    "bid_close": "last",
    "ask_open": "first",
    "ask_high": "max",
    "ask_low": "min",
    "ask_close": "last",
    "volume": "sum",
}


def ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    if ticks.empty:
        return empty_bars()
    t = ticks.sort_values("ts", kind="stable")
    g = t.groupby(t["ts"].dt.floor("1min"), sort=True)
    bars = pd.DataFrame(
        {
            "bid_open": g["bid"].first(),
            "bid_high": g["bid"].max(),
            "bid_low": g["bid"].min(),
            "bid_close": g["bid"].last(),
            "ask_open": g["ask"].first(),
            "ask_high": g["ask"].max(),
            "ask_low": g["ask"].min(),
            "ask_close": g["ask"].last(),
            "volume": g["bid_volume"].sum().astype("float64")
            + g["ask_volume"].sum().astype("float64"),
            "tick_count": g.size(),
        }
    )
    bars.index.name = "ts"
    return normalize_bars(bars.reset_index())


def merge_candle_sides(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """Join Dukascopy BID and ASK M1 candles into the canonical bar layout.

    Only minutes present on both sides are kept: one-sided minutes have no usable spread.
    """
    if bid.empty or ask.empty:
        return empty_bars()
    b = bid.set_index("ts").add_prefix("bid_")
    a = ask.set_index("ts").add_prefix("ask_")
    joined = b.join(a, how="inner").sort_index()
    joined["volume"] = joined.pop("bid_volume") + joined.pop("ask_volume")
    joined["tick_count"] = pd.NA
    joined.index.name = "ts"
    return normalize_bars(joined.reset_index())


def resample_bars(
    bars: pd.DataFrame, timeframe: Timeframe, calendar: MarketCalendar
) -> pd.DataFrame:
    """Aggregate bars (normally M1) into ``timeframe`` bars aligned to the trading day."""
    if timeframe in (Timeframe.TICK, Timeframe.M1):
        raise ValueError(f"cannot resample into {timeframe}")
    if bars.empty:
        return empty_bars()
    src = bars.sort_values("ts", kind="stable")
    keys = calendar.bucket_start(src["ts"], timeframe)
    g = src.groupby(keys.to_numpy(), sort=True)
    out = g.agg(_OHLC_AGG)
    out["tick_count"] = g["tick_count"].sum(min_count=1)
    out.index.name = "ts"
    return normalize_bars(out.reset_index()[BAR_COLUMNS])

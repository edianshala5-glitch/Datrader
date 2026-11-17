"""Canonical in-memory and on-disk shapes of market data.

Every frame passed between data components uses one of these two layouts, with a
``ts`` column of dtype ``datetime64[ns, UTC]`` in memory. On disk timestamps are stored at
microsecond precision (ticks are millisecond-stamped), which is DuckDB's native
TIMESTAMPTZ resolution.

Bars keep *both* bid and ask OHLC. The backtester can then fill buys at the ask and
sells at the bid, and trigger a long stop on the bid low and a short stop on the ask high,
so spread costs come from data instead of a guessed constant.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import pyarrow as pa

TS_DTYPE: Final = "datetime64[ns, UTC]"

TICK_COLUMNS = ["ts", "bid", "ask", "bid_volume", "ask_volume"]

BAR_PRICE_COLUMNS = [
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
]
BAR_COLUMNS = ["ts", *BAR_PRICE_COLUMNS, "volume", "tick_count"]

TICK_ARROW_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("us", tz="UTC")),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("bid_volume", pa.float32()),
        ("ask_volume", pa.float32()),
    ]
)

BAR_ARROW_SCHEMA = pa.schema(
    [("ts", pa.timestamp("us", tz="UTC"))]
    + [(c, pa.float64()) for c in BAR_PRICE_COLUMNS]
    + [("volume", pa.float64()), ("tick_count", pa.int32())]  # tick_count null for candles
)


def empty_ticks() -> pd.DataFrame:
    return normalize_ticks(pd.DataFrame({c: [] for c in TICK_COLUMNS}))


def empty_bars() -> pd.DataFrame:
    return normalize_bars(pd.DataFrame({c: [] for c in BAR_COLUMNS}))


def normalize_ticks(df: pd.DataFrame) -> pd.DataFrame:
    out = df[TICK_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True).astype(TS_DTYPE)
    out[["bid", "ask"]] = out[["bid", "ask"]].astype("float64")
    out[["bid_volume", "ask_volume"]] = out[["bid_volume", "ask_volume"]].astype("float32")
    return out.reset_index(drop=True)


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = df[BAR_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True).astype(TS_DTYPE)
    out[BAR_PRICE_COLUMNS] = out[BAR_PRICE_COLUMNS].astype("float64")
    out["volume"] = out["volume"].astype("float64")
    out["tick_count"] = out["tick_count"].astype("Int32")
    return out.reset_index(drop=True)


def mid(bars: pd.DataFrame, field: str = "close") -> pd.Series:
    return (bars[f"bid_{field}"] + bars[f"ask_{field}"]) / 2.0

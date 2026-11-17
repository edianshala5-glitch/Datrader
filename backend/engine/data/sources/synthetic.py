"""Synthetic XAUUSD-like M1 data for tests, demos and offline development.

Not a market model. It only reproduces the *shape* the pipeline must handle: the trading
calendar (weekend close, daily 17:00-18:00 NY break), intraday volatility seasonality
(London/NY sessions busier), fat-tailed returns, volatility clustering, and a
time-varying bid/ask spread. Fully deterministic for a given seed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

import numpy as np
import pandas as pd

from app.models.enums import DataSource
from engine.data.calendar import MarketCalendar
from engine.data.schema import empty_bars, normalize_bars
from engine.data.sources.base import Chunk, month_windows


def generate_m1(
    start: datetime,
    end: datetime,
    calendar: MarketCalendar,
    *,
    seed: int = 7,
    start_price: float = 2000.0,
    annual_vol: float = 0.15,
) -> pd.DataFrame:
    minutes = calendar.open_minutes(pd.Timestamp(start), pd.Timestamp(end))
    n = len(minutes)
    if n == 0:
        return empty_bars()
    rng = np.random.default_rng(seed)

    hour_ny = np.asarray(minutes.tz_convert(calendar.tz).hour)
    seasonal = np.where((hour_ny >= 3) & (hour_ny < 12), 1.6, 0.7)  # London + NY busier
    # GARCH-ish clustering: slowly varying volatility multiplier.
    log_vol = np.cumsum(rng.normal(0, 0.02, n))
    log_vol -= pd.Series(log_vol).rolling(5000, min_periods=1).mean().to_numpy()
    vol_mult = np.exp(np.clip(log_vol, -1.0, 1.0))

    per_min = annual_vol / np.sqrt(252 * 23 * 60)
    shocks = rng.standard_t(df=4, size=n) / np.sqrt(2.0)  # fat tails, ~unit variance
    rets = shocks * per_min * seasonal * vol_mult
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, per_min * seasonal * vol_mult * start_price, (2, n)))
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]

    tick = 0.01
    spread = np.round((0.18 + 0.15 * (seasonal < 1) + rng.exponential(0.05, n)) / tick) * tick
    ohlc = {"open": open_, "high": high, "low": low, "close": close}
    bars = pd.DataFrame({"ts": minutes})
    for k, v in ohlc.items():
        bars[f"bid_{k}"] = np.round(v / tick) * tick
        bars[f"ask_{k}"] = bars[f"bid_{k}"] + spread
    bars["volume"] = np.round(rng.gamma(2.0, 0.05, n) * seasonal, 4)
    bars["tick_count"] = rng.poisson(40 * seasonal).astype("int32") + 1
    return normalize_bars(bars)


class SyntheticSource:
    source = DataSource.SYNTHETIC

    def __init__(self, symbol: str, calendar: MarketCalendar, *, seed: int = 7) -> None:
        self.symbol = symbol.upper()
        self.calendar = calendar
        self.seed = seed

    async def chunks(self, start: datetime, end: datetime) -> AsyncIterator[Chunk]:
        # Generate the whole range once so the price path is continuous across months.
        bars = generate_m1(start, end, self.calendar, seed=self.seed)
        windows = month_windows(start, end)
        for i, (ws, we) in enumerate(windows):
            part = bars[(bars["ts"] >= ws) & (bars["ts"] < we)].reset_index(drop=True)
            yield Chunk(label=ws.strftime("%Y-%m"), m1=part, progress=(i + 1) / len(windows))

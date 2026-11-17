"""Completed M1 bars → completed strategy-timeframe bars, aligned exactly like the batch
resampler (``MarketCalendar.bucket_start``), so live and backtest bars are identical."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.models.enums import Timeframe
from engine.data.calendar import MarketCalendar

Bar = dict[str, Any]


class BarAggregator:
    def __init__(
        self, timeframe: Timeframe, calendar: MarketCalendar, grace_seconds: float = 5.0
    ) -> None:
        self.tf = timeframe
        self.cal = calendar
        self.grace = pd.Timedelta(seconds=grace_seconds)
        self.bucket: pd.Timestamp | None = None
        self.acc: Bar | None = None

    def _bucket(self, ts: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(self.cal.bucket_start(pd.DatetimeIndex([ts]), self.tf).iloc[0])

    def _emit(self) -> list[Bar]:
        if self.acc is None:
            return []
        out = dict(self.acc)
        self.acc, self.bucket = None, None
        return [out]

    def add(self, m1: Bar) -> list[Bar]:
        ts = pd.Timestamp(m1["ts"])
        b = self._bucket(ts)
        out = self._emit() if self.bucket is not None and b != self.bucket else []
        if self.acc is None:
            self.bucket = b
            self.acc = {**m1, "ts": b}
        else:
            a = self.acc
            for side in ("bid", "ask"):
                a[f"{side}_high"] = max(a[f"{side}_high"], m1[f"{side}_high"])
                a[f"{side}_low"] = min(a[f"{side}_low"], m1[f"{side}_low"])
                a[f"{side}_close"] = m1[f"{side}_close"]
            a["volume"] = a["volume"] + m1["volume"]
            tc = m1.get("tick_count")
            a["tick_count"] = (a.get("tick_count") or 0) + (
                tc if tc == tc and tc is not None else 0
            )
        # The bucket's last minute just arrived → emit without waiting for the next bar.
        if ts + pd.Timedelta(minutes=1) >= b + pd.Timedelta(seconds=self.tf.seconds):
            out += self._emit()
        return out

    def flush(self, now: pd.Timestamp) -> list[Bar]:
        """Emit a bucket whose time is up even if its last minute had no ticks."""
        if (
            self.bucket is not None
            and now >= self.bucket + pd.Timedelta(seconds=self.tf.seconds) + self.grace
        ):
            return self._emit()
        return []

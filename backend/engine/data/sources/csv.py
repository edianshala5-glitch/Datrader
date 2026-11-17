"""Import bar history from CSV files, e.g. MT5 exports from your own broker.

Supported layouts (auto-detected from the header):

* **MT5** (History Center / "Export bars"), tab separated::

      <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>
      2024.01.02  14:00:00  2066.22  2067.10  2065.80  2066.95  812  0  33

  Prices are bid. ``<SPREAD>`` is in *points*, so ask = bid + spread * point.
* **Generic**: columns ``time, open, high, low, close`` plus optional ``volume`` and
  ``spread`` (price units), any common delimiter.

Timestamps in the file are interpreted in ``tz``: any IANA zone, or ``"NY+7"`` for
the common "New York close" broker server time (GMT+2 in winter, GMT+3 in summer,
switching on US DST dates).
"""

from __future__ import annotations

import csv
import io
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.models.enums import DataSource
from engine.data.schema import empty_bars, normalize_bars
from engine.data.sources.base import Chunk, month_windows

NY_CLOSE_TZ = "NY+7"


def to_utc(local: pd.Series, tz: str) -> pd.Series:
    if tz.upper() == NY_CLOSE_TZ:
        # Server clock = New York clock + 7h. Shift back to NY wall time, localize, convert.
        ny = (local - pd.Timedelta(hours=7)).dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
        )
        out: pd.Series = ny.dt.tz_convert("UTC")
        return out
    out = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert(
        "UTC"
    )
    return out


def parse_bars_csv(text: str, *, tz: str = "UTC", point: float = 0.01) -> pd.DataFrame:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        sep = dialect.delimiter
    except csv.Error:
        sep = "\t" if "\t" in sample else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    if {"date", "time"} <= set(df.columns):  # MT5
        local = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
        spread = df["spread"].astype(float) * point if "spread" in df else 0.0
        volume = df["tickvol"] if "tickvol" in df else df.get("vol", 0)
    elif "time" in df.columns:
        local = pd.to_datetime(df["time"], utc=False)
        if getattr(local.dt, "tz", None) is not None:
            local = local.dt.tz_convert("UTC").dt.tz_localize(None)
            tz = "UTC"
        spread = df["spread"].astype(float) if "spread" in df else 0.0
        volume = df.get("volume", 0)
    else:
        raise ValueError(f"unrecognized CSV header: {list(df.columns)}")

    ts = to_utc(pd.Series(local), tz)
    bars = pd.DataFrame({"ts": ts})
    for f in ("open", "high", "low", "close"):
        bars[f"bid_{f}"] = df[f].astype(float)
        bars[f"ask_{f}"] = df[f].astype(float) + spread
    bars["volume"] = pd.Series(volume, index=df.index).astype(float)
    bars["tick_count"] = pd.NA
    bars = bars.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")
    return normalize_bars(bars)


class CsvSource:
    """Yields the file's bars as monthly chunks. The file's bar size is kept as-is.
    Ingest M1 exports to get every higher timeframe."""

    source = DataSource.CSV

    def __init__(self, path: Path, symbol: str, *, tz: str = "UTC", point: float = 0.01):
        self.path = Path(path)
        self.symbol = symbol.upper()
        self.tz = tz
        self.point = point

    async def chunks(self, start: datetime, end: datetime) -> AsyncIterator[Chunk]:
        bars = parse_bars_csv(self.path.read_text(), tz=self.tz, point=self.point)
        if bars.empty:
            return
        s = max(pd.Timestamp(start), bars["ts"].iloc[0])
        e = min(pd.Timestamp(end), bars["ts"].iloc[-1] + pd.Timedelta(minutes=1))
        windows = month_windows(s, e)
        for i, (ws, we) in enumerate(windows):
            part = bars[(bars["ts"] >= ws) & (bars["ts"] < we)]
            yield Chunk(
                label=ws.strftime("%Y-%m"),
                m1=part.reset_index(drop=True) if not part.empty else empty_bars(),
                progress=(i + 1) / len(windows),
            )

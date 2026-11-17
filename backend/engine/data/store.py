"""Parquet market-data lake with DuckDB reads.

Layout (Hive-style partitions so DuckDB can prune by year)::

    {root}/ticks/symbol=XAUUSD/source=dukascopy/year=2024/month=01/part.parquet
    {root}/bars/symbol=XAUUSD/source=dukascopy/timeframe=5m/year=2024/part.parquet

Writes are idempotent merges with atomic replacement, so re-ingesting a range never
duplicates rows and a crash never leaves a half-written partition.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.models.enums import DataSource, Timeframe
from engine.data.schema import (
    BAR_ARROW_SCHEMA,
    BAR_COLUMNS,
    TICK_ARROW_SCHEMA,
    TICK_COLUMNS,
    empty_bars,
    empty_ticks,
    normalize_bars,
    normalize_ticks,
)


@dataclass(frozen=True)
class Coverage:
    start: pd.Timestamp | None
    end: pd.Timestamp | None  # timestamp of the last row
    rows: int


def _ts(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ paths
    def _tick_dir(self, symbol: str, source: DataSource) -> Path:
        return self.root / "ticks" / f"symbol={symbol}" / f"source={source.value}"

    def _bar_dir(self, symbol: str, source: DataSource, timeframe: Timeframe) -> Path:
        return (
            self.root
            / "bars"
            / f"symbol={symbol}"
            / f"source={source.value}"
            / f"timeframe={timeframe.value}"
        )

    def tick_partition(self, symbol: str, source: DataSource, year: int, month: int) -> Path:
        return (
            self._tick_dir(symbol, source) / f"year={year}" / f"month={month:02d}" / "part.parquet"
        )

    def bar_partition(
        self, symbol: str, source: DataSource, timeframe: Timeframe, year: int
    ) -> Path:
        return self._bar_dir(symbol, source, timeframe) / f"year={year}" / "part.parquet"

    # ------------------------------------------------------------------ io helpers
    @staticmethod
    def _write_atomic(df: pd.DataFrame, path: Path, schema: pa.Schema) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(path)

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        df: pd.DataFrame = pq.read_table(path).to_pandas()
        return df

    # ------------------------------------------------------------------ writes
    def write_ticks(self, symbol: str, source: DataSource, ticks: pd.DataFrame) -> list[Path]:
        """Merge ticks into monthly partitions. Existing ticks inside any *hour* present in
        ``ticks`` are replaced. Hours are the download unit, and duplicate timestamps
        are legitimate tick data, so de-duplicating on ``ts`` would be wrong."""
        if ticks.empty:
            return []
        ticks = normalize_ticks(ticks)
        written = []
        month_key = ticks["ts"].dt.year * 100 + ticks["ts"].dt.month
        for key in sorted(int(k) for k in month_key.unique()):
            part = ticks[month_key == key]
            path = self.tick_partition(symbol, source, key // 100, key % 100)
            existing = self._read_file(path)
            if existing is not None and not existing.empty:
                existing = normalize_ticks(existing)
                hours = part["ts"].dt.floor("h").unique()
                existing = existing[~existing["ts"].dt.floor("h").isin(hours)]
                part = pd.concat([existing, part], ignore_index=True)
            part = part.sort_values("ts", kind="stable")[TICK_COLUMNS]
            self._write_atomic(part, path, TICK_ARROW_SCHEMA)
            written.append(path)
        return written

    def write_bars(
        self,
        symbol: str,
        source: DataSource,
        timeframe: Timeframe,
        bars: pd.DataFrame,
        *,
        replace_years: bool = False,
    ) -> list[Path]:
        """Merge bars into yearly partitions (newer rows win on equal ``ts``).

        ``replace_years=True`` overwrites each touched year outright. Used when
        higher timeframes are rebuilt from complete M1 data.
        """
        if bars.empty:
            return []
        bars = normalize_bars(bars)
        written = []
        years = bars["ts"].dt.year
        for year in sorted(int(y) for y in years.unique()):
            part = bars[years == year]
            path = self.bar_partition(symbol, source, timeframe, year)
            existing = None if replace_years else self._read_file(path)
            if existing is not None and not existing.empty:
                part = pd.concat([normalize_bars(existing), part], ignore_index=True)
                part = part.drop_duplicates("ts", keep="last")
            part = part.sort_values("ts", kind="stable")[BAR_COLUMNS]
            self._write_atomic(part, path, BAR_ARROW_SCHEMA)
            written.append(path)
        return written

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _query(
        glob: str,
        columns: Iterable[str],
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
        limit: int | None = None,
        tail: bool = False,
    ) -> pd.DataFrame:
        # Timestamps leave DuckDB as integer epoch-µs: fast, and needs no tz objects.
        cols = ", ".join("epoch_us(ts) AS ts" if c == "ts" else f'"{c}"' for c in columns)
        where, params = ["TRUE"], []
        if start is not None:
            where.append("ts >= ? AND year >= ?")
            params += [start, start.year]
        if end is not None:
            where.append("ts < ? AND year <= ?")
            params += [end, end.year]
        order = "DESC" if tail else "ASC"
        sql = (
            f"SELECT {cols} FROM read_parquet(?, hive_partitioning = true) "
            f"WHERE {' AND '.join(where)} ORDER BY ts {order}"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with duckdb.connect() as con:
            con.execute("SET TimeZone = 'UTC'")
            df = con.execute(sql, [glob, *params]).df()
        if tail:
            df = df.iloc[::-1].reset_index(drop=True)
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="us", utc=True)
        return df

    def read_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        source: DataSource = DataSource.DUKASCOPY,
        start: datetime | pd.Timestamp | str | None = None,
        end: datetime | pd.Timestamp | str | None = None,
        limit: int | None = None,
        tail: bool = False,
    ) -> pd.DataFrame:
        """Bars with ``start <= ts < end``. ``tail=True`` with ``limit`` returns the latest
        ``limit`` bars (still in ascending order)."""
        base = self._bar_dir(symbol, source, timeframe)
        if not base.exists():
            return empty_bars()
        df = self._query(
            str(base / "year=*" / "*.parquet"),
            BAR_COLUMNS,
            _ts(start) if start is not None else None,
            _ts(end) if end is not None else None,
            limit,
            tail,
        )
        return normalize_bars(df)

    def read_ticks(
        self,
        symbol: str,
        *,
        source: DataSource = DataSource.DUKASCOPY,
        start: datetime | pd.Timestamp | str | None = None,
        end: datetime | pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        base = self._tick_dir(symbol, source)
        if not base.exists():
            return empty_ticks()
        df = self._query(
            str(base / "year=*" / "month=*" / "*.parquet"),
            TICK_COLUMNS,
            _ts(start) if start is not None else None,
            _ts(end) if end is not None else None,
        )
        return normalize_ticks(df)

    def bar_coverage(
        self, symbol: str, timeframe: Timeframe, source: DataSource = DataSource.DUKASCOPY
    ) -> Coverage:
        base = self._bar_dir(symbol, source, timeframe)
        return self._coverage(str(base / "year=*" / "*.parquet"), base.exists())

    def tick_coverage(self, symbol: str, source: DataSource = DataSource.DUKASCOPY) -> Coverage:
        base = self._tick_dir(symbol, source)
        return self._coverage(str(base / "year=*" / "month=*" / "*.parquet"), base.exists())

    @staticmethod
    def _coverage(glob: str, exists: bool) -> Coverage:
        if not exists:
            return Coverage(None, None, 0)
        with duckdb.connect() as con:
            con.execute("SET TimeZone = 'UTC'")
            row = con.execute(
                "SELECT epoch_us(min(ts)), epoch_us(max(ts)), count(*) "
                "FROM read_parquet(?, hive_partitioning = true)",
                [glob],
            ).fetchone()
        if row is None or not row[2]:
            return Coverage(None, None, 0)
        return Coverage(
            pd.Timestamp(row[0], unit="us", tz="UTC"),
            pd.Timestamp(row[1], unit="us", tz="UTC"),
            int(row[2]),
        )

    def bar_years(
        self, symbol: str, timeframe: Timeframe, source: DataSource = DataSource.DUKASCOPY
    ) -> list[int]:
        base = self._bar_dir(symbol, source, timeframe)
        if not base.exists():
            return []
        return sorted(int(p.name.split("=")[1]) for p in base.glob("year=*") if p.is_dir())

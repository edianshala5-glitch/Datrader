"""Common interface for historical data sources.

A source yields :class:`Chunk` objects (typically one per month) so ingestion can
write, report progress and checkpoint incrementally instead of holding years of data in
memory.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from app.models.enums import DataSource


@dataclass
class Chunk:
    label: str  # e.g. "2024-01"
    m1: pd.DataFrame  # canonical bar layout
    ticks: pd.DataFrame | None = None  # canonical tick layout, when the source has ticks
    progress: float = 0.0  # 0..1 through the requested range


class HistoricalSource(Protocol):
    source: DataSource
    symbol: str

    def chunks(self, start: datetime, end: datetime) -> AsyncIterator[Chunk]: ...


def month_windows(
    start: datetime, end: datetime
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) into calendar-month windows (UTC)."""
    s, e = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
    edges = [s, *pd.date_range(s.normalize() + pd.offsets.MonthBegin(1), e, freq="MS"), e]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if a < b]

"""Dukascopy public datafeed: free historical ticks and M1 candles with bid/ask.

File layout (all times UTC, **month is zero-based** in the URL):

* ticks:   ``{SYM}/{YYYY}/{MM-1}/{DD}/{HH}h_ticks.bi5``        one file per hour
* candles: ``{SYM}/{YYYY}/{MM-1}/{DD}/{BID|ASK}_candles_min_1.bi5``  one file per day

``.bi5`` files are LZMA-compressed arrays of big-endian records:

* tick   (20 bytes): ms-from-hour int32, ask int32, bid int32, ask_vol f32, bid_vol f32
* candle (24 bytes): s-from-day int32, open, close, low, high int32, volume f32

Integer prices are divided by a per-instrument scale (XAUUSD: 1000).

Dukascopy rate-limits aggressively (HTTP 429), and shared cloud IPs especially. The
client therefore applies a global request rate, a *shared* cooldown that pauses every
worker when any request is throttled, exponential backoff with jitter, and a raw-file
cache. An interrupted download resumes where it stopped and never fetches a file twice.
"""

from __future__ import annotations

import asyncio
import logging
import lzma
import os
import random
import time as _time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import TracebackType

import httpx
import numpy as np
import pandas as pd

from app.models.enums import DataSource
from engine.data.calendar import MarketCalendar
from engine.data.resample import merge_candle_sides, ticks_to_m1
from engine.data.schema import TS_DTYPE, empty_bars, empty_ticks, normalize_ticks
from engine.data.sources.base import Chunk, month_windows

logger = logging.getLogger(__name__)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

PRICE_SCALE: dict[str, int] = {"XAUUSD": 1000, "XAGUSD": 1000}
DEFAULT_PRICE_SCALE = 100_000  # most FX majors

TICK_DTYPE = np.dtype(
    [("ms", ">i4"), ("ask", ">i4"), ("bid", ">i4"), ("ask_vol", ">f4"), ("bid_vol", ">f4")]
)
CANDLE_DTYPE = np.dtype(
    [
        ("sec", ">i4"),
        ("open", ">i4"),
        ("close", ">i4"),
        ("low", ">i4"),
        ("high", ">i4"),
        ("volume", ">f4"),
    ]
)


def price_scale(symbol: str) -> int:
    return PRICE_SCALE.get(symbol.upper(), DEFAULT_PRICE_SCALE)


# --------------------------------------------------------------------------- paths
def tick_path(symbol: str, hour: datetime) -> str:
    h = hour.astimezone(UTC)
    return f"{symbol.upper()}/{h.year:04d}/{h.month - 1:02d}/{h.day:02d}/{h.hour:02d}h_ticks.bi5"


def candle_path(symbol: str, day: date, side: str) -> str:
    side = side.upper()
    if side not in ("BID", "ASK"):
        raise ValueError("side must be BID or ASK")
    folder = f"{symbol.upper()}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}"
    return f"{folder}/{side}_candles_min_1.bi5"


# --------------------------------------------------------------------------- codec
def _decompress(raw: bytes, dtype: np.dtype) -> np.ndarray:
    if not raw:
        return np.zeros(0, dtype=dtype)
    data = lzma.decompress(raw)
    if len(data) % dtype.itemsize:
        raise ValueError(f"corrupt bi5: {len(data)} bytes is not a multiple of {dtype.itemsize}")
    return np.frombuffer(data, dtype=dtype)


def decode_ticks(raw: bytes, hour_start: datetime, scale: int) -> pd.DataFrame:
    arr = _decompress(raw, TICK_DTYPE)
    if arr.size == 0:
        return empty_ticks()
    base = pd.Timestamp(hour_start).tz_convert("UTC").as_unit("ns")
    ts = base + pd.to_timedelta(arr["ms"].astype(np.int64), unit="ms")
    return normalize_ticks(
        pd.DataFrame(
            {
                "ts": ts,
                "bid": arr["bid"].astype(np.float64) / scale,
                "ask": arr["ask"].astype(np.float64) / scale,
                "bid_volume": arr["bid_vol"].astype(np.float32),
                "ask_volume": arr["ask_vol"].astype(np.float32),
            }
        )
    )


def encode_ticks(ticks: pd.DataFrame, hour_start: datetime, scale: int) -> bytes:
    """Inverse of :func:`decode_ticks`. Used for test fixtures."""
    base = pd.Timestamp(hour_start).tz_convert("UTC")
    arr = np.zeros(len(ticks), dtype=TICK_DTYPE)
    arr["ms"] = ((ticks["ts"] - base) // pd.Timedelta(milliseconds=1)).to_numpy()
    arr["ask"] = np.round(ticks["ask"].to_numpy() * scale)
    arr["bid"] = np.round(ticks["bid"].to_numpy() * scale)
    arr["ask_vol"] = ticks["ask_volume"].to_numpy()
    arr["bid_vol"] = ticks["bid_volume"].to_numpy()
    return lzma.compress(arr.tobytes(), format=lzma.FORMAT_ALONE)


def decode_candles(raw: bytes, day_start: datetime, scale: int) -> pd.DataFrame:
    """Return ``ts, open, high, low, close, volume`` for one side. Zero-volume candles
    (Dukascopy's flat fillers for minutes with no ticks) are dropped, which keeps
    candle-mode bars consistent with bars resampled from ticks."""
    arr = _decompress(raw, CANDLE_DTYPE)
    arr = arr[arr["volume"] > 0]
    body_hi = np.maximum(arr["open"], arr["close"])
    body_lo = np.minimum(arr["open"], arr["close"])
    if np.any((arr["high"] < body_hi) | (arr["low"] > body_lo)):
        # A wrong field order shows up here immediately rather than as silent bad bars.
        raise ValueError("candle file failed OHLC sanity check: unexpected record layout")
    base = pd.Timestamp(day_start).tz_convert("UTC").as_unit("ns")
    return pd.DataFrame(
        {
            "ts": (base + pd.to_timedelta(arr["sec"].astype(np.int64), unit="s")).astype(TS_DTYPE),
            "open": arr["open"] / scale,
            "high": arr["high"] / scale,
            "low": arr["low"] / scale,
            "close": arr["close"] / scale,
            "volume": arr["volume"].astype(np.float64),
        }
    )


def encode_candles(candles: pd.DataFrame, day_start: datetime, scale: int) -> bytes:
    base = pd.Timestamp(day_start).tz_convert("UTC")
    arr = np.zeros(len(candles), dtype=CANDLE_DTYPE)
    arr["sec"] = ((candles["ts"] - base) // pd.Timedelta(seconds=1)).to_numpy()
    for col in ("open", "close", "low", "high"):
        arr[col] = np.round(candles[col].to_numpy() * scale)
    arr["volume"] = candles["volume"].to_numpy()
    return lzma.compress(arr.tobytes(), format=lzma.FORMAT_ALONE)


# --------------------------------------------------------------------------- client
class DownloadError(RuntimeError):
    pass


class _RateLimiter:
    """Spaces request starts at least ``1/rate`` seconds apart across all workers."""

    def __init__(self, rate: float, clock: Callable[[], float] = _time.monotonic) -> None:
        self._interval = 1.0 / rate if rate > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()
        self._clock = clock

    async def acquire(self) -> None:
        async with self._lock:
            now = self._clock()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


class DukascopyClient:
    """Async downloader with caching, global rate limiting and shared 429 cooldown.

    ``fetch`` returns raw bytes. ``b""`` means Dukascopy has no data for that file
    (market closed, or before the instrument's history starts).
    """

    RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        cache_dir: Path,
        *,
        max_concurrency: int = 4,
        requests_per_second: float = 4.0,
        max_retries: int = 8,
        backoff_base: float = 2.0,
        backoff_max: float = 300.0,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._limiter = _RateLimiter(requests_per_second)
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._sleep = sleep
        self._resume_at = 0.0  # monotonic time before which nobody may send
        self._throttle_streak = 0
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "swarm-trader-data/0.1"},
            follow_redirects=True,
        )
        self.stats = {"requests": 0, "cache_hits": 0, "throttled": 0, "errors": 0}

    async def __aenter__(self) -> DukascopyClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _cache_file(self, path: str) -> Path:
        return self.cache_dir / path

    @staticmethod
    def _write_atomic(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(target)

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self._backoff_max)
            except ValueError:
                pass
        delay = self._backoff_base * (2 ** min(attempt, 10))
        return float(min(delay, self._backoff_max) * random.uniform(0.75, 1.25))

    async def _wait_cooldown(self) -> None:
        while (remaining := self._resume_at - _time.monotonic()) > 0:
            await self._sleep(remaining)

    async def fetch(self, path: str, *, cache: bool = True) -> bytes:
        cached = self._cache_file(path)
        if cache and cached.exists():
            self.stats["cache_hits"] += 1
            return cached.read_bytes()

        last_error: str = ""
        for attempt in range(self._max_retries + 1):
            async with self._sem:
                await self._wait_cooldown()
                await self._limiter.acquire()
                self.stats["requests"] += 1
                try:
                    resp = await self._client.get(f"/{path}")
                except httpx.TransportError as exc:
                    resp = None
                    last_error = f"{type(exc).__name__}: {exc}"

            if resp is not None and resp.status_code == 200:
                self._throttle_streak = 0
                data = resp.content
                if cache:
                    self._write_atomic(cached, data)
                return data
            if resp is not None and resp.status_code == 404:
                if cache:
                    self._write_atomic(cached, b"")
                return b""

            if resp is not None:
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in self.RETRY_STATUSES:
                    self.stats["errors"] += 1
                    raise DownloadError(f"{path}: {last_error}")
                if resp.status_code == 429:
                    self.stats["throttled"] += 1
                    self._throttle_streak += 1
            delay = self._backoff(
                max(attempt, self._throttle_streak - 1),
                resp.headers.get("Retry-After") if resp is not None else None,
            )
            # Shared cooldown: every worker pauses, so we stop hammering a throttled server.
            self._resume_at = max(self._resume_at, _time.monotonic() + delay)
            logger.warning(
                "dukascopy retry",
                extra={"path": path, "error": last_error, "attempt": attempt, "delay_s": delay},
            )

        self.stats["errors"] += 1
        raise DownloadError(f"{path}: gave up after {self._max_retries} retries ({last_error})")


def hour_range(start: datetime, end: datetime) -> pd.DatetimeIndex:
    """Hour starts covering [start, end)."""
    s = pd.Timestamp(start).tz_convert("UTC").floor("h")
    return pd.date_range(s, pd.Timestamp(end).tz_convert("UTC"), freq="h", inclusive="left")


def is_complete(period_end: datetime, now: datetime | None = None) -> bool:
    """Dukascopy publishes a file once its period ends. Allow a margin before caching."""
    now = now or datetime.now(UTC)
    return period_end <= now - timedelta(hours=2)


# --------------------------------------------------------------------------- source
class DukascopySource:
    """Historical XAUUSD (or any Dukascopy symbol) as monthly chunks.

    ``mode="candles"`` (default): 2 requests per trading day (BID + ASK M1 candles),
    about 12x fewer requests than ticks. Enough for 1m-and-up research.
    ``mode="ticks"``: 1 request per open hour. Keeps raw ticks for exact intrabar
    stop/target resolution and spread analysis.
    """

    source = DataSource.DUKASCOPY

    def __init__(
        self,
        client: DukascopyClient,
        symbol: str,
        calendar: MarketCalendar,
        *,
        mode: str = "candles",
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_skipped: int = 50,
    ) -> None:
        if mode not in ("candles", "ticks"):
            raise ValueError("mode must be 'candles' or 'ticks'")
        self.client = client
        self.symbol = symbol.upper()
        self.calendar = calendar
        self.mode = mode
        self.scale = price_scale(self.symbol)
        self._now = now
        # Files that still failed after all retries. They are skipped (a gap that the next run
        # fills: failures are never cached) so one bad file can't stop a 10-year download.
        self.skipped: list[str] = []
        self.max_skipped = max_skipped

    async def _fetch_or_skip(self, path: str, *, cache: bool) -> bytes | None:
        try:
            return await self.client.fetch(path, cache=cache)
        except DownloadError as exc:
            self.skipped.append(path)
            logger.warning("dukascopy file skipped", extra={"path": path, "error": str(exc)})
            if len(self.skipped) > self.max_skipped:
                raise DownloadError(
                    f"{len(self.skipped)} files failed: Dukascopy is unreachable or blocking "
                    "this connection. Try again later (downloaded files are kept)."
                ) from exc
            return None

    async def chunks(self, start: datetime, end: datetime) -> AsyncIterator[Chunk]:
        end = min(pd.Timestamp(end).tz_convert("UTC"), pd.Timestamp(self._now()))
        windows = month_windows(start, end)
        for i, (w_start, w_end) in enumerate(windows):
            if self.mode == "ticks":
                ticks = await self._month_ticks(w_start, w_end)
                m1 = ticks_to_m1(ticks)
            else:
                ticks = None
                m1 = await self._month_candles(w_start, w_end)
            m1 = m1[(m1["ts"] >= w_start) & (m1["ts"] < w_end)]
            yield Chunk(
                label=w_start.strftime("%Y-%m"),
                m1=m1.reset_index(drop=True),
                ticks=ticks,
                progress=(i + 1) / len(windows),
            )

    async def _month_ticks(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        hours = hour_range(start, end)
        hours = hours[self.calendar.hour_has_trading(hours)]
        now = self._now()

        async def one(hour: pd.Timestamp) -> pd.DataFrame:
            h = hour.to_pydatetime()
            raw = await self._fetch_or_skip(
                tick_path(self.symbol, h), cache=is_complete(h + timedelta(hours=1), now)
            )
            return empty_ticks() if raw is None else decode_ticks(raw, h, self.scale)

        frames = await asyncio.gather(*(one(h) for h in hours))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return empty_ticks()
        ticks = pd.concat(frames, ignore_index=True).sort_values("ts", kind="stable")
        return ticks[(ticks["ts"] >= start) & (ticks["ts"] < end)].reset_index(drop=True)

    async def _month_candles(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        days = pd.date_range(start.normalize(), end, freq="D", inclusive="left")
        # A UTC day is worth requesting if any of its hours trade.
        days = pd.DatetimeIndex(
            [
                d
                for d in days
                if self.calendar.hour_has_trading(pd.date_range(d, periods=24, freq="h")).any()
            ]
        )
        now = self._now()

        async def one(day: pd.Timestamp) -> pd.DataFrame:
            d = day.to_pydatetime()
            cacheable = is_complete(d + timedelta(days=1), now)
            bid_raw, ask_raw = await asyncio.gather(
                self._fetch_or_skip(candle_path(self.symbol, d.date(), "BID"), cache=cacheable),
                self._fetch_or_skip(candle_path(self.symbol, d.date(), "ASK"), cache=cacheable),
            )
            if bid_raw is None or ask_raw is None:  # both sides are needed for a bar
                return empty_bars()
            return merge_candle_sides(
                decode_candles(bid_raw, d, self.scale), decode_candles(ask_raw, d, self.scale)
            )

        frames = [f for f in await asyncio.gather(*(one(d) for d in days)) if not f.empty]
        if not frames:
            return empty_bars()
        return pd.concat(frames, ignore_index=True).sort_values("ts", kind="stable")

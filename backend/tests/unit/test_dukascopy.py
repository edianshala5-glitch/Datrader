from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from engine.data.calendar import MarketCalendar
from engine.data.sources import dukascopy as d

FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "dukascopy" / "XAUUSD_2024-01-02_14h_ticks.bi5"
)
HOUR = datetime(2024, 1, 2, 14, tzinfo=UTC)


def test_paths_use_zero_based_month() -> None:
    assert d.tick_path("xauusd", HOUR) == "XAUUSD/2024/00/02/14h_ticks.bi5"
    assert d.candle_path("XAUUSD", date(2024, 12, 31), "bid") == (
        "XAUUSD/2024/11/31/BID_candles_min_1.bi5"
    )
    with pytest.raises(ValueError):
        d.candle_path("XAUUSD", date(2024, 1, 1), "MID")


def test_decode_real_fixture() -> None:
    """A real Dukascopy hour (2024-01-02 14:00 UTC), checked against known values."""
    ticks = d.decode_ticks(FIXTURE.read_bytes(), HOUR, d.price_scale("XAUUSD"))
    assert len(ticks) == 15463
    first = ticks.iloc[0]
    assert first["ts"] == pd.Timestamp("2024-01-02 14:00:00.072", tz="UTC")
    assert (first["bid"], first["ask"]) == (2066.225, 2066.565)
    assert ticks["ts"].is_monotonic_increasing
    assert ticks["ts"].iloc[-1] < pd.Timestamp(HOUR) + pd.Timedelta(hours=1)
    assert (ticks["ask"] >= ticks["bid"]).all()
    assert 0.2 < (ticks["ask"] - ticks["bid"]).median() < 0.5


def test_tick_codec_roundtrip() -> None:
    ticks = d.decode_ticks(FIXTURE.read_bytes(), HOUR, 1000)
    assert d.decode_ticks(d.encode_ticks(ticks, HOUR, 1000), HOUR, 1000).equals(ticks)


def test_empty_and_corrupt_payloads() -> None:
    assert d.decode_ticks(b"", HOUR, 1000).empty
    import lzma

    with pytest.raises(ValueError, match="corrupt"):
        d.decode_ticks(lzma.compress(b"x" * 21, format=lzma.FORMAT_ALONE), HOUR, 1000)


def _candles(day: datetime, n: int = 5, base: float = 2000.0) -> pd.DataFrame:
    ts = pd.date_range(day, periods=n, freq="1min", tz="UTC")
    o = base + np.arange(n) * 0.5
    return pd.DataFrame(
        {"ts": ts, "open": o, "high": o + 1, "low": o - 1, "close": o + 0.25, "volume": 1.0}
    )


def test_candle_codec_drops_zero_volume_and_validates_layout() -> None:
    day = datetime(2024, 1, 2, tzinfo=UTC)
    c = _candles(day)
    c.loc[2, "volume"] = 0.0  # Dukascopy filler minute
    out = d.decode_candles(d.encode_candles(c, day, 1000), day, 1000)
    assert len(out) == 4 and out["close"].iloc[0] == 2000.25

    swapped = c.rename(columns={"high": "low", "low": "high"})  # wrong field order
    with pytest.raises(ValueError, match="sanity"):
        d.decode_candles(d.encode_candles(swapped, day, 1000), day, 1000)


# --------------------------------------------------------------------------- client
def _client(tmp_path: Path, handler, **kw) -> d.DukascopyClient:  # type: ignore[no-untyped-def]
    kw.setdefault("requests_per_second", 1000)
    kw.setdefault("backoff_base", 0.001)
    return d.DukascopyClient(tmp_path, transport=httpx.MockTransport(handler), **kw)


async def test_client_retries_429_then_caches(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": "Too Many Requests"})
        return httpx.Response(200, content=b"payload")

    async with _client(tmp_path, handler) as client:
        assert await client.fetch("XAUUSD/2024/00/02/14h_ticks.bi5") == b"payload"
        assert await client.fetch("XAUUSD/2024/00/02/14h_ticks.bi5") == b"payload"
        assert client.stats["throttled"] == 2 and client.stats["cache_hits"] == 1
    assert len(calls) == 3
    assert (tmp_path / "XAUUSD/2024/00/02/14h_ticks.bi5").read_bytes() == b"payload"


async def test_client_404_is_cached_as_no_data(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    async with _client(tmp_path, handler) as client:
        assert await client.fetch("a/b.bi5") == b""
        assert await client.fetch("a/b.bi5") == b""
    assert calls == 1


async def test_client_does_not_cache_incomplete_periods(tmp_path: Path) -> None:
    async with _client(tmp_path, lambda r: httpx.Response(200, content=b"partial")) as client:
        await client.fetch("x/now.bi5", cache=False)
    assert not (tmp_path / "x/now.bi5").exists()


async def test_client_gives_up_and_raises(tmp_path: Path) -> None:
    async with _client(tmp_path, lambda r: httpx.Response(503), max_retries=2) as client:
        with pytest.raises(d.DownloadError, match="gave up"):
            await client.fetch("x.bi5")
    async with _client(tmp_path, lambda r: httpx.Response(403)) as client:
        with pytest.raises(d.DownloadError, match="HTTP 403"):
            await client.fetch("y.bi5")


async def test_client_retries_transport_errors(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("reset by peer")
        return httpx.Response(200, content=b"ok")

    async with _client(tmp_path, handler) as client:
        assert await client.fetch("z.bi5") == b"ok"


# --------------------------------------------------------------------------- source
async def test_source_ticks_mode_only_requests_open_hours(tmp_path: Path) -> None:
    requested: list[str] = []
    real = FIXTURE.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/datafeed/")
        requested.append(path)
        return httpx.Response(200, content=real if path.endswith("02/14h_ticks.bi5") else b"")

    async with _client(tmp_path, handler) as client:
        src = d.DukascopySource(client, "XAUUSD", MarketCalendar.xauusd(), mode="ticks")
        chunks = [
            c
            async for c in src.chunks(
                datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 8, tzinfo=UTC)
            )
        ]
    assert len(chunks) == 1 and chunks[0].label == "2024-01"
    assert chunks[0].ticks is not None and len(chunks[0].ticks) == 15463
    assert len(chunks[0].m1) == 60
    # No requests for Saturday (Jan 6) or Sunday before the 18:00 NY open (23:00 UTC).
    assert not any("/00/06/" in p for p in requested)
    sunday = sorted(p for p in requested if "/00/07/" in p)
    assert sunday == ["XAUUSD/2024/00/07/23h_ticks.bi5"]
    # Daily break 22:00-23:00 UTC (17-18 EST) is skipped on weekdays.
    assert "XAUUSD/2024/00/03/22h_ticks.bi5" not in requested


async def test_source_candles_mode_merges_sides(tmp_path: Path) -> None:
    day = datetime(2024, 1, 3, tzinfo=UTC)
    bid = _candles(day + pd.Timedelta(hours=10))
    ask = bid.assign(**{k: bid[k] + 0.3 for k in ("open", "high", "low", "close")})
    payloads = {
        "XAUUSD/2024/00/03/BID_candles_min_1.bi5": d.encode_candles(bid, day, 1000),
        "XAUUSD/2024/00/03/ASK_candles_min_1.bi5": d.encode_candles(ask, day, 1000),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=payloads.get(request.url.path.removeprefix("/datafeed/"), b"")
        )

    async with _client(tmp_path, handler) as client:
        src = d.DukascopySource(client, "XAUUSD", MarketCalendar.xauusd(), mode="candles")
        [chunk] = [c async for c in src.chunks(day, day + pd.Timedelta(days=1))]
    assert chunk.ticks is None and len(chunk.m1) == 5
    spread = chunk.m1["ask_close"] - chunk.m1["bid_close"]
    assert np.allclose(spread, 0.3)
    assert chunk.m1["tick_count"].isna().all()


def test_real_candles_match_real_ticks() -> None:
    """Real Dukascopy files for 2024-01-02: candle-mode M1 bars must equal bars built
    from raw ticks. This pins the candle record layout (open, close, low, high)."""
    from engine.data.resample import merge_candle_sides, ticks_to_m1

    day = datetime(2024, 1, 2, tzinfo=UTC)
    fx = FIXTURE.parent
    bid = d.decode_candles((fx / "XAUUSD_2024-01-02_BID_candles_min_1.bi5").read_bytes(), day, 1000)
    ask = d.decode_candles((fx / "XAUUSD_2024-01-02_ASK_candles_min_1.bi5").read_bytes(), day, 1000)
    candles = merge_candle_sides(bid, ask)
    assert len(candles) == 23 * 60  # full session: 24h minus the 17:00-18:00 NY break

    from_ticks = ticks_to_m1(d.decode_ticks(FIXTURE.read_bytes(), HOUR, 1000)).set_index("ts")
    hour = candles.set_index("ts").loc[from_ticks.index]
    cols = [c for c in from_ticks.columns if c.startswith(("bid_", "ask_"))]
    np.testing.assert_allclose(hour[cols].to_numpy(), from_ticks[cols].to_numpy(), atol=1e-9)

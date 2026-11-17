"""Market data feeds that yield *completed* M1 bars in the canonical bid/ask layout.

* :class:`ReplayFeed`: replays stored M1 bars (tests, demos, dry runs). Optionally paced.
* :class:`OandaFeed`: OANDA v20 practice API (free demo account). Streams bid/ask prices
  for XAU_USD, builds M1 bars, and backfills recent M1 candles on start. Reconnects with
  backoff when the stream drops or goes silent (no heartbeat).

Authentication: ``Authorization: Bearer <token>``. In a managed cloud session the network
proxy can inject the token for the API host, so code may pass a placeholder.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import pandas as pd

from engine.data.schema import BAR_COLUMNS, normalize_bars

logger = logging.getLogger(__name__)
Bar = dict[str, Any]  # one canonical bar row (ts + bid/ask OHLC + volume + tick_count)


class BarFeed(Protocol):
    name: str

    async def backfill(self, count: int) -> pd.DataFrame: ...

    def bars(self) -> AsyncIterator[Bar]: ...


@dataclass
class _Minute:
    ts: pd.Timestamp
    bo: float
    bh: float
    bl: float
    bc: float
    ao: float
    ah: float
    al: float
    ac: float
    n: int

    def bar(self) -> Bar:
        return {
            "ts": self.ts,
            "bid_open": self.bo,
            "bid_high": self.bh,
            "bid_low": self.bl,
            "bid_close": self.bc,
            "ask_open": self.ao,
            "ask_high": self.ah,
            "ask_low": self.al,
            "ask_close": self.ac,
            "volume": float(self.n),
            "tick_count": self.n,
        }


class M1Builder:
    """Ticks → completed M1 bars. A minute is emitted when a tick from a later minute arrives,
    or when :meth:`flush` is called ``grace`` seconds after the minute ended (quiet markets)."""

    def __init__(self, grace_seconds: float = 2.0) -> None:
        self.cur: _Minute | None = None
        self.grace = pd.Timedelta(seconds=grace_seconds)
        self.last_emitted: pd.Timestamp | None = None  # ticks for emitted minutes are ignored

    def add(self, ts: pd.Timestamp, bid: float, ask: float) -> list[Bar]:
        minute = ts.floor("1min")
        if self.last_emitted is not None and minute <= self.last_emitted:
            return []  # replayed after a reconnect, or late
        out = []
        if self.cur is not None and minute > self.cur.ts:
            out.append(self.cur.bar())
            self.last_emitted = self.cur.ts
            self.cur = None
        if self.cur is None:
            self.cur = _Minute(minute, bid, bid, bid, bid, ask, ask, ask, ask, 1)
        elif minute == self.cur.ts:
            c = self.cur
            c.bh, c.bl, c.bc = max(c.bh, bid), min(c.bl, bid), bid
            c.ah, c.al, c.ac = max(c.ah, ask), min(c.al, ask), ask
            c.n += 1
        return out  # ticks older than the open minute are ignored (late / out of order)

    def flush(self, now: pd.Timestamp) -> list[Bar]:
        if self.cur is not None and now >= self.cur.ts + pd.Timedelta(minutes=1) + self.grace:
            bar = self.cur.bar()
            self.last_emitted = self.cur.ts
            self.cur = None
            return [bar]
        return []


class ReplayFeed:
    """Replays M1 bars after ``start``. Earlier bars are available as history/backfill."""

    name = "replay"

    def __init__(
        self, m1: pd.DataFrame, start: pd.Timestamp, *, bars_per_second: float = 0.0
    ) -> None:
        self.m1 = m1.reset_index(drop=True)
        self.start = pd.Timestamp(start)
        self.rate = bars_per_second

    async def backfill(self, count: int) -> pd.DataFrame:
        hist = self.m1[self.m1["ts"] < self.start]
        return hist.iloc[-count:].reset_index(drop=True)

    async def bars(self) -> AsyncIterator[Bar]:
        live = self.m1[self.m1["ts"] >= self.start]
        for row in live.to_dict(orient="records"):
            yield {str(k): v for k, v in row.items()}
            await asyncio.sleep(1.0 / self.rate if self.rate > 0 else 0)


class OandaFeed:
    name = "oanda"

    def __init__(
        self,
        instrument: str = "XAU_USD",
        *,
        token: str | None = None,
        account_id: str | None = None,
        practice: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
        heartbeat_timeout: float = 20.0,
        max_reconnects: int | None = None,
    ) -> None:
        env = "fxpractice" if practice else "fxtrade"
        self.api = f"https://api-{env}.oanda.com"
        self.stream = f"https://stream-{env}.oanda.com"
        self.instrument = instrument
        self.account_id = account_id
        self.heartbeat_timeout = heartbeat_timeout
        self.max_reconnects = max_reconnects
        headers = {
            "Authorization": f"Bearer {token or 'injected-by-proxy'}",
            "Accept-Datetime-Format": "RFC3339",
        }
        self._client = httpx.AsyncClient(
            headers=headers, transport=transport, timeout=httpx.Timeout(30, read=None)
        )
        self.status: dict[str, Any] = {"connected": False, "reconnects": 0, "last_tick": None}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _account(self) -> str:
        if self.account_id is None:
            r = await self._client.get(f"{self.api}/v3/accounts")
            r.raise_for_status()
            self.account_id = r.json()["accounts"][0]["id"]
        return self.account_id

    async def backfill(self, count: int = 5000) -> pd.DataFrame:
        r = await self._client.get(
            f"{self.api}/v3/instruments/{self.instrument}/candles",
            params={"granularity": "M1", "price": "BA", "count": min(count, 5000)},
        )
        r.raise_for_status()
        rows = []
        for c in r.json().get("candles", []):
            if not c.get("complete"):
                continue
            b, a = c["bid"], c["ask"]
            rows.append(
                {
                    "ts": pd.Timestamp(c["time"]),
                    "bid_open": float(b["o"]),
                    "bid_high": float(b["h"]),
                    "bid_low": float(b["l"]),
                    "bid_close": float(b["c"]),
                    "ask_open": float(a["o"]),
                    "ask_high": float(a["h"]),
                    "ask_low": float(a["l"]),
                    "ask_close": float(a["c"]),
                    "volume": float(c.get("volume", 0)),
                    "tick_count": int(c.get("volume", 0)),
                }
            )
        return normalize_bars(pd.DataFrame(rows, columns=BAR_COLUMNS))

    async def bars(self) -> AsyncIterator[Bar]:
        builder = M1Builder()
        attempt = 0  # consecutive failures (drives backoff); status["reconnects"] is the total
        while True:
            try:
                account = await self._account()
                url = f"{self.stream}/v3/accounts/{account}/pricing/stream"
                async with self._client.stream(
                    "GET", url, params={"instruments": self.instrument}
                ) as resp:
                    resp.raise_for_status()
                    self.status["connected"] = True
                    attempt = 0
                    lines = resp.aiter_lines()
                    while True:
                        line = await asyncio.wait_for(anext(lines), self.heartbeat_timeout)
                        if not line.strip():
                            continue
                        msg = json.loads(line)
                        if msg.get("type") == "PRICE" and msg.get("tradeable", True):
                            ts = pd.Timestamp(msg["time"])
                            bid, ask = (
                                float(msg["bids"][0]["price"]),
                                float(msg["asks"][0]["price"]),
                            )
                            self.status["last_tick"] = ts.isoformat()
                            for bar in builder.add(ts, bid, ask):
                                yield bar
                        elif msg.get("type") == "HEARTBEAT":
                            for bar in builder.flush(pd.Timestamp(msg["time"])):
                                yield bar
            except (
                StopAsyncIteration,
                TimeoutError,
                httpx.HTTPError,
                json.JSONDecodeError,
                KeyError,
            ) as exc:
                self.status["connected"] = False
                self.status["reconnects"] += 1
                attempt += 1
                delay = min(60.0, 2 ** min(attempt, 6))
                logger.warning(
                    "oanda stream dropped, reconnecting",
                    extra={"error": repr(exc), "delay_s": delay},
                )
                if (
                    self.max_reconnects is not None
                    and self.status["reconnects"] > self.max_reconnects
                ):
                    return
                await asyncio.sleep(delay if math.isfinite(delay) else 1)

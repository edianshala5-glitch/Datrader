"""Capital.com feed: free demo account, REST backfill + WebSocket quotes → completed M1 bars.

API reference: the official Postman collection (github.com/capital-com-sv/capital-api-postman).

* ``POST /api/v1/session`` with header ``X-CAP-API-KEY`` and body ``identifier`` / ``password``
  returns the ``CST`` and ``X-SECURITY-TOKEN`` headers. Both expire after 10 minutes of
  inactivity, and the endpoint allows 1 request per second.
* ``GET /api/v1/prices/{epic}?resolution=MINUTE&max=1000&to=...`` returns bid/ask candles;
  ``from`` / ``to`` filter on ``snapshotTimeUTC``.
* ``wss://api-streaming-capital.backend-capital.com/connect``: ``marketData.subscribe``
  pushes ``quote`` messages (``bid``, ``ofr``, ``timestamp`` in ms). The service must be
  pinged at least every 10 minutes.

Capital.com API keys can trade (there are no read-only keys). This feed only reads prices,
and the system never sends orders. Use a demo account and the key's custom password.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

import httpx
import pandas as pd
from websockets.exceptions import WebSocketException

from engine.data.schema import BAR_COLUMNS, normalize_bars
from engine.live.feeds import Bar, M1Builder

logger = logging.getLogger(__name__)

DEMO_API = "https://demo-api-capital.backend-capital.com/api/v1"
LIVE_API = "https://api-capital.backend-capital.com/api/v1"
STREAM_URL = "wss://api-streaming-capital.backend-capital.com/connect"
MAX_CANDLES = 1000  # per /prices request


class CapitalAuthError(RuntimeError):
    """Login refused (wrong key, identifier or password). Retrying won't help."""


class WebSocketLike(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...


WsConnect = Callable[[str], contextlib.AbstractAsyncContextManager[WebSocketLike]]


def _default_ws_connect(url: str) -> contextlib.AbstractAsyncContextManager[WebSocketLike]:
    from websockets.asyncio.client import connect

    # Protocol-level pings detect dead connections; proxies come from HTTPS_PROXY.
    return connect(url, ping_interval=20, ping_timeout=20, open_timeout=20)


def _utc(text: str) -> pd.Timestamp:
    ts = pd.Timestamp(text)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _candle(row: dict[str, Any]) -> Bar:
    o, h, lo, c = row["openPrice"], row["highPrice"], row["lowPrice"], row["closePrice"]
    vol = row.get("lastTradedVolume") or 0
    return {
        "ts": _utc(row["snapshotTimeUTC"]),
        "bid_open": float(o["bid"]),
        "bid_high": float(h["bid"]),
        "bid_low": float(lo["bid"]),
        "bid_close": float(c["bid"]),
        "ask_open": float(o["ask"]),
        "ask_high": float(h["ask"]),
        "ask_low": float(lo["ask"]),
        "ask_close": float(c["ask"]),
        "volume": float(vol),
        "tick_count": int(vol),
    }


class CapitalFeed:
    name = "capital"

    def __init__(
        self,
        epic: str = "GOLD",
        *,
        api_key: str | None,
        identifier: str | None,
        password: str | None,
        demo: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
        ws_connect: WsConnect | None = None,
        stream_url: str = STREAM_URL,
        keepalive_seconds: float = 300.0,
        flush_seconds: float = 1.0,
        max_reconnects: int | None = None,
        clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
    ) -> None:
        if not (api_key and identifier and password):
            raise CapitalAuthError(
                "Capital.com feed needs SWARM_CAPITAL_API_KEY, SWARM_CAPITAL_IDENTIFIER "
                "and SWARM_CAPITAL_PASSWORD"
            )
        self.epic = epic
        self.api = DEMO_API if demo else LIVE_API
        self.stream_url = stream_url
        self.keepalive = keepalive_seconds
        self.flush_every = flush_seconds
        self.max_reconnects = max_reconnects
        self._clock = clock
        self._api_key, self._identifier, self._password = api_key, identifier, password
        self._ws_connect = ws_connect or _default_ws_connect
        self._client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(30))
        self._tokens: dict[str, str] | None = None
        self._last_login = 0.0
        self._corr = itertools.count(1)
        self.status: dict[str, Any] = {"connected": False, "reconnects": 0, "last_tick": None}

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ REST
    async def _login(self) -> dict[str, str]:
        wait = 1.1 - (time.monotonic() - self._last_login)  # POST /session: 1 request/s
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_login = time.monotonic()
        r = await self._client.post(
            f"{self.api}/session",
            headers={"X-CAP-API-KEY": self._api_key},
            json={"identifier": self._identifier, "password": self._password},
        )
        if r.status_code in (400, 401, 403):
            # Never echo the body: it can contain the identifier.
            raise CapitalAuthError(f"Capital.com login refused (HTTP {r.status_code})")
        r.raise_for_status()
        cst, sec = r.headers.get("CST"), r.headers.get("X-SECURITY-TOKEN")
        if not cst or not sec:
            raise CapitalAuthError("Capital.com login returned no session tokens")
        self._tokens = {"CST": cst, "X-SECURITY-TOKEN": sec}
        host = (r.json() or {}).get("streamingHost")
        if isinstance(host, str) and host.startswith("wss://"):
            self.stream_url = host.rstrip("/") + "/connect"
        return self._tokens

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in (1, 2):
            tokens = self._tokens or await self._login()
            r = await self._client.get(f"{self.api}{path}", params=params, headers=tokens)
            if r.status_code == 401 and attempt == 1:  # session expired: log in again
                self._tokens = None
                continue
            r.raise_for_status()
            data: dict[str, Any] = r.json()
            return data
        raise CapitalAuthError("Capital.com session rejected after re-login")

    async def backfill(self, count: int = 5000) -> pd.DataFrame:
        """The latest ``count`` completed M1 bars, paged backwards 1,000 at a time."""
        now = self._clock()
        rows: dict[pd.Timestamp, Bar] = {}
        to: pd.Timestamp | None = None
        while len(rows) < count:
            params: dict[str, Any] = {"resolution": "MINUTE", "max": MAX_CANDLES}
            if to is not None:
                params["to"] = to.strftime("%Y-%m-%dT%H:%M:%S")
            prices = (await self._get(f"/prices/{self.epic}", params)).get("prices") or []
            page = [_candle(p) for p in prices]
            page = [b for b in page if b["ts"] + pd.Timedelta(minutes=1) <= now]  # completed
            new = [b for b in page if b["ts"] not in rows]
            if not new:
                break
            for b in new:
                rows[b["ts"]] = b
            to = min(b["ts"] for b in new) - pd.Timedelta(minutes=1)
            if len(prices) < MAX_CANDLES:
                break
        bars = sorted(rows.values(), key=lambda b: b["ts"])[-count:]
        return normalize_bars(pd.DataFrame(bars, columns=BAR_COLUMNS))

    # ------------------------------------------------------------------ streaming
    def _msg(self, destination: str, payload: dict[str, Any] | None = None) -> str:
        assert self._tokens is not None
        body: dict[str, Any] = {
            "destination": destination,
            "correlationId": str(next(self._corr)),
            "cst": self._tokens["CST"],
            "securityToken": self._tokens["X-SECURITY-TOKEN"],
        }
        if payload is not None:
            body["payload"] = payload
        return json.dumps(body)

    async def bars(self) -> AsyncIterator[Bar]:
        builder = M1Builder()
        attempt = 0
        bid = ask = None
        while True:
            try:
                await self._login()  # fresh tokens for every (re)connect
                async with self._ws_connect(self.stream_url) as ws:
                    await ws.send(self._msg("marketData.subscribe", {"epics": [self.epic]}))
                    self.status["connected"] = True
                    last_ping = time.monotonic()
                    while True:
                        if time.monotonic() - last_ping >= self.keepalive:
                            await ws.send(self._msg("ping"))
                            last_ping = time.monotonic()
                        try:
                            raw = await asyncio.wait_for(ws.recv(), self.flush_every)
                        except TimeoutError:
                            for bar in builder.flush(self._clock()):
                                yield bar
                            continue
                        msg = json.loads(raw)
                        if msg.get("destination") == "marketData.subscribe":
                            subs = (msg.get("payload") or {}).get("subscriptions") or {}
                            if msg.get("status") != "OK" or subs.get(self.epic) != "PROCESSED":
                                raise ConnectionError(f"subscription refused: {subs or msg}")
                            attempt = 0
                            continue
                        if msg.get("destination") != "quote":
                            continue
                        p = msg.get("payload") or {}
                        if p.get("epic") != self.epic:
                            continue
                        bid = float(p["bid"]) if p.get("bid") is not None else bid
                        ask = float(p["ofr"]) if p.get("ofr") is not None else ask
                        if bid is None or ask is None:
                            continue
                        ts = pd.Timestamp(int(p["timestamp"]), unit="ms", tz="UTC")
                        self.status["last_tick"] = ts.isoformat()
                        for bar in builder.add(ts, bid, ask):
                            yield bar
            except CapitalAuthError:
                self.status["connected"] = False
                raise  # bad credentials: let the supervisor report it and back off
            except (
                ConnectionError,
                OSError,
                TimeoutError,
                WebSocketException,
                httpx.HTTPError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as exc:
                self.status["connected"] = False
                self.status["reconnects"] += 1
                attempt += 1
                if (
                    self.max_reconnects is not None
                    and self.status["reconnects"] > self.max_reconnects
                ):
                    return
                delay = min(60.0, float(2 ** min(attempt, 6)))
                logger.warning(
                    "capital.com stream dropped, reconnecting",
                    extra={"error": type(exc).__name__, "delay_s": delay},
                )
                await asyncio.sleep(delay)

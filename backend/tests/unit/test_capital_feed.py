"""Capital.com feed against a simulated REST API and WebSocket stream."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pandas as pd
import pytest

from app.core.config import Settings
from app.services.feeds import check_live_feed, make_live_feed
from engine.live.capital import CapitalAuthError, CapitalFeed

NOW = pd.Timestamp("2024-03-05 14:30:20", tz="UTC")
KEY, USER, PW = "test-key", "me@example.com", "custom-pw"


def _candle(ts: pd.Timestamp, px: float) -> dict[str, Any]:
    def side(p: float) -> dict[str, float]:
        return {"bid": p, "ask": round(p + 0.3, 2)}

    return {
        "snapshotTime": (ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
        "snapshotTimeUTC": ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "openPrice": side(px),
        "highPrice": side(px + 1),
        "lowPrice": side(px - 1),
        "closePrice": side(px + 0.5),
        "lastTradedVolume": 42,
    }


class FakeCapital:
    """REST side: /session and /prices with 1,000-candle pages filtered on snapshotTimeUTC."""

    def __init__(self, n_minutes: int = 2500, expire_first_get: bool = False) -> None:
        # Minutes up to and including the forming one (14:30), which must be dropped.
        last = NOW.floor("1min")
        self.minutes = [last - pd.Timedelta(minutes=i) for i in range(n_minutes)][::-1]
        self.logins: list[dict[str, Any]] = []
        self.price_calls: list[dict[str, str]] = []
        self.expire_first_get = expire_first_get
        self.reject_login = False
        self.reject_code = f"error.invalid.details {USER}"  # unsafe: contains the e-mail

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/session") and req.method == "POST":
            self.logins.append({"key": req.headers.get("X-CAP-API-KEY"), **json.loads(req.content)})
            if self.reject_login:
                return httpx.Response(401, json={"errorCode": self.reject_code})
            n = len(self.logins)
            return httpx.Response(
                200,
                headers={"CST": f"cst-{n}", "X-SECURITY-TOKEN": f"sec-{n}"},
                json={"streamingHost": "wss://stream.example/"},
            )
        if "/prices/GOLD" in req.url.path:
            params = dict(req.url.params)
            self.price_calls.append({**params, "cst": req.headers.get("CST", "")})
            if self.expire_first_get:
                self.expire_first_get = False
                return httpx.Response(401, json={"errorCode": "error.invalid.session.token"})
            assert req.headers.get("X-SECURITY-TOKEN", "").startswith("sec-")
            mins = self.minutes
            if "to" in params:
                to = pd.Timestamp(params["to"], tz="UTC")
                mins = [m for m in mins if m <= to]
            page = mins[-int(params["max"]) :]
            return httpx.Response(200, json={"prices": [_candle(m, 2000.0) for m in page]})
        return httpx.Response(404)


class FakeWs:
    def __init__(self, script: list[Any]) -> None:
        self.sent: list[dict[str, Any]] = []
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        for item in script:
            self.queue.put_nowait(item)

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        item = await self.queue.get()
        if isinstance(item, BaseException):
            raise item
        if item == "idle":  # simulate a quiet market
            await asyncio.sleep(3600)
        return json.dumps(item)


def _feed(api: FakeCapital, sockets: list[FakeWs], **kw: Any) -> tuple[CapitalFeed, list[str]]:
    urls: list[str] = []

    @contextlib.asynccontextmanager
    async def connect(url: str) -> AsyncIterator[FakeWs]:
        urls.append(url)
        yield sockets.pop(0)

    feed = CapitalFeed(
        api_key=KEY,
        identifier=USER,
        password=PW,
        transport=httpx.MockTransport(api.handler),
        ws_connect=connect,  # type: ignore[arg-type]
        clock=lambda: NOW,
        **kw,
    )
    return feed, urls


def _quote(ts: str, bid: float | None, ofr: float | None, epic: str = "GOLD") -> dict[str, Any]:
    ms = int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)
    return {
        "status": "OK",
        "destination": "quote",
        "payload": {"epic": epic, "product": "CFD", "bid": bid, "ofr": ofr, "timestamp": ms},
    }


SUB_OK = {
    "status": "OK",
    "destination": "marketData.subscribe",
    "correlationId": "1",
    "payload": {"subscriptions": {"GOLD": "PROCESSED"}},
}


async def test_backfill_pages_and_drops_forming_minute() -> None:
    api = FakeCapital(n_minutes=2500)
    feed, _ = _feed(api, [])
    bars = await feed.backfill(2200)
    await feed.aclose()

    assert api.logins == [{"key": KEY, "identifier": USER, "password": PW}]
    assert len(api.price_calls) == 3 and "to" not in api.price_calls[0]
    # First page: 14:30 back to 21:51 the day before (999 complete minutes + the forming one).
    assert api.price_calls[1]["to"] == "2024-03-04T21:50:00"
    assert len(bars) == 2200 and bars["ts"].is_monotonic_increasing
    assert bars["ts"].iloc[-1] == pd.Timestamp("2024-03-05 14:29", tz="UTC")  # 14:30 is forming
    row = bars.iloc[-1]
    assert (row["bid_open"], row["ask_close"], row["volume"]) == (2000.0, 2000.8, 42.0)


async def test_expired_session_relogs_in() -> None:
    api = FakeCapital(n_minutes=50, expire_first_get=True)
    feed, _ = _feed(api, [])
    bars = await feed.backfill(100)
    await feed.aclose()
    assert len(api.logins) == 2 and len(bars) == 49
    assert api.price_calls[-1]["cst"] == "cst-2"


async def test_bad_credentials_raise_without_leaking_body() -> None:
    api = FakeCapital()
    api.reject_login = True
    feed, _ = _feed(api, [])
    with pytest.raises(CapitalAuthError) as exc:
        await feed.backfill(10)
    await feed.aclose()
    # The safe error code is shown with a hint; the e-mail inside the code text is not.
    assert "401" in str(exc.value) and USER not in str(exc.value)
    api2 = FakeCapital()
    api2.reject_login = True
    api2.reject_code = "error.invalid.api.key"
    feed2, _ = _feed(api2, [])
    with pytest.raises(CapitalAuthError, match=r"error\.invalid\.api\.key\): the API key is wrong"):
        await feed2.backfill(10)
    await feed2.aclose()
    with pytest.raises(CapitalAuthError, match="SWARM_CAPITAL_API_KEY"):
        CapitalFeed(api_key=None, identifier=USER, password=PW)


async def test_stream_builds_m1_bars_and_keeps_alive() -> None:
    ws = FakeWs(
        [
            SUB_OK,
            _quote("2024-03-05 14:31:05", 2001.0, 2001.3),
            _quote("2024-03-05 14:31:20", 2003.0, None),  # bid-only update keeps last ask
            _quote("2024-03-05 14:31:40", 1999.5, 1999.8),
            _quote("2024-03-05 14:31:50", 5.0, 6.0, epic="SILVER"),  # other markets ignored
            _quote("2024-03-05 14:32:02", 2000.0, 2000.3),  # next minute closes 14:31
            "idle",
        ]
    )
    api = FakeCapital()
    feed, urls = _feed(api, [ws], keepalive_seconds=0.0, flush_seconds=0.05)
    agen = feed.bars()
    bar = await asyncio.wait_for(anext(agen), 5)
    await agen.aclose()
    await feed.aclose()

    assert urls == ["wss://stream.example/connect"]  # streamingHost from the session
    sub = ws.sent[0]
    assert sub["destination"] == "marketData.subscribe" and sub["payload"] == {"epics": ["GOLD"]}
    assert (sub["cst"], sub["securityToken"]) == ("cst-1", "sec-1")
    assert any(m["destination"] == "ping" for m in ws.sent)  # session keep-alive
    assert bar["ts"] == pd.Timestamp("2024-03-05 14:31", tz="UTC")
    assert (bar["bid_open"], bar["bid_high"], bar["bid_low"], bar["bid_close"]) == (
        2001.0,
        2003.0,
        1999.5,
        1999.5,
    )
    assert (bar["ask_open"], bar["ask_high"], bar["ask_close"]) == (2001.3, 2001.3, 1999.8)
    assert bar["tick_count"] == 3 and feed.status["connected"]


async def test_quiet_market_flushes_last_minute() -> None:
    ws = FakeWs([SUB_OK, _quote("2024-03-05 14:28:10", 2001.0, 2001.3), "idle"])
    feed, _ = _feed(FakeCapital(), [ws], flush_seconds=0.05)  # clock is 14:30:20
    agen = feed.bars()
    bar = await asyncio.wait_for(anext(agen), 5)
    await agen.aclose()
    await feed.aclose()
    assert bar["ts"] == pd.Timestamp("2024-03-05 14:28", tz="UTC")


async def test_reconnects_with_fresh_session_after_drop() -> None:
    first = FakeWs([SUB_OK, ConnectionResetError("dropped")])
    second = FakeWs(
        [SUB_OK, _quote("2024-03-05 14:31:05", 2001.0, 2001.3), _quote("2024-03-05 14:32:01", 1, 2)]
    )
    api = FakeCapital()
    feed, urls = _feed(api, [first, second], flush_seconds=0.05)
    feed_sleep = asyncio.sleep

    async def no_wait(_: float) -> None:
        await feed_sleep(0)

    import engine.live.capital as mod

    mod.asyncio.sleep = no_wait  # type: ignore[assignment]
    try:
        agen = feed.bars()
        bar = await asyncio.wait_for(anext(agen), 5)
        await agen.aclose()
    finally:
        mod.asyncio.sleep = feed_sleep  # type: ignore[assignment]
        await feed.aclose()
    assert len(urls) == 2 and len(api.logins) == 2 and feed.status["reconnects"] == 1
    assert second.sent[0]["cst"] == "cst-2"
    assert bar["ts"] == pd.Timestamp("2024-03-05 14:31", tz="UTC")


async def test_refused_subscription_reconnects_then_gives_up() -> None:
    refused = {**SUB_OK, "payload": {"subscriptions": {"GOLD": "ERROR"}}}
    feed, urls = _feed(FakeCapital(), [FakeWs([refused])], max_reconnects=0)
    bars = [b async for b in feed.bars()]
    await feed.aclose()
    assert bars == [] and len(urls) == 1 and feed.status["reconnects"] == 1


def test_settings_select_feed() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.live_feed == "capital"
    with pytest.raises(CapitalAuthError, match="SWARM_CAPITAL_IDENTIFIER"):
        check_live_feed(s, "capital")
    check_live_feed(s, "oanda")  # OANDA tokens may be injected by a proxy
    ok = Settings(  # type: ignore[call-arg]
        _env_file=None, capital_api_key="k", capital_identifier="u", capital_password="p"
    )
    check_live_feed(ok, "capital")
    feed = make_live_feed(ok, "capital")
    assert isinstance(feed, CapitalFeed) and feed.api.startswith("https://demo-api-capital")


async def test_real_websocket_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default `websockets` client path against a local server speaking the protocol."""
    from websockets.asyncio.server import serve

    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    received: list[dict[str, Any]] = []

    async def handler(conn: Any) -> None:
        received.append(json.loads(await conn.recv()))
        await conn.send(json.dumps(SUB_OK))
        await conn.send(json.dumps(_quote("2024-03-05 14:31:05", 2001.0, 2001.3)))
        await conn.send(json.dumps(_quote("2024-03-05 14:32:01", 2002.0, 2002.3)))
        await asyncio.sleep(5)

    class NoHost(FakeCapital):
        def handler(self, req: httpx.Request) -> httpx.Response:
            res = super().handler(req)
            if req.url.path.endswith("/session"):
                return httpx.Response(200, headers=res.headers, json={})
            return res

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        feed = CapitalFeed(
            api_key=KEY,
            identifier=USER,
            password=PW,
            transport=httpx.MockTransport(NoHost().handler),
            stream_url=f"ws://127.0.0.1:{port}/connect",
            clock=lambda: NOW,
        )
        agen = feed.bars()
        bar = await asyncio.wait_for(anext(agen), 10)
        await agen.aclose()
        await feed.aclose()
    assert received[0]["destination"] == "marketData.subscribe"
    assert bar["ts"] == pd.Timestamp("2024-03-05 14:31", tz="UTC") and bar["bid_close"] == 2001.0

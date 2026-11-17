"""Live layer: bar building, aggregation, backtest parity, rationale, OANDA protocol, notifiers."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd
import pytest

from app.models.enums import Timeframe
from app.services.seed import XAUUSD
from engine.backtest.runner import CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.data.resample import resample_bars
from engine.data.sources.synthetic import generate_m1
from engine.live.aggregator import BarAggregator
from engine.live.feeds import M1Builder, OandaFeed, ReplayFeed
from engine.live.notify import TelegramNotifier, WebhookNotifier
from engine.live.signal_engine import LiveStrategy, SignalEvent, StrategyRunner
from engine.strategy.templates import all_templates

CAL = MarketCalendar.xauusd()
SESSIONS = XAUUSD.model_dump(mode="json")["sessions"]
COSTS = CostConfig.from_instrument(XAUUSD)
LIVE_START = pd.Timestamp("2023-03-05 23:00", tz="UTC")  # Sunday open


@pytest.fixture(scope="module")
def m1() -> pd.DataFrame:
    return generate_m1(
        datetime(2022, 12, 4, tzinfo=UTC), datetime(2023, 3, 26, tzinfo=UTC), CAL, seed=5
    )


# --------------------------------------------------------------------------- bars
def test_m1_builder() -> None:
    b = M1Builder(grace_seconds=2)
    t0 = pd.Timestamp("2024-03-05 14:00:05", tz="UTC")
    assert b.add(t0, 2000.0, 2000.3) == []
    assert b.add(t0 + pd.Timedelta(seconds=20), 2001.0, 2001.3) == []
    assert b.add(t0 + pd.Timedelta(seconds=40), 1999.5, 1999.8) == []
    [bar] = b.add(t0 + pd.Timedelta(seconds=60), 2000.5, 2000.8)  # next minute closes the first
    assert bar["ts"] == pd.Timestamp("2024-03-05 14:00", tz="UTC")
    assert (bar["bid_open"], bar["bid_high"], bar["bid_low"], bar["bid_close"]) == (
        2000.0,
        2001.0,
        1999.5,
        1999.5,
    )
    assert bar["ask_high"] == 2001.3 and bar["tick_count"] == 3
    assert b.flush(pd.Timestamp("2024-03-05 14:02:01", tz="UTC")) == []  # grace not over
    [quiet] = b.flush(pd.Timestamp("2024-03-05 14:02:03", tz="UTC"))
    assert quiet["ts"] == pd.Timestamp("2024-03-05 14:01", tz="UTC")


@pytest.mark.parametrize("tf", [Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H4])
def test_aggregator_matches_batch_resampler(m1: pd.DataFrame, tf: Timeframe) -> None:
    week = m1[(m1["ts"] >= "2023-01-08") & (m1["ts"] < "2023-01-14")].reset_index(drop=True)
    agg = BarAggregator(tf, CAL)
    live = [b for row in week.to_dict(orient="records") for b in agg.add(row)]
    batch = resample_bars(week, tf, CAL)
    assert len(live) == len(batch)
    cols = [c for c in batch.columns if c.startswith(("bid_", "ask_"))]
    np.testing.assert_allclose(pd.DataFrame(live)[cols].to_numpy(), batch[cols].to_numpy())
    assert [pd.Timestamp(b["ts"]) for b in live] == list(batch["ts"])


def test_aggregator_flushes_on_timer() -> None:
    agg = BarAggregator(Timeframe.M15, CAL, grace_seconds=5)
    bar = {
        "ts": pd.Timestamp("2024-03-05 21:44", tz="UTC"),
        **{f"{s}_{f}": 1.0 for s in ("bid", "ask") for f in ("open", "high", "low", "close")},
        "volume": 1.0,
        "tick_count": 1,
    }
    assert agg.add({**bar, "ts": pd.Timestamp("2024-03-05 21:40", tz="UTC")}) == []
    assert agg.flush(pd.Timestamp("2024-03-05 21:45:03", tz="UTC")) == []
    assert len(agg.flush(pd.Timestamp("2024-03-05 21:45:06", tz="UTC"))) == 1


# --------------------------------------------------------------------------- parity
async def _replay(m1: pd.DataFrame, genome, window: int) -> list[SignalEvent]:  # type: ignore[no-untyped-def]
    feed = ReplayFeed(m1, LIVE_START)
    hist = resample_bars(await feed.backfill(10**9), Timeframe.M15, CAL)
    runner = StrategyRunner(
        LiveStrategy("s", "t", genome, Timeframe.M15),
        hist,
        calendar=CAL,
        sessions=SESSIONS,
        costs=COSTS,
        window=window,
    )
    agg = BarAggregator(Timeframe.M15, CAL)
    events: list[SignalEvent] = []
    async for b in feed.bars():
        for tb in agg.add(b):
            events += runner.on_bar(tb)
    return events


@pytest.mark.parametrize("name", ["ema_pullback", "sweep_reversal"])
def test_live_signals_match_backtest_exactly(m1: pd.DataFrame, name: str) -> None:
    g = all_templates()[name]
    m15 = resample_bars(m1, Timeframe.M15, CAL)
    bt = run_backtest(
        g,
        m15,
        Timeframe.M15,
        calendar=CAL,
        sessions=SESSIONS,
        costs=COSTS,
        start=LIVE_START,
        with_context=False,
    ).trades
    bt = bt[bt["entry_ts"] > LIVE_START]
    events = asyncio.run(_replay(m1, g, window=3000))
    fills = {e.entry_ts: e for e in events if e.type == "fill" and e.entry_ts > LIVE_START}
    exits = {e.entry_ts: e for e in events if e.type == "exit" and e.entry_ts > LIVE_START}
    assert len(bt) >= 10
    assert sorted(fills) == sorted(bt["entry_ts"])
    for r in bt.itertuples():
        assert fills[r.entry_ts].price == pytest.approx(r.entry_price, abs=1e-9)
        assert fills[r.entry_ts].stop == pytest.approx(r.stop_price, abs=1e-9)
        if r.exit_reason != "end_of_data":
            assert exits[r.entry_ts].exit_price == pytest.approx(r.exit_price, abs=1e-9)
            assert exits[r.entry_ts].reason == r.exit_reason
            assert exits[r.entry_ts].pnl_r == pytest.approx(r.r, abs=1e-9)
    entries = [e for e in events if e.type == "entry"]
    assert len(entries) == len(fills) and not [e for e in events if e.type == "expire"]


def test_rationale_lists_met_conditions(m1: pd.DataFrame) -> None:
    g = all_templates()["sweep_reversal"]
    events = asyncio.run(_replay(m1, g, window=3000))
    entries = [e for e in events if e.type == "entry"]
    assert entries
    for e in entries:
        r = e.rationale
        holds = {"all": all, "any": any}
        assert holds[r["conditions_op"]](
            c["met"] for c in r["conditions"]
        )  # it fired, so the entry holds
        assert holds[r["filters_op"]](c["met"] for c in r["filters"])
        assert r["filters_op"] == "any"  # sweep_reversal: London OR New York
        assert e.text.startswith(e.side.upper()) and "Why:" in e.text and "stop" in e.text
        rsi = next(c for c in e.rationale["conditions"] if "RSI" in c["label"])
        assert rsi["value"] is not None and (
            rsi["value"] < 45 if e.side == "long" else rsi["value"] > 55
        )
        assert e.stop is not None and (
            (e.stop < e.price) if e.side == "long" else (e.stop > e.price)
        )


def test_runner_ignores_duplicate_bars(m1: pd.DataFrame) -> None:
    m15 = resample_bars(m1, Timeframe.M15, CAL)
    hist, live = m15.iloc[:-50], m15.iloc[-50:]
    runner = StrategyRunner(
        LiveStrategy("s", "t", all_templates()["ema_pullback"], Timeframe.M15),
        hist,
        calendar=CAL,
        sessions=SESSIONS,
        costs=COSTS,
        window=3000,
    )
    first = [e for row in live.to_dict(orient="records") for e in runner.on_bar(row)]
    again = [e for row in live.to_dict(orient="records") for e in runner.on_bar(row)]
    assert again == []  # replayed bars are ignored, no duplicate events
    assert len(runner.bars) == min(3000, len(m15))
    assert pd.Timestamp(runner.bars["ts"].iloc[-1]) == m15["ts"].iloc[-1]
    assert isinstance(first, list)


# --------------------------------------------------------------------------- OANDA protocol
def _oanda_transport(drop_first_stream: bool = False) -> httpx.MockTransport:
    state = {"streams": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["Authorization"] == "Bearer test-token"
        if req.url.path == "/v3/accounts":
            return httpx.Response(200, json={"accounts": [{"id": "101-004-1"}]})
        if req.url.path.endswith("/candles"):
            assert req.url.params["price"] == "BA" and req.url.params["granularity"] == "M1"
            c = {"o": "2000.10", "h": "2001.00", "l": "1999.90", "c": "2000.50"}
            a = {k: str(float(v) + 0.3) for k, v in c.items()}
            return httpx.Response(
                200,
                json={
                    "candles": [
                        {
                            "time": "2024-03-05T14:00:00.000000000Z",
                            "complete": True,
                            "volume": 42,
                            "bid": c,
                            "ask": a,
                        },
                        {
                            "time": "2024-03-05T14:01:00.000000000Z",
                            "complete": False,
                            "volume": 3,
                            "bid": c,
                            "ask": a,
                        },
                    ]
                },
            )
        if req.url.path.endswith("/pricing/stream"):
            state["streams"] += 1
            if drop_first_stream and state["streams"] == 1:
                return httpx.Response(503)
            lines = []
            for i, (sec, bid) in enumerate([(5, 2000.0), (30, 2001.0), (65, 2000.5), (70, 2000.6)]):
                ts = pd.Timestamp("2024-03-05 14:02", tz="UTC") + pd.Timedelta(seconds=sec)
                lines.append(
                    {
                        "type": "PRICE",
                        "time": ts.isoformat(),
                        "tradeable": True,
                        "bids": [{"price": str(bid)}],
                        "asks": [{"price": str(bid + 0.3)}],
                    }
                )
                if i == 1:
                    lines.append({"type": "HEARTBEAT", "time": ts.isoformat()})
            lines.append({"type": "HEARTBEAT", "time": "2024-03-05T14:04:10Z"})
            return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_oanda_backfill_and_stream() -> None:
    feed = OandaFeed(token="test-token", transport=_oanda_transport(), max_reconnects=0)
    hist = await feed.backfill(500)
    assert len(hist) == 1  # incomplete candle dropped
    assert hist["bid_close"].iloc[0] == 2000.5 and hist["ask_close"].iloc[0] == pytest.approx(
        2000.8
    )
    bars = []
    async for b in feed.bars():
        bars.append(b)
    await feed.aclose()
    assert [pd.Timestamp(b["ts"]).strftime("%H:%M") for b in bars] == ["14:02", "14:03"]
    assert (bars[0]["bid_high"], bars[0]["bid_close"], bars[0]["tick_count"]) == (2001.0, 2001.0, 2)
    assert feed.account_id == "101-004-1"


async def test_oanda_reconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("engine.live.feeds.asyncio.sleep", no_sleep)
    feed = OandaFeed(
        token="test-token", transport=_oanda_transport(drop_first_stream=True), max_reconnects=1
    )
    bars = [b async for b in feed.bars()]
    await feed.aclose()
    assert feed.status["reconnects"] >= 1 and len(bars) == 2


# --------------------------------------------------------------------------- notifiers
async def test_notifiers() -> None:
    sent: list[dict] = []  # type: ignore[type-arg]

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append({"url": str(req.url), "body": json.loads(req.content)})
        return httpx.Response(200)

    t = httpx.MockTransport(handler)
    await WebhookNotifier("https://hooks.example/x", transport=t).send("LONG XAUUSD", {"id": "1"})
    await TelegramNotifier("TOKEN", "42", transport=t).send("LONG XAUUSD", {})
    assert sent[0]["body"]["content"] == "LONG XAUUSD" and sent[0]["body"]["signal"] == {"id": "1"}
    assert sent[1]["url"].endswith("/botTOKEN/sendMessage") and sent[1]["body"]["chat_id"] == "42"

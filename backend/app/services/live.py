"""Live signaling service: feed → bar aggregation → champion runners → signals.

Signals are persisted idempotently (unique per strategy/timeframe/bar/kind), streamed on the
SIGNALS channel, and pushed to notifiers for *active* champions only. Incubating champions
paper-trade silently. After every closed trade the champion roster is re-evaluated
(activation, ranking, demotion) and runners are added or removed accordingly.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.events import Channel, Event, EventBus
from app.models import Champion, ContextRule, Instrument, Signal, Strategy
from app.models.enums import ChampionStatus, DataSource, Side, SignalKind, SignalStatus, Timeframe
from app.services.champions import ChampionPolicy, evaluate_champions
from engine.backtest.runner import CostConfig
from engine.data.calendar import MarketCalendar
from engine.data.resample import resample_bars
from engine.data.store import ParquetStore
from engine.live.aggregator import BarAggregator
from engine.live.feeds import BarFeed
from engine.live.notify import Notifier
from engine.live.signal_engine import LiveStrategy, SignalEvent, StrategyRunner
from engine.strategy.genome import Genome

logger = logging.getLogger(__name__)
EXIT_STATUS = {
    "target": SignalStatus.TARGET_HIT,
    "stop": SignalStatus.STOP_HIT,
    "trailing_stop": SignalStatus.STOP_HIT,
}


def signal_dict(s: Signal) -> dict[str, Any]:
    return {
        "id": str(s.id),
        "champion_id": str(s.champion_id) if s.champion_id else None,
        "strategy_id": str(s.strategy_id),
        "timeframe": s.timeframe.value,
        "kind": s.kind.value,
        "side": s.side.value,
        "status": s.status.value,
        "bar_ts": s.bar_ts.isoformat(),
        "emitted_at": s.emitted_at.isoformat(),
        "entry_price": s.entry_price,
        "stop_price": s.stop_price,
        "target_price": s.target_price,
        "risk_reward": s.risk_reward,
        "confidence": s.confidence,
        "exit_price": s.exit_price,
        "pnl_r": s.pnl_r,
        "rationale": s.rationale,
        "rationale_text": s.rationale_text,
        "closed_at": s.closed_at.isoformat() if s.closed_at else None,
    }


class LiveService:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        store: ParquetStore,
        bus: EventBus,
        *,
        symbol: str = "XAUUSD",
        history_source: DataSource = DataSource.DUKASCOPY,
        notifiers: list[Notifier] | None = None,
        policy: ChampionPolicy | None = None,
        window: int = 20_000,
    ) -> None:
        self.sm, self.store, self.bus = sessionmaker, store, bus
        self.symbol = symbol
        self.history_source = history_source
        self.notifiers = notifiers or []
        self.policy = policy or ChampionPolicy()
        self.window = window
        self.runners: dict[str, StrategyRunner] = {}
        self.champion_of: dict[str, tuple[Any, ChampionStatus]] = {}
        self.aggs: dict[Timeframe, BarAggregator] = {}
        self.dirty = True  # roster must be (re)loaded
        self.status: dict[str, Any] = {
            "running": False,
            "feed": None,
            "last_bar": None,
            "signals": 0,
            "champions": 0,
            "bars": 0,
        }
        self._m1_history = pd.DataFrame()

    # ------------------------------------------------------------------ roster
    async def _load_roster(self) -> None:
        async with self.sm() as s:
            inst = await s.scalar(select(Instrument).where(Instrument.symbol == self.symbol))
            assert inst is not None
            self.cal = MarketCalendar.from_instrument_config(
                inst.session_timezone, inst.trading_hours
            )
            self.sessions, self.costs, self.instrument_id = (
                inst.sessions,
                CostConfig.from_instrument(inst),
                inst.id,
            )
            rows = (
                await s.execute(
                    select(Champion, Strategy)
                    .join(Strategy, Champion.strategy_id == Strategy.id)
                    .where(
                        Champion.status.in_((ChampionStatus.INCUBATING, ChampionStatus.ACTIVE)),
                        Strategy.instrument_id == inst.id,
                    )
                )
            ).all()
            rules = {
                r.id: r.description
                for r in await s.scalars(
                    select(ContextRule).where(
                        ContextRule.id.in_(
                            [c.context_rule_id for c, _ in rows if c.context_rule_id]
                        )
                    )
                )
            }
        wanted = {str(st.id) for _, st in rows}
        for key in list(self.runners):
            if key not in wanted:
                del self.runners[key]
        for champ, st in rows:
            key = str(st.id)
            snap = champ.promotion_snapshot or {}
            meta = {
                "champion_id": str(champ.id),
                "status": champ.status.value,
                "rank": champ.rank,
                "context_rule": rules.get(champ.context_rule_id) if champ.context_rule_id else None,
                "stats": snap,
            }
            self.champion_of[key] = (champ.id, champ.status)
            if key in self.runners:
                self.runners[key].s.meta = meta
                continue
            tf = st.timeframe
            self.aggs.setdefault(tf, BarAggregator(tf, self.cal))
            self.runners[key] = StrategyRunner(
                LiveStrategy(key, st.name, Genome.model_validate(st.genome), tf, meta),
                self._history(tf),
                calendar=self.cal,
                sessions=self.sessions,
                costs=self.costs,
                symbol=self.symbol,
                window=self.window,
            )
        self.status["champions"] = len(self.runners)
        self.dirty = False

    def _history(self, tf: Timeframe) -> pd.DataFrame:
        recent = (
            resample_bars(self._m1_history, tf, self.cal)
            if len(self._m1_history)
            else pd.DataFrame()
        )
        end = recent["ts"].iloc[0] if len(recent) else None
        stored = self.store.read_bars(
            self.symbol, tf, source=self.history_source, end=end, limit=self.window, tail=True
        )
        frames = [f for f in (stored, recent) if len(f)]
        if not frames:
            return stored
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("ts", keep="last")
            .sort_values("ts")
        )

    # ------------------------------------------------------------------ persistence
    async def _handle(self, ev: SignalEvent, runner: StrategyRunner) -> None:
        champ_id, cstatus = self.champion_of.get(ev.strategy_key, (None, ChampionStatus.INCUBATING))
        tf = runner.s.timeframe
        strategy_id = uuid.UUID(runner.s.key)
        now = datetime.now(UTC)
        async with self.sm() as s:
            active_entry = await s.scalar(
                select(Signal)
                .where(
                    Signal.strategy_id == strategy_id,
                    Signal.kind == SignalKind.ENTRY,
                    Signal.status.in_((SignalStatus.PENDING, SignalStatus.ACTIVE)),
                )
                .order_by(Signal.bar_ts.desc())
            )
            sig: Signal | None = None
            if ev.type == "entry":
                exists = await s.scalar(
                    select(Signal).where(
                        Signal.strategy_id == strategy_id,
                        Signal.timeframe == tf,
                        Signal.bar_ts == ev.bar_ts.to_pydatetime(),
                        Signal.kind == SignalKind.ENTRY,
                    )
                )
                if exists is not None:
                    return  # idempotent: replayed bar after a restart
                rr = None
                if (
                    ev.target is not None
                    and ev.stop is not None
                    and ev.price is not None
                    and ev.price != ev.stop
                ):
                    rr = abs(ev.target - ev.price) / abs(ev.price - ev.stop)
                sig = Signal(
                    champion_id=champ_id,
                    strategy_id=strategy_id,
                    instrument_id=self.instrument_id,
                    timeframe=tf,
                    kind=SignalKind.ENTRY,
                    side=Side(ev.side),
                    status=SignalStatus.PENDING,
                    bar_ts=ev.bar_ts.to_pydatetime(),
                    emitted_at=now,
                    entry_price=ev.price,
                    stop_price=ev.stop,
                    target_price=ev.target,
                    risk_reward=rr,
                    confidence=(runner.s.meta.get("stats") or {}).get("oos_win_rate"),
                    rationale={**ev.rationale, "champion_status": cstatus.value},
                    rationale_text=ev.text,
                )
                s.add(sig)
            elif ev.type == "fill":
                sig = active_entry
                if sig is None:
                    # Already recorded and closed (bars replayed after a restart)? Nothing to do.
                    recent = await s.scalars(
                        select(Signal)
                        .where(
                            Signal.strategy_id == strategy_id,
                            Signal.kind == SignalKind.ENTRY,
                        )
                        .order_by(Signal.bar_ts.desc())
                        .limit(50)
                    )
                    iso = ev.entry_ts.isoformat() if ev.entry_ts is not None else None
                    if any((r.rationale or {}).get("entry_ts") == iso for r in recent):
                        return
                if sig is None:  # restarted between signal and fill: record the position anyway
                    sig = Signal(
                        champion_id=champ_id,
                        strategy_id=strategy_id,
                        instrument_id=self.instrument_id,
                        timeframe=tf,
                        kind=SignalKind.ENTRY,
                        side=Side(ev.side),
                        bar_ts=ev.bar_ts.to_pydatetime(),
                        emitted_at=now,
                        rationale={},
                        rationale_text="",
                    )
                    s.add(sig)
                sig.status, sig.entry_price, sig.stop_price, sig.target_price = (
                    SignalStatus.ACTIVE,
                    ev.price,
                    ev.stop,
                    ev.target,
                )
                sig.rationale = {
                    **(sig.rationale or {}),
                    "entry_ts": ev.entry_ts.isoformat() if ev.entry_ts else None,
                }
            elif (
                ev.type == "expire"
                and active_entry is not None
                and active_entry.status is SignalStatus.PENDING
            ):
                sig = active_entry
                sig.status, sig.closed_at = SignalStatus.EXPIRED, now
            elif ev.type in ("exit_pending", "exit") and active_entry is not None:
                pending_exit = await s.scalar(
                    select(Signal).where(
                        Signal.parent_signal_id == active_entry.id, Signal.kind == SignalKind.EXIT
                    )
                )
                sig = pending_exit or Signal(
                    champion_id=champ_id,
                    strategy_id=strategy_id,
                    instrument_id=self.instrument_id,
                    timeframe=tf,
                    kind=SignalKind.EXIT,
                    side=active_entry.side,
                    status=SignalStatus.PENDING,
                    bar_ts=ev.bar_ts.to_pydatetime(),
                    emitted_at=now,
                    parent_signal_id=active_entry.id,
                    rationale={"reason": ev.reason},
                    rationale_text=f"EXIT {active_entry.side.value} ({ev.reason})",
                )
                if pending_exit is None:
                    s.add(sig)
                if ev.type == "exit":
                    sig.status, sig.exit_price, sig.pnl_r, sig.closed_at = (
                        SignalStatus.CLOSED,
                        ev.exit_price,
                        ev.pnl_r,
                        now,
                    )
                    sig.rationale = {**(sig.rationale or {}), "reason": ev.reason}
                    sig.rationale_text = (
                        f"EXIT {active_entry.side.value} @ {ev.exit_price:.2f} "
                        f"({ev.reason}, {ev.pnl_r:+.2f}R)"
                    )
                    active_entry.status = EXIT_STATUS.get(ev.reason or "", SignalStatus.CLOSED)
                    active_entry.exit_price, active_entry.pnl_r, active_entry.closed_at = (
                        ev.exit_price,
                        ev.pnl_r,
                        now,
                    )
            if sig is None:
                return
            await s.flush()
            payload = signal_dict(sig)
            changes = []
            if ev.type == "exit":
                changes = await evaluate_champions(s, self.policy)
            await s.commit()
        self.status["signals"] += 1
        await self.bus.publish(Event(channel=Channel.SIGNALS, type=ev.type, payload=payload))
        if changes:
            self.dirty = True
            await self.bus.publish(
                Event(
                    channel=Channel.SIGNALS, type="champions_changed", payload={"changes": changes}
                )
            )
        if (
            cstatus is ChampionStatus.ACTIVE
            and ev.type in ("entry", "exit", "exit_pending")
            and self.notifiers
        ):
            text = payload["rationale_text"] or f"{ev.type} {ev.side}"
            await asyncio.gather(
                *(n.send(text, payload) for n in self.notifiers), return_exceptions=True
            )

    # ------------------------------------------------------------------ main loop
    async def run(
        self, feed: BarFeed, stop: threading.Event | None = None, backfill: int = 5000
    ) -> None:
        stop = stop or threading.Event()
        self.status.update(running=True, feed=feed.name)
        try:
            self._m1_history = await feed.backfill(backfill)
            await self._load_roster()
            await self.bus.publish(
                Event(
                    channel=Channel.SIGNALS,
                    type="live_started",
                    payload={"feed": feed.name, "champions": len(self.runners)},
                )
            )
            async for m1 in feed.bars():
                if stop.is_set():
                    break
                if self.dirty:
                    await self._load_roster()
                for tf, agg in list(self.aggs.items()):
                    for tb in agg.flush(pd.Timestamp(m1["ts"])) + agg.add(m1):
                        self.status["bars"] += 1
                        self.status["last_bar"] = pd.Timestamp(tb["ts"]).isoformat()
                        for runner in [r for r in self.runners.values() if r.s.timeframe is tf]:
                            for ev in runner.on_bar(tb):
                                await self._handle(ev, runner)
        finally:
            self.status["running"] = False
            await self.bus.publish(
                Event(channel=Channel.SIGNALS, type="live_stopped", payload=dict(self.status))
            )

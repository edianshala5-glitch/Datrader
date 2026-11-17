"""Live signal engine with backtest parity by construction.

On every completed bar, a :class:`StrategyRunner` re-runs the *same* compiler and numba
simulator the backtester uses over a rolling window (history + live bars), then diffs the
result with what it reported before:

* a trade whose entry is new → ``fill`` (the pending entry filled at this bar's open)
* a trade whose exit is new → ``exit`` (stop / target / trailing / signal / time / session)
* flat at this close and entry conditions true → ``entry`` (fills at the next open)
* open position whose exit signal / time limit fires at this close → ``exit_pending``

The simulator closes any open position at the last bar ("end of data"). Live, that's an
artefact, so such trades are treated as still open.

Session flats use only the calendar (scheduled closes). The batch backtester also treats
unexpected >1h data gaps as closes, which a live engine can't know in advance. The two
agree everywhere except around unscheduled outages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.models.enums import Timeframe
from engine.backtest.runner import CostConfig, execution_prices, simulate_signals
from engine.backtest.simulator import EXIT_REASONS, TRADE_FIELDS
from engine.data.calendar import MarketCalendar
from engine.data.schema import BAR_COLUMNS, normalize_bars
from engine.live.rationale import build_rationale, rationale_text
from engine.strategy.compiler import FeatureProvider, compile_genome
from engine.strategy.genome import Genome


@dataclass
class _Trade:
    signal_idx: float
    entry_idx: float
    exit_idx: float
    side: float
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    lots: float
    pnl: float
    r: float
    mae_r: float
    mfe_r: float
    reason: float
    commission: float


@dataclass
class LiveStrategy:
    key: str  # strategy id
    name: str
    genome: Genome
    timeframe: Timeframe
    meta: dict[str, Any] = field(
        default_factory=dict
    )  # champion id/status/rank, stats, context rule


@dataclass
class SignalEvent:
    strategy_key: str
    type: str  # entry | fill | expire | exit_pending | exit
    side: str
    bar_ts: pd.Timestamp  # open time of the bar whose close triggered the event
    decided_at: pd.Timestamp  # close time of that bar
    entry_ts: pd.Timestamp | None = None  # identity of the trade (open of its entry bar)
    price: float | None = None
    stop: float | None = None
    target: float | None = None
    exit_price: float | None = None
    reason: str | None = None
    pnl_r: float | None = None
    rationale: dict[str, Any] = field(default_factory=dict)
    text: str = ""


def calendar_flats(
    ts: pd.DatetimeIndex, tf: Timeframe, cal: MarketCalendar, mode: str
) -> np.ndarray:
    if mode == "none":
        return np.zeros(len(ts), dtype=bool)
    end = ts + pd.Timedelta(seconds=tf.seconds)
    closed = ~cal.is_open(end)
    if mode == "weekend":  # only closes longer than the daily break
        nxt = end + pd.Timedelta(hours=2)
        return np.asarray(closed & ~cal.is_open(nxt))
    return np.asarray(closed)


class StrategyRunner:
    def __init__(
        self,
        strategy: LiveStrategy,
        history: pd.DataFrame,
        *,
        calendar: MarketCalendar,
        sessions: dict[str, Any] | None,
        costs: CostConfig,
        symbol: str = "XAUUSD",
        window: int = 20_000,
        live_since: pd.Timestamp | None = None,
    ) -> None:
        self.s = strategy
        self.cal = calendar
        self.sessions = sessions
        self.costs = costs
        self.symbol = symbol
        self.window = window
        self.bars = normalize_bars(history).iloc[-window:].reset_index(drop=True)
        # Events are only reported for trades entered at/after this time (no history replay).
        if live_since is not None:
            self.live_since: pd.Timestamp = pd.Timestamp(live_since)
        elif len(self.bars):
            self.live_since = pd.Timestamp(self.bars["ts"].iloc[-1]) + pd.Timedelta(
                seconds=strategy.timeframe.seconds
            )
        else:
            self.live_since = pd.Timestamp("1970-01-01", tz="UTC")
        self.filled: set[pd.Timestamp] = set()
        self.exited: set[pd.Timestamp] = set()
        self.pending: dict[str, Any] | None = None  # entry announced, waiting for the next open
        self.exit_announced: set[pd.Timestamp] = set()

    def on_bar(self, bar: dict[str, Any]) -> list[SignalEvent]:
        row = normalize_bars(pd.DataFrame([bar], columns=BAR_COLUMNS))
        if len(self.bars) and row["ts"].iloc[0] <= self.bars["ts"].iloc[-1]:
            return []  # duplicate / out-of-order bar (e.g. after a reconnect)
        self.bars = (
            pd.concat([self.bars, row], ignore_index=True)
            .iloc[-self.window :]
            .reset_index(drop=True)
        )
        return self._evaluate()

    def _evaluate(self) -> list[SignalEvent]:
        tf, g = self.s.timeframe, self.s.genome
        bars = self.bars
        n = len(bars)
        if n < 3:
            return []
        fp = FeatureProvider(bars, tf, self.cal, self.sessions)
        sig = compile_genome(g, fp)
        sig.flat_at = calendar_flats(pd.DatetimeIndex(bars["ts"]), tf, self.cal, g.exit.flat_at)
        raw, _eq, _pos = simulate_signals(
            execution_prices(bars, self.costs.min_spread),
            sig,
            self.costs,
            g.risk.risk_per_trade,
            g.risk.max_lots,
        )
        trades = pd.DataFrame(raw, columns=list(TRADE_FIELDS))
        ts = pd.DatetimeIndex(bars["ts"])
        t = n - 1
        bar_ts, decided = ts[t], ts[t] + pd.Timedelta(seconds=tf.seconds)
        events: list[SignalEvent] = []
        open_trade: _Trade | None = None

        records: list[dict[str, Any]] = trades.to_dict(orient="records")  # type: ignore[assignment]
        for rec in records:
            tr = _Trade(**rec)
            e_ts = ts[int(tr.entry_idx)]
            if e_ts < self.live_since:
                continue
            side = "long" if tr.side == 1 else "short"
            reason = EXIT_REASONS[int(tr.reason)]
            is_open = reason == "end_of_data" and int(tr.exit_idx) == t
            if e_ts not in self.filled:
                self.filled.add(e_ts)
                self.pending = None
                events.append(
                    SignalEvent(
                        self.s.key,
                        "fill",
                        side,
                        bar_ts,
                        decided,
                        e_ts,
                        price=float(tr.entry_price),
                        stop=float(tr.stop_price),
                        target=None if np.isnan(tr.target_price) else float(tr.target_price),
                    )
                )
            if is_open:
                open_trade = tr
            elif e_ts not in self.exited:
                self.exited.add(e_ts)
                events.append(
                    SignalEvent(
                        self.s.key,
                        "exit",
                        side,
                        ts[int(tr.exit_idx)],
                        decided,
                        e_ts,
                        exit_price=float(tr.exit_price),
                        reason=reason,
                        pnl_r=float(tr.r),
                    )
                )

        if self.pending is not None and not any(e.type == "fill" for e in events):
            p = self.pending
            self.pending = None
            events.append(
                SignalEvent(
                    self.s.key, "expire", p["side"], p["bar_ts"], decided, reason="not filled"
                )
            )

        if open_trade is not None:
            e_ts = ts[int(open_trade.entry_idx)]
            long = open_trade.side == 1
            held = t - int(open_trade.entry_idx) + 1
            exit_now = (sig.exit_long[t] if long else sig.exit_short[t]) or (
                sig.max_bars and held >= sig.max_bars
            )
            if exit_now and not sig.flat_at[t] and e_ts not in self.exit_announced:
                self.exit_announced.add(e_ts)
                events.append(
                    SignalEvent(
                        self.s.key,
                        "exit_pending",
                        "long" if long else "short",
                        bar_ts,
                        decided,
                        e_ts,
                        reason="signal"
                        if (sig.exit_long[t] if long else sig.exit_short[t])
                        else "time",
                    )
                )
            return events

        # Flat at this close: would the backtester enter at the next open?
        nxt_flat = calendar_flats(pd.DatetimeIndex([decided]), tf, self.cal, g.exit.flat_at)[
            0
        ] or not bool(self.cal.is_open(pd.DatetimeIndex([decided]))[0])
        go_long, go_short = bool(sig.entry_long[t]), bool(sig.entry_short[t])
        if (
            go_long != go_short
            and not sig.flat_at[t]
            and not nxt_flat
            and decided >= self.live_since
        ):
            side = "long" if go_long else "short"
            ref = float(bars["ask_close"].iloc[t] if go_long else bars["bid_close"].iloc[t])
            ref = (
                max(ref, float(bars["bid_close"].iloc[t]) + self.costs.min_spread)
                if go_long
                else ref
            )
            lvl = sig.stop_level_long[t] if go_long else sig.stop_level_short[t]
            dist = sig.stop_dist_long[t] if go_long else sig.stop_dist_short[t]
            sgn = 1 if go_long else -1
            stop = float(lvl) if not np.isnan(lvl) else ref - sgn * float(dist)
            risk = (ref - stop) * sgn
            tdist = sig.target_dist_long[t] if go_long else sig.target_dist_short[t]
            target = (
                ref + sgn * sig.target_rr * risk
                if not np.isnan(sig.target_rr)
                else (ref + sgn * float(tdist) if not np.isnan(tdist) else None)
            )
            rat = build_rationale(g, side, fp, t, {**self.s.meta, "name": self.s.name})
            text = rationale_text(self.symbol, tf.value, side, ref, stop, target, rat)
            self.pending = {"side": side, "bar_ts": bar_ts}
            events.append(
                SignalEvent(
                    self.s.key,
                    "entry",
                    side,
                    bar_ts,
                    decided,
                    price=ref,
                    stop=stop,
                    target=target,
                    rationale=rat,
                    text=text,
                )
            )
        return events

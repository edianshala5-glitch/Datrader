"""High-level backtest: genome + bars → trades, equity curve, metrics, entry context."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.models.enums import Timeframe
from engine.backtest.metrics import compute_metrics
from engine.backtest.simulator import EXIT_REASONS, TRADE_FIELDS, simulate
from engine.data.calendar import MarketCalendar
from engine.indicators import compute
from engine.indicators.kernels import bars_since
from engine.strategy.compiler import FeatureProvider, Signals, compile_genome
from engine.strategy.genome import Genome

ENGINE_VERSION = "4.0"


@dataclass(frozen=True)
class CostConfig:
    """Execution assumptions. ``from_instrument`` reads the instrument's cost model."""

    initial_equity: float = 100_000.0
    contract_size: float = 100.0  # XAUUSD: 100 oz per lot
    tick_size: float = 0.01
    commission_per_lot_rt: float = 7.0
    slippage_ticks: float = 2.0
    min_spread: float = 0.20  # floor applied to bid/ask data
    lot_step: float = 0.01
    min_lot: float = 0.01

    @property
    def slippage(self) -> float:
        return self.slippage_ticks * self.tick_size

    @classmethod
    def from_instrument(cls, instrument: Any, **overrides: Any) -> CostConfig:
        cm = instrument.cost_model
        cm = cm if isinstance(cm, dict) else cm.model_dump()
        return cls(
            contract_size=float(instrument.contract_size),
            tick_size=float(instrument.tick_size),
            commission_per_lot_rt=cm["commission_per_lot_round_turn"],
            slippage_ticks=cm["slippage_ticks"],
            min_spread=cm["spread"],
            **overrides,
        )

    def to_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class BacktestRun:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict[str, Any]
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    costs: CostConfig
    signals: dict[str, int] = field(default_factory=dict)
    engine_version: str = ENGINE_VERSION


def _context_frame(fp: FeatureProvider) -> dict[str, np.ndarray]:
    """Features snapshotted at each trade's decision bar: Meta-Analyzer input (Step 7)."""
    inp = fp.inputs
    out: dict[str, np.ndarray] = {}
    if inp.calendar is not None:
        t = compute(inp, "time")
        out.update(
            weekday=t["weekday"],
            hour_ny=t["hour_ny"],
            minute_ny=t["minute_ny"],
            us_holiday=t["us_holiday"],
        )
        if inp.sessions:
            for k, v in compute(inp, "sessions").items():
                out[f"session_{k}"] = v
    reg = compute(inp, "regime")
    out.update(
        vol_regime=reg["vol_regime"], vol_rank=reg["vol_rank"], trend_regime=reg["trend_regime"]
    )
    st = compute(inp, "structure")
    out["structure_trend"] = st["trend"]
    for ev in ("bos_bull", "bos_bear", "choch_bull", "choch_bear", "sweep_high", "sweep_low"):
        out[f"bars_since_{ev}"] = bars_since(st[ev])
    dv = compute(inp, "divergence")
    out["bars_since_div_bull"] = bars_since(dv["regular_bull"] | dv["hidden_bull"])
    out["bars_since_div_bear"] = bars_since(dv["regular_bear"] | dv["hidden_bear"])
    out["rsi"] = compute(inp, "rsi")["value"]
    out["atr_pct"] = compute(inp, "atr")["pct"]
    out["spread"] = inp.spread
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else round(float(v), 6)
    return v


def execution_prices(bars: pd.DataFrame, min_spread: float) -> dict[str, np.ndarray]:
    """Bid OHLC and ask OHLC with the spread floor applied (never assume a tighter market
    than the configured minimum)."""
    out = {
        f"bid_{f}": bars[f"bid_{f}"].to_numpy(np.float64) for f in ("open", "high", "low", "close")
    }
    for f in ("open", "high", "low", "close"):
        out[f"ask_{f}"] = np.maximum(
            bars[f"ask_{f}"].to_numpy(np.float64), out[f"bid_{f}"] + min_spread
        )
    return out


def simulate_signals(
    px: dict[str, np.ndarray],
    sig: Signals,
    costs: CostConfig,
    risk_frac: float,
    max_lots: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bid = {f: px[f"bid_{f}"] for f in ("open", "high", "low", "close")}
    ask = {f: px[f"ask_{f}"] for f in ("open", "high", "low", "close")}
    out: tuple[np.ndarray, np.ndarray, np.ndarray] = simulate(
        bid["open"],
        bid["high"],
        bid["low"],
        bid["close"],
        ask["open"],
        ask["high"],
        ask["low"],
        ask["close"],
        sig.entry_long,
        sig.entry_short,
        sig.exit_long,
        sig.exit_short,
        sig.stop_level_long,
        sig.stop_level_short,
        sig.stop_dist_long,
        sig.stop_dist_short,
        sig.target_dist_long,
        sig.target_dist_short,
        sig.target_rr,
        sig.trail_dist,
        sig.trail_activate_r,
        sig.max_bars,
        sig.flat_at,
        costs.initial_equity,
        risk_frac,
        costs.contract_size,
        costs.lot_step,
        costs.min_lot,
        max_lots,
        costs.commission_per_lot_rt,
        costs.slippage,
    )
    return out


def run_backtest(
    genome: Genome,
    bars: pd.DataFrame,
    timeframe: Timeframe,
    *,
    calendar: MarketCalendar | None = None,
    sessions: dict[str, Any] | None = None,
    costs: CostConfig | None = None,
    htf_bars: dict[Timeframe, pd.DataFrame] | None = None,
    start: datetime | pd.Timestamp | None = None,
    end: datetime | pd.Timestamp | None = None,
    with_context: bool = True,
    provider: FeatureProvider | None = None,
) -> BacktestRun:
    """Backtest over ``[start, end)``.

    Indicators are computed on *all* supplied bars, so history before ``start`` acts as
    warm-up. Trading only happens inside the window, and any position is closed on the
    window's last bar. Pass a shared ``provider`` to reuse features across many genomes.
    """
    costs = costs or CostConfig()
    bars = bars.reset_index(drop=True)
    fp = provider or FeatureProvider(bars, timeframe, calendar, sessions, htf_bars)
    sig = compile_genome(genome, fp)

    ts = pd.DatetimeIndex(bars["ts"])
    lo = int(ts.searchsorted(pd.Timestamp(start))) if start is not None else 0
    hi = int(ts.searchsorted(pd.Timestamp(end))) if end is not None else len(bars)
    window = slice(lo, hi)
    wsig = sig.window(window)
    wbars = bars.iloc[window].reset_index(drop=True)
    wts = ts[window]
    if len(wbars) == 0:
        raise ValueError("no bars in the backtest window")

    px_full = fp.memo(
        ("exec_px", costs.min_spread), lambda: execution_prices(bars, costs.min_spread)
    )
    px = {k: v[window] for k, v in px_full.items()}
    raw, equity_arr, position = simulate_signals(
        px, wsig, costs, genome.risk.risk_per_trade, genome.risk.max_lots
    )
    trades = pd.DataFrame(raw, columns=list(TRADE_FIELDS))
    for c in ("signal_idx", "entry_idx", "exit_idx", "side", "reason"):
        trades[c] = trades[c].astype(np.int64)
    trades["side"] = np.where(trades["side"] == 1, "long", "short")
    trades["exit_reason"] = [EXIT_REASONS[i] for i in trades.pop("reason")]
    trades["entry_ts"] = wts[trades["entry_idx"].to_numpy()]
    trades["exit_ts"] = wts[trades["exit_idx"].to_numpy()]  # open time of the exit bar
    trades["bars_held"] = trades["exit_idx"] - trades["entry_idx"] + 1

    if with_context and len(trades):
        ctx = _context_frame(fp)
        idx = trades["signal_idx"].to_numpy() + lo  # decision bar in full-series coordinates
        trades["context"] = pd.Series(
            [{k: _jsonable(v[i]) for k, v in ctx.items()} for i in idx],
            index=trades.index,
            dtype=object,
        )

    equity = pd.Series(equity_arr, index=wts, name="equity")
    cal = calendar or fp.calendar
    day_full = fp.memo(
        "trading_day",
        lambda: cal.trading_day(ts) if cal is not None else ts.tz_convert("UTC").normalize(),
    )
    trading_day = day_full[window]
    metrics = compute_metrics(trades, equity, trading_day, costs.initial_equity, position)
    return BacktestRun(
        trades=trades,
        equity=equity,
        metrics=metrics,
        period_start=wts[0],
        period_end=wts[-1] + pd.Timedelta(seconds=timeframe.seconds),
        costs=costs,
        signals={"long": int(wsig.entry_long.sum()), "short": int(wsig.entry_short.sum())},
    )

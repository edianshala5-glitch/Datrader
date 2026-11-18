"""Compile a genome into per-bar signal arrays.

Timing convention (shared with the backtester and, later, the live engine): every array value
at bar ``t`` is computed from information available at the **close of bar t**. The
backtester acts on it at the **open of bar t+1**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.models.enums import Timeframe
from engine.data.calendar import MarketCalendar
from engine.indicators import Inputs, compute
from engine.indicators.kernels import bars_since, confirmed_swings
from engine.indicators.mtf import align_htf
from engine.indicators.smc import _ffill
from engine.indicators.standard import atr_values
from engine.memcache import BoundedCache, cache_budget_bytes
from engine.strategy.genome import (
    AtrStop,
    AtrTarget,
    Compare,
    ConstOperand,
    Cross,
    EventCond,
    FeatureOperand,
    FeatureSpec,
    FlagCond,
    Genome,
    Group,
    PercentStop,
    PriceOperand,
    RangeCond,
    RRTarget,
    StateCond,
    SwingStop,
    ZoneStop,
    ZoneTarget,
)

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


class FeatureProvider:
    """Feature arrays aligned to the strategy's bars, including leak-free higher timeframes."""

    def __init__(
        self,
        bars: pd.DataFrame,
        timeframe: Timeframe,
        calendar: MarketCalendar | None,
        sessions: dict[str, Any] | None = None,
        htf_bars: dict[Timeframe, pd.DataFrame] | None = None,
    ) -> None:
        self.bars = bars.reset_index(drop=True)
        self.timeframe = timeframe
        self.inputs = Inputs.from_bars(self.bars, timeframe, calendar, sessions)
        # Memory budget (see engine.memcache): base-timeframe indicators live in the bounded
        # Inputs cache; this one holds only higher-timeframe arrays aligned to our bars.
        budget = cache_budget_bytes()
        self.inputs._cache.max_bytes = budget * 2 // 3
        self._htf_bars = htf_bars or {}
        self._htf_inputs: dict[Timeframe, Inputs] = {}
        self._htf_budget = budget // 12
        self._cache = BoundedCache(budget // 6)
        self._calendar = calendar
        self._sessions = sessions

    def __len__(self) -> int:
        return len(self.bars)

    def get(self, spec: FeatureSpec) -> npt.NDArray[Any]:
        if spec.timeframe is None or spec.timeframe is self.timeframe:
            return compute(self.inputs, spec.indicator, spec.params)[spec.output]
        if spec in self._cache:
            return self._cache[spec]  # type: ignore[no-any-return]
        tf = spec.timeframe
        if tf not in self._htf_bars:
            raise KeyError(f"genome needs {tf} bars; pass them via htf_bars")
        if tf not in self._htf_inputs:
            self._htf_inputs[tf] = Inputs.from_bars(
                self._htf_bars[tf], tf, self._calendar, self._sessions
            )
            self._htf_inputs[tf]._cache.max_bytes = self._htf_budget
        htf = self._htf_inputs[tf]
        raw = compute(htf, spec.indicator, spec.params)[spec.output]
        aligned = align_htf(self.inputs.ts, self.timeframe, htf.ts, tf, pd.DataFrame({"v": raw}))[
            "v"
        ]
        values: npt.NDArray[Any]
        if raw.dtype == np.bool_:
            values = aligned.fillna(False).to_numpy(dtype=bool)
        else:
            values = aligned.to_numpy(dtype=np.float64)
        self._cache[spec] = values
        return values

    def price(self, field_name: str) -> FloatArray:
        arr: FloatArray = getattr(self.inputs, field_name)
        return arr

    def memo(self, key: Any, fn: Any) -> Any:
        """Cache for genome-independent arrays (flat masks, trading days, exec prices)."""
        return self.inputs.cached(("provider", key), fn)

    @property
    def calendar(self) -> MarketCalendar | None:
        return self._calendar


def _shift(x: FloatArray, k: int) -> FloatArray:
    if k == 0:
        return x
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    return out


def _operand(op: Any, fp: FeatureProvider) -> FloatArray:
    if isinstance(op, ConstOperand):
        return np.full(len(fp), op.value)
    if isinstance(op, PriceOperand):
        return _shift(fp.price(op.field), op.shift)
    assert isinstance(op, FeatureOperand)
    return _shift(np.asarray(fp.get(op.feature), dtype=np.float64), op.shift)


def evaluate(cond: Any, fp: FeatureProvider) -> BoolArray:
    """Vectorized truth value of a condition at every bar (NaN → False)."""
    with np.errstate(invalid="ignore"):
        if isinstance(cond, Group):
            parts = [evaluate(c, fp) for c in cond.conditions]
            reducer = np.logical_and if cond.op == "all" else np.logical_or
            out: BoolArray = np.asarray(reducer.reduce(parts), dtype=bool)
            return ~out if cond.negate else out
        if isinstance(cond, Compare):
            a, b = _operand(cond.left, fp), _operand(cond.right, fp)
            fn = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal}[
                cond.op
            ]
            return np.asarray(fn(a, b), dtype=bool)
        if isinstance(cond, Cross):
            a, b = _operand(cond.left, fp), _operand(cond.right, fp)
            pa, pb = _shift(a, 1), _shift(b, 1)
            if cond.direction == "above":
                return (a > b) & (pa <= pb)
            return (a < b) & (pa >= pb)
        if isinstance(cond, EventCond):
            ev = np.asarray(fp.get(cond.feature), dtype=bool)
            if cond.within == 0:
                return ev
            since = bars_since(ev)
            return (since >= 0) & (since <= cond.within)
        if isinstance(cond, FlagCond):
            flag = np.asarray(fp.get(cond.feature), dtype=bool)
            return flag if cond.value else ~flag
        if isinstance(cond, StateCond):
            return np.isin(np.asarray(fp.get(cond.feature)), cond.values)
        if isinstance(cond, RangeCond):
            v = np.asarray(fp.get(cond.feature), dtype=np.float64)
            return (v >= cond.lo) & (v <= cond.hi)
    raise TypeError(f"unknown condition {type(cond).__name__}")


@dataclass
class Signals:
    entry_long: BoolArray
    entry_short: BoolArray
    exit_long: BoolArray
    exit_short: BoolArray
    stop_level_long: FloatArray  # absolute stop price (swing stops), NaN if distance-based
    stop_level_short: FloatArray
    stop_dist_long: FloatArray  # stop distance from the fill (ATR / percent stops)
    stop_dist_short: FloatArray
    target_dist_long: FloatArray  # NaN = no distance target
    target_dist_short: FloatArray
    target_rr: float  # NaN = no R-multiple target
    trail_dist: FloatArray  # NaN = no trailing stop
    trail_activate_r: float
    max_bars: int  # 0 = unlimited
    flat_at: BoolArray  # bar t is the last bar before a forced flat
    meta: dict[str, Any] = field(default_factory=dict)

    def window(self, sl: slice) -> Signals:
        """The same signals restricted to bars ``sl`` (scalars unchanged)."""
        return Signals(
            entry_long=self.entry_long[sl],
            entry_short=self.entry_short[sl],
            exit_long=self.exit_long[sl],
            exit_short=self.exit_short[sl],
            stop_level_long=self.stop_level_long[sl],
            stop_level_short=self.stop_level_short[sl],
            stop_dist_long=self.stop_dist_long[sl],
            stop_dist_short=self.stop_dist_short[sl],
            target_dist_long=self.target_dist_long[sl],
            target_dist_short=self.target_dist_short[sl],
            target_rr=self.target_rr,
            trail_dist=self.trail_dist[sl],
            trail_activate_r=self.trail_activate_r,
            max_bars=self.max_bars,
            flat_at=self.flat_at[sl],
            meta=dict(self.meta),
        )


def flat_bars(
    ts: pd.DatetimeIndex, timeframe: Timeframe, calendar: MarketCalendar | None, mode: str
) -> BoolArray:
    """Bars after whose close a position must be flat.

    ``daily``: the market is closed when the bar ends (17:00 NY break, weekend), or the data
    shows a gap of over an hour to the next bar. The gap rule is a proxy for scheduled holiday
    early closes, which the weekly calendar doesn't know. ``weekend``: only gaps of 24h+.
    ``none``: never.
    """
    n = len(ts)
    if mode == "none" or n == 0:
        return np.zeros(n, dtype=bool)
    bar_end = ts + pd.Timedelta(seconds=timeframe.seconds)
    gap = np.zeros(n, dtype=bool)
    nxt = np.asarray(ts[1:] - bar_end[:-1], dtype="timedelta64[s]").astype(np.int64)
    if mode == "weekend":
        gap[:-1] = nxt >= 24 * 3600
        return gap
    gap[:-1] = nxt > 3600
    closed = ~calendar.is_open(bar_end) if calendar is not None else np.zeros(n, dtype=bool)
    return np.asarray(gap | closed)


def _zone(genome: Genome, fp: FeatureProvider, output: str) -> FloatArray:
    """A level of the zone the entry conditions use, aligned to the strategy's bars."""
    spec = genome.zone_feature()
    assert spec is not None  # guaranteed by Genome validation
    values: FloatArray = fp.get(spec.model_copy(update={"output": output}))
    return values


def compile_genome(genome: Genome, fp: FeatureProvider) -> Signals:
    n = len(fp)
    false = np.zeros(n, dtype=bool)
    filt = evaluate(genome.filter, fp) if genome.filter else ~false
    el = (evaluate(genome.entry_long, fp) & filt) if genome.entry_long else false
    es = (evaluate(genome.entry_short, fp) & filt) if genome.entry_short else false
    xl = evaluate(genome.exit.signal_long, fp) if genome.exit.signal_long else false
    xs = evaluate(genome.exit.signal_short, fp) if genome.exit.signal_short else false

    close = fp.inputs.close
    nan = np.full(n, np.nan)
    stop = genome.exit.stop
    lvl_l, lvl_s, dist_l, dist_s = nan, nan, nan, nan
    if isinstance(stop, AtrStop):
        dist_l = dist_s = stop.mult * atr_values(fp.inputs, stop.length)
    elif isinstance(stop, PercentStop):
        dist_l = dist_s = close * stop.pct / 100.0
    elif isinstance(stop, SwingStop):
        atr = atr_values(fp.inputs, 14)
        _, _, sh_p, sl_p, _, _ = confirmed_swings(
            fp.inputs.high, fp.inputs.low, stop.length, stop.length
        )
        lvl_l = _ffill(sl_p) - stop.buffer_atr * atr
        lvl_s = _ffill(sh_p) + stop.buffer_atr * atr
        with np.errstate(invalid="ignore"):
            too_wide_l = (close - lvl_l) > stop.max_atr * atr
            too_wide_s = (lvl_s - close) > stop.max_atr * atr
            lvl_l = np.where(too_wide_l | (lvl_l >= close), np.nan, lvl_l)
            lvl_s = np.where(too_wide_s | (lvl_s <= close), np.nan, lvl_s)
    elif isinstance(stop, ZoneStop):
        d_lo, d_hi = _zone(genome, fp, "demand_low"), _zone(genome, fp, "demand_high")
        s_lo, s_hi = _zone(genome, fp, "supply_low"), _zone(genome, fp, "supply_high")
        with np.errstate(invalid="ignore"):
            lvl_l = d_lo - stop.buffer_frac * (d_hi - d_lo)
            lvl_s = s_hi + stop.buffer_frac * (s_hi - s_lo)
            if stop.min_pct > 0:
                lvl_l = np.minimum(lvl_l, close * (1 - stop.min_pct / 100))
                lvl_s = np.maximum(lvl_s, close * (1 + stop.min_pct / 100))
            bad_l, bad_s = lvl_l >= close, lvl_s <= close
            if stop.max_pct is not None:
                bad_l |= (close - lvl_l) > close * stop.max_pct / 100
                bad_s |= (lvl_s - close) > close * stop.max_pct / 100
            lvl_l = np.where(bad_l, np.nan, lvl_l)
            lvl_s = np.where(bad_s, np.nan, lvl_s)

    # A trade needs a defined stop at decision time.
    valid_l = ~np.isnan(lvl_l) | ~np.isnan(dist_l)
    valid_s = ~np.isnan(lvl_s) | ~np.isnan(dist_s)

    tgt = genome.exit.target
    target_rr = tgt.ratio if isinstance(tgt, RRTarget) else np.nan
    t_dist = tgt.mult * atr_values(fp.inputs, tgt.length) if isinstance(tgt, AtrTarget) else nan
    t_dist_l = t_dist_s = t_dist
    if isinstance(tgt, ZoneTarget):
        # Distance from the decision close to the impulse extreme (fills at the next open).
        with np.errstate(invalid="ignore"):
            t_dist_l = _zone(genome, fp, "demand_target") - close
            t_dist_s = close - _zone(genome, fp, "supply_target")
            risk_l = np.where(np.isnan(lvl_l), dist_l, close - lvl_l)
            risk_s = np.where(np.isnan(lvl_s), dist_s, lvl_s - close)
            valid_l &= (t_dist_l > 0) & (t_dist_l >= tgt.min_rr * risk_l)
            valid_s &= (t_dist_s > 0) & (t_dist_s >= tgt.min_rr * risk_s)

    tr = genome.exit.trailing
    trail = tr.mult * atr_values(fp.inputs, tr.length) if tr else nan

    return Signals(
        entry_long=el & valid_l,
        entry_short=es & valid_s,
        exit_long=xl,
        exit_short=xs,
        stop_level_long=np.asarray(lvl_l, dtype=np.float64),
        stop_level_short=np.asarray(lvl_s, dtype=np.float64),
        stop_dist_long=np.asarray(dist_l, dtype=np.float64),
        stop_dist_short=np.asarray(dist_s, dtype=np.float64),
        target_dist_long=np.asarray(t_dist_l, dtype=np.float64),
        target_dist_short=np.asarray(t_dist_s, dtype=np.float64),
        target_rr=float(target_rr),
        trail_dist=np.asarray(trail, dtype=np.float64),
        trail_activate_r=tr.activate_r if tr else 0.0,
        max_bars=genome.exit.max_bars or 0,
        flat_at=fp.memo(
            ("flat", genome.exit.flat_at),
            lambda: flat_bars(fp.inputs.ts, fp.timeframe, fp.calendar, genome.exit.flat_at),
        ),
        meta={"raw_long": int(el.sum()), "raw_short": int(es.sum())},
    )

"""Render genomes as plain English for the UI, the leaderboard and live-signal rationale."""

from __future__ import annotations

from typing import Any

from engine.indicators import get_spec
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

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

_NICE = {
    ("sd_zones", "demand_ready"): "demand zone ready (BOS + slow pullback)",
    ("sd_zones", "supply_ready"): "supply zone ready (BOS + slow pullback)",
    ("sd_zones", "demand_high"): "demand zone top",
    ("sd_zones", "demand_low"): "demand zone bottom",
    ("sd_zones", "supply_high"): "supply zone top",
    ("sd_zones", "supply_low"): "supply zone bottom",
    ("sd_zones", "demand_target"): "demand impulse high",
    ("sd_zones", "supply_target"): "supply impulse low",
    ("candles", "bull_rejection"): "bullish rejection candle",
    ("candles", "bear_rejection"): "bearish rejection candle",
    ("candles", "bull_engulfing"): "bullish engulfing",
    ("candles", "bear_engulfing"): "bearish engulfing",
    ("candles", "bull_three"): "three rising bullish candles",
    ("candles", "bear_three"): "three falling bearish candles",
    ("candles", "bull_any"): "bullish confirmation candle",
    ("candles", "bear_any"): "bearish confirmation candle",
    ("structure", "bos_bull"): "bullish BOS",
    ("structure", "bos_bear"): "bearish BOS",
    ("structure", "choch_bull"): "bullish CHoCH",
    ("structure", "choch_bear"): "bearish CHoCH",
    ("structure", "sweep_high"): "liquidity sweep of a swing high",
    ("structure", "sweep_low"): "liquidity sweep of a swing low",
    ("structure", "eqh"): "equal highs",
    ("structure", "eql"): "equal lows",
    ("structure", "trend"): "market-structure trend",
    ("fvg", "new_bull"): "new bullish FVG",
    ("fvg", "new_bear"): "new bearish FVG",
    ("fvg", "touch_bull"): "retest of a bullish FVG",
    ("fvg", "touch_bear"): "retest of a bearish FVG",
    ("order_blocks", "touch_bull"): "retest of a bullish order block",
    ("order_blocks", "touch_bear"): "retest of a bearish order block",
    ("divergence", "regular_bull"): "regular bullish divergence",
    ("divergence", "regular_bear"): "regular bearish divergence",
    ("divergence", "hidden_bull"): "hidden bullish divergence",
    ("divergence", "hidden_bear"): "hidden bearish divergence",
    ("time", "weekday"): "trading day",
    ("time", "hour_ny"): "NY hour",
    ("time", "minute_ny"): "NY time",
    ("time", "us_holiday"): "US holiday session",
    ("regime", "vol_regime"): "volatility regime",
    ("regime", "trend_regime"): "trend regime",
}
_SESSIONS = {
    "asia": "Asian session",
    "london": "London session",
    "new_york": "New York session",
    "london_ny_overlap": "London/NY overlap",
    "comex": "COMEX hours",
    "asia_kz": "Asian kill zone",
    "london_kz": "London kill zone",
    "ny_am_kz": "NY AM kill zone",
    "ny_pm_kz": "NY PM kill zone",
}


def feature_label(f: FeatureSpec) -> str:
    if f.indicator == "sessions":
        base = _SESSIONS.get(f.output, f.output)
    else:
        nice = _NICE.get((f.indicator, f.output))
        if nice is not None:
            base = nice
        else:
            spec = get_spec(f.indicator)
            resolved = spec.resolve(f.params)
            args = ",".join(str(v) for v in resolved.values())
            name = f"{f.indicator.upper()}({args})" if args else f.indicator.upper()
            single = len(spec.outputs) == 1 or f.output == f.indicator
            base = name if single else f"{name} {f.output.replace('_', ' ')}"
    non_default = {
        k: v for k, v in f.params.items() if get_spec(f.indicator).params[k].default != v
    }
    if (f.indicator, f.output) in _NICE and non_default:
        base += " (" + ", ".join(f"{k}={v}" for k, v in non_default.items()) + ")"
    return f"{f.timeframe.value} {base}" if f.timeframe else base


def _operand(op: Any) -> str:
    if isinstance(op, ConstOperand):
        return f"{op.value:g}"
    s: str
    if isinstance(op, PriceOperand):
        s = op.field
    else:
        assert isinstance(op, FeatureOperand)
        s = feature_label(op.feature)
    return f"{s} {op.shift} bars ago" if op.shift else s


def _minutes(v: float) -> str:
    m = int(v)
    return f"{m // 60:02d}:{m % 60:02d}"


def condition_text(c: Any) -> str:
    if isinstance(c, Group):
        joiner = " AND " if c.op == "all" else " OR "
        inner = joiner.join(condition_text(x) for x in c.conditions)
        if len(c.conditions) > 1:
            inner = f"({inner})"
        return f"NOT {inner}" if c.negate else inner
    if isinstance(c, Compare):
        return f"{_operand(c.left)} {c.op} {_operand(c.right)}"
    if isinstance(c, Cross):
        return f"{_operand(c.left)} crosses {c.direction} {_operand(c.right)}"
    if isinstance(c, EventCond):
        label = feature_label(c.feature)
        return label if c.within == 0 else f"{label} within the last {c.within} bars"
    if isinstance(c, FlagCond):
        label = feature_label(c.feature)
        return (
            f"in {label}"
            if c.value and c.feature.indicator == "sessions"
            else (label if c.value else f"not {label}")
        )
    if isinstance(c, StateCond):
        label = feature_label(c.feature)
        if (c.feature.indicator, c.feature.output) == ("time", "weekday"):
            return f"{label} is " + "/".join(WEEKDAYS[v] for v in c.values)
        if c.feature.output in ("trend", "trend_regime"):
            names = {-1: "bearish", 0: "neutral", 1: "bullish"}
            return f"{label} is " + "/".join(names[v] for v in c.values)
        return f"{label} in {c.values}"
    if isinstance(c, RangeCond):
        label = feature_label(c.feature)
        if c.feature.output == "minute_ny":
            return f"NY time between {_minutes(c.lo)} and {_minutes(c.hi)}"
        return f"{c.lo:g} <= {label} <= {c.hi:g}"
    raise TypeError(type(c).__name__)


def explain(genome: Genome) -> str:
    lines = []
    if genome.entry_long:
        lines.append("LONG when " + condition_text(genome.entry_long))
    if genome.entry_short:
        lines.append("SHORT when " + condition_text(genome.entry_short))
    if genome.filter:
        lines.append("Only if " + condition_text(genome.filter))
    ex = genome.exit
    stop = ex.stop
    if isinstance(stop, AtrStop):
        s = f"stop {stop.mult:g}x ATR({stop.length})"
    elif isinstance(stop, SwingStop):
        s = f"stop beyond the last swing (length {stop.length}) + {stop.buffer_atr:g} ATR"
    elif isinstance(stop, ZoneStop):
        s = f"stop beyond the zone + {stop.buffer_frac:.0%} of its height"
        if stop.min_pct:
            s += f", at least {stop.min_pct:g}% away"
        if stop.max_pct:
            s += f" (skip if wider than {stop.max_pct:g}%)"
    else:
        assert isinstance(stop, PercentStop)
        s = f"stop {stop.pct:g}%"
    parts = [s]
    if isinstance(ex.target, RRTarget):
        parts.append(f"target {ex.target.ratio:g}R")
    elif isinstance(ex.target, AtrTarget):
        parts.append(f"target {ex.target.mult:g}x ATR({ex.target.length})")
    elif isinstance(ex.target, ZoneTarget):
        parts.append(f"target the impulse high/low (min {ex.target.min_rr:g}R)")
    if ex.trailing:
        parts.append(f"trail {ex.trailing.mult:g}x ATR after +{ex.trailing.activate_r:g}R")
    if ex.max_bars:
        parts.append(f"max {ex.max_bars} bars")
    if ex.signal_long:
        parts.append("exit longs when " + condition_text(ex.signal_long))
    if ex.signal_short:
        parts.append("exit shorts when " + condition_text(ex.signal_short))
    parts.append(
        {
            "daily": "flat before 17:00 NY",
            "weekend": "flat before the weekend",
            "none": "may hold overnight",
        }[ex.flat_at]
    )
    lines.append("Exit: " + ", ".join(parts))
    lines.append(f"Risk {genome.risk.risk_per_trade * 100:g}% of equity per trade")
    return "\n".join(lines)

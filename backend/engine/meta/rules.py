"""Context features of trades and the rules built from them.

Every feature is *direction-aligned* (phrased for the trade's own side) and maps to an
executable genome condition phrased for the LONG side. The short side gets the
:func:`~engine.evolution.operators.mirror` of it, exactly like the GA's symmetric genomes.
Any rule the Meta-Analyzer finds can therefore be added to the strategy and traded live.

Examples (long phrasing; mirrored for shorts):
    weekday in {1}              → state  time.weekday in [1]               (filter, neutral)
    hour_ny in [9, 10]          → range  time.minute_ny 540..659           (filter, neutral)
    bars_since_choch <= 5       → event  structure.choch_bull within 5     (entry, directional)
    trend_aligned in {1}        → state  structure.trend in [1]            (entry, directional)
    rsi_aligned <= 40           → rsi < 40                                 (entry, directional)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from engine.evolution.operators import mirror
from engine.strategy.genome import MAX_CONDITIONS, Genome, Group, walk

NEVER = 10_000  # bars_since value when the event never happened
SESSIONS = [
    "asia",
    "london",
    "new_york",
    "london_ny_overlap",
    "comex",
    "asia_kz",
    "london_kz",
    "ny_am_kz",
    "ny_pm_kz",
]


@dataclass(frozen=True)
class FeatureDef:
    name: str
    kind: str  # "ordinal" (small ints), "bool", "numeric"
    directional: bool
    label: str


FEATURES: dict[str, FeatureDef] = {
    f.name: f
    for f in [
        FeatureDef("weekday", "ordinal", False, "trading day"),
        FeatureDef("hour_ny", "ordinal", False, "NY hour"),
        FeatureDef("us_holiday", "bool", False, "US holiday session"),
        *[FeatureDef(f"session_{s}", "bool", False, s.replace("_", " ")) for s in SESSIONS],
        FeatureDef("vol_regime", "ordinal", False, "volatility regime"),
        FeatureDef("vol_rank", "numeric", False, "ATR percentile"),
        FeatureDef("atr_pct", "numeric", False, "ATR % of price"),
        FeatureDef("trend_aligned", "ordinal", True, "market structure vs. trade side"),
        FeatureDef("trend_regime_aligned", "ordinal", True, "trend regime vs. trade side"),
        FeatureDef("bars_since_bos", "numeric", True, "bars since BOS in trade direction"),
        FeatureDef("bars_since_choch", "numeric", True, "bars since CHoCH in trade direction"),
        FeatureDef(
            "bars_since_sweep", "numeric", True, "bars since liquidity sweep against the trade"
        ),
        FeatureDef("bars_since_div", "numeric", True, "bars since divergence in trade direction"),
        FeatureDef(
            "rsi_aligned", "numeric", True, "RSI on the trade's side (long RSI / 100 - short RSI)"
        ),
    ]
}


def trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Direction-aligned feature matrix from the runner's per-trade ``context`` dicts."""
    ctx = pd.DataFrame(list(trades["context"]), index=trades.index)
    long = (trades["side"] == "long").to_numpy()
    sign = np.where(long, 1, -1)

    def since(bull: str, bear: str) -> np.ndarray:
        v = np.where(long, ctx[bull], ctx[bear]).astype(float)
        return np.where(v < 0, NEVER, v)

    out = pd.DataFrame(index=trades.index)
    out["weekday"] = ctx["weekday"].astype(int)
    out["hour_ny"] = ctx["hour_ny"].astype(int)
    out["us_holiday"] = ctx["us_holiday"].astype(bool)
    for s in SESSIONS:
        out[f"session_{s}"] = ctx.get(f"session_{s}", False)
        out[f"session_{s}"] = out[f"session_{s}"].astype(bool)
    out["vol_regime"] = ctx["vol_regime"].astype(int)
    out["vol_rank"] = ctx["vol_rank"].astype(float)
    out["atr_pct"] = ctx["atr_pct"].astype(float)
    out["trend_aligned"] = (ctx["structure_trend"].astype(int) * sign).astype(int)
    out["trend_regime_aligned"] = (ctx["trend_regime"].astype(int) * sign).astype(int)
    out["bars_since_bos"] = since("bars_since_bos_bull", "bars_since_bos_bear")
    out["bars_since_choch"] = since("bars_since_choch_bull", "bars_since_choch_bear")
    out["bars_since_sweep"] = since("bars_since_sweep_low", "bars_since_sweep_high")
    out["bars_since_div"] = since("bars_since_div_bull", "bars_since_div_bear")
    out["rsi_aligned"] = np.where(long, ctx["rsi"].astype(float), 100.0 - ctx["rsi"].astype(float))
    return out


# --------------------------------------------------------------------------- rules
@dataclass(frozen=True)
class Clause:
    feature: str
    op: str  # "in" (value = tuple of ints/bools), "<=", ">"
    value: Any

    def mask(self, x: pd.DataFrame) -> np.ndarray:
        col = x[self.feature].to_numpy()
        if self.op == "in":
            return np.isin(col, list(self.value))
        if self.op == "<=":
            return np.asarray(col <= self.value, dtype=bool)
        return np.asarray(col > self.value, dtype=bool)

    def to_dict(self) -> dict[str, Any]:
        v = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"feature": self.feature, "op": self.op, "value": v}

    def describe(self) -> str:
        f = FEATURES[self.feature]
        if self.feature == "weekday":
            days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            return "trading day " + "/".join(days[int(d)] for d in self.value)
        if self.feature == "hour_ny":
            hs = sorted(int(h) for h in self.value)
            return f"NY time {hs[0]:02d}:00-{hs[-1] + 1:02d}:00"
        if f.kind == "bool":
            yes = True in self.value
            name = "US holiday session" if self.feature == "us_holiday" else f"in {f.label}"
            return name if yes else f"not {name}"
        if self.feature == "trend_aligned":
            names = {1: "with", 0: "neutral to", -1: "against"}
            return "structure trend " + "/".join(names[int(v)] for v in self.value) + " the trade"
        if self.feature == "trend_regime_aligned":
            names = {1: "with", 0: "neutral to", -1: "against"}
            return "trend regime " + "/".join(names[int(v)] for v in self.value) + " the trade"
        if self.feature == "vol_regime":
            names = {0: "low", 1: "normal", 2: "high"}
            return "volatility " + "/".join(names[int(v)] for v in self.value)
        if self.feature.startswith("bars_since"):
            what = f.label.replace("bars since ", "")
            return (
                f"{what} within the last {int(self.value)} bars"
                if self.op == "<="
                else (f"no {what} in the last {int(self.value)} bars")
            )
        sym = "<=" if self.op == "<=" else ">"
        return f"{f.label} {sym} {self.value:g}"


@dataclass(frozen=True)
class Rule:
    clauses: tuple[Clause, ...]
    miner: str

    def mask(self, x: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(x), dtype=bool)
        for c in self.clauses:
            m &= c.mask(x)
        return m

    def describe(self) -> str:
        return " AND ".join(c.describe() for c in self.clauses)

    def key(self) -> tuple[Any, ...]:
        return tuple(sorted((c.feature, c.op, str(c.value)) for c in self.clauses))


# --------------------------------------------------------------------------- → genome
def _f(indicator: str, output: str) -> dict[str, Any]:
    return {"indicator": indicator, "output": output, "params": {}}


def _cmp(indicator: str, output: str, op: str, value: float) -> dict[str, Any]:
    return {
        "type": "compare",
        "left": {"kind": "feature", "feature": _f(indicator, output)},
        "op": "<" if op == "<=" else ">",
        "right": {"kind": "const", "value": round(float(value), 4)},
    }


_EVENT_FEATURES = {
    "bars_since_bos": [("structure", "bos_bull")],
    "bars_since_choch": [("structure", "choch_bull")],
    "bars_since_sweep": [("structure", "sweep_low")],
    "bars_since_div": [("divergence", "regular_bull"), ("divergence", "hidden_bull")],
}


def clause_condition(c: Clause) -> dict[str, Any] | None:
    """Long-phrased genome condition for a clause (None if it can't be expressed)."""
    if c.feature == "weekday":
        return {
            "type": "state",
            "feature": _f("time", "weekday"),
            "values": sorted(int(v) for v in c.value),
        }
    if c.feature == "hour_ny":
        hs = sorted(int(h) for h in c.value)
        if hs != list(range(hs[0], hs[-1] + 1)):
            return None  # only contiguous windows
        return {
            "type": "range",
            "feature": _f("time", "minute_ny"),
            "lo": hs[0] * 60,
            "hi": hs[-1] * 60 + 59,
        }
    if c.feature == "us_holiday" or c.feature.startswith("session_"):
        ind, out = (
            ("time", "us_holiday")
            if c.feature == "us_holiday"
            else ("sessions", c.feature[len("session_") :])
        )
        return {"type": "flag", "feature": _f(ind, out), "value": bool(True in c.value)}
    if c.feature == "vol_regime":
        return {
            "type": "state",
            "feature": _f("regime", "vol_regime"),
            "values": sorted(int(v) for v in c.value),
        }
    if c.feature == "trend_aligned":
        return {
            "type": "state",
            "feature": _f("structure", "trend"),
            "values": sorted(int(v) for v in c.value),
        }
    if c.feature == "trend_regime_aligned":
        return {
            "type": "state",
            "feature": _f("regime", "trend_regime"),
            "values": sorted(int(v) for v in c.value),
        }
    if c.feature == "vol_rank":
        return _cmp("regime", "vol_rank", c.op, c.value)
    if c.feature == "atr_pct":
        return _cmp("atr", "pct", c.op, c.value)
    if c.feature == "rsi_aligned":
        return _cmp("rsi", "value", c.op, c.value)
    if c.feature in _EVENT_FEATURES:
        n = int(c.value)
        if n > 200:
            return None
        events = [
            {"type": "event", "feature": _f(i, o), "within": n}
            for i, o in _EVENT_FEATURES[c.feature]
        ]
        inner: dict[str, Any] = (
            events[0] if len(events) == 1 else {"type": "group", "op": "any", "conditions": events}
        )
        if c.op == "<=":
            return inner
        return {"type": "group", "op": "all", "negate": True, "conditions": [inner]}
    return None


def refine_genome(genome: Genome, rule: Rule) -> Genome | None:
    """Base genome + rule. Directional clauses go into the entry trees (mirrored for shorts),
    neutral ones into the shared filter. Returns None if inexpressible or too complex."""
    d = genome.canonical()
    for c in rule.clauses:
        cond = clause_condition(c)
        if cond is None:
            return None
        if FEATURES[c.feature].directional:
            if d.get("entry_long"):
                d["entry_long"]["conditions"].append(copy.deepcopy(cond))
            if d.get("entry_short"):
                d["entry_short"]["conditions"].append(mirror(cond))
        else:
            d["filter"] = d.get("filter") or {
                "type": "group",
                "op": "all",
                "negate": False,
                "conditions": [],
            }
            d["filter"]["conditions"].append(cond)
    try:
        refined = Genome.model_validate(d)
    except (ValidationError, ValueError):
        return None
    total = sum(
        1 for r in refined.condition_roots() for n, _ in walk(r) if not isinstance(n, Group)
    )
    return refined if total <= MAX_CONDITIONS else None

"""Genetic operators on genomes: random generation, mirroring, mutation, crossover.

Operators work on canonical dicts and always return a *validated* Genome. Anything invalid is
retried or falls back to the parent, so the evaluator never sees nonsense.

Symmetry: an individual is ``long``, ``short`` or ``both``. For ``both``, the short side is
derived by :func:`mirror` from the long side (bull ↔ bear events, oscillator thresholds
reflected about their range midpoint, comparisons flipped). This halves the search space
and makes strategies treat both directions alike unless evolution decides otherwise.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from pydantic import ValidationError

from app.models.enums import Timeframe
from engine.indicators import get_spec
from engine.indicators.base import relevant_params
from engine.strategy.genome import MAX_CONDITIONS, Genome
from engine.strategy.mutate import perturb

Rng = np.random.Generator
Cond = dict[str, Any]

# --------------------------------------------------------------------------- mirroring
_SWAP_PAIRS = [
    ("bos_bull", "bos_bear"),
    ("choch_bull", "choch_bear"),
    ("sweep_low", "sweep_high"),
    ("eql", "eqh"),
    ("hh", "ll"),
    ("new_bull", "new_bear"),
    ("touch_bull", "touch_bear"),
    ("regular_bull", "regular_bear"),
    ("hidden_bull", "hidden_bear"),
    ("bull_top", "bear_bottom"),
    ("bull_bottom", "bear_top"),
    ("dist_bull_atr", "dist_bear_atr"),
    ("upper", "lower"),
    ("last_high", "last_low"),
    ("level_high", "level_low"),
    ("plus_di", "minus_di"),
    ("high", "low"),
]
_SWAP = {a: b for a, b in _SWAP_PAIRS} | {b: a for a, b in _SWAP_PAIRS}
# Outputs whose value has no direction: mirroring leaves them untouched.
_NEUTRAL = {
    ("adx", "adx"),
    ("atr", "value"),
    ("atr", "pct"),
    ("bbands", "width"),
    ("regime", "vol_rank"),
    ("regime", "vol_regime"),
    ("regime", "efficiency"),
}
_NEUTRAL_INDICATORS = {"time", "sessions"}
_DIRECTIONAL_STATES = {("structure", "trend"), ("regime", "trend_regime")}
_FLIP_OP = {">": "<", "<": ">", ">=": "<=", "<=": ">="}


def _is_neutral(f: dict[str, Any]) -> bool:
    return f["indicator"] in _NEUTRAL_INDICATORS or (f["indicator"], f["output"]) in _NEUTRAL


def _mirror_feature(f: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(f)
    if not _is_neutral(f) and f["output"] in _SWAP:
        out["output"] = _SWAP[f["output"]]
    return out


def _reflect(value: float, f: dict[str, Any]) -> float:
    decl = get_spec(f["indicator"]).outputs[f["output"]]
    if decl.lo is not None and decl.hi is not None:
        return round(decl.lo + decl.hi - value, 6)
    return -value


def _value_flips(op: Cond) -> bool:
    """Does mirroring this operand reflect its value (vs. just renaming or nothing)?"""
    if op["kind"] != "feature":
        return bool(op["kind"] == "price")
    f = op["feature"]
    if _is_neutral(f):
        return False
    return (
        f["output"] not in _SWAP
        or get_spec(f["indicator"]).outputs[f["output"]].kind.value == "price"
    )


def mirror(cond: Cond) -> Cond:
    c = copy.deepcopy(cond)
    t = c["type"]
    if t == "group":
        c["conditions"] = [mirror(x) for x in c["conditions"]]
        return c
    if t in ("compare", "cross"):
        left, right = c["left"], c["right"]
        directional = _value_flips(left) or _value_flips(right)
        for side in ("left", "right"):
            op = c[side]
            if op["kind"] == "feature":
                op["feature"] = _mirror_feature(op["feature"])
        if directional:
            for a, b in (("left", "right"), ("right", "left")):
                if c[a]["kind"] == "const" and c[b]["kind"] == "feature":
                    c[a]["value"] = _reflect(c[a]["value"], c[b]["feature"])
            if t == "compare":
                c["op"] = _FLIP_OP[c["op"]]
            else:
                c["direction"] = "below" if c["direction"] == "above" else "above"
        return c
    f = c["feature"]
    if t in ("event", "flag"):
        c["feature"] = _mirror_feature(f)
    elif t == "state" and (f["indicator"], f["output"]) in _DIRECTIONAL_STATES:
        c["values"] = sorted(-v for v in c["values"])
    elif t == "range" and not _is_neutral(f):
        lo, hi = _reflect(c["hi"], f), _reflect(c["lo"], f)
        c["lo"], c["hi"] = min(lo, hi), max(lo, hi)
    return c


# --------------------------------------------------------------------------- random generation
def _feat(
    indicator: str, output: str, rng: Rng, timeframe: Timeframe | None = None, **fixed: Any
) -> dict[str, Any]:
    spec = get_spec(indicator)
    used = set(relevant_params(indicator, output))
    params: dict[str, Any] = {}
    for name, p in spec.params.items():
        if name not in used:
            continue
        if name in fixed:
            params[name] = fixed[name]
        elif rng.random() < 0.6:  # keep many defaults: smaller, more readable genomes
            continue
        elif p.choices:
            params[name] = p.choices[int(rng.integers(len(p.choices)))]
        elif p.kind == "int" and p.low is not None and p.high is not None:
            hi = min(p.high, p.low + 3 * (p.default - p.low) + 2)  # bias towards practical values
            params[name] = int(rng.integers(int(p.low), int(hi) + 1))
        elif p.low is not None and p.high is not None:
            step = p.step or 0.01
            params[name] = round(float(rng.uniform(p.low, p.high)) / step) * step
    for a, b in spec.ordered:  # e.g. MACD fast < slow: swap a crossed pair, else use defaults
        va = params.get(a, spec.params[a].default)
        vb = params.get(b, spec.params[b].default)
        if va < vb:
            continue
        pa, pb = spec.params[a], spec.params[b]
        if (
            va != vb
            and (pa.low is None or vb >= pa.low)
            and (pa.high is None or vb <= pa.high)
            and (pb.low is None or va >= pb.low)
            and (pb.high is None or va <= pb.high)
        ):
            params[a], params[b] = vb, va
        else:
            params.pop(a, None)
            params.pop(b, None)
    params = spec.resolve(params)
    params = {k: v for k, v in params.items() if v != spec.params[k].default}
    out: dict[str, Any] = {"indicator": indicator, "output": output, "params": params}
    if timeframe is not None:
        out["timeframe"] = timeframe.value
    return out


def _fop(f: dict[str, Any]) -> Cond:
    return {"kind": "feature", "feature": f}


def _const(v: float) -> Cond:
    return {"kind": "const", "value": v}


BULL_EVENTS = [
    ("structure", "bos_bull"),
    ("structure", "choch_bull"),
    ("structure", "sweep_low"),
    ("structure", "eql"),
    ("swings", "hh"),
    ("fvg", "new_bull"),
    ("fvg", "touch_bull"),
    ("order_blocks", "touch_bull"),
    ("order_blocks", "new_bull"),
    ("divergence", "regular_bull"),
    ("divergence", "hidden_bull"),
]
SESSION_FLAGS = [
    "asia",
    "london",
    "new_york",
    "london_ny_overlap",
    "comex",
    "london_kz",
    "ny_am_kz",
    "ny_pm_kz",
]


def random_entry_condition(rng: Rng, htf: tuple[Timeframe, ...] = ()) -> Cond:
    """A random condition phrased for the LONG side."""
    kind = rng.choice(
        ["event", "oscillator", "price", "cross", "state", "context"],
        p=[0.34, 0.2, 0.14, 0.1, 0.12, 0.1],
    )
    if kind == "event":
        ind, out = BULL_EVENTS[int(rng.integers(len(BULL_EVENTS)))]
        return {
            "type": "event",
            "feature": _feat(ind, out, rng),
            "within": int(rng.choice([0, 0, 1, 2, 3, 5, 8, 12, 20])),
        }
    if kind == "oscillator":
        choice = int(rng.integers(4))
        if choice == 0:
            th = float(rng.integers(20, 61))
            return {
                "type": "compare",
                "left": _fop(_feat("rsi", "value", rng)),
                "op": "<",
                "right": _const(th),
            }
        if choice == 1:
            th = float(rng.integers(10, 51))
            return {
                "type": "compare",
                "left": _fop(_feat("stoch", "k", rng)),
                "op": "<",
                "right": _const(th),
            }
        if choice == 2:
            th = round(float(rng.uniform(-0.2, 0.6)), 2)
            return {
                "type": "compare",
                "left": _fop(_feat("bbands", "pct_b", rng)),
                "op": "<",
                "right": _const(th),
            }
        return {
            "type": "compare",
            "left": _fop(_feat("macd", "hist", rng)),
            "op": ">",
            "right": _const(0.0),
        }
    if kind == "price":
        if rng.random() < 0.5:
            return {
                "type": "compare",
                "left": {"kind": "price", "field": "close"},
                "op": ">",
                "right": _fop(_feat(str(rng.choice(["ema", "sma", "vwap"])), "value", rng)),
            }
        fast, slow = sorted(
            int(x) for x in rng.choice([9, 20, 21, 34, 50, 100, 200], 2, replace=False)
        )
        return {
            "type": "compare",
            "left": _fop(_feat("ema", "value", rng, length=fast)),
            "op": ">",
            "right": _fop(_feat("ema", "value", rng, length=slow)),
        }
    if kind == "cross":
        if rng.random() < 0.5:
            th = float(rng.integers(25, 56))
            return {
                "type": "cross",
                "left": _fop(_feat("rsi", "value", rng)),
                "right": _const(th),
                "direction": "above",
            }
        return {
            "type": "cross",
            "left": {"kind": "price", "field": "close"},
            "right": _fop(_feat("ema", "value", rng)),
            "direction": "above",
        }
    if kind == "state":
        tf = htf[int(rng.integers(len(htf)))] if htf and rng.random() < 0.4 else None
        if rng.random() < 0.6:
            return {"type": "state", "feature": _feat("structure", "trend", rng, tf), "values": [1]}
        return {"type": "state", "feature": _feat("regime", "trend_regime", rng, tf), "values": [1]}
    return random_context_condition(rng)


def random_context_condition(rng: Rng) -> Cond:
    """Direction-neutral filters: sessions, NY time windows, weekdays, volatility regime."""
    kind = rng.choice(
        ["session", "window", "weekday", "vol", "adx"], p=[0.35, 0.25, 0.15, 0.15, 0.1]
    )
    if kind == "session":
        return {"type": "flag", "feature": _feat("sessions", str(rng.choice(SESSION_FLAGS)), rng)}
    if kind == "window":
        start = int(rng.integers(0, 96)) * 15
        length = int(rng.choice([15, 30, 60, 90, 120, 180, 240]))
        return {
            "type": "range",
            "feature": _feat("time", "minute_ny", rng),
            "lo": start,
            "hi": min(1439, start + length - 1),
        }
    if kind == "weekday":
        days = sorted(int(d) for d in rng.choice(5, int(rng.integers(2, 5)), replace=False))
        return {"type": "state", "feature": _feat("time", "weekday", rng), "values": days}
    if kind == "vol":
        vals = sorted(int(v) for v in rng.choice(3, int(rng.integers(1, 3)), replace=False))
        return {"type": "state", "feature": _feat("regime", "vol_regime", rng), "values": vals}
    return {
        "type": "compare",
        "left": _fop(_feat("adx", "adx", rng)),
        "op": ">",
        "right": _const(float(rng.integers(15, 36))),
    }


def random_exit(rng: Rng) -> dict[str, Any]:
    r = rng.random()
    if r < 0.6:
        stop: dict[str, Any] = {"type": "atr", "mult": round(float(rng.uniform(0.8, 3.0)), 2)}
    elif r < 0.9:
        stop = {
            "type": "swing",
            "length": int(rng.integers(3, 11)),
            "buffer_atr": round(float(rng.uniform(0, 0.5)), 2),
        }
    else:
        stop = {"type": "percent", "pct": round(float(rng.uniform(0.1, 0.8)), 2)}
    r = rng.random()
    target: dict[str, Any] | None
    if r < 0.6:
        target = {"type": "rr", "ratio": round(float(rng.uniform(1.0, 4.0)), 2)}
    elif r < 0.8:
        target = {"type": "atr", "mult": round(float(rng.uniform(1.0, 6.0)), 2)}
    else:
        target = None
    out: dict[str, Any] = {"stop": stop, "target": target, "flat_at": "daily"}
    if target is None or rng.random() < 0.25:
        out["trailing"] = {
            "type": "atr",
            "mult": round(float(rng.uniform(1.0, 4.0)), 2),
            "activate_r": round(float(rng.uniform(0.5, 2.0)), 2),
        }
    if rng.random() < 0.35:
        out["max_bars"] = int(rng.choice([4, 8, 12, 16, 24, 48, 96]))
    return out


def assemble(entry: list[Cond], filt: list[Cond], exit_: dict[str, Any], mode: str) -> Genome:
    long_tree = {"type": "group", "op": "all", "conditions": entry}
    d: dict[str, Any] = {"exit": exit_}
    if mode in ("long", "both"):
        d["entry_long"] = long_tree
    if mode in ("short", "both"):
        d["entry_short"] = mirror(long_tree)
    if filt:
        d["filter"] = {"type": "group", "op": "all", "conditions": filt}
    return Genome.model_validate(d)


def decompose(genome: Genome) -> tuple[list[Cond], list[Cond], dict[str, Any], str]:
    """Inverse of :func:`assemble`: the long-phrased entry conditions, filters, exit and mode."""
    d = genome.canonical()
    if d.get("entry_long") and d.get("entry_short"):
        mode, entry = "both", d["entry_long"]["conditions"]
    elif d.get("entry_long"):
        mode, entry = "long", d["entry_long"]["conditions"]
    else:
        mode, entry = "short", mirror(d["entry_short"])["conditions"]
    filt = d["filter"]["conditions"] if d.get("filter") else []
    return copy.deepcopy(entry), copy.deepcopy(filt), d["exit"], mode


def _budget(mode: str) -> int:
    """How many entry+filter conditions fit under MAX_CONDITIONS for this mode."""
    return MAX_CONDITIONS // 2 if mode == "both" else MAX_CONDITIONS


def random_genome(
    rng: Rng, *, max_entry: int = 3, htf: tuple[Timeframe, ...] = (), tries: int = 50
) -> Genome:
    for _ in range(tries):
        mode = str(rng.choice(["both", "long", "short"], p=[0.6, 0.2, 0.2]))
        entry = [
            random_entry_condition(rng, htf) for _ in range(int(rng.integers(1, max_entry + 1)))
        ]
        filt = [random_context_condition(rng) for _ in range(int(rng.choice([0, 0, 1, 1, 2])))]
        try:
            return assemble(entry, filt, random_exit(rng), mode)
        except (ValidationError, ValueError):
            continue
    raise RuntimeError("could not generate a valid random genome")


# --------------------------------------------------------------------------- variation
def mutate(
    genome: Genome, rng: Rng, *, htf: tuple[Timeframe, ...] = (), max_entry: int = 5
) -> Genome:
    """Apply one structural mutation (or a numeric perturbation) and return a valid genome."""
    for _ in range(20):
        entry, filt, exit_, mode = decompose(genome)
        op = rng.choice(
            ["perturb", "add", "remove", "replace", "filter", "exit", "mode"],
            p=[0.35, 0.15, 0.12, 0.15, 0.1, 0.1, 0.03],
        )
        if op == "perturb":
            out = perturb(genome, rng, scale=0.2, prob=0.4)
            if out != genome:
                return out
            continue
        if op == "add" and len(entry) < max_entry:
            entry.append(random_entry_condition(rng, htf))
        elif op == "remove" and len(entry) > 1:
            entry.pop(int(rng.integers(len(entry))))
        elif op == "replace":
            entry[int(rng.integers(len(entry)))] = random_entry_condition(rng, htf)
        elif op == "filter":
            if filt and rng.random() < 0.4:
                filt.pop(int(rng.integers(len(filt))))
            elif len(filt) < 2:
                filt.append(random_context_condition(rng))
            else:
                filt[int(rng.integers(len(filt)))] = random_context_condition(rng)
        elif op == "exit":
            fresh = random_exit(rng)
            part = str(rng.choice(["stop", "target", "trailing", "max_bars"]))
            exit_[part] = fresh.get(part)
        elif op == "mode":
            mode = str(rng.choice([m for m in ("both", "long", "short") if m != mode]))
        else:
            continue
        if len(entry) + len(filt) > _budget(mode):
            continue
        try:
            child = assemble(entry, filt, exit_, mode)
        except (ValidationError, ValueError):
            continue
        if child != genome:
            return child
    return genome


def crossover(a: Genome, b: Genome, rng: Rng) -> Genome:
    """Uniform crossover of entry conditions and filters; exit and mode from either parent."""
    ea, fa, xa, ma = decompose(a)
    eb, fb, xb, mb = decompose(b)
    for _ in range(10):
        pool = ea + eb
        k = int(rng.integers(1, min(len(pool), 4) + 1))
        entry = [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]
        fpool = fa + fb
        filt = (
            [
                fpool[i]
                for i in sorted(
                    rng.choice(len(fpool), min(len(fpool), int(rng.integers(0, 3))), replace=False)
                )
            ]
            if fpool
            else []
        )
        exit_ = copy.deepcopy(xa if rng.random() < 0.5 else xb)
        mode = ma if rng.random() < 0.5 else mb
        if len(entry) + len(filt) > _budget(mode):
            continue
        try:
            return assemble(entry, filt, exit_, mode)
        except (ValidationError, ValueError):
            continue
    return a

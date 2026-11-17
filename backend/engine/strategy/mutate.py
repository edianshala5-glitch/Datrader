"""Numeric knobs of a genome and perturbations of them.

Used by the robustness check ("does the edge survive small parameter changes?") and by
the GA's mutation operator (Step 6). Every knob carries bounds from the source of truth:
indicator ``Param`` ranges, an output's declared range for thresholds, and pydantic field
constraints for exit rules. A perturbed genome is therefore always valid.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast

import annotated_types as at
import numpy as np
from pydantic import BaseModel, ValidationError

from engine.indicators import get_spec
from engine.strategy import genome as g

Path = tuple[str | int, ...]


@dataclass(frozen=True)
class Knob:
    path: Path
    value: float
    lo: float
    hi: float
    integer: bool
    step: float = 0.01  # values are snapped to this grid (readable, de-duplicable genomes)


def _grid(lo: float, hi: float) -> float:
    span = hi - lo
    return 1.0 if span >= 100 else 0.1 if span >= 10 else 0.01


_EXIT_MODELS: dict[tuple[str, str], type[BaseModel]] = {
    ("stop", "atr"): g.AtrStop,
    ("stop", "swing"): g.SwingStop,
    ("stop", "percent"): g.PercentStop,
    ("target", "rr"): g.RRTarget,
    ("target", "atr"): g.AtrTarget,
    ("trailing", "atr"): g.AtrTrail,
}


def _field_bounds(model: type[BaseModel], name: str) -> tuple[float, float] | None:
    lo = hi = None
    for m in model.model_fields[name].metadata:
        if isinstance(m, (at.Ge, at.Gt)):
            lo = float(cast(float, m.ge if isinstance(m, at.Ge) else m.gt))
        if isinstance(m, (at.Le, at.Lt)):
            hi = float(cast(float, m.le if isinstance(m, at.Le) else m.lt))
    return (lo, hi) if lo is not None and hi is not None else None


def _feature_knobs(feat: dict[str, Any], path: Path) -> list[Knob]:
    spec = get_spec(feat["indicator"])
    out = []
    for name, p in spec.params.items():
        if p.kind not in ("int", "float") or p.low is None or p.high is None:
            continue
        value = feat.get("params", {}).get(name, p.default)
        step = p.step or (1.0 if p.kind == "int" else _grid(p.low, p.high))
        out.append(
            Knob((*path, "params", name), float(value), p.low, p.high, p.kind == "int", step)
        )
    return out


def _output_range(operand: dict[str, Any]) -> tuple[float | None, float | None]:
    if operand.get("kind") != "feature":
        return None, None
    f = operand["feature"]
    decl = get_spec(f["indicator"]).outputs[f["output"]]
    return decl.lo, decl.hi


def _walk_condition(node: dict[str, Any], path: Path, out: list[Knob]) -> None:
    t = node["type"]
    if t == "group":
        for i, child in enumerate(node["conditions"]):
            _walk_condition(child, (*path, "conditions", i), out)
        return
    if t in ("compare", "cross"):
        for side, other in (("left", "right"), ("right", "left")):
            op = node[side]
            if op["kind"] == "feature":
                out.extend(_feature_knobs(op["feature"], (*path, side, "feature")))
            elif op["kind"] == "const":
                lo, hi = _output_range(node[other])
                v = float(op["value"])
                if lo is None or hi is None:  # unbounded output: allow +-50%
                    span = max(abs(v) * 0.5, 1e-6)
                    lo, hi = v - span, v + span
                out.append(Knob((*path, side, "value"), v, lo, hi, False, _grid(lo, hi)))
        return
    out.extend(_feature_knobs(node["feature"], (*path, "feature")))
    if t == "event" and node.get("within", 0) > 0:
        w = node["within"]
        out.append(
            Knob((*path, "within"), w, max(1, round(w * 0.5)), min(200, round(w * 1.5) + 1), True)
        )
    if t == "range":
        decl = get_spec(node["feature"]["indicator"]).outputs[node["feature"]["output"]]
        lo = decl.lo if decl.lo is not None else node["lo"] - abs(node["lo"]) * 0.5
        hi = decl.hi if decl.hi is not None else node["hi"] + abs(node["hi"]) * 0.5
        out.append(Knob((*path, "lo"), node["lo"], lo, hi, False, _grid(lo, hi)))
        out.append(Knob((*path, "hi"), node["hi"], lo, hi, False, _grid(lo, hi)))


def knobs(genome: g.Genome) -> list[Knob]:
    d = genome.canonical()
    out: list[Knob] = []
    for root in ("entry_long", "entry_short", "filter"):
        if d.get(root):
            _walk_condition(d[root], (root,), out)
    ex = d["exit"]
    for root in ("signal_long", "signal_short"):
        if ex.get(root):
            _walk_condition(ex[root], ("exit", root), out)
    for part in ("stop", "target", "trailing"):
        rule = ex.get(part)
        if not rule:
            continue
        model = _EXIT_MODELS[(part, rule["type"])]
        for name, value in rule.items():
            if name == "type" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            bounds = _field_bounds(model, name)
            if bounds:
                integer = isinstance(value, int)
                out.append(
                    Knob(
                        ("exit", part, name),
                        float(value),
                        *bounds,
                        integer,
                        1.0 if integer else 0.05,
                    )
                )
    if ex.get("max_bars"):
        out.append(Knob(("exit", "max_bars"), ex["max_bars"], 1, 5000, True))
    return out


def _set(d: dict[str, Any], path: Path, value: Any) -> None:
    node: Any = d
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def perturb(
    genome: g.Genome,
    rng: np.random.Generator,
    *,
    scale: float = 0.15,
    prob: float = 1.0,
    max_tries: int = 20,
) -> g.Genome:
    """Multiply each knob by ``1 + N(0, scale)`` (with probability ``prob``), clipped to its
    bounds and rounded for integers. Returns the original genome if no valid variant is
    found (e.g. lo/hi crossing in a range condition)."""
    ks = knobs(genome)
    base = genome.canonical()
    for _ in range(max_tries):
        d = copy.deepcopy(base)
        changed = False
        for k in ks:
            if rng.random() > prob:
                continue
            ref = k.value if k.value != 0 else (k.hi - k.lo) * 0.1
            v = k.value + ref * rng.normal(0.0, scale)
            v = float(np.clip(round(v / k.step) * k.step, k.lo, k.hi))
            v = round(v, 6)
            if k.integer:
                v = round(v)
                if v == k.value and scale > 0:  # make integer knobs actually move sometimes
                    v = int(np.clip(v + rng.choice([-1, 1]), k.lo, k.hi))
            changed |= v != k.value
            _set(d, k.path, v)
        if not changed:
            continue
        try:
            return g.Genome.model_validate(d)
        except (ValidationError, ValueError):
            continue
    return genome

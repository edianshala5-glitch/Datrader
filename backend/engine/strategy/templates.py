"""Hand-written starting strategies: documentation by example, and seeds for the GA's
initial population (Step 6). None of them is claimed to be profitable. Finding out which
contexts make them work is the swarm's job."""

from __future__ import annotations

from typing import Any

from engine.strategy.genome import Genome


def f(indicator: str, output: str, **params: Any) -> dict[str, Any]:
    return {"indicator": indicator, "output": output, "params": params}


def feat(indicator: str, output: str, **params: Any) -> dict[str, Any]:
    return {"kind": "feature", "feature": f(indicator, output, **params)}


def const(v: float) -> dict[str, Any]:
    return {"kind": "const", "value": v}


def group(*conds: dict[str, Any], op: str = "all") -> dict[str, Any]:
    return {"type": "group", "op": op, "conditions": list(conds)}


TEMPLATES: dict[str, dict[str, Any]] = {
    # SMC continuation: CHoCH confirms the reversal, enter on the FVG retest in the NY AM kill zone.
    "smc_choch_fvg": {
        "name": "SMC CHoCH + FVG retest (NY AM)",
        "entry_long": group(
            {"type": "event", "feature": f("structure", "choch_bull"), "within": 12},
            {"type": "event", "feature": f("fvg", "touch_bull")},
        ),
        "entry_short": group(
            {"type": "event", "feature": f("structure", "choch_bear"), "within": 12},
            {"type": "event", "feature": f("fvg", "touch_bear")},
        ),
        "filter": group({"type": "flag", "feature": f("sessions", "ny_am_kz")}),
        "exit": {
            "stop": {"type": "swing", "length": 5, "buffer_atr": 0.2},
            "target": {"type": "rr", "ratio": 2.0},
            "flat_at": "daily",
        },
    },
    # Liquidity sweep reversal during London / NY.
    "sweep_reversal": {
        "name": "Liquidity sweep reversal",
        "entry_long": group(
            {"type": "event", "feature": f("structure", "sweep_low"), "within": 2},
            {"type": "compare", "left": feat("rsi", "value"), "op": "<", "right": const(45)},
        ),
        "entry_short": group(
            {"type": "event", "feature": f("structure", "sweep_high"), "within": 2},
            {"type": "compare", "left": feat("rsi", "value"), "op": ">", "right": const(55)},
        ),
        "filter": group(
            {"type": "flag", "feature": f("sessions", "london")},
            {"type": "flag", "feature": f("sessions", "new_york")},
            op="any",
        ),
        "exit": {
            "stop": {"type": "atr", "mult": 1.2},
            "target": {"type": "rr", "ratio": 1.5},
            "max_bars": 24,
            "flat_at": "daily",
        },
    },
    # Trend pullback: EMA alignment, RSI dip, re-cross up.
    "ema_pullback": {
        "name": "EMA trend pullback",
        "entry_long": group(
            {
                "type": "compare",
                "left": feat("ema", "value", length=50),
                "op": ">",
                "right": feat("ema", "value", length=200),
            },
            {
                "type": "cross",
                "left": feat("rsi", "value"),
                "right": const(40),
                "direction": "above",
            },
        ),
        "entry_short": group(
            {
                "type": "compare",
                "left": feat("ema", "value", length=50),
                "op": "<",
                "right": feat("ema", "value", length=200),
            },
            {
                "type": "cross",
                "left": feat("rsi", "value"),
                "right": const(60),
                "direction": "below",
            },
        ),
        "exit": {
            "stop": {"type": "atr", "mult": 2.0},
            "target": None,
            "trailing": {"type": "atr", "mult": 2.5, "activate_r": 1.0},
            "flat_at": "daily",
        },
    },
    # Divergence at the Bollinger extremes, with a higher-timeframe trend filter.
    "divergence_bbands": {
        "name": "Divergence at band extremes",
        "entry_long": group(
            {"type": "event", "feature": f("divergence", "regular_bull"), "within": 3},
            {"type": "compare", "left": feat("bbands", "pct_b"), "op": "<", "right": const(0.2)},
        ),
        "entry_short": group(
            {"type": "event", "feature": f("divergence", "regular_bear"), "within": 3},
            {"type": "compare", "left": feat("bbands", "pct_b"), "op": ">", "right": const(0.8)},
        ),
        "exit": {
            "stop": {"type": "atr", "mult": 1.5},
            "target": {"type": "atr", "mult": 2.5},
            "flat_at": "daily",
        },
    },
}


def template(name: str) -> Genome:
    return Genome.model_validate(TEMPLATES[name])


def all_templates() -> dict[str, Genome]:
    return {k: template(k) for k in TEMPLATES}

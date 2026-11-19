"""Hand-specified setups, built as genomes so they go through the same validation,
promotion and live pipeline as evolved strategies.

``sd_zone_setup`` encodes the user's discretionary trade:

    zone whose swing broke structure (BOS); with ``zone_type="ob"`` an aggressive order
    block left by a displacement  ->  pullback  ->  retest of the zone  ->  optional
    confirmation candle on the execution timeframe (1m)  ->  entry; stop beyond the zone;
    target at the impulse extreme. Only with the market-structure trend of the zone
    timeframe ("micro" trend) and optionally of higher timeframes.
"""

from __future__ import annotations

from typing import Any, Literal

from app.models.enums import Timeframe
from engine.strategy.genome import Genome

Confirm = Literal["touch", "rejection", "engulfing", "three", "any"]
CONFIRMS: tuple[Confirm, ...] = ("touch", "rejection", "engulfing", "three", "any")
# Bars back (on the execution timeframe) in which price must have touched the zone. Three
# candles in a row need room: the touch is usually on the first of them.
# "touch" enters on the retest itself (no confirmation candle).
_TOUCH_LOOKBACK = {"touch": 0, "rejection": 0, "engulfing": 1, "three": 2, "any": 2}


def _zone(output: str, tf: Timeframe, params: dict[str, Any]) -> dict[str, Any]:
    return {"indicator": "sd_zones", "output": output, "params": params, "timeframe": tf.value}


def _side(
    long: bool,
    tf: Timeframe,
    confirm: Confirm,
    zone_params: dict[str, Any],
    wick_ratio: float,
    trend: tuple[Timeframe, ...],
) -> dict[str, Any]:
    side = "demand" if long else "supply"
    near, far = ("high", "low") if long else ("low", "high")
    touch_field = "low" if long else "high"
    touch_op, hold_op = ("<=", ">=") if long else (">=", "<=")
    touched = [
        {
            "type": "compare",
            "left": {"kind": "price", "field": touch_field, "shift": k},
            "op": touch_op,
            "right": {"kind": "feature", "feature": _zone(f"{side}_{near}", tf, zone_params)},
        }
        for k in range(_TOUCH_LOOKBACK[confirm] + 1)
    ]
    pattern = f"{'bull' if long else 'bear'}_{confirm}"
    trend_conds = [  # only trade with the market structure of these timeframes
        {
            "type": "state",
            "feature": {"indicator": "structure", "output": "trend", "timeframe": t.value},
            "values": [1 if long else -1],
        }
        for t in trend
    ]
    confirmation = (
        []
        if confirm == "touch"
        else [
            {
                "type": "event",
                "feature": {
                    "indicator": "candles",
                    "output": pattern,
                    "params": {"wick_ratio": wick_ratio},
                },
            }
        ]
    )
    return {
        "type": "group",
        "op": "all",
        "conditions": [
            *trend_conds,
            {"type": "flag", "feature": _zone(f"{side}_ready", tf, zone_params)},
            touched[0]
            if len(touched) == 1
            else {"type": "group", "op": "any", "conditions": touched},
            {  # closed back on the right side: the zone held
                "type": "compare",
                "left": {"kind": "price", "field": "close"},
                "op": hold_op,
                "right": {"kind": "feature", "feature": _zone(f"{side}_{far}", tf, zone_params)},
            },
            *confirmation,
        ],
    }


def sd_zone_setup(
    zone_tf: Timeframe = Timeframe.M15,
    confirm: Confirm = "any",
    *,
    zone_params: dict[str, Any] | None = None,
    wick_ratio: float = 2.0,
    trend: tuple[Timeframe, ...] | None = None,
    buffer_frac: float = 0.1,
    min_stop_pct: float = 0.3,
    max_stop_pct: float | None = 0.6,
    min_rr: float = 1.0,
    flat_at: Literal["daily", "weekend", "none"] = "daily",
) -> Genome:
    """The supply/demand setup, run on the execution timeframe (1m) with zones from
    ``zone_tf``, trading only with the market-structure trend of every timeframe in
    ``trend`` (default: the zone timeframe itself, the "micro" trend; () = no filter).
    ``zone_params`` override ``sd_zones`` defaults (swing length, impulse size, approach
    bars, max candle size, stair steps, zone age, touch window, zones tracked).

    Stops sit beyond the zone but at least ``min_stop_pct`` % of price away (0.3% is about
    $8 with gold at 2,650); setups needing more than ``max_stop_pct`` % are skipped."""
    zp = dict(zone_params or {})
    tr = (zone_tf,) if trend is None else trend
    kind = "OB" if zp.get("zone_type") == "ob" else "S/D zone"
    trend_txt = "+".join(t.value for t in tr) or "none"
    return Genome.model_validate(
        {
            "name": f"{kind} {zone_tf.value} + {confirm}, trend {trend_txt}",
            "entry_long": _side(True, zone_tf, confirm, zp, wick_ratio, tr),
            "entry_short": _side(False, zone_tf, confirm, zp, wick_ratio, tr),
            "exit": {
                "stop": {
                    "type": "zone",
                    "buffer_frac": buffer_frac,
                    "min_pct": min_stop_pct,
                    "max_pct": max_stop_pct,
                },
                "target": {"type": "zone", "min_rr": min_rr},
                "flat_at": flat_at,
            },
        }
    )


def _harmonic_side(
    long: bool,
    tf: Timeframe,
    confirm: Confirm,
    params: dict[str, Any],
    wick_ratio: float,
    trend: tuple[Timeframe, ...],
) -> dict[str, Any]:
    side = "bull" if long else "bear"

    def feat(output: str) -> dict[str, Any]:
        return {"indicator": "harmonics", "output": output, "params": params, "timeframe": tf.value}

    near, far = ("prz_high", "prz_low") if long else ("prz_low", "prz_high")
    touch_field = "low" if long else "high"
    touch_op, hold_op = ("<=", ">=") if long else (">=", "<=")
    touched = [
        {
            "type": "compare",
            "left": {"kind": "price", "field": touch_field, "shift": k},
            "op": touch_op,
            "right": {"kind": "feature", "feature": feat(f"{side}_{near}")},
        }
        for k in range(_TOUCH_LOOKBACK[confirm] + 1)
    ]
    conditions: list[dict[str, Any]] = [
        {
            "type": "state",
            "feature": {"indicator": "structure", "output": "trend", "timeframe": t.value},
            "values": [1 if long else -1],
        }
        for t in trend
    ]
    conditions += [
        {"type": "flag", "feature": feat(f"{side}_ready")},
        touched[0] if len(touched) == 1 else {"type": "group", "op": "any", "conditions": touched},
        {  # closed back on the reversal side of the PRZ
            "type": "compare",
            "left": {"kind": "price", "field": "close"},
            "op": hold_op,
            "right": {"kind": "feature", "feature": feat(f"{side}_{far}")},
        },
    ]
    if confirm != "touch":
        conditions.append(
            {
                "type": "event",
                "feature": {
                    "indicator": "candles",
                    "output": f"{side}_{confirm}",
                    "params": {"wick_ratio": wick_ratio},
                },
            }
        )
    return {"type": "group", "op": "all", "conditions": conditions}


def harmonic_setup(
    pattern_tf: Timeframe = Timeframe.M5,
    confirm: Confirm = "any",
    *,
    params: dict[str, Any] | None = None,
    wick_ratio: float = 2.0,
    trend: tuple[Timeframe, ...] = (),
    buffer_frac: float = 0.1,
    min_stop_pct: float = 0.15,
    max_stop_pct: float | None = 0.8,
    min_rr: float = 1.0,
    flat_at: Literal["daily", "weekend", "none"] = "daily",
) -> Genome:
    """Harmonic XABCD patterns on ``pattern_tf``, entered on the 1m when price reaches the
    PRZ and a confirmation candle closes back on the reversal side. Stop beyond X (or the
    PRZ for patterns completing past X) plus ``buffer_frac`` of the PRZ height, at least
    ``min_stop_pct`` % of price; skipped if wider than ``max_stop_pct`` %. Target: back
    towards A (``target_frac`` param, default 61.8% of the AD leg)."""
    hp = dict(params or {})
    which = hp.get("pattern", "all")
    trend_txt = "+".join(t.value for t in trend) or "none"
    return Genome.model_validate(
        {
            "name": f"Harmonic {which} {pattern_tf.value} + {confirm}, trend {trend_txt}",
            "entry_long": _harmonic_side(True, pattern_tf, confirm, hp, wick_ratio, trend),
            "entry_short": _harmonic_side(False, pattern_tf, confirm, hp, wick_ratio, trend),
            "exit": {
                "stop": {
                    "type": "zone",
                    "buffer_frac": buffer_frac,
                    "min_pct": min_stop_pct,
                    "max_pct": max_stop_pct,
                },
                "target": {"type": "zone", "min_rr": min_rr},
                "flat_at": flat_at,
            },
        }
    )

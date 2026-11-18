"""Hand-specified setups, built as genomes so they go through the same validation,
promotion and live pipeline as evolved strategies.

``sd_zone_setup`` encodes the user's discretionary trade:

    supply/demand zone whose swing broke structure (BOS)  ->  slow, stair-stepped pullback
    ->  price reaches the zone  ->  confirmation candle on the execution timeframe (1m)
    ->  entry; stop beyond the zone; target at the impulse extreme.
"""

from __future__ import annotations

from typing import Any, Literal

from app.models.enums import Timeframe
from engine.strategy.genome import Genome

Confirm = Literal["rejection", "engulfing", "three", "any"]
CONFIRMS: tuple[Confirm, ...] = ("rejection", "engulfing", "three", "any")
# Bars back (on the execution timeframe) in which price must have touched the zone. Three
# candles in a row need room: the touch is usually on the first of them.
_TOUCH_LOOKBACK = {"rejection": 0, "engulfing": 1, "three": 2, "any": 2}


def _zone(output: str, tf: Timeframe, params: dict[str, Any]) -> dict[str, Any]:
    return {"indicator": "sd_zones", "output": output, "params": params, "timeframe": tf.value}


def _side(
    long: bool, tf: Timeframe, confirm: Confirm, zone_params: dict[str, Any], wick_ratio: float
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
    return {
        "type": "group",
        "op": "all",
        "conditions": [
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
            {
                "type": "event",
                "feature": {
                    "indicator": "candles",
                    "output": pattern,
                    "params": {"wick_ratio": wick_ratio},
                },
            },
        ],
    }


def sd_zone_setup(
    zone_tf: Timeframe = Timeframe.M15,
    confirm: Confirm = "any",
    *,
    zone_params: dict[str, Any] | None = None,
    wick_ratio: float = 2.0,
    buffer_frac: float = 0.1,
    min_rr: float = 1.0,
    flat_at: Literal["daily", "weekend", "none"] = "daily",
) -> Genome:
    """The supply/demand setup, run on the execution timeframe (1m) with zones from
    ``zone_tf``. ``zone_params`` override ``sd_zones`` defaults (swing length, impulse size,
    approach bars, max candle size, stair steps, zone age, touch window)."""
    zp = dict(zone_params or {})
    return Genome.model_validate(
        {
            "name": f"S/D zone {zone_tf.value} + {confirm}",
            "entry_long": _side(True, zone_tf, confirm, zp, wick_ratio),
            "entry_short": _side(False, zone_tf, confirm, zp, wick_ratio),
            "exit": {
                "stop": {"type": "zone", "buffer_frac": buffer_frac},
                "target": {"type": "zone", "min_rr": min_rr},
                "flat_at": flat_at,
            },
        }
    )

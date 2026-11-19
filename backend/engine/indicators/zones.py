"""Supply/demand zones with a "slow approach" filter, and candle confirmation patterns.

Encodes this discretionary setup:

1. A **demand zone** is the base of the swing whose up-move breaks structure (bullish BOS):
   the candle at the lowest low between the broken swing high and the breaking close. The
   zone spans that candle's low to its body top. The impulse out of the zone must be strong
   (``impulse_atr`` x ATR from the zone low to the breaking close).
2. The **target** is the highest high the impulse makes before price turns back.
3. The zone is **ready** while price retraces towards it *slowly*: at least
   ``approach_bars`` bars since the impulse high, no candle larger than ``max_candle_atr``
   x ATR, and at least ``min_steps`` small lower highs (a stair-step pullback that leaves
   liquidity behind).
4. The zone dies when a close goes through it, when it gets older than ``max_age`` bars, or
   ``touch_bars`` bars after price first reaches it (first touch only). Up to ``max_zones``
   zones per side live at once (a new one replaces the oldest); the output is the nearest
   one to price, preferring zones whose pullback is ready.

Supply zones are the mirror image (computed on negated prices). Everything is evaluated at
bar close with confirmed swings, so the values at bar ``t`` use only bars ``<= t``.

``candles`` flags the entry confirmations on the execution timeframe (e.g. 1m): rejection
(pin bar), engulfing, and three consecutive candles in the trade direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from engine.indicators.base import (
    Category,
    Inputs,
    Output,
    OutputKind,
    Outputs,
    Param,
    indicator,
)
from engine.indicators.kernels import confirmed_swings
from engine.indicators.standard import atr_values

NAN = np.nan


@njit(cache=True)
def _demand_zones(  # type: ignore[no-untyped-def]
    o,
    h,
    lo,
    c,
    atr,
    sh_event,
    sh_price,
    sh_pivot,
    impulse_atr,
    approach_bars,
    max_candle_atr,
    min_steps,
    max_age,
    touch_bars,
    max_zones,
    ob_mode,
    displacement_atr,
    max_leg_bars,
):
    """Track up to ``max_zones`` demand zones at once; output the nearest one below price,
    preferring zones whose pullback qualifies as slow ("ready")."""
    n = h.shape[0]
    z_low = np.full(n, NAN)
    z_high = np.full(n, NAN)
    z_target = np.full(n, NAN)
    ready = np.zeros(n, dtype=np.bool_)

    k = max_zones
    alive = np.zeros(k, dtype=np.bool_)
    zl = np.full(k, NAN)
    zh = np.full(k, NAN)
    peak = np.full(k, NAN)
    peak_bar = np.full(k, -1, dtype=np.int64)
    created = np.full(k, -1, dtype=np.int64)
    touched = np.zeros(k, dtype=np.bool_)
    touch_bar = np.full(k, -1, dtype=np.int64)
    frozen = np.zeros(k, dtype=np.bool_)

    level = NAN  # latest unbroken confirmed swing high
    level_pivot = -1

    for i in range(n):
        if sh_event[i]:
            level = sh_price[i]
            level_pivot = sh_pivot[i]

        # --- bullish break of structure creates a zone (replacing the oldest if full)
        if level_pivot >= 0 and not np.isnan(level) and c[i] > level:
            b = level_pivot
            for j in range(level_pivot, i + 1):
                if lo[j] < lo[b]:
                    b = j
            level = NAN  # this swing high is broken; wait for the next one
            level_pivot = -1
            a = atr[i]
            valid = not np.isnan(a) and a > 0 and (c[i] - lo[b]) >= impulse_atr * a
            bottom = lo[b]
            top = max(o[b], c[b])
            if valid and ob_mode:
                # Order block: the last down candle at the origin of the move (up to 3 bars
                # before the low), spanning its full range plus the sweep low.
                ob = b
                for j in range(b, max(b - 3, 0) - 1, -1):
                    if c[j] < o[j]:
                        ob = j
                        break
                top = h[ob]
                bottom = min(lo[ob], lo[b])
                # Aggressive: a big-bodied candle in the leg, and the break comes quickly.
                biggest_body = 0.0
                for j in range(ob + 1, i + 1):
                    body = c[j] - o[j]
                    if body > biggest_body:
                        biggest_body = body
                valid = biggest_body >= displacement_atr * a and (i - b) <= max_leg_bars
            elif valid and top - lo[b] < 0.1 * a:  # doji base: use its high instead
                top = h[b]
            if valid and top < c[i]:
                slot = -1
                for s in range(k):
                    if not alive[s]:
                        slot = s
                        break
                if slot < 0:
                    slot = 0
                    for s in range(1, k):
                        if created[s] < created[slot]:
                            slot = s
                alive[slot] = True
                zl[slot], zh[slot] = bottom, top
                created[slot] = i
                peak[slot], peak_bar[slot] = h[b], b
                for j in range(b, i + 1):
                    if h[j] >= peak[slot]:
                        peak[slot], peak_bar[slot] = h[j], j
                touched[slot] = False
                touch_bar[slot] = -1
                frozen[slot] = False

        # --- update every live zone
        for s in range(k):
            if not alive[s]:
                continue
            if c[i] < zl[s] or i - created[s] > max_age:  # broken through, or stale
                alive[s] = False
                continue
            if not touched[s]:
                if h[i] >= peak[s]:  # impulse still extending: the pullback starts later
                    peak[s], peak_bar[s] = h[i], i
                pb = peak_bar[s]
                ok = i - pb >= approach_bars
                if ok:
                    biggest = 0.0
                    for j in range(pb + 1, i + 1):
                        r = h[j] - lo[j]
                        if r > biggest:
                            biggest = r
                    ok = biggest <= max_candle_atr * atr[i]
                if ok and min_steps > 0:
                    steps = 0
                    for j in range(pb + 2, i):
                        if h[j] > h[j - 1] and h[j] >= h[j + 1]:
                            steps += 1
                    ok = steps >= min_steps
                if lo[i] <= zh[s]:  # first touch: keep the verdict on the approach *before* it
                    touched[s] = True
                    touch_bar[s] = i
                else:
                    frozen[s] = ok
            elif i - touch_bar[s] >= touch_bars:
                alive[s] = False  # first touch only
                continue

        # --- output: the nearest live zone below price, ready ones first
        best = -1
        for s in range(k):
            if not alive[s]:
                continue
            if best < 0:
                best = s
            elif frozen[s] != frozen[best]:
                if frozen[s]:
                    best = s
            elif zh[s] > zh[best]:
                best = s
        if best >= 0:
            z_low[i], z_high[i], z_target[i] = zl[best], zh[best], peak[best]
            ready[i] = frozen[best]
    return z_low, z_high, z_target, ready


ZONE_PARAMS = {
    "length": Param(3, 2, 10, 1),  # swing length of the structure that gets broken
    "impulse_atr": Param(1.5, 0.5, 5.0, 0.1),
    "approach_bars": Param(4, 1, 30, 1),
    "max_candle_atr": Param(1.8, 0.3, 4.0, 0.05),
    "min_steps": Param(1, 0, 5, 1),
    "max_age": Param(300, 10, 2000, 1),
    "touch_bars": Param(2, 1, 10, 1),
    "max_zones": Param(3, 1, 10, 1),  # zones per side tracked at the same time
    # "base": zone = base candle (low .. body top). "ob": aggressive order block = last down
    # candle before a displacement (big body >= displacement_atr x ATR, BOS within
    # max_leg_bars of the low), full candle range.
    "zone_type": Param("base", choices=("base", "ob")),
    "displacement_atr": Param(1.0, 0.2, 4.0, 0.1),
    "max_leg_bars": Param(12, 2, 60, 1),
}
_PRICE = Output(OutputKind.PRICE)
_WICK = ("wick_ratio",)
_FLAG = Output(OutputKind.FLAG)


def sd_zone_values(inp: Inputs, **p: Any) -> Outputs:
    """All zone outputs. ``compute`` caches the result per parameter set."""
    atr = atr_values(inp, 14)
    args = (
        float(p["impulse_atr"]),
        int(p["approach_bars"]),
        float(p["max_candle_atr"]),
        int(p["min_steps"]),
        int(p["max_age"]),
        int(p["touch_bars"]),
        int(p["max_zones"]),
        p["zone_type"] == "ob",
        float(p["displacement_atr"]),
        int(p["max_leg_bars"]),
    )
    length = int(p["length"])
    sh_e, _, sh_p, _, sh_i, _ = confirmed_swings(inp.high, inp.low, length, length)
    dl, dh, dt, dr = _demand_zones(
        inp.open, inp.high, inp.low, inp.close, atr, sh_e, sh_p, sh_i, *args
    )
    # Supply = demand on the mirrored series (negate prices; highs and lows swap).
    mh, ml = -inp.low, -inp.high
    msh_e, _, msh_p, _, msh_i, _ = confirmed_swings(mh, ml, length, length)
    sl_, sh_, st, sr = _demand_zones(-inp.open, mh, ml, -inp.close, atr, msh_e, msh_p, msh_i, *args)
    return {
        "demand_low": dl,
        "demand_high": dh,
        "demand_target": dt,
        "demand_ready": dr,
        "supply_low": -sh_,
        "supply_high": -sl_,
        "supply_target": -st,
        "supply_ready": sr,
    }


@indicator(
    "sd_zones",
    category=Category.STRUCTURE,
    params=ZONE_PARAMS,
    outputs={
        "demand_low": _PRICE,
        "demand_high": _PRICE,
        "demand_target": _PRICE,
        "demand_ready": _FLAG,
        "supply_low": _PRICE,
        "supply_high": _PRICE,
        "supply_target": _PRICE,
        "supply_ready": _FLAG,
    },
    description=(
        "Supply/demand zone at the base of the swing that broke structure, ready while price "
        "pulls back slowly (small candles, enough bars, stair steps). Target = impulse extreme."
    ),
)
def _sd_zones(inp: Inputs, **p: Any) -> Outputs:
    return sd_zone_values(inp, **p)


@indicator(
    "candles",
    category=Category.STRUCTURE,
    params={"wick_ratio": Param(2.0, 1.0, 5.0, 0.1)},
    outputs={
        "bull_rejection": Output(
            OutputKind.EVENT, description="bullish pin bar", uses=("wick_ratio",)
        ),
        "bear_rejection": Output(
            OutputKind.EVENT, description="bearish pin bar", uses=("wick_ratio",)
        ),
        "bull_engulfing": Output(OutputKind.EVENT, uses=()),
        "bear_engulfing": Output(OutputKind.EVENT, uses=()),
        "bull_three": Output(OutputKind.EVENT, description="3 rising bullish candles", uses=()),
        "bear_three": Output(OutputKind.EVENT, description="3 falling bearish candles", uses=()),
        "bull_any": Output(OutputKind.EVENT, uses=_WICK),
        "bear_any": Output(OutputKind.EVENT, uses=_WICK),
    },
    description="Entry confirmation candles: rejection (pin bar), engulfing, three in a row.",
)
def _candles(inp: Inputs, wick_ratio: float) -> Outputs:
    o, h, lo, c = inp.open, inp.high, inp.low, inp.close
    rng = h - lo
    body = np.abs(c - o)
    lower = np.minimum(o, c) - lo
    upper = h - np.maximum(o, c)
    ref = np.maximum(body, 0.1 * rng)
    with np.errstate(invalid="ignore"):
        bull_rej = (rng > 0) & (lower >= wick_ratio * ref) & (lower >= 0.5 * rng)
        bull_rej &= c >= lo + 0.5 * rng
        bear_rej = (rng > 0) & (upper >= wick_ratio * ref) & (upper >= 0.5 * rng)
        bear_rej &= c <= h - 0.5 * rng

    def prev(x: np.ndarray, k: int = 1) -> np.ndarray:
        out = np.empty_like(x)
        out[:k] = np.nan if x.dtype.kind == "f" else False
        out[k:] = x[:-k]
        return out

    up, down = c > o, c < o
    po, pc = prev(o), prev(c)
    with np.errstate(invalid="ignore"):
        bull_eng = up & (pc < po) & (c >= po) & (o <= pc) & (body > np.abs(pc - po))
        bear_eng = down & (pc > po) & (c <= po) & (o >= pc) & (body > np.abs(pc - po))
        bull_three = up & prev(up) & prev(up, 2) & (c > pc) & (pc > prev(c, 2))
        bear_three = down & prev(down) & prev(down, 2) & (c < pc) & (pc < prev(c, 2))
    bull_three[:2] = False
    bear_three[:2] = False
    return {
        "bull_rejection": bull_rej,
        "bear_rejection": bear_rej,
        "bull_engulfing": bull_eng,
        "bear_engulfing": bear_eng,
        "bull_three": bull_three,
        "bear_three": bear_three,
        "bull_any": bull_rej | bull_eng | bull_three,
        "bear_any": bear_rej | bear_eng | bear_three,
    }

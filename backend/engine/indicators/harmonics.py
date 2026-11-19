"""Harmonic patterns (XABCD): Gartley, Bat, Butterfly, Crab, Cypher, Shark.

Swings come from the same confirmed fractals as market structure, joined into an
alternating zigzag. When a new swing high C is confirmed after X (low), A (high) and
B (low), the leg ratios are checked against each pattern and the **potential reversal
zone (PRZ)** for D is projected: the overlap of the XA-retracement and BC-extension levels
(Cypher: 0.786 of XC). The bullish pattern is armed until price reaches the PRZ (and for
``touch_bars`` bars after), unless price first makes a new high above C, closes beyond the
stop reference, or the pattern gets older than ``max_age`` bars. Bearish patterns are the
mirror image (negated prices).

Ratios (bullish; B = AB/XA, C = BC/AB, D = XA retracement and BC extension):

    Gartley    B 0.618         C 0.382-0.886  D 0.786       BC ext 1.272-1.618
    Bat        B 0.382-0.50    C 0.382-0.886  D 0.886       BC ext 1.618-2.618
    Butterfly  B 0.786         C 0.382-0.886  D 1.27-1.618  BC ext 1.618-2.24
    Crab       B 0.382-0.618   C 0.382-0.886  D 1.618       BC ext 2.24-3.618
    Cypher     B 0.382-0.618   C 1.13-1.414 of XA (beyond A)   D 0.786 of XC
    Shark      B 0.382-0.886   C 1.13-1.618 of AB (beyond A)   D 0.886-1.13, BC ext 1.618-2.24

Every ratio gets a relative tolerance ``tol``. Values at bar ``t`` use bars ``<= t`` only.

Outputs per side: PRZ low/high, stop reference (beyond X, or beyond the PRZ for patterns
that complete past X), target (``target_frac`` of the way from the PRZ middle back to the
leg's extreme), the pattern kind (1..6) and the X/A/B/C pivots (prices and bar indices,
for charts).
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

NAN = np.nan
KINDS = ("gartley", "bat", "butterfly", "crab", "cypher", "shark")
# columns: b_lo, b_hi, c_lo, c_hi, d_lo, d_hi, e_lo, e_hi   (mode: 0 standard, 1 cypher, 2 shark)
_TABLE = np.array(
    [
        [0.618, 0.618, 0.382, 0.886, 0.786, 0.786, 1.272, 1.618],
        [0.382, 0.500, 0.382, 0.886, 0.886, 0.886, 1.618, 2.618],
        [0.786, 0.786, 0.382, 0.886, 1.270, 1.618, 1.618, 2.240],
        [0.382, 0.618, 0.382, 0.886, 1.618, 1.618, 2.240, 3.618],
        [0.382, 0.618, 1.130, 1.414, 0.786, 0.786, 0.0, 0.0],
        [0.382, 0.886, 1.130, 1.618, 0.886, 1.130, 1.618, 2.240],
    ]
)
_MODES = np.array([0, 0, 0, 0, 1, 2], dtype=np.int64)


@njit(cache=True)
def _within(x, lo, hi, tol):  # type: ignore[no-untyped-def]
    return lo * (1.0 - tol) <= x <= hi * (1.0 + tol)


@njit(cache=True)
def _match(X, A, B, C, table, modes, enabled, tol):  # type: ignore[no-untyped-def]
    """First enabled bullish pattern matching X<A>B<C (prices); returns kind, PRZ lo/hi."""
    xa = A - X
    ab = A - B
    bc = C - B
    if xa <= 0 or ab <= 0 or bc <= 0 or B <= X:
        return 0, NAN, NAN
    b_ret = ab / xa
    for p in range(table.shape[0]):
        if not enabled[p]:
            continue
        r = table[p]
        mode = modes[p]
        if not _within(b_ret, r[0], r[1], tol):
            continue
        if mode == 0:
            if not (_within(bc / ab, r[2], r[3], tol) and C < A):
                continue
        elif mode == 1:  # cypher: C beyond A, measured on XA
            if not _within((C - X) / xa, r[2], r[3], tol):
                continue
        elif not _within(bc / ab, r[2], r[3], tol):  # shark: C beyond A, measured on AB
            continue
        if mode == 1:
            xc = C - X
            lo, hi = C - r[5] * (1 + tol) * xc, C - r[4] * (1 - tol) * xc
        else:
            lo1, hi1 = A - r[5] * (1 + tol) * xa, A - r[4] * (1 - tol) * xa
            lo2, hi2 = C - r[7] * (1 + tol) * bc, C - r[6] * (1 - tol) * bc
            lo, hi = max(lo1, lo2), min(hi1, hi2)
        if lo < hi:
            return p + 1, lo, hi
    return 0, NAN, NAN


@njit(cache=True)
def _bullish_harmonics(  # type: ignore[no-untyped-def]
    h,
    lo,
    c,
    sh_event,
    sh_price,
    sh_pivot,
    sl_event,
    sl_price,
    sl_pivot,
    table,
    modes,
    enabled,
    tol,
    target_frac,
    max_age,
    touch_bars,
):
    n = h.shape[0]
    out = np.full((8, n), NAN)  # prz_lo, prz_hi, stop, target, X, A, B, C
    idx_out = np.full((4, n), NAN)  # bar indices of X, A, B, C
    kind_out = np.zeros(n, dtype=np.int64)
    ready = np.zeros(n, dtype=np.bool_)

    pp = np.empty(n + 1)
    pi = np.empty(n + 1, dtype=np.int64)
    pt = np.empty(n + 1, dtype=np.int64)  # 1 high, -1 low
    m = 0

    active = False
    kind = 0
    z_lo = z_hi = stop = target = 0.0
    px = np.zeros(4)
    pix = np.zeros(4, dtype=np.int64)
    created = touch_bar = -1
    touched = False

    for t in range(n):
        new_high = False
        # Add confirmed pivots in pivot-time order (a bar can confirm both).
        order = (0, 1) if sl_pivot[t] <= sh_pivot[t] else (1, 0)
        for which in order:
            if which == 0 and sl_event[t]:
                price, idx = sl_price[t], sl_pivot[t]
                if m > 0 and pt[m - 1] == -1:
                    if price < pp[m - 1]:
                        pp[m - 1], pi[m - 1] = price, idx
                else:
                    pp[m], pi[m], pt[m] = price, idx, -1
                    m += 1
            elif which == 1 and sh_event[t]:
                price, idx = sh_price[t], sh_pivot[t]
                if m > 0 and pt[m - 1] == 1:
                    if price > pp[m - 1]:
                        pp[m - 1], pi[m - 1] = price, idx
                        new_high = True
                else:
                    pp[m], pi[m], pt[m] = price, idx, 1
                    m += 1
                    new_high = True

        if new_high and m >= 4 and pt[m - 4] == -1:
            X, A, B, C = pp[m - 4], pp[m - 3], pp[m - 2], pp[m - 1]
            k, plo, phi = _match(X, A, B, C, table, modes, enabled, tol)
            if k > 0 and c[t] > plo:
                active = True
                kind = k
                z_lo, z_hi = plo, phi
                stop = min(X, plo)
                mid = 0.5 * (plo + phi)
                target = mid + target_frac * (max(A, C) - mid)
                for q in range(4):
                    px[q] = pp[m - 4 + q]
                    pix[q] = pi[m - 4 + q]
                created = t
                touched = False
                touch_bar = -1

        if not active:
            continue
        if (not touched and h[t] > px[3]) or c[t] < stop or t - created > max_age:
            active = False
            continue
        if not touched:
            if lo[t] <= z_hi:
                touched = True
                touch_bar = t
        elif t - touch_bar >= touch_bars:
            active = False
            continue
        out[0, t], out[1, t], out[2, t], out[3, t] = z_lo, z_hi, stop, target
        for q in range(4):
            out[4 + q, t] = px[q]
            idx_out[q, t] = pix[q]
        kind_out[t] = kind
        ready[t] = True
    return out, idx_out, kind_out, ready


_PRICE = Output(OutputKind.PRICE)
_IDX = Output(OutputKind.UNBOUNDED, description="bar index of the pivot (for charts)")
_KIND = Output(
    OutputKind.STATE, values=(0, 1, 2, 3, 4, 5, 6), description="0 none, 1..6 " + ", ".join(KINDS)
)


def _side_outputs(prefix: str) -> dict[str, Output]:
    out: dict[str, Output] = {
        f"{prefix}_ready": Output(OutputKind.FLAG, description=f"{prefix} pattern armed"),
        f"{prefix}_prz_low": _PRICE,
        f"{prefix}_prz_high": _PRICE,
        f"{prefix}_stop": _PRICE,
        f"{prefix}_target": _PRICE,
        f"{prefix}_kind": _KIND,
    }
    for p in "xabc":
        out[f"{prefix}_{p}"] = _PRICE
        out[f"{prefix}_{p}_idx"] = _IDX
    return out


@indicator(
    "harmonics",
    category=Category.STRUCTURE,
    params={
        "length": Param(3, 2, 10, 1),
        "tol": Param(0.05, 0.0, 0.2, 0.01),
        "pattern": Param("all", choices=("all", *KINDS)),
        "target_frac": Param(0.618, 0.2, 1.0, 0.01),
        "max_age": Param(100, 10, 1000, 1),
        "touch_bars": Param(3, 1, 20, 1),
    },
    outputs={**_side_outputs("bull"), **_side_outputs("bear")},
    description="Harmonic XABCD patterns (Gartley, Bat, Butterfly, Crab, Cypher, Shark): "
    "PRZ for D, stop beyond X/PRZ, target back towards A.",
)
def _harmonics(inp: Inputs, **p: Any) -> Outputs:
    enabled = np.array([p["pattern"] in ("all", k) for k in KINDS])
    length = int(p["length"])
    args = (
        _TABLE,
        _MODES,
        enabled,
        float(p["tol"]),
        float(p["target_frac"]),
        int(p["max_age"]),
        int(p["touch_bars"]),
    )

    def run(h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> tuple[Any, ...]:
        sh_e, sl_e, sh_p, sl_p, sh_i, sl_i = confirmed_swings(h, lo, length, length)
        return _bullish_harmonics(h, lo, c, sh_e, sh_p, sh_i, sl_e, sl_p, sl_i, *args)  # type: ignore[no-any-return]

    res: Outputs = {}
    bull = run(inp.high, inp.low, inp.close)
    bear = run(-inp.low, -inp.high, -inp.close)  # mirror: bearish = bullish on negated prices
    for prefix, (vals, idx, kind, ready), sign in (("bull", bull, 1.0), ("bear", bear, -1.0)):
        lo_v, hi_v = vals[0] * sign, vals[1] * sign
        res[f"{prefix}_prz_low"] = np.minimum(lo_v, hi_v)
        res[f"{prefix}_prz_high"] = np.maximum(lo_v, hi_v)
        res[f"{prefix}_stop"] = vals[2] * sign
        res[f"{prefix}_target"] = vals[3] * sign
        for q, name in enumerate("xabc"):
            res[f"{prefix}_{name}"] = vals[4 + q] * sign
            res[f"{prefix}_{name}_idx"] = idx[q]
        res[f"{prefix}_kind"] = kind
        res[f"{prefix}_ready"] = ready
    return res

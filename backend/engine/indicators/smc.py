"""Smart Money Concepts: swings, market structure (BOS / CHoCH), liquidity sweeps,
equal highs/lows, fair value gaps, order blocks and divergences.

Every event is stamped at the bar where it becomes *knowable*. A swing pivot ``right`` bars
after the pivot, a structure break at the close that breaks, an FVG at the close of its
third candle. Backtests therefore see exactly what a live trader would have seen.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.indicators.base import (
    Category,
    Inputs,
    Output,
    OutputKind,
    Outputs,
    Param,
    indicator,
)
from engine.indicators.kernels import (
    confirmed_swings,
    divergences,
    fair_value_gaps,
    market_structure,
    order_blocks,
)
from engine.indicators.standard import atr_values, macd_values, rsi_values

EVENT = Output(OutputKind.EVENT)
FLAG = Output(OutputKind.FLAG)
PRICE = Output(OutputKind.PRICE)
SWING = Param(5, 2, 30, 1)  # bars each side of a pivot
# eq_tol only affects equal highs/lows; break_on doesn't affect them.
STRUCT_USES = ("length", "break_on")
EQ_USES = ("length", "eq_tol")
STRUCT_EVENT = Output(OutputKind.EVENT, uses=STRUCT_USES)


def swings(inp: Inputs, length: int) -> tuple[np.ndarray, ...]:
    return inp.cached(  # type: ignore[no-any-return]
        ("swings", length), lambda: confirmed_swings(inp.high, inp.low, length, length)
    )


def structure(inp: Inputs, length: int, break_on: str, eq_tol: float) -> tuple[Any, ...]:
    def build() -> tuple[Any, ...]:
        sh_e, sl_e, sh_p, sl_p, _, _ = swings(inp, length)
        return market_structure(  # type: ignore[no-any-return]
            inp.high,
            inp.low,
            inp.close,
            atr_values(inp, 14),
            sh_e,
            sl_e,
            sh_p,
            sl_p,
            break_on == "close",
            eq_tol,
        )

    return inp.cached(("structure", length, break_on, eq_tol), build)  # type: ignore[no-any-return]


@indicator(
    "swings",
    category=Category.STRUCTURE,
    params={"length": SWING},
    outputs={
        "high": Output(OutputKind.EVENT, description="swing high confirmed this bar"),
        "low": Output(OutputKind.EVENT, description="swing low confirmed this bar"),
        "last_high": Output(OutputKind.PRICE, description="latest confirmed swing high"),
        "last_low": Output(OutputKind.PRICE, description="latest confirmed swing low"),
        "hh": Output(OutputKind.EVENT, description="new swing high above the previous one"),
        "ll": Output(OutputKind.EVENT, description="new swing low below the previous one"),
    },
    description="Fractal swing points, emitted when confirmed (`length` bars after the pivot).",
)
def _swings(inp: Inputs, length: int) -> Outputs:
    sh_e, sl_e, sh_p, sl_p, _, _ = swings(inp, length)
    hh = sh_e & (sh_p > _prev_event_value(sh_p, sh_e))
    ll = sl_e & (sl_p < _prev_event_value(sl_p, sl_e))
    return {
        "high": sh_e,
        "low": sl_e,
        "last_high": _ffill(sh_p),
        "last_low": _ffill(sl_p),
        "hh": hh,
        "ll": ll,
    }


@indicator(
    "structure",
    category=Category.STRUCTURE,
    params={
        "length": SWING,
        "break_on": Param("close", choices=("close", "wick")),
        "eq_tol": Param(0.1, 0.01, 1.0, 0.01),
    },
    outputs={
        "trend": Output(
            OutputKind.STATE, values=(-1, 0, 1), description="-1 bear, 1 bull", uses=STRUCT_USES
        ),
        "bos_bull": STRUCT_EVENT,
        "bos_bear": STRUCT_EVENT,
        "choch_bull": STRUCT_EVENT,
        "choch_bear": STRUCT_EVENT,
        "sweep_high": Output(
            OutputKind.EVENT,
            description="wick above swing high, close back below",
            uses=STRUCT_USES,
        ),
        "sweep_low": Output(
            OutputKind.EVENT, description="wick below swing low, close back above", uses=STRUCT_USES
        ),
        "eqh": Output(
            OutputKind.EVENT, description="equal highs (within eq_tol x ATR)", uses=EQ_USES
        ),
        "eql": Output(OutputKind.EVENT, description="equal lows", uses=EQ_USES),
        "level_high": Output(
            OutputKind.PRICE, description="unbroken swing high being watched", uses=STRUCT_USES
        ),
        "level_low": Output(
            OutputKind.PRICE, description="unbroken swing low being watched", uses=STRUCT_USES
        ),
    },
    description="Market structure: BOS (continuation), CHoCH (reversal), sweeps, EQH/EQL.",
)
def _structure(inp: Inputs, length: int, break_on: str, eq_tol: float) -> Outputs:
    (trend, bos_b, bos_s, ch_b, ch_s, sw_h, sw_l, eqh, eql, lvl_h, lvl_l) = structure(
        inp, length, break_on, eq_tol
    )
    return {
        "trend": trend,
        "bos_bull": bos_b,
        "bos_bear": bos_s,
        "choch_bull": ch_b,
        "choch_bear": ch_s,
        "sweep_high": sw_h,
        "sweep_low": sw_l,
        "eqh": eqh,
        "eql": eql,
        "level_high": lvl_h,
        "level_low": lvl_l,
    }


def fvg_raw(inp: Inputs, min_size_atr: float) -> tuple[Any, ...]:
    return inp.cached(  # type: ignore[no-any-return]
        ("fvg", min_size_atr),
        lambda: fair_value_gaps(
            inp.high, inp.low, inp.close, atr_values(inp, 14), min_size_atr, 64
        ),
    )


@indicator(
    "fvg",
    category=Category.LIQUIDITY,
    params={"min_size_atr": Param(0.1, 0.0, 2.0, 0.05)},
    outputs={
        "new_bull": EVENT,
        "new_bear": EVENT,
        "touch_bull": Output(
            OutputKind.EVENT, description="price retraced into a live bullish FVG"
        ),
        "touch_bear": Output(
            OutputKind.EVENT, description="price retraced into a live bearish FVG"
        ),
        "bull_top": PRICE,
        "bull_bottom": PRICE,
        "bear_top": PRICE,
        "bear_bottom": PRICE,
        "dist_bull_atr": Output(OutputKind.RATIO, description="(close - nearest bull FVG top)/ATR"),
        "dist_bear_atr": Output(
            OutputKind.RATIO, description="(nearest bear FVG bottom - close)/ATR"
        ),
    },
    description="Fair value gaps (3-candle imbalances) with mitigation tracking.",
)
def _fvg(inp: Inputs, min_size_atr: float) -> Outputs:
    nb, ns, tb, ts, bt, bb, st, sb, _, _, _ = fvg_raw(inp, min_size_atr)
    atr = atr_values(inp, 14)
    return {
        "new_bull": nb,
        "new_bear": ns,
        "touch_bull": tb,
        "touch_bear": ts,
        "bull_top": bt,
        "bull_bottom": bb,
        "bear_top": st,
        "bear_bottom": sb,
        "dist_bull_atr": (inp.close - bt) / atr,
        "dist_bear_atr": (sb - inp.close) / atr,
    }


@indicator(
    "order_blocks",
    category=Category.LIQUIDITY,
    params={"length": SWING, "lookback": Param(10, 2, 50, 1)},
    outputs={
        "new_bull": EVENT,
        "new_bear": EVENT,
        "touch_bull": Output(OutputKind.EVENT, description="price traded into the live bullish OB"),
        "touch_bear": Output(OutputKind.EVENT, description="price traded into the live bearish OB"),
        "bull_top": PRICE,
        "bull_bottom": PRICE,
        "bear_top": PRICE,
        "bear_bottom": PRICE,
    },
    description="Order blocks: last opposite candle before a BOS/CHoCH, live until closed through.",
)
def _order_blocks(inp: Inputs, length: int, lookback: int) -> Outputs:
    st = structure(inp, length, "close", 0.1)
    bull_break = st[1] | st[3]
    bear_break = st[2] | st[4]
    nb, ns, tb, ts, bt, bb, sb_t, sb_b = order_blocks(
        inp.open, inp.high, inp.low, inp.close, bull_break, bear_break, lookback
    )
    return {
        "new_bull": nb,
        "new_bear": ns,
        "touch_bull": tb,
        "touch_bear": ts,
        "bull_top": bt,
        "bull_bottom": bb,
        "bear_top": sb_t,
        "bear_bottom": sb_b,
    }


@indicator(
    "divergence",
    ordered=(("min_dist", "max_dist"),),
    category=Category.DIVERGENCE,
    params={
        "osc": Param("rsi", choices=("rsi", "macd_hist")),
        "length": Param(5, 2, 20, 1),
        "min_dist": Param(5, 2, 50, 1),
        "max_dist": Param(60, 10, 200, 1),
    },
    outputs={
        "regular_bull": EVENT,
        "hidden_bull": EVENT,
        "regular_bear": EVENT,
        "hidden_bear": EVENT,
    },
    description="Price/oscillator divergences between confirmed pivots (RSI-14 or MACD histogram).",
)
def _divergence(inp: Inputs, osc: str, length: int, min_dist: int, max_dist: int) -> Outputs:
    if min_dist >= max_dist:
        raise ValueError("divergence: min_dist must be < max_dist")
    series = rsi_values(inp.close, 14) if osc == "rsi" else macd_values(inp.close, 12, 26, 9)[2]
    rb, hb, rs, hs = divergences(inp.high, inp.low, series, length, length, min_dist, max_dist)
    return {"regular_bull": rb, "hidden_bull": hb, "regular_bear": rs, "hidden_bear": hs}


# --------------------------------------------------------------------------- helpers
def _ffill(x: np.ndarray) -> np.ndarray:
    idx = np.where(~np.isnan(x), np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)
    return out


def _prev_event_value(values: np.ndarray, events: np.ndarray) -> np.ndarray:
    """At each event bar, the value of the *previous* event (NaN if none)."""
    filled = _ffill(np.where(events, values, np.nan))
    prev = np.concatenate([[np.nan], filled[:-1]])
    return np.where(events, prev, np.nan)

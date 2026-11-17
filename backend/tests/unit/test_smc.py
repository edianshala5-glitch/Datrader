"""Smart Money Concepts on hand-built price paths where every event's bar is known."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.enums import Timeframe
from engine.data.calendar import MarketCalendar
from engine.indicators import Inputs, compute
from engine.indicators.kernels import bars_since

CAL = MarketCalendar.xauusd()


def make(hlc: list[tuple[float, float, float]], opens: list[float] | None = None) -> Inputs:
    arr = np.array(hlc, dtype=float)
    h, lo, c = arr[:, 0], arr[:, 1], arr[:, 2]
    o = np.array(opens, float) if opens else np.concatenate([[c[0]], c[:-1]])
    n = len(c)
    return Inputs(
        ts=pd.date_range("2024-03-05 14:00", periods=n, freq="5min", tz="UTC"),
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=np.ones(n),
        spread=np.zeros(n),
        timeframe=Timeframe.M5,
        calendar=CAL,
    )


# Swing high 105 @3 (confirmed @5), swing low 100 @6 (confirmed @8), close 106 @9 breaks up.
BASE = [
    (101, 100, 100.5),
    (102, 101, 101.5),
    (103, 102, 102.5),
    (105, 103.5, 104),
    (104, 102, 102.5),
    (103, 101, 101.5),
    (102, 100, 100.5),
    (103, 101, 102),
    (104, 102, 103.5),
]
STRUCT = [
    *BASE,
    (106.5, 104, 106),  # 9: BOS up (first break from undefined trend)
    (107, 105, 106.5),  # 10: swing high 107 (confirmed @12)
    (106, 104, 104.5),  # 11
    (105, 102, 102.5),  # 12
    (103, 99, 99.5),  # 13: close below swing low 100 while bullish: CHoCH down
]


def idx(a: np.ndarray) -> list[int]:  # type: ignore[type-arg]
    return np.flatnonzero(a).tolist()


def test_swings_are_emitted_at_confirmation() -> None:
    sw = compute(make(STRUCT), "swings", {"length": 2})
    assert idx(sw["high"]) == [5, 12]  # pivots at 3 and 10, +2 bars
    assert idx(sw["low"]) == [8]  # pivot at 6
    assert np.isnan(sw["last_high"][4]) and sw["last_high"][5] == 105
    assert sw["last_high"][12] == 107 and sw["last_low"][13] == 100
    assert idx(sw["hh"]) == [12]


def test_bos_then_choch() -> None:
    st = compute(make(STRUCT), "structure", {"length": 2})
    assert idx(st["bos_bull"]) == [9]
    assert idx(st["choch_bear"]) == [13]
    assert not st["choch_bull"].any() and not st["bos_bear"].any()
    assert st["trend"].tolist() == [0] * 9 + [1] * 4 + [-1]
    assert np.isnan(st["level_high"][9])  # consumed by the break
    assert st["level_high"][12] == 107  # new swing high being watched


def test_wick_break_mode_and_sweep_then_break() -> None:
    # Close mode: the 105.5 wick at bar 9 is a sweep (close back below 105). The level
    # stays live, so bar 10 closing at 105.8 is still a genuine BOS. A second wick does
    # not re-fire the sweep.
    path = [*BASE, (105.5, 103, 104.8), (105.6, 104, 104.9), (106, 104, 105.8)]
    close_mode = compute(make(path), "structure", {"length": 2})
    assert idx(close_mode["sweep_high"]) == [9]
    assert idx(close_mode["bos_bull"]) == [11]
    # Wick mode: the first wick through the level is already the break.
    wick_mode = compute(make(path), "structure", {"length": 2, "break_on": "wick"})
    assert idx(wick_mode["bos_bull"]) == [9] and not wick_mode["sweep_high"].any()


def test_equal_highs() -> None:
    path = [*BASE, (105, 103, 104), (104, 102, 102.5), (103, 101, 101.5)]
    st = compute(make(path), "structure", {"length": 2})
    assert idx(st["eqh"]) == [11]  # second 105 swing high confirmed at 9 + 2


def test_order_block_on_bullish_break() -> None:
    ob = compute(make(STRUCT), "order_blocks", {"length": 2, "lookback": 10})
    assert idx(ob["new_bull"]) == [9]
    # last down-close candle before the break is bar 6 (open 101.5, close 100.5)
    assert (ob["bull_top"][9], ob["bull_bottom"][9]) == (102, 100)
    assert idx(ob["touch_bull"]) == [12]  # low 102 tags the top
    assert np.isnan(ob["bull_top"][13])  # close 99.5 < 100 invalidates it


def test_fair_value_gap_lifecycle() -> None:
    path = [
        (100, 99, 99.5),
        (103, 100.5, 102.5),
        (104, 101, 103.5),  # 2: low 101 > high[0] 100 → bullish FVG [100, 101]
        (103.5, 101.8, 102.8),
        (102, 100.6, 101.5),  # 4: trades into the gap → touch
        (101, 99.5, 99.8),  # 5: low through 100 → filled (dies), and a bearish FVG forms
        (101.2, 100.5, 100.9),
    ]
    f = compute(make(path), "fvg", {"min_size_atr": 0.0})
    assert idx(f["new_bull"]) == [2]
    assert (f["bull_top"][2], f["bull_bottom"][2]) == (101, 100)
    assert idx(f["touch_bull"]) == [4]
    assert np.isnan(f["bull_top"][5])
    assert idx(f["new_bear"]) == [5]  # high[5] 101 < low[3] 101.8


def test_regular_bullish_divergence() -> None:
    closes = np.concatenate(
        [
            np.linspace(120, 100, 20),  # fast selloff → RSI deeply oversold at low A (bar 19)
            np.linspace(101, 106, 6),  # rebound
            np.linspace(105.5, 99.5, 13),  # slow grind to a lower low B (bar 38), RSI higher
            np.linspace(100.5, 104, 6),
        ]
    )
    o = np.concatenate([[closes[0]], closes[:-1]])
    hlc = [(max(a, b) + 0.1, min(a, b) - 0.1, b) for a, b in zip(o, closes, strict=True)]
    d = compute(make(hlc), "divergence", {"length": 3, "min_dist": 5, "max_dist": 60})
    assert idx(d["regular_bull"]) == [38 + 3]
    assert not d["hidden_bull"].any() and not d["regular_bear"].any()


def test_bars_since() -> None:
    ev = np.array([False, True, False, False, True, False])
    assert bars_since(ev).tolist() == [-1, 0, 1, 2, 0, 1]

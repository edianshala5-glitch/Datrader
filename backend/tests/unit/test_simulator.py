"""Execution rules on hand-built bars with hand-computed fills and P&L.

Defaults: bid OHLC as given, ask = bid + 0.20, contract 100, equity 100k, risk 1%
(= $1,000), stop distance 2.0 → 5 lots. Zero commission/slippage unless a test says so.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from engine.backtest.runner import CostConfig, execution_prices, simulate_signals
from engine.backtest.simulator import EXIT_REASONS, TRADE_FIELDS
from engine.strategy.compiler import Signals

FREE = CostConfig(commission_per_lot_rt=0.0, slippage_ticks=0.0, min_spread=0.0)


def bars(ohlc: list[tuple[float, float, float, float]], spread: float = 0.2) -> pd.DataFrame:
    a = np.array(ohlc, dtype=float)
    df = pd.DataFrame(
        {"ts": pd.date_range("2024-03-05 14:00", periods=len(a), freq="15min", tz="UTC")}
    )
    for i, f in enumerate(("open", "high", "low", "close")):
        df[f"bid_{f}"] = a[:, i]
        df[f"ask_{f}"] = a[:, i] + spread
    df["volume"] = 1.0
    df["tick_count"] = 1
    return df


def signals(
    n: int, *, long_at: tuple[int, ...] = (), short_at: tuple[int, ...] = (), **kw: object
) -> Signals:
    nan = np.full(n, np.nan)
    false = np.zeros(n, dtype=bool)
    el, es = false.copy(), false.copy()
    el[list(long_at)] = True
    es[list(short_at)] = True
    base = Signals(
        entry_long=el,
        entry_short=es,
        exit_long=false.copy(),
        exit_short=false.copy(),
        stop_level_long=nan,
        stop_level_short=nan,
        stop_dist_long=np.full(n, 2.0),
        stop_dist_short=np.full(n, 2.0),
        target_dist_long=nan,
        target_dist_short=nan,
        target_rr=2.0,
        trail_dist=nan,
        trail_activate_r=0.0,
        max_bars=0,
        flat_at=false.copy(),
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def run(b: pd.DataFrame, s: Signals, costs: CostConfig = FREE, risk: float = 0.01) -> pd.DataFrame:
    raw, equity, _ = simulate_signals(execution_prices(b, costs.min_spread), s, costs, risk, 50.0)
    t = pd.DataFrame(raw, columns=list(TRADE_FIELDS))
    t["reason"] = [EXIT_REASONS[int(i)] for i in t["reason"]]
    t.attrs["equity"] = equity
    return t


T0 = (100, 100.5, 99.5, 100)  # signal bar
T1 = (100, 101, 99.8, 100.5)  # entry bar: long fills at ask open 100.20 → stop 98.20, target 104.20


def test_long_target_hit() -> None:
    t = run(
        bars([T0, T1, (100.5, 104.5, 100.2, 104), (104, 104.5, 103, 104)]), signals(4, long_at=(0,))
    )
    [tr] = t.itertuples()
    assert (tr.entry_idx, tr.exit_idx, tr.reason) == (1, 2, "target")
    assert (tr.entry_price, tr.stop_price, tr.target_price, tr.lots) == pytest.approx(
        (100.2, 98.2, 104.2, 5.0)
    )
    assert tr.exit_price == pytest.approx(104.2)
    assert tr.pnl == pytest.approx(2000.0) and tr.r == pytest.approx(2.0)


def test_bar_touching_stop_and_target_counts_as_stop() -> None:
    t = run(bars([T0, T1, (100.5, 104.5, 98.0, 104), T1]), signals(4, long_at=(0,)))
    assert t["reason"].tolist() == ["stop"]
    assert t["pnl"].iloc[0] == pytest.approx(-1000.0) and t["r"].iloc[0] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open() -> None:
    t = run(bars([T0, T1, (97.0, 97.5, 96.5, 97), T1]), signals(4, long_at=(0,)))
    assert t["exit_price"].iloc[0] == pytest.approx(97.0)
    assert t["r"].iloc[0] == pytest.approx(-1.6)  # worse than -1R: gap risk is real


def test_gap_through_target_fills_at_better_open() -> None:
    t = run(bars([T0, T1, (105.0, 106, 104.8, 105.5), T1]), signals(4, long_at=(0,)))
    assert t["reason"].tolist() == ["target"] and t["exit_price"].iloc[0] == pytest.approx(105.0)


def test_short_uses_ask_for_stops() -> None:
    # Short fills at bid open 100.00; stop 102.00 checked against ASK high (bid + 0.2).
    # Ask high is 102.1, which is > 102 even though bid high (101.9) is not.
    t = run(bars([T0, T1, (100.5, 101.9, 100.0, 101), T1]), signals(4, short_at=(0,)))
    assert t["reason"].tolist() == ["stop"]
    assert (t["entry_price"].iloc[0], t["exit_price"].iloc[0]) == pytest.approx((100.0, 102.0))


def test_trailing_stop() -> None:
    s = signals(5, long_at=(0,), target_rr=np.nan, trail_dist=np.full(5, 1.0), trail_activate_r=0.5)
    b = bars([T0, T1, (100.5, 102.0, 100.6, 101.8), (101.8, 101.9, 100.9, 101), T1])
    [tr] = run(b, s).itertuples()
    # After bar 2 (MFE 1.8 ≥ 0.5R = 1.0) the stop trails to 102 - 1 = 101. Bar 3 tags it.
    assert (tr.reason, tr.exit_idx) == ("trailing_stop", 3)
    assert tr.exit_price == pytest.approx(101.0) and tr.pnl == pytest.approx(400.0)
    assert tr.mfe_r == pytest.approx(0.9)


def test_max_bars_and_signal_exits_fill_next_open() -> None:
    flat_bar = (100.5, 100.8, 100.3, 100.6)
    b = bars([T0, T1, flat_bar, (100.7, 100.9, 100.4, 100.8), flat_bar])
    t = run(b, signals(5, long_at=(0,), max_bars=2))
    assert t[["exit_idx", "reason"]].values.tolist() == [[3, "time"]]
    assert t["exit_price"].iloc[0] == pytest.approx(100.7)

    xl = np.zeros(5, dtype=bool)
    xl[2] = True
    t = run(b, signals(5, long_at=(0,), exit_long=xl))
    assert t[["exit_idx", "reason"]].values.tolist() == [[3, "signal"]]


def test_session_flat_exits_at_close_and_blocks_entries() -> None:
    flat = np.zeros(5, dtype=bool)
    flat[2] = True
    b = bars([T0, T1, (100.5, 100.8, 100.3, 100.6), T1, T1])
    t = run(b, signals(5, long_at=(0, 1, 2), flat_at=flat))
    # Entered at bar 1, forced out at bar 2's close. The signal on the flat bar (2) and the
    # entry into bar 2 (signal at 1) are both refused.
    assert t[["entry_idx", "exit_idx", "reason"]].values.tolist() == [[1, 2, "session_close"]]
    assert t["exit_price"].iloc[0] == pytest.approx(100.6)


def test_open_position_closed_at_end_of_data() -> None:
    t = run(bars([T0, T1, (100.5, 101, 100.2, 100.9)]), signals(3, long_at=(0,)))
    assert t["reason"].tolist() == ["end_of_data"] and t["exit_price"].iloc[0] == pytest.approx(
        100.9
    )


def test_conflicting_signals_skip() -> None:
    assert run(bars([T0, T1, T1]), signals(3, long_at=(0,), short_at=(0,))).empty


def test_costs_slippage_commission_and_spread_floor() -> None:
    costs = CostConfig(commission_per_lot_rt=7.0, slippage_ticks=2.0, min_spread=0.5)
    b = bars([T0, T1, (100.5, 104.8, 100.4, 104), T1], spread=0.0)
    [tr] = run(b, signals(4, long_at=(0,)), costs).itertuples()
    # ask = max(bid, bid + 0.5 floor) → fill 100.5 + 0.02 slippage
    assert tr.entry_price == pytest.approx(100.52)
    assert tr.target_price == pytest.approx(104.52)  # bid high 104.8 reaches it
    assert tr.lots == pytest.approx(5.0)
    assert tr.pnl == pytest.approx(4.0 * 5 * 100 - 35.0)  # $35 = 7 x 5 lots
    assert tr.commission == pytest.approx(35.0)


def test_position_sizing_rounds_down_and_respects_min_lot() -> None:
    b = bars([T0, T1, (100.5, 101, 100, 100.8)])
    t = run(b, signals(3, long_at=(0,), stop_dist_long=np.full(3, 3.0)))
    assert t["lots"].iloc[0] == pytest.approx(3.33)  # $1000 / (3.0 x 100) = 3.333 → 3.33
    assert run(b, signals(3, long_at=(0,)), risk=0.000001).empty  # < 0.01 lot → skipped


def test_swing_stop_level_and_invalid_risk() -> None:
    lvl = np.full(4, np.nan)
    lvl[0] = 99.2
    b = bars([T0, T1, (100.5, 104, 99.0, 100), T1])
    [tr] = run(b, signals(4, long_at=(0,), stop_level_long=lvl, target_rr=np.nan)).itertuples()
    assert tr.stop_price == pytest.approx(99.2) and tr.reason == "stop"
    assert tr.lots == pytest.approx(10.0)  # risk 1.0 per oz → $1000 / 100
    lvl[0] = 101.0  # stop above the fill → no trade
    assert run(b, signals(4, long_at=(0,), stop_level_long=lvl)).empty


def test_equity_marks_to_market() -> None:
    t = run(bars([T0, T1, (100.5, 101, 100.2, 100.9)]), signals(3, long_at=(0,)))
    eq = t.attrs["equity"]
    assert eq[0] == 100_000
    assert eq[1] == pytest.approx(100_000 + (100.5 - 100.2) * 500)  # bid close vs fill
    assert eq[2] == pytest.approx(100_000 + t["pnl"].iloc[0])

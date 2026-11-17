"""Standard indicators vs independent textbook implementations and hand-computed values."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from app.models.enums import Timeframe
from app.services.seed import XAUUSD
from engine.data.calendar import MarketCalendar
from engine.data.resample import resample_bars
from engine.data.sources.synthetic import generate_m1
from engine.indicators import Inputs, compute
from engine.indicators.standard import ema, rma

CAL = MarketCalendar.xauusd()


@pytest.fixture(scope="module")
def inp() -> Inputs:
    m1 = generate_m1(
        datetime(2024, 3, 3, tzinfo=UTC), datetime(2024, 3, 23, tzinfo=UTC), CAL, seed=3
    )
    m5 = resample_bars(m1, Timeframe.M5, CAL)
    return Inputs.from_bars(m5, Timeframe.M5, CAL, XAUUSD.model_dump(mode="json")["sessions"])


def ref_rsi(c: np.ndarray, n: int) -> list[float]:  # type: ignore[type-arg]
    out = [math.nan] * len(c)
    gains = [max(c[i] - c[i - 1], 0.0) for i in range(1, len(c))]
    losses = [max(c[i - 1] - c[i], 0.0) for i in range(1, len(c))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, len(c)):
        ag = (ag * (n - 1) + gains[i - 1]) / n
        al = (al * (n - 1) + losses[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def ref_atr(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int) -> list[float]:  # type: ignore[type-arg]
    tr = [h[0] - lo[0]] + [
        max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))
    ]
    out = [math.nan] * len(c)
    a = sum(tr[:n]) / n
    out[n - 1] = a
    for i in range(n, len(c)):
        a = (a * (n - 1) + tr[i]) / n
        out[i] = a
    return out


def test_ema_and_rma_are_sma_seeded() -> None:
    x = np.array([1.0, 2, 3, 4, 5])
    np.testing.assert_allclose(ema(x, 3), [np.nan, np.nan, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(rma(x, 2), [np.nan, 1.5, 2.25, 3.125, 4.0625])
    # leading NaNs are skipped, not propagated
    np.testing.assert_allclose(ema(np.array([np.nan, 1.0, 2, 3]), 2), [np.nan, np.nan, 1.5, 2.5])


def test_rsi_matches_wilder_reference(inp: Inputs) -> None:
    ours = compute(inp, "rsi", {"length": 14})["value"]
    np.testing.assert_allclose(ours, ref_rsi(inp.close, 14), rtol=1e-10, equal_nan=True)
    assert np.nanmin(ours) >= 0 and np.nanmax(ours) <= 100


def test_rsi_extremes() -> None:
    up = np.arange(1.0, 40.0)
    rsi_up = compute(_inputs_from_close(up), "rsi")["value"]
    assert np.all(rsi_up[14:] == 100.0)
    rsi_down = compute(_inputs_from_close(up[::-1].copy()), "rsi")["value"]
    assert np.all(rsi_down[14:] == 0.0)


def test_atr_matches_wilder_reference(inp: Inputs) -> None:
    ours = compute(inp, "atr", {"length": 14})["value"]
    np.testing.assert_allclose(
        ours, ref_atr(inp.high, inp.low, inp.close, 14), rtol=1e-10, equal_nan=True
    )


def test_macd_bbands_stoch(inp: Inputs) -> None:
    m = compute(inp, "macd")
    np.testing.assert_allclose(m["line"], ema(inp.close, 12) - ema(inp.close, 26), equal_nan=True)
    np.testing.assert_allclose(m["hist"], m["line"] - ema(m["line"], 9), equal_nan=True)

    bb = compute(inp, "bbands", {"length": 20, "mult": 2.0})
    window = inp.close[100:120]
    assert bb["mid"][119] == pytest.approx(window.mean())
    assert bb["upper"][119] == pytest.approx(window.mean() + 2 * window.std(ddof=0))

    st = compute(inp, "stoch")
    assert np.nanmin(st["k"]) >= 0 and np.nanmax(st["k"]) <= 100


def test_adx_detects_trend() -> None:
    trend = np.linspace(2000, 2100, 300) + np.sin(np.arange(300)) * 0.5
    adx = compute(_inputs_from_close(trend), "adx")
    assert adx["adx"][-1] > 40
    assert adx["plus_di"][-1] > adx["minus_di"][-1]


def test_vwap_resets_at_ny_roll(inp: Inputs) -> None:
    vw = compute(inp, "vwap")["value"]
    ny = inp.ts.tz_convert("America/New_York")
    first_of_day = np.flatnonzero((ny.hour == 18) & (ny.minute == 0))
    assert len(first_of_day) >= 10
    tp = (inp.high + inp.low + inp.close) / 3
    np.testing.assert_allclose(vw[first_of_day], tp[first_of_day])


def test_time_context(inp: Inputs) -> None:
    t = compute(inp, "time")
    ny = inp.ts.tz_convert("America/New_York")
    i = int(np.flatnonzero((ny.dayofweek == 0) & (ny.hour == 18))[0])  # Monday 18:00 NY
    assert t["weekday"][i] == 1  # belongs to Tuesday's session
    assert t["minutes_into_day"][i] == 0
    j = int(np.flatnonzero((ny.dayofweek == 1) & (ny.hour == 9) & (ny.minute == 30))[0])
    assert (t["weekday"][j], t["hour_ny"][j], t["minute_ny"][j]) == (1, 9, 570)

    s = compute(inp, "sessions")
    assert s["ny_am_kz"][j] and s["new_york"][j] and s["london"][j] and s["london_ny_overlap"][j]
    assert not s["asia"][j]


def test_param_validation(inp: Inputs) -> None:
    with pytest.raises(ValueError, match="outside"):
        compute(inp, "rsi", {"length": 1})
    with pytest.raises(ValueError, match="unknown params"):
        compute(inp, "rsi", {"period": 14})
    with pytest.raises(ValueError, match="fast must be < slow"):
        compute(inp, "macd", {"fast": 30, "slow": 20})
    with pytest.raises(ValueError, match="not in"):
        compute(inp, "structure", {"break_on": "body"})
    with pytest.raises(ValueError, match="integer"):
        compute(inp, "ema", {"length": 2.5})


def _inputs_from_close(close: np.ndarray) -> Inputs:  # type: ignore[type-arg]
    n = len(close)
    o = np.concatenate([[close[0]], close[:-1]])
    return Inputs(
        ts=pd.date_range("2024-03-05 14:00", periods=n, freq="5min", tz="UTC"),
        open=o,
        high=np.maximum(o, close) + 0.1,
        low=np.minimum(o, close) - 0.1,
        close=close.astype(float),
        volume=np.ones(n),
        spread=np.zeros(n),
        timeframe=Timeframe.M5,
        calendar=CAL,
    )

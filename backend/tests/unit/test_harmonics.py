"""Harmonic XABCD patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.enums import Timeframe
from engine.indicators import Inputs, compute
from engine.strategy.explain import explain
from engine.strategy.setups import harmonic_setup


def _path(points: list[tuple[float, int]], start: float) -> list[tuple[float, float, float, float]]:
    """Straight legs between pivot prices: (target price, bars to get there)."""
    rows, price = [], start
    for target, bars in points:
        step = (target - price) / bars
        for _ in range(bars):
            o, c = price, price + step
            rows.append((o, max(o, c) + 0.05, min(o, c) - 0.05, c))
            price = c
    return rows


def _inputs(rows: list[tuple[float, float, float, float]]) -> Inputs:
    o, h, lo, c = (np.array(x) for x in zip(*rows, strict=True))
    ts = pd.date_range("2024-01-02 03:00", periods=len(rows), freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {
            "ts": ts,
            **{
                f"bid_{k}": v - 0.15
                for k, v in (("open", o), ("high", h), ("low", lo), ("close", c))
            },
            **{
                f"ask_{k}": v + 0.15
                for k, v in (("open", o), ("high", h), ("low", lo), ("close", c))
            },
            "volume": np.ones(len(rows)),
            "tick_count": np.ones(len(rows)),
        }
    )
    return Inputs.from_bars(bars, Timeframe.M5)


# X 100 -> A 110 -> B 103.82 (0.618 XA) -> C 107.64 (0.618 AB) -> D ~102.1 (0.786 XA)
GARTLEY = [(100.0, 8), (110.0, 10), (103.82, 8), (107.64, 6), (102.1, 8), (106.0, 8)]


def test_bullish_gartley_prz() -> None:
    rows = _path(GARTLEY, 105.0)
    out = compute(_inputs(rows), "harmonics", {"length": 3})
    armed = np.flatnonzero(out["bull_ready"])
    assert armed.size, "the Gartley should be detected"
    i = armed[0]
    assert i >= (8 + 10 + 8 + 6 - 1) + 3  # only once pivot C (bar 31) is confirmed, 3 bars later
    assert out["bull_kind"][i] == 1  # gartley
    assert 101.6 < out["bull_prz_low"][i] < out["bull_prz_high"][i] < 102.6
    assert np.isclose(out["bull_x"][i], 99.95) and np.isclose(out["bull_a"][i], 110.05)
    assert out["bull_stop"][i] == out["bull_x"][i]  # D inside XA: stop beyond X
    assert out["bull_prz_high"][i] < out["bull_target"][i] < out["bull_a"][i]
    # The later legs (110 -> 103.8 -> 107.7 -> 102.1) really are a bearish Cypher:
    # B = 0.62 of XA, C = 1.27 x XA beyond A. Nothing bearish before that.
    bear = np.flatnonzero(out["bear_ready"])
    assert bear.size and (out["bear_kind"][bear] == 5).all() and bear[0] > i


def test_bad_ratios_are_not_a_pattern() -> None:
    shallow = [(100.0, 8), (110.0, 10), (107.0, 6), (109.0, 6), (104.0, 8), (106.0, 8)]  # B = 0.3
    out = compute(_inputs(_path(shallow, 105.0)), "harmonics", {"length": 3})
    assert not out["bull_ready"].any()


def test_bearish_mirror_and_pattern_filter() -> None:
    mirrored = [(200.0 - p, n) for p, n in GARTLEY]
    out = compute(_inputs(_path(mirrored, 95.0)), "harmonics", {"length": 3})
    i = np.flatnonzero(out["bear_ready"])[0]
    assert out["bear_kind"][i] == 1
    assert 97.4 < out["bear_prz_low"][i] < out["bear_prz_high"][i] < 98.4
    assert out["bear_stop"][i] == out["bear_x"][i] > out["bear_prz_high"][i]
    only_bat = compute(_inputs(_path(GARTLEY, 105.0)), "harmonics", {"length": 3, "pattern": "bat"})
    assert not only_bat["bull_ready"].any()


def test_harmonics_never_use_future_bars() -> None:
    rows = _path(GARTLEY, 105.0)
    full = compute(_inputs(rows), "harmonics", {"length": 3})
    for cut in range(20, len(rows)):
        part = compute(_inputs(rows[:cut]), "harmonics", {"length": 3})
        for k in ("bull_ready", "bull_prz_low", "bull_prz_high", "bull_kind", "bull_target"):
            np.testing.assert_array_equal(part[k], full[k][:cut], err_msg=f"{k} at {cut}")


def test_harmonic_setup_genome() -> None:
    g = harmonic_setup(Timeframe.M5, "rejection", params={"pattern": "all"})
    text = explain(g)
    assert "bullish harmonic pattern armed" in text and "bullish rejection candle" in text
    assert g.zone_feature() is not None and g.zone_feature().indicator == "harmonics"  # type: ignore[union-attr]

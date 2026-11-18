"""Supply/demand zones, confirmation candles and the zone stop/target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.enums import Timeframe
from engine.indicators import Inputs, compute
from engine.strategy.compiler import FeatureProvider, compile_genome
from engine.strategy.explain import explain
from engine.strategy.genome import Genome
from engine.strategy.setups import sd_zone_setup

WARM = [(100.5, 101.0, 100.0, 100.5)] * 20
SETUP = [
    (101.0, 101.5, 100.5, 101.0),
    (101.0, 101.5, 100.5, 101.0),
    (101.0, 101.5, 100.5, 101.0),
    (101.0, 101.5, 100.5, 101.0),
    (101.0, 103.0, 100.8, 102.0),  # swing high 103 (pivot)
    (102.0, 102.2, 100.8, 101.0),
    (101.0, 101.2, 99.8, 100.0),
    (100.0, 100.2, 99.0, 99.5),  # base: zone = 99.0 .. 100.0 (body top)
    (99.6, 102.2, 99.5, 102.0),
    (102.0, 104.8, 101.9, 104.5),  # close above 103: bullish BOS
    (104.5, 105.5, 104.2, 105.0),  # impulse high 105.5 = target
]
SLOW = [  # small candles, takes time, one stair step (bar 13 is a minor lower high)
    (105.0, 105.1, 104.2, 104.4),
    (104.4, 104.5, 103.7, 103.9),
    (103.9, 104.6, 103.8, 104.2),
    (104.2, 104.3, 103.3, 103.5),
    (103.5, 103.6, 102.6, 102.8),
    (102.8, 102.9, 101.9, 102.1),
    (102.1, 102.2, 101.2, 101.4),
    (101.4, 101.5, 100.4, 100.6),
]
TOUCH = [(100.6, 100.7, 99.8, 100.2), (100.2, 100.6, 100.0, 100.4), (100.4, 100.8, 100.2, 100.6)]
FAST = [(105.0, 105.1, 100.4, 100.6)]  # one big candle straight back to the zone


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    o, h, lo, c = (np.array(x, dtype=float) for x in zip(*rows, strict=True))
    ts = pd.date_range("2024-01-02 03:00", periods=len(rows), freq="15min", tz="UTC")
    half = 0.15
    return pd.DataFrame(
        {
            "ts": ts,
            "bid_open": o - half,
            "bid_high": h - half,
            "bid_low": lo - half,
            "bid_close": c - half,
            "ask_open": o + half,
            "ask_high": h + half,
            "ask_low": lo + half,
            "ask_close": c + half,
            "volume": np.ones(len(rows)),
            "tick_count": np.ones(len(rows)),
        }
    )


PARAMS = {"length": 2, "max_candle_atr": 1.5, "approach_bars": 4, "min_steps": 1}


def _zones(rows: list[tuple[float, float, float, float]]) -> dict[str, np.ndarray]:
    inp = Inputs.from_bars(_bars(rows), Timeframe.M15)
    return compute(inp, "sd_zones", PARAMS)


def test_demand_zone_after_bos_and_slow_pullback() -> None:
    z = _zones(WARM + SETUP + SLOW + TOUCH)
    bos, last_slow, touch = 20 + 9, 20 + len(SETUP) + len(SLOW) - 1, 20 + len(SETUP) + len(SLOW)
    assert np.isnan(z["demand_low"][bos - 1])  # nothing before the break
    assert z["demand_low"][bos] == 99.0 and z["demand_high"][bos] == 100.0
    assert not z["demand_ready"][bos]  # no pullback yet
    assert z["demand_target"][last_slow] == 105.5
    assert z["demand_ready"][last_slow]  # slow, stair-stepped approach
    assert z["demand_ready"][touch]  # verdict frozen on the approach before the touch
    assert np.isnan(z["demand_low"][touch + 2])  # first touch only (touch_bars=2)
    assert not z["supply_ready"].any()


def test_fast_pullback_is_not_ready() -> None:
    z = _zones(WARM + SETUP + FAST + TOUCH)
    assert not z["demand_ready"].any()


def test_zone_values_never_use_future_bars() -> None:
    rows = WARM + SETUP + SLOW + TOUCH
    full = _zones(rows)
    for cut in range(len(WARM) + 5, len(rows)):
        part = _zones(rows[:cut])
        for k in ("demand_low", "demand_high", "demand_target", "demand_ready"):
            np.testing.assert_array_equal(part[k], full[k][:cut], err_msg=f"{k} at {cut}")


def test_candle_patterns() -> None:
    rows = [
        (100.0, 100.2, 99.0, 100.1),  # bullish pin bar: long lower wick
        (100.1, 100.3, 99.9, 100.0),
        (100.0, 100.1, 99.6, 99.7),  # bearish
        (99.6, 100.4, 99.5, 100.3),  # bullish engulfing
        (100.3, 100.6, 100.2, 100.5),  # 2nd up
        (100.5, 100.9, 100.4, 100.8),  # 3rd up: three rising bullish candles
    ]
    out = compute(Inputs.from_bars(_bars(rows), Timeframe.M1), "candles", {"wick_ratio": 2.0})
    assert out["bull_rejection"][0] and not out["bear_rejection"][0]
    assert out["bull_engulfing"][3] and not out["bull_engulfing"][4]
    assert out["bull_three"][5] and not out["bull_three"][4]
    assert out["bull_any"][[0, 3, 5]].all()


def test_setup_genome_compiles_with_zone_stop_and_target() -> None:
    g = sd_zone_setup(Timeframe.M15, "any", zone_params=PARAMS, min_stop_pct=0.0, max_stop_pct=None)
    assert Genome.model_validate(g.canonical()) == g
    text = explain(g)
    assert "demand zone ready" in text and "stop beyond the zone" in text
    # Run on the zone timeframe itself (no HTF) to check the stop/target arrays.
    g15 = Genome.model_validate(
        {
            **g.canonical(),
            "entry_long": {
                "type": "group",
                "op": "all",
                "conditions": [
                    {
                        "type": "flag",
                        "feature": {
                            "indicator": "sd_zones",
                            "output": "demand_ready",
                            "params": PARAMS,
                        },
                    }
                ],
            },
            "entry_short": None,
        }
    )
    sig = compile_genome(g15, FeatureProvider(_bars(WARM + SETUP + SLOW), Timeframe.M15, None))
    i = len(WARM) + len(SETUP) + len(SLOW) - 1
    assert sig.entry_long[i]
    assert np.isclose(sig.stop_level_long[i], 99.0 - 0.1 * 1.0)  # zone low - 10% of height
    assert np.isclose(sig.target_dist_long[i], 105.5 - 100.6)  # impulse high - close


def test_zone_stop_requires_a_zone_feature() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="needs an sd_zones feature"):
        Genome.model_validate(
            {
                "entry_long": {
                    "type": "group",
                    "op": "all",
                    "conditions": [
                        {"type": "flag", "feature": {"indicator": "candles", "output": "bull_any"}}
                    ],
                },
                "exit": {"stop": {"type": "zone"}, "target": {"type": "zone"}},
            }
        )


def test_tracking_several_zones_finds_more_setups() -> None:
    rng = np.random.default_rng(7)
    n = 6000
    close = 2000 + np.cumsum(rng.normal(0, 1.0, n))
    opn = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0, 0.6, n))
    rows = list(
        zip(
            opn,
            np.maximum(opn, close) + wick,
            np.minimum(opn, close) - wick,
            close,
            strict=True,
        )
    )
    inp = Inputs.from_bars(_bars(rows), Timeframe.M15)
    one = compute(inp, "sd_zones", {"max_zones": 1})
    three = compute(inp, "sd_zones", {"max_zones": 3})
    assert np.isfinite(three["demand_low"]).sum() > np.isfinite(one["demand_low"]).sum()
    assert three["demand_ready"].sum() >= one["demand_ready"].sum()
    with np.errstate(invalid="ignore"):  # every zone sits below its target
        ok = ~np.isfinite(three["demand_high"]) | (three["demand_high"] < three["demand_target"])
    assert ok.all()


def test_zone_stop_minimum_and_maximum_distance() -> None:
    base = sd_zone_setup(Timeframe.M15, "any", zone_params=PARAMS).canonical()
    entry = {
        "type": "group",
        "op": "all",
        "conditions": [
            {
                "type": "flag",
                "feature": {"indicator": "sd_zones", "output": "demand_ready", "params": PARAMS},
            }
        ],
    }
    bars = _bars(WARM + SETUP + SLOW)
    i = len(WARM) + len(SETUP) + len(SLOW) - 1  # close 100.6, zone stop 98.9

    def compiled(min_pct: float, max_pct: float | None):  # type: ignore[no-untyped-def]
        g = Genome.model_validate(
            {
                **base,
                "entry_long": entry,
                "entry_short": None,
                "exit": {
                    **base["exit"],
                    "stop": {
                        "type": "zone",
                        "buffer_frac": 0.1,
                        "min_pct": min_pct,
                        "max_pct": max_pct,
                    },
                    "target": {"type": "zone", "min_rr": 2.0},
                },
            }
        )
        return compile_genome(g, FeatureProvider(bars, Timeframe.M15, None))

    wide = compiled(3.0, None)  # 3% minimum: stop pushed to 100.6 * 0.97
    assert np.isclose(wide.stop_level_long[i], 100.6 * 0.97)
    assert not wide.entry_long[i]  # target 105.5 is < 2R away with a 3% stop
    assert not compiled(0.0, 1.0).entry_long[i]  # zone stop is 1.7% away: too wide
    assert compiled(0.0, 2.0).entry_long[i]  # 4.9 to target vs 1.7 risk: 2.9R

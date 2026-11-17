"""Genome validation, compilation, explanation, metrics, and end-to-end backtest integrity."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.models.enums import Timeframe
from app.services.seed import XAUUSD
from engine.backtest.metrics import compute_metrics, max_drawdown
from engine.backtest.runner import CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.data.resample import resample_bars
from engine.data.sources.synthetic import generate_m1
from engine.strategy.compiler import FeatureProvider, compile_genome, evaluate
from engine.strategy.explain import explain
from engine.strategy.genome import Genome, Group
from engine.strategy.templates import TEMPLATES, all_templates, const, f, feat, group

CAL = MarketCalendar.xauusd()
SESSIONS = XAUUSD.model_dump(mode="json")["sessions"]
STOP = {"stop": {"type": "atr", "mult": 1.5}}


def genome(**kw: object) -> Genome:
    return Genome.model_validate({"exit": STOP, **kw})


@pytest.fixture(scope="module")
def m1() -> pd.DataFrame:
    return generate_m1(
        datetime(2024, 1, 7, tzinfo=UTC), datetime(2024, 4, 7, tzinfo=UTC), CAL, seed=5
    )


@pytest.fixture(scope="module")
def m15(m1: pd.DataFrame) -> pd.DataFrame:
    return resample_bars(m1, Timeframe.M15, CAL)


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (
            group(
                {"type": "compare", "left": feat("ema", "value"), "op": ">", "right": const(2000)}
            ),
            "price-level",
        ),
        (
            group(
                {"type": "compare", "left": feat("rsi", "value"), "op": ">", "right": const(140)}
            ),
            "outside",
        ),
        (
            group(
                {
                    "type": "compare",
                    "left": feat("rsi", "value"),
                    "op": ">",
                    "right": feat("ema", "value"),
                }
            ),
            "price-level",
        ),
        (
            group(
                {
                    "type": "compare",
                    "left": feat("rsi", "value"),
                    "op": ">",
                    "right": feat("macd", "hist"),
                }
            ),
            "cannot compare",
        ),
        (group({"type": "event", "feature": f("rsi", "value")}), "not an event"),
        (group({"type": "state", "feature": f("structure", "trend"), "values": [2]}), "not in"),
        (group({"type": "flag", "feature": f("rsi", "value")}), "not a flag"),
        (group({"type": "event", "feature": f("nope", "x")}), "unknown indicator"),
        (group({"type": "event", "feature": f("structure", "choch_bull", length=99)}), "outside"),
        (group({"type": "range", "feature": f("rsi", "value"), "lo": 60, "hi": 40}), "lo > hi"),
    ],
)
def test_invalid_genomes_are_rejected(entry: dict, match: str) -> None:  # type: ignore[type-arg]
    with pytest.raises((ValidationError, KeyError), match=match):
        genome(entry_long=entry)


def test_structural_limits() -> None:
    with pytest.raises(ValidationError, match="entry_long and/or entry_short"):
        genome()
    deep = group(group(group(group({"type": "event", "feature": f("structure", "bos_bull")}))))
    with pytest.raises(ValidationError, match="deeper"):
        genome(entry_long=deep)
    many = group(
        *[{"type": "event", "feature": f("structure", "bos_bull"), "within": i} for i in range(8)]
    )
    with pytest.raises(ValidationError, match="exceed"):
        genome(entry_long=many, entry_short=many)


def test_canonical_roundtrip_keeps_null_target() -> None:
    g = genome(
        entry_long=group({"type": "event", "feature": f("structure", "bos_bull")}),
        exit={**STOP, "target": None},
    )
    again = Genome.model_validate(g.canonical())
    assert again == g and again.exit.target is None


def test_templates_validate_and_explain() -> None:
    gs = all_templates()
    assert set(gs) == set(TEMPLATES)
    text = explain(gs["smc_choch_fvg"])
    assert "bullish CHoCH within the last 12 bars" in text and "NY AM kill zone" in text
    assert "flat before 17:00 NY" in text
    assert "trail 2.5x ATR after +1R" in explain(gs["ema_pullback"])


# --------------------------------------------------------------------------- compiler
def test_condition_semantics(m15: pd.DataFrame) -> None:
    fp = FeatureProvider(m15, Timeframe.M15, CAL, SESSIONS)
    rsi = fp.get(
        Genome.model_validate(
            {
                "exit": STOP,
                "entry_long": group(
                    {"type": "compare", "left": feat("rsi", "value"), "op": "<", "right": const(30)}
                ),
            }
        )
        .entry_long.conditions[0]
        .left.feature
    )  # type: ignore[union-attr]
    g = genome(
        entry_long=group(
            {
                "type": "cross",
                "left": feat("rsi", "value"),
                "right": const(30),
                "direction": "above",
            }
        )
    )
    cross = evaluate(g.entry_long, fp)
    expected = (rsi > 30) & (np.concatenate([[np.nan], rsi[:-1]]) <= 30)
    np.testing.assert_array_equal(cross, expected)
    assert not cross[:15].any()  # NaN warm-up never triggers

    ev = genome(
        entry_long=group({"type": "event", "feature": f("structure", "bos_bull"), "within": 3})
    )
    within = evaluate(ev.entry_long, fp)
    bos = np.flatnonzero(fp.get(ev.entry_long.conditions[0].feature))  # type: ignore[union-attr]
    for i in bos[:20]:
        assert within[i : i + 4].all()

    neg = Group.model_validate(
        {
            "type": "group",
            "negate": True,
            "conditions": [{"type": "flag", "feature": f("sessions", "london")}],
        }
    )
    np.testing.assert_array_equal(evaluate(neg, fp), ~fp.get(neg.conditions[0].feature))  # type: ignore[union-attr]

    wd = Group.model_validate(
        {
            "type": "group",
            "conditions": [
                {"type": "state", "feature": f("time", "weekday"), "values": [1]},
                {"type": "range", "feature": f("time", "minute_ny"), "lo": 540, "hi": 660},
            ],
        }
    )
    hits = fp.inputs.ts[evaluate(wd, fp)].tz_convert("America/New_York")
    assert (
        len(hits) and (hits.dayofweek == 1).all() and ((hits.hour >= 9) & (hits.hour <= 11)).all()
    )


def test_htf_features_need_htf_bars(m1: pd.DataFrame, m15: pd.DataFrame) -> None:
    g = genome(
        entry_long=group(
            {
                "type": "state",
                "feature": {**f("structure", "trend"), "timeframe": "1h"},
                "values": [1],
            }
        )
    )
    with pytest.raises(KeyError, match="1h"):
        compile_genome(g, FeatureProvider(m15, Timeframe.M15, CAL))
    h1 = resample_bars(m1, Timeframe.H1, CAL)
    sig = compile_genome(g, FeatureProvider(m15, Timeframe.M15, CAL, htf_bars={Timeframe.H1: h1}))
    assert 0 < sig.entry_long.sum() < len(m15)


def test_flat_mask_marks_last_bar_before_break(m15: pd.DataFrame) -> None:
    g = all_templates()["ema_pullback"]
    sig = compile_genome(g, FeatureProvider(m15, Timeframe.M15, CAL))
    ny_end = (pd.DatetimeIndex(m15["ts"]) + pd.Timedelta(minutes=15)).tz_convert("America/New_York")
    flagged = ny_end[sig.flat_at]
    assert len(flagged) and (flagged.hour == 17).all() and (flagged.minute == 0).all()


# --------------------------------------------------------------------------- metrics
def test_metrics_hand_computed() -> None:
    idx = pd.date_range("2024-03-04 15:00", periods=4, freq="1D", tz="UTC")
    equity = pd.Series([100_100.0, 100_050.0, 100_250.0, 100_200.0], index=idx)
    trades = pd.DataFrame(
        {
            "pnl": [100.0, -50.0, 200.0, -50.0],
            "r": [1.0, -0.5, 2.0, -0.5],
            "entry_idx": [0, 1, 2, 3],
            "exit_idx": [0, 1, 2, 3],
            "commission": 0.0,
            "side": ["long", "short", "long", "long"],
            "exit_reason": ["target", "stop", "target", "stop"],
        }
    )
    m = compute_metrics(
        trades, equity, pd.DatetimeIndex(idx.normalize()), 100_000.0, np.array([1, 1, 0, 1])
    )
    assert m["profit_factor"] == pytest.approx(3.0)
    assert (m["win_rate"], m["expectancy_r"], m["max_consecutive_losses"]) == (0.5, 0.5, 1)
    dr = np.array([100_100 / 100_000, 100_050 / 100_100, 100_250 / 100_050, 100_200 / 100_250]) - 1
    assert m["sharpe"] == pytest.approx(dr.mean() / dr.std(ddof=1) * math.sqrt(252))
    assert m["sortino"] == pytest.approx(
        dr.mean() / math.sqrt(np.mean(np.minimum(dr, 0) ** 2)) * math.sqrt(252)
    )
    assert m["exposure"] == 0.75 and (m["long_trades"], m["short_trades"]) == (3, 1)
    assert max_drawdown(np.array([100, 120, 90, 130.0])) == pytest.approx(0.25)


def test_no_trades_metrics(m15: pd.DataFrame) -> None:
    never = genome(
        entry_long=group(
            {"type": "compare", "left": feat("rsi", "value"), "op": ">", "right": const(100)}
        )
    )
    run = run_backtest(never, m15, Timeframe.M15, calendar=CAL)
    assert run.metrics["n_trades"] == 0 and run.metrics["total_return"] == 0.0


# --------------------------------------------------------------------------- integrity
def test_backtest_has_no_lookahead(m15: pd.DataFrame) -> None:
    """Trades completed before a cut must be identical whether or not later bars exist."""
    g = all_templates()["sweep_reversal"]
    full = run_backtest(g, m15, Timeframe.M15, calendar=CAL, sessions=SESSIONS, with_context=False)
    assert full.metrics["n_trades"] > 50
    for cut in (1500, 3000, len(m15) - 50):
        part = run_backtest(
            g, m15.iloc[:cut], Timeframe.M15, calendar=CAL, sessions=SESSIONS, with_context=False
        )
        done = full.trades[full.trades["exit_idx"] < cut - 1].reset_index(drop=True)
        got = part.trades.iloc[: len(done)].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, done)


def test_window_uses_history_as_warmup(m15: pd.DataFrame) -> None:
    g = all_templates()["ema_pullback"]
    start = m15["ts"].iloc[3000]
    run = run_backtest(g, m15, Timeframe.M15, calendar=CAL, start=start)
    assert run.period_start == start
    assert (run.trades["entry_ts"] > start).all()
    assert run.equity.index[0] == start


def test_random_walk_has_no_edge() -> None:
    """On edge-free synthetic data a correct engine must find ~zero expectancy without costs,
    and negative expectancy with costs. A significantly positive result would mean leaked
    future information."""
    m1 = generate_m1(
        datetime(2020, 1, 5, tzinfo=UTC), datetime(2022, 1, 2, tzinfo=UTC), CAL, seed=42
    )
    m5 = resample_bars(m1, Timeframe.M5, CAL)
    zero_spread = m5.copy()
    for fld in ("open", "high", "low", "close"):
        zero_spread[f"ask_{fld}"] = zero_spread[f"bid_{fld}"]
    free = CostConfig(commission_per_lot_rt=0.0, slippage_ticks=0.0, min_spread=0.0)
    fp_free = FeatureProvider(zero_spread, Timeframe.M5, CAL, SESSIONS)
    fp_cost = FeatureProvider(m5, Timeframe.M5, CAL, SESSIONS)
    for name in ("ema_pullback", "sweep_reversal", "divergence_bbands"):
        g = all_templates()[name]
        r = run_backtest(
            g, zero_spread, Timeframe.M5, costs=free, provider=fp_free, with_context=False
        ).trades["r"]
        t_stat = r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))
        assert len(r) > 500, name
        assert abs(t_stat) < 3.5, f"{name}: suspicious edge on a random walk (t={t_stat:.2f})"
        costly = run_backtest(
            g, m5, Timeframe.M5, costs=CostConfig(), provider=fp_cost, with_context=False
        )
        assert costly.metrics["expectancy_r"] < r.mean(), name


def test_trades_carry_entry_context(m15: pd.DataFrame) -> None:
    run = run_backtest(
        all_templates()["sweep_reversal"], m15, Timeframe.M15, calendar=CAL, sessions=SESSIONS
    )
    ctx = run.trades["context"].iloc[0]
    assert {
        "weekday",
        "hour_ny",
        "session_ny_am_kz",
        "vol_regime",
        "structure_trend",
        "bars_since_sweep_low",
        "bars_since_choch_bull",
        "rsi",
        "spread",
    } <= set(ctx)
    # sweep_reversal requires a sweep within 2 bars of the decision bar
    for c, side in zip(run.trades["context"], run.trades["side"], strict=True):
        key = "bars_since_sweep_low" if side == "long" else "bars_since_sweep_high"
        assert 0 <= c[key] <= 2

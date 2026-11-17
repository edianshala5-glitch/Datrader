"""Meta-Analyzer: feature alignment, rule→genome conversion (proved by backtest), mining,
and the controls (planted Tuesday-morning edge found; edge-free data → nothing kept)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from app.models.enums import Timeframe
from app.services.seed import XAUUSD
from engine.backtest.runner import CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.data.resample import resample_bars
from engine.data.sources.synthetic import generate_m1
from engine.meta.analyzer import MetaConfig, analyze
from engine.meta.miner import MinerConfig, candidate_clauses, mine
from engine.meta.rules import NEVER, Clause, Rule, clause_condition, refine_genome, trade_features
from engine.strategy.genome import Genome
from engine.strategy.templates import all_templates
from engine.validation.splits import make_splits

CAL = MarketCalendar.xauusd()
SESSIONS = XAUUSD.model_dump(mode="json")["sessions"]
COSTS = CostConfig.from_instrument(XAUUSD)
SPLITS = make_splits("2021-01-03", "2024-01-01")
ALWAYS_IN = Genome.model_validate(
    {
        "entry_long": {
            "type": "group",
            "conditions": [
                {
                    "type": "range",
                    "feature": {"indicator": "time", "output": "minute_ny"},
                    "lo": 0,
                    "hi": 1439,
                }
            ],
        },
        "exit": {
            "stop": {"type": "atr", "mult": 3.0},
            "target": None,
            "max_bars": 4,
            "flat_at": "daily",
        },
    }
)


def _bars(mu: float, seed: int) -> pd.DataFrame:
    def drift(m: pd.DatetimeIndex) -> np.ndarray:  # type: ignore[type-arg]
        ny = m.tz_convert("America/New_York")
        return np.where((ny.day_of_week == 1) & (ny.hour >= 9) & (ny.hour < 11), mu, 0.0)

    m1 = generate_m1(
        datetime(2021, 1, 3, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
        CAL,
        seed=seed,
        drift=drift,
    )
    return resample_bars(m1, Timeframe.M15, CAL)


@pytest.fixture(scope="module")
def flat() -> pd.DataFrame:
    return _bars(0.0, 9)


# --------------------------------------------------------------------------- features
def test_features_are_direction_aligned() -> None:
    ctx = {
        "weekday": 1,
        "hour_ny": 9,
        "us_holiday": False,
        "vol_regime": 1,
        "vol_rank": 0.5,
        "atr_pct": 0.1,
        "structure_trend": 1,
        "trend_regime": -1,
        "rsi": 30.0,
        "bars_since_bos_bull": 3,
        "bars_since_bos_bear": -1,
        "bars_since_choch_bull": 7,
        "bars_since_choch_bear": 2,
        "bars_since_sweep_low": 1,
        "bars_since_sweep_high": 9,
        "bars_since_div_bull": -1,
        "bars_since_div_bear": 4,
    }
    trades = pd.DataFrame({"side": ["long", "short"], "context": [ctx, ctx]})
    x = trade_features(trades)
    assert x["trend_aligned"].tolist() == [1, -1]
    assert x["trend_regime_aligned"].tolist() == [-1, 1]
    assert x["bars_since_bos"].tolist() == [3, NEVER]
    assert x["bars_since_choch"].tolist() == [7, 2]
    assert x["bars_since_sweep"].tolist() == [1, 9]  # long: sweep of lows; short: sweep of highs
    assert x["bars_since_div"].tolist() == [NEVER, 4]
    assert x["rsi_aligned"].tolist() == [30.0, 70.0]


# --------------------------------------------------------------------------- rule → genome
def test_clause_conversions() -> None:
    assert clause_condition(Clause("hour_ny", "in", (9, 10))) == {
        "type": "range",
        "feature": {"indicator": "time", "output": "minute_ny", "params": {}},
        "lo": 540,
        "hi": 659,
    }
    assert clause_condition(Clause("hour_ny", "in", (9, 11))) is None  # non-contiguous
    neg = clause_condition(Clause("bars_since_choch", ">", 10))
    assert neg is not None and neg["negate"] and neg["conditions"][0]["within"] == 10
    assert clause_condition(Clause("bars_since_bos", "<=", 500)) is None  # beyond event window
    div = clause_condition(Clause("bars_since_div", "<=", 5))
    assert div is not None and div["op"] == "any" and len(div["conditions"]) == 2


def test_refine_places_directional_and_neutral_clauses() -> None:
    base = all_templates()["sweep_reversal"]
    rule = Rule(
        (
            Clause("weekday", "in", (1, 2)),
            Clause("trend_aligned", "in", (1,)),
            Clause("rsi_aligned", "<=", 40.0),
        ),
        "test",
    )
    refined = refine_genome(base, rule)
    assert refined is not None
    long_c = refined.entry_long.conditions  # type: ignore[union-attr]
    short_c = refined.entry_short.conditions  # type: ignore[union-attr]
    assert long_c[-2].values == [1] and short_c[-2].values == [-1]  # trend mirrored
    assert (long_c[-1].op, long_c[-1].right.value) == ("<", 40.0)  # type: ignore[union-attr]
    assert (short_c[-1].op, short_c[-1].right.value) == (">", 60.0)  # type: ignore[union-attr]
    assert refined.filter.conditions[-1].values == [1, 2]  # type: ignore[union-attr]
    too_much = Rule(tuple(Clause("weekday", "in", (d,)) for d in range(5)) * 3, "x")
    assert refine_genome(base, too_much) is None


@pytest.mark.parametrize(
    "clause",
    [
        Clause("weekday", "in", (1, 2)),
        Clause("hour_ny", "in", (8, 9, 10)),
        Clause("session_london", "in", (True,)),
        Clause("vol_regime", "in", (0, 1)),
        Clause("trend_aligned", "in", (1,)),
        Clause("bars_since_choch", "<=", 20),
        Clause("bars_since_bos", ">", 10),
        Clause("rsi_aligned", "<=", 50.0),
    ],
)
def test_refined_genome_trades_only_inside_the_rule(flat: pd.DataFrame, clause: Clause) -> None:
    """The real proof of conversion: every trade the refined genome takes satisfies the rule."""
    base = all_templates()["ema_pullback"]  # trades both directions
    refined = refine_genome(base, Rule((clause,), "t"))
    assert refined is not None
    window = flat[(flat["ts"] >= "2022-01-01") & (flat["ts"] < "2022-07-01")].reset_index(drop=True)
    run = run_backtest(refined, window, Timeframe.M15, calendar=CAL, sessions=SESSIONS, costs=COSTS)
    assert len(run.trades) > 5
    x = trade_features(run.trades)
    assert Rule((clause,), "t").mask(x).all(), clause


# --------------------------------------------------------------------------- miner
def test_miner_finds_planted_feature_interaction() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = pd.DataFrame(
        {
            "weekday": rng.integers(0, 5, n),
            "hour_ny": rng.integers(0, 24, n),
            "us_holiday": False,
            **{
                f"session_{s}": rng.random(n) < 0.3
                for s in [
                    "asia",
                    "london",
                    "new_york",
                    "london_ny_overlap",
                    "comex",
                    "asia_kz",
                    "london_kz",
                    "ny_am_kz",
                    "ny_pm_kz",
                ]
            },
            "vol_regime": rng.integers(0, 3, n),
            "vol_rank": rng.random(n),
            "atr_pct": rng.random(n) * 0.3,
            "trend_aligned": rng.integers(-1, 2, n),
            "trend_regime_aligned": rng.integers(-1, 2, n),
            "bars_since_bos": rng.integers(0, 50, n).astype(float),
            "bars_since_choch": rng.integers(0, 50, n).astype(float),
            "bars_since_sweep": rng.integers(0, 50, n).astype(float),
            "bars_since_div": rng.integers(0, 50, n).astype(float),
            "rsi_aligned": rng.random(n) * 100,
        }
    )
    r = rng.normal(-0.05, 1.0, n)
    planted = (x["weekday"] == 2) & x["hour_ny"].between(14, 15)
    r[planted.to_numpy()] += 1.0
    cands = mine(x, r, MinerConfig())
    best = cands[0]
    feats = {c.feature for c in best.rule.clauses}
    assert {"weekday", "hour_ny"} <= feats, best.rule.describe()
    assert best.mean_r > 0.5
    assert len(candidate_clauses(x)) > 200


# --------------------------------------------------------------------------- analyzer controls
def test_meta_analyzer_finds_planted_tuesday_morning_edge() -> None:
    rep = analyze(
        ALWAYS_IN,
        _bars(5e-5, 9),
        Timeframe.M15,
        SPLITS,
        calendar=CAL,
        sessions=SESSIONS,
        costs=COSTS,
    )
    assert rep.selected, [r.reasons for r in rep.rules]
    top = rep.selected[0]
    feats = {c.feature for c in top.rule.clauses}
    assert {"weekday", "hour_ny"} <= feats
    wd = next(c for c in top.rule.clauses if c.feature == "weekday")
    assert wd.value == (1,)  # Tuesday
    hours = next(c for c in top.rule.clauses if c.feature == "hour_ny").value
    assert set(hours) & {8, 9, 10}
    assert top.refined_run is not None and top.refined_run.metrics["expectancy_r"] > 0.2
    assert top.refined_dsr is not None and top.refined_dsr > 0.95
    assert all(r.p_adjusted >= r.p_value for r in rep.rules)
    assert len(rep.selected) <= MetaConfig().max_rules


def test_meta_analyzer_keeps_nothing_on_edge_free_data(flat: pd.DataFrame) -> None:
    for genome in (ALWAYS_IN, all_templates()["sweep_reversal"]):
        rep = analyze(
            genome, flat, Timeframe.M15, SPLITS, calendar=CAL, sessions=SESSIONS, costs=COSTS
        )
        assert rep.rules and not rep.validated
        # mined rules look great in-sample and fall apart out of sample
        assert max(r.candidate.mean_r for r in rep.rules) > 0
        assert np.mean([r.oos_mean_r for r in rep.rules]) < np.mean(
            [r.candidate.mean_r for r in rep.rules]
        )


def test_meta_skips_strategies_with_too_few_trades(flat: pd.DataFrame) -> None:
    rare = Genome.model_validate(
        {
            "entry_long": {
                "type": "group",
                "conditions": [
                    {
                        "type": "range",
                        "feature": {"indicator": "time", "output": "minute_ny"},
                        "lo": 600,
                        "hi": 600,
                    },
                    {
                        "type": "state",
                        "feature": {"indicator": "time", "output": "weekday"},
                        "values": [2],
                    },
                ],
            }
        }
    )
    # once a week for ~2.4 years is ~125 trades: below a 200-trade minimum
    rep = analyze(
        rare,
        flat,
        Timeframe.M15,
        SPLITS,
        calendar=CAL,
        sessions=SESSIONS,
        costs=COSTS,
        config=MetaConfig(min_base_trades=200),
    )
    assert rep.skipped_reason and "trades" in rep.skipped_reason and not rep.rules

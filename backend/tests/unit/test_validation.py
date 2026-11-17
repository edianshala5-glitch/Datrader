"""Validation harness: statistics, PBO, splits, perturbation, Monte Carlo, and the three
behavioural controls (no edge → reject, planted edge → pass, lucky winner → reject)."""

from __future__ import annotations

import math
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
from engine.strategy.compiler import FeatureProvider
from engine.strategy.genome import Genome
from engine.strategy.mutate import knobs, perturb
from engine.strategy.templates import all_templates
from engine.validation import statistics as st
from engine.validation.pbo import pbo_cscv
from engine.validation.robustness import monte_carlo
from engine.validation.splits import DataSplits, make_splits
from engine.validation.validator import evaluate_holdout, returns_matrix, validate_genome

CAL = MarketCalendar.xauusd()
SESSIONS = XAUUSD.model_dump(mode="json")["sessions"]
COSTS = CostConfig.from_instrument(XAUUSD)
FREE = CostConfig(commission_per_lot_rt=0.0, slippage_ticks=0.0, min_spread=0.0)
SPLITS = make_splits("2021-01-03", "2024-01-01")


# --------------------------------------------------------------------------- statistics
def test_probabilistic_sharpe_formula() -> None:
    # SR 0.1/period over 251 periods, normal returns: z = 0.1 * sqrt(250) / sqrt(1 + 0.5*0.01)
    expected = st.NormalDist().cdf(0.1 * math.sqrt(250) / math.sqrt(1 + 0.5 * 0.01))
    assert st.probabilistic_sharpe(0.1, 251) == pytest.approx(expected)
    # negative skew / fat tails lower confidence
    assert st.probabilistic_sharpe(0.1, 251, skew=-1, kurtosis=8) < expected


def test_deflation_grows_with_trials() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)  # true per-period SR 0.1
    assert st.expected_max_sharpe(1, 0.002) == 0.0
    emax = [st.expected_max_sharpe(n, 1 / 500) for n in (10, 100, 1000, 100_000)]
    assert emax == sorted(emax) and emax[0] > 0
    dsr = [st.deflated_sharpe(r, n) for n in (1, 10, 100, 1000)]
    assert dsr == sorted(dsr, reverse=True)
    assert dsr[0] == pytest.approx(st.probabilistic_sharpe(*_psr_args(r)))
    assert dsr[-1] < 0.5 < dsr[0]


def _psr_args(r: np.ndarray) -> tuple[float, int, float, float]:  # type: ignore[type-arg]
    m = st.moments(r)
    return m.sharpe, m.n, m.skew, m.kurtosis


def test_multiple_testing_adjustments() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    assert st.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert st.benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_bootstrap_ci_and_t_test() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.2, 1.0, 400)
    lo, hi = st.bootstrap_ci(x, n_boot=1000)
    assert lo < 0.2 < hi and hi - lo < 0.3
    lo_b, hi_b = st.bootstrap_ci(x, n_boot=500, block=10)
    assert lo_b < x.mean() < hi_b
    t, p = st.t_test_mean(x)
    assert t > 2 and p < 0.05


# --------------------------------------------------------------------------- PBO
def test_pbo_noise_vs_skill() -> None:
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.01, (1000, 40))
    assert 0.3 <= pbo_cscv(noise, n_blocks=10).pbo <= 0.7  # theory: 0.5
    drifts = np.linspace(-0.002, 0.003, 40)  # persistent, different true edges
    skill = rng.normal(0, 0.01, (1000, 40)) + drifts
    res = pbo_cscv(skill, n_blocks=10)
    assert res.pbo < 0.1 and res.prob_oos_loss < 0.1 and res.n_splits == 252
    with pytest.raises(ValueError):
        pbo_cscv(noise, n_blocks=7)


# --------------------------------------------------------------------------- splits
def test_walk_forward_splits() -> None:
    sp = SPLITS
    assert sp.holdout_start < sp.holdout_end and len(sp.folds) == 5
    prev_end = None
    for f in sp.folds:
        assert f.train_start == sp.research_start  # anchored
        assert f.test_start - f.train_end == pd.Timedelta(days=1)  # embargo
        assert f.test_end <= sp.holdout_start
        if prev_end is not None:
            assert f.test_start == prev_end
        prev_end = f.test_end
    assert sp.folds[-1].test_end == sp.holdout_start
    rolling = make_splits("2021-01-03", "2024-01-01", anchored=False)
    lens = {f.train_end - f.train_start for f in rolling.folds}
    assert len(lens) == 1  # every rolling train window has exactly the same length
    assert all(f.train_start > rolling.research_start for f in rolling.folds[1:])
    assert DataSplits.from_dict(sp.to_dict()) == sp
    with pytest.raises(ValueError):
        make_splits("2024-01-01", "2024-01-20", n_folds=10)


# --------------------------------------------------------------------------- perturbation
def test_knobs_and_perturbation() -> None:
    g = all_templates()["sweep_reversal"]
    ks = knobs(g)
    paths = {"/".join(map(str, k.path)) for k in ks}
    assert "entry_long/conditions/1/right/value" in paths  # RSI threshold
    assert "exit/stop/mult" in paths and "exit/max_bars" in paths
    assert all(k.lo <= k.value <= k.hi for k in ks)

    rng = np.random.default_rng(5)
    variants = [perturb(g, rng) for _ in range(20)]
    assert len({v.model_dump_json() for v in variants}) > 15
    for v in variants:
        for k in knobs(v):
            assert k.lo <= k.value <= k.hi
            assert abs(k.value / k.step - round(k.value / k.step)) < 1e-6  # on the grid
    a = perturb(g, np.random.default_rng(9))
    b = perturb(g, np.random.default_rng(9))
    assert a == b  # deterministic per seed


def test_monte_carlo() -> None:
    mc = monte_carlo(np.full(50, 1.0), risk_frac=0.01, n_paths=200)
    assert mc["max_dd_p95"] == 0.0 and mc["prob_loss"] == 0.0
    mixed = monte_carlo(np.array([2.0, -1.0] * 100), risk_frac=0.01, n_paths=300)
    assert mixed["max_dd_p95"] >= mixed["max_dd_p50"] > 0
    assert mixed["prob_loss"] < 0.05


# --------------------------------------------------------------------------- behavioural controls
def window_genome(minute: int, bars: int, side: str = "long") -> Genome:
    return Genome.model_validate(
        {
            f"entry_{side}": {
                "type": "group",
                "conditions": [
                    {
                        "type": "range",
                        "feature": {"indicator": "time", "output": "minute_ny"},
                        "lo": minute,
                        "hi": minute,
                    }
                ],
            },
            "exit": {
                "stop": {"type": "atr", "mult": 4.0},
                "target": None,
                "max_bars": bars,
                "flat_at": "daily",
            },
        }
    )


def _data(mu: float, seed: int, zero_spread: bool = False) -> pd.DataFrame:
    def drift(m: pd.DatetimeIndex) -> np.ndarray:  # type: ignore[type-arg]
        h = m.tz_convert("America/New_York").hour
        return np.where((h >= 8) & (h < 11), mu, 0.0)

    m1 = generate_m1(
        datetime(2021, 1, 3, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
        CAL,
        seed=seed,
        drift=drift,
    )
    bars = resample_bars(m1, Timeframe.M15, CAL)
    if zero_spread:
        for f in ("open", "high", "low", "close"):
            bars[f"ask_{f}"] = bars[f"bid_{f}"]
    return bars


@pytest.fixture(scope="module")
def flat_data() -> pd.DataFrame:
    return _data(0.0, seed=9)


def _validate(g: Genome, bars: pd.DataFrame, **kw: object):  # type: ignore[no-untyped-def]
    kw.setdefault("costs", COSTS)
    return validate_genome(g, bars, Timeframe.M15, SPLITS, calendar=CAL, sessions=SESSIONS, **kw)  # type: ignore[arg-type]


def test_negative_control_rejects_templates_on_edge_free_data(flat_data: pd.DataFrame) -> None:
    for name in ("sweep_reversal", "ema_pullback"):
        rep = _validate(all_templates()[name], flat_data)
        assert not rep.passed, name
        assert rep.overfit_risk > 0.6
        assert not rep.gates["deflated_sharpe"]["passed"]
        assert rep.oos_metrics["n_trades"] >= 100  # rejected on merit, not lack of data


def test_positive_control_accepts_planted_edge() -> None:
    bars = _data(6e-6, seed=9)
    rep = _validate(window_genome(480, 8), bars)  # buy the 08:15 NY open, hold 2h
    assert rep.passed, rep.reasons
    assert rep.overfit_risk < 0.2 and rep.dsr > 0.95 and rep.positive_fold_ratio >= 0.8
    # the identical rule on edge-free data is rejected
    assert not _validate(window_genome(480, 8), _data(0.0, seed=9)).passed


def test_lucky_winner_needs_trial_correction() -> None:
    """Best of 50 random strategies on edge-free, cost-free data looks robust unless the
    number of trials is accounted for."""
    bars = _data(0.0, seed=21, zero_spread=True)
    rng = np.random.default_rng(3)
    cands = [
        window_genome(
            int(rng.integers(0, 96)) * 15,
            int(rng.integers(2, 12)),
            str(rng.choice(["long", "short"])),
        )
        for _ in range(50)
    ]
    research = bars[bars["ts"] < SPLITS.holdout_start].reset_index(drop=True)
    fp = FeatureProvider(research, Timeframe.M15, CAL, SESSIONS)
    runs = {
        str(i): run_backtest(
            g, research, Timeframe.M15, costs=FREE, provider=fp, with_context=False
        )
        for i, g in enumerate(cands)
    }
    best = max(runs, key=lambda k: runs[k].metrics["sharpe"] or -99)
    assert runs[best].metrics["sharpe"] > 0.8  # looks great in-sample

    pbo = pbo_cscv(returns_matrix(runs, CAL).to_numpy(), n_blocks=10).pbo
    assert 0.35 <= pbo <= 0.65  # selection has no skill

    naive = _validate(cands[int(best)], bars, costs=FREE, n_trials=1)
    honest = _validate(cands[int(best)], bars, costs=FREE, n_trials=50, pbo=pbo)
    assert honest.dsr < naive.dsr and honest.dsr < 0.5
    assert honest.overfit_risk > naive.overfit_risk
    assert not honest.passed
    assert not honest.gates["pbo"]["passed"]


def test_validation_never_sees_holdout(flat_data: pd.DataFrame) -> None:
    g = all_templates()["ema_pullback"]
    a = _validate(g, flat_data, config=None).to_dict()
    vandalised = flat_data.copy()
    late = vandalised["ts"] >= SPLITS.holdout_start
    cols = [c for c in vandalised.columns if c.startswith(("bid_", "ask_"))]
    vandalised.loc[late, cols] = vandalised.loc[late, cols] * 1.5
    b = _validate(g, vandalised).to_dict()
    assert a == b
    hold = evaluate_holdout(g, flat_data, Timeframe.M15, SPLITS, calendar=CAL, sessions=SESSIONS)
    assert hold.period_start >= SPLITS.holdout_start
    assert (hold.trades["entry_ts"] >= SPLITS.holdout_start).all()

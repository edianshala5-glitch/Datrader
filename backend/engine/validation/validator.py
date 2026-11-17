"""Full out-of-sample validation of one genome, ending in an overfit-risk score and gates.

Pipeline (everything inside the *research* period; the holdout is never touched here):
1. Walk-forward: backtest every fold's train window (IS) and test window (OOS).
2. Stitch the OOS windows into one track record: trades plus a daily equity curve.
3. Deflated Sharpe of the OOS daily returns, penalized for ``n_trials``.
4. Walk-forward efficiency (OOS / IS expectancy) and fold consistency.
5. Parameter robustness: K perturbed neighbours over the research period.
6. Cost stress: doubled commission and slippage, plus a wider spread floor.
7. Monte Carlo on the OOS trades.
8. Overfit-risk score (0 robust .. 1 noise) and hard gates for VALIDATED status.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from app.models.enums import Timeframe
from engine.backtest.metrics import compute_metrics
from engine.backtest.runner import BacktestRun, CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.strategy.compiler import FeatureProvider
from engine.strategy.genome import Genome
from engine.strategy.mutate import perturb
from engine.validation.robustness import monte_carlo
from engine.validation.splits import DataSplits
from engine.validation.statistics import deflated_sharpe, moments, probabilistic_sharpe


@dataclass(frozen=True)
class ValidationConfig:
    min_oos_trades: int = 100
    dsr_min: float = 0.95
    min_positive_folds: float = 0.6
    min_robustness: float = 0.6
    max_pbo: float = 0.3  # 0.5 = picking the best backtest is no better than random
    n_perturbations: int = 16
    perturb_scale: float = 0.15
    cost_multiplier: float = 2.0  # commission and slippage x2
    spread_floor_multiplier: float = 1.5
    mc_paths: int = 1000

    # Weights of the overfit-risk components (renormalized over those available).
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "dsr": 0.30,
            "folds": 0.15,
            "wfe": 0.15,
            "robustness": 0.15,
            "sample": 0.10,
            "pbo": 0.10,
            "costs": 0.05,
        }
    )


@dataclass
class FoldResult:
    index: int
    is_run: BacktestRun
    oos_run: BacktestRun

    def summary(self) -> dict[str, Any]:
        def pick(m: dict[str, Any]) -> dict[str, Any]:
            return {
                k: m.get(k)
                for k in (
                    "n_trades",
                    "expectancy_r",
                    "win_rate",
                    "profit_factor",
                    "sharpe",
                    "max_drawdown",
                )
            }

        return {
            "index": self.index,
            "test_start": self.oos_run.period_start.isoformat(),
            "test_end": self.oos_run.period_end.isoformat(),
            "is": pick(self.is_run.metrics),
            "oos": pick(self.oos_run.metrics),
        }


@dataclass
class ValidationReport:
    folds: list[FoldResult]
    oos_trades: pd.DataFrame
    oos_equity: pd.Series
    oos_metrics: dict[str, Any]
    dsr: float
    psr: float
    n_trials: int
    wfe: float
    positive_fold_ratio: float
    robustness: dict[str, Any]
    cost_stress: dict[str, Any]
    monte_carlo: dict[str, Any]
    pbo: float | None
    components: dict[str, float]
    overfit_risk: float
    gates: dict[str, dict[str, Any]]
    passed: bool

    @property
    def reasons(self) -> list[str]:
        return [
            f"{k}: {g['value']} (needs {g['rule']})"
            for k, g in self.gates.items()
            if not g["passed"]
        ]

    def to_dict(self) -> dict[str, Any]:
        def clean(v: Any) -> Any:
            if isinstance(v, float):
                return None if not math.isfinite(v) else round(v, 6)
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v

        out: dict[str, Any] = clean(
            {
                "passed": self.passed,
                "overfit_risk": self.overfit_risk,
                "reasons": self.reasons,
                "gates": self.gates,
                "components": self.components,
                "oos": self.oos_metrics,
                "dsr": self.dsr,
                "psr": self.psr,
                "n_trials": self.n_trials,
                "wfe": self.wfe,
                "positive_fold_ratio": self.positive_fold_ratio,
                "robustness": self.robustness,
                "cost_stress": self.cost_stress,
                "monte_carlo": self.monte_carlo,
                "pbo": self.pbo,
                "folds": [f.summary() for f in self.folds],
            }
        )
        return out


def _stitch(runs: list[BacktestRun], initial: float) -> pd.Series:
    """Chain OOS equity curves into one track record (each fold starts where the last ended)."""
    parts, level = [], initial
    for r in runs:
        eq = r.equity / r.costs.initial_equity * level
        parts.append(eq)
        level = float(eq.iloc[-1])
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def _exp(run: BacktestRun) -> float:
    v = run.metrics.get("expectancy_r")
    return float(v) if v is not None and run.metrics["n_trades"] > 0 else 0.0


def validate_genome(
    genome: Genome,
    bars: pd.DataFrame,
    timeframe: Timeframe,
    splits: DataSplits,
    *,
    calendar: MarketCalendar | None = None,
    sessions: dict[str, Any] | None = None,
    costs: CostConfig | None = None,
    htf_bars: dict[Timeframe, pd.DataFrame] | None = None,
    n_trials: int = 1,
    sr_variance: float | None = None,
    pbo: float | None = None,
    config: ValidationConfig | None = None,
    seed: int = 0,
) -> ValidationReport:
    cfg = config or ValidationConfig()
    costs = costs or CostConfig()
    # Never let holdout bars into validation, not even as data the provider could see.
    research = bars[bars["ts"] < splits.holdout_start].reset_index(drop=True)
    htf = {
        tf: b[b["ts"] < splits.holdout_start].reset_index(drop=True)
        for tf, b in (htf_bars or {}).items()
    }
    fp = FeatureProvider(research, timeframe, calendar, sessions, htf)

    def bt(
        g: Genome,
        start: pd.Timestamp,
        end: pd.Timestamp,
        c: CostConfig = costs,
        provider: FeatureProvider | None = fp,
        ctx: bool = False,
    ) -> BacktestRun:
        return run_backtest(
            g,
            research,
            timeframe,
            calendar=calendar,
            sessions=sessions,
            costs=c,
            htf_bars=htf,
            start=start,
            end=end,
            with_context=ctx,
            provider=provider,
        )

    folds = [
        FoldResult(
            f.index,
            bt(genome, f.train_start, f.train_end),
            bt(genome, f.test_start, f.test_end, ctx=True),
        )
        for f in splits.folds
    ]
    oos_runs = [f.oos_run for f in folds]
    oos_trades = pd.concat(
        [r.trades.assign(fold=f.index) for f, r in zip(folds, oos_runs, strict=True)],
        ignore_index=True,
    )
    equity = _stitch(oos_runs, costs.initial_equity)
    day = (
        calendar.trading_day(pd.DatetimeIndex(equity.index))
        if calendar is not None
        else pd.DatetimeIndex(equity.index).normalize()
    )
    oos_metrics = compute_metrics(oos_trades, equity, day, costs.initial_equity)

    from engine.backtest.metrics import daily_returns

    dr = daily_returns(equity, day, costs.initial_equity).to_numpy()
    m = moments(dr)
    dsr = deflated_sharpe(dr, n_trials, sr_variance)
    psr = probabilistic_sharpe(m.sharpe, m.n, m.skew, m.kurtosis)

    is_exp = float(np.mean([_exp(f.is_run) for f in folds]))
    oos_exp = float(oos_metrics["expectancy_r"] or 0.0)
    wfe = (
        float(np.clip(oos_exp / is_exp, -1.0, 2.0))
        if is_exp > 0
        else (0.0 if oos_exp <= 0 else 1.0)
    )
    positive_folds = float(np.mean([_exp(f.oos_run) > 0 for f in folds]))

    # --- robustness: do nearby parameter sets still have positive expectancy?
    rng = np.random.default_rng(seed)
    rs, re = splits.research_start, splits.holdout_start
    base = bt(genome, rs, re)
    neighbours = []
    for _ in range(cfg.n_perturbations):
        variant = perturb(genome, rng, scale=cfg.perturb_scale)
        if variant == genome:
            continue
        neighbours.append(_exp(bt(variant, rs, re)))
    base_exp = _exp(base)
    frac_pos: float = float(np.mean([e > 0 for e in neighbours])) if neighbours else 0.0
    robustness: dict[str, Any] = {
        "n": len(neighbours),
        "base_expectancy_r": base_exp,
        "fraction_positive": frac_pos,
        "median_expectancy_r": float(np.median(neighbours)) if neighbours else None,
        "median_retention": (float(np.median(neighbours)) / base_exp)
        if neighbours and base_exp > 0
        else None,
    }

    # --- cost stress
    stressed = replace(
        costs,
        commission_per_lot_rt=costs.commission_per_lot_rt * cfg.cost_multiplier,
        slippage_ticks=costs.slippage_ticks * cfg.cost_multiplier,
        min_spread=costs.min_spread * cfg.spread_floor_multiplier,
    )
    stress_run = bt(genome, rs, re, c=stressed)
    cost_stress = {
        "expectancy_r": _exp(stress_run),
        "survives": _exp(stress_run) > 0,
        "n_trades": stress_run.metrics["n_trades"],
    }

    mc = monte_carlo(
        oos_trades["r"].to_numpy() if len(oos_trades) else [],
        risk_frac=genome.risk.risk_per_trade,
        n_paths=cfg.mc_paths,
        seed=seed,
    )

    n_oos = int(oos_metrics["n_trades"])
    comps = {
        "dsr": 1.0 - dsr,
        "folds": 1.0 - positive_folds,
        "wfe": 1.0 - float(np.clip(wfe, 0.0, 1.0)),
        "robustness": 1.0 - frac_pos,
        "sample": float(np.clip(1.0 - n_oos / (2 * cfg.min_oos_trades), 0.0, 1.0)),
        "costs": 0.0 if cost_stress["survives"] else 1.0,
    }
    if pbo is not None:
        comps["pbo"] = float(pbo)
    w = {k: cfg.weights[k] for k in comps}
    risk = sum(comps[k] * w[k] for k in comps) / sum(w.values())

    def gate(value: Any, ok: bool, rule: str) -> dict[str, Any]:
        return {"value": value, "passed": bool(ok), "rule": rule}

    gates = {
        "oos_trades": gate(n_oos, n_oos >= cfg.min_oos_trades, f">= {cfg.min_oos_trades}"),
        "oos_expectancy": gate(round(oos_exp, 4), oos_exp > 0, "> 0R after costs"),
        "deflated_sharpe": gate(round(dsr, 4), dsr >= cfg.dsr_min, f">= {cfg.dsr_min}"),
        "positive_folds": gate(
            round(positive_folds, 3),
            positive_folds >= cfg.min_positive_folds,
            f">= {cfg.min_positive_folds}",
        ),
        "robustness": gate(
            round(frac_pos, 3),
            frac_pos >= cfg.min_robustness,
            f">= {cfg.min_robustness}",
        ),
        "cost_stress": gate(
            round(cost_stress["expectancy_r"], 4), cost_stress["survives"], "> 0R at 2x costs"
        ),
    }
    if pbo is not None:
        gates["pbo"] = gate(round(pbo, 3), pbo <= cfg.max_pbo, f"<= {cfg.max_pbo}")

    return ValidationReport(
        folds=folds,
        oos_trades=oos_trades,
        oos_equity=equity,
        oos_metrics=oos_metrics,
        dsr=dsr,
        psr=psr,
        n_trials=n_trials,
        wfe=wfe,
        positive_fold_ratio=positive_folds,
        robustness=robustness,
        cost_stress=cost_stress,
        monte_carlo=mc,
        pbo=pbo,
        components={k: round(v, 4) for k, v in comps.items()},
        overfit_risk=round(float(risk), 4),
        gates=gates,
        passed=all(g["passed"] for g in gates.values()),
    )


def evaluate_holdout(
    genome: Genome,
    bars: pd.DataFrame,
    timeframe: Timeframe,
    splits: DataSplits,
    **kwargs: Any,
) -> BacktestRun:
    """One backtest over the locked holdout (earlier bars serve only as indicator warm-up).
    Callers must enforce "once per strategy" (see app.services.validation)."""
    return run_backtest(
        genome, bars, timeframe, start=splits.holdout_start, end=splits.holdout_end, **kwargs
    )


def returns_matrix(runs: dict[str, BacktestRun], calendar: MarketCalendar | None) -> pd.DataFrame:
    """Daily returns of many strategies aligned on trading days (input for PBO/CSCV)."""
    from engine.backtest.metrics import daily_returns

    cols = {}
    for key, r in runs.items():
        idx = pd.DatetimeIndex(r.equity.index)
        day = calendar.trading_day(idx) if calendar is not None else idx.normalize()
        cols[key] = daily_returns(r.equity, day, r.costs.initial_equity)
    return pd.DataFrame(cols).fillna(0.0)

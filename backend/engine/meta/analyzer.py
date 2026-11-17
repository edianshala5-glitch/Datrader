"""Meta-Analyzer: find the contexts in which a strategy actually works, and prove it.

Pipeline for one genome (research period only, holdout untouched):
1. Backtest over the research period with per-trade context.
2. Split trades by entry time: **mining** (first ``mine_fraction``) | embargo | **testing**.
3. Mine candidate rules on the mining trades (:mod:`engine.meta.miner`).
4. Test the top candidates on the testing trades. A rule is VALIDATED only if, out of sample:
   it keeps >= ``min_oos_trades`` trades, those trades have positive expectancy, and their
   lift over the excluded trades is significant after a **Holm** correction across all
   candidates tested.
5. For each validated rule, build the refined genome (base + rule) and backtest *it* on the
   testing window. Filtering entries changes which trades happen, so the refined strategy
   is measured directly, with a Deflated Sharpe charged for every candidate tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from app.models.enums import Timeframe
from engine.backtest.metrics import daily_returns
from engine.backtest.runner import BacktestRun, CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.meta.miner import Candidate, MinerConfig, importance, mine
from engine.meta.rules import Rule, refine_genome, trade_features
from engine.strategy.compiler import FeatureProvider
from engine.strategy.explain import explain
from engine.strategy.genome import Genome
from engine.validation.splits import DataSplits
from engine.validation.statistics import deflated_sharpe, holm

_N = NormalDist()


@dataclass(frozen=True)
class MetaConfig:
    mine_fraction: float = 0.6
    embargo_days: float = 1.0
    min_oos_trades: int = 30
    alpha: float = 0.05
    min_base_trades: int = 100
    max_rules: int = 3  # distinct validated rules kept per strategy
    max_overlap: float = 0.5  # Jaccard overlap of OOS kept trades that counts as "the same rule"
    miner: MinerConfig = field(default_factory=MinerConfig)


@dataclass
class RuleResult:
    candidate: Candidate
    oos_n: int
    oos_mean_r: float
    oos_win_rate: float
    oos_profit_factor: float | None
    oos_coverage: float
    oos_lift: float  # mean kept - mean excluded
    p_value: float
    p_adjusted: float = 1.0
    validated: bool = False
    selected: bool = False  # validated AND distinct from better rules (persisted / shown)
    reasons: list[str] = field(default_factory=list)
    oos_mask: np.ndarray | None = field(default=None, repr=False)
    refined: Genome | None = None
    refined_run: BacktestRun | None = None
    refined_dsr: float | None = None

    @property
    def rule(self) -> Rule:
        return self.candidate.rule

    def to_dict(self) -> dict[str, Any]:
        c = self.candidate
        out = {
            "rule": self.rule.describe(),
            "clauses": [cl.to_dict() for cl in self.rule.clauses],
            "miner": self.rule.miner,
            "mining": {
                "n": c.n,
                "coverage": c.coverage,
                "mean_r": c.mean_r,
                "win_rate": c.win_rate,
                "profit_factor": c.profit_factor,
                "lift": c.lift,
                "score": c.score,
            },
            "oos": {
                "n": self.oos_n,
                "coverage": self.oos_coverage,
                "mean_r": self.oos_mean_r,
                "win_rate": self.oos_win_rate,
                "profit_factor": self.oos_profit_factor,
                "lift": self.oos_lift,
            },
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "validated": self.validated,
            "selected": self.selected,
            "reasons": self.reasons,
        }
        if self.refined is not None:
            out["refined_explanation"] = explain(self.refined)
        if self.refined_run is not None:
            m = self.refined_run.metrics
            out["refined_oos"] = {
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
            out["refined_dsr"] = self.refined_dsr
        return _clean_dict(out)


@dataclass
class MetaReport:
    base_mining: dict[str, Any]
    base_oos: dict[str, Any]
    importance: list[dict[str, Any]]
    rules: list[RuleResult]
    mine_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_trades: int
    skipped_reason: str | None = None

    @property
    def validated(self) -> list[RuleResult]:
        return [r for r in self.rules if r.validated]

    @property
    def selected(self) -> list[RuleResult]:
        return [r for r in self.rules if r.selected]

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "n_trades": self.n_trades,
                "skipped_reason": self.skipped_reason,
                "windows": {
                    "mine_end": self.mine_end.isoformat(),
                    "test_start": self.test_start.isoformat(),
                    "test_end": self.test_end.isoformat(),
                },
                "base": {"mining": self.base_mining, "oos": self.base_oos},
                "importance": self.importance,
                "candidates_tested": len(self.rules),
                "validated": len(self.validated),
                "selected": len(self.selected),
                "rules": [r.to_dict() for r in self.rules],
            }
        )


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 6)
    if isinstance(v, (np.floating, np.integer)):
        return _clean(v.item())
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    return v


def _clean_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = _clean(d)
    return out


def _summary(r: np.ndarray) -> dict[str, Any]:
    if len(r) == 0:
        return {"n": 0}
    return {"n": len(r), "mean_r": float(r.mean()), "win_rate": float((r > 0).mean())}


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided p-value for mean(a) > mean(b) (normal approximation; samples are large)."""
    if len(a) < 2 or len(b) < 2:
        return 1.0
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return 1.0
    return 1.0 - _N.cdf((a.mean() - b.mean()) / se)


def analyze(
    genome: Genome,
    bars: pd.DataFrame,
    timeframe: Timeframe,
    splits: DataSplits,
    *,
    calendar: MarketCalendar | None = None,
    sessions: dict[str, Any] | None = None,
    costs: CostConfig | None = None,
    htf_bars: dict[Timeframe, pd.DataFrame] | None = None,
    config: MetaConfig | None = None,
    extra_trials: int = 0,
) -> MetaReport:
    """``extra_trials``: rule candidates already tested on the same OOS data elsewhere (e.g.
    other strategies in the same run). It's added to the refined strategies' DSR trial count."""
    cfg = config or MetaConfig()
    costs = costs or CostConfig()
    research = bars[bars["ts"] < splits.holdout_start].reset_index(drop=True)
    htf = {
        tf: b[b["ts"] < splits.holdout_start].reset_index(drop=True)
        for tf, b in (htf_bars or {}).items()
    }
    fp = FeatureProvider(research, timeframe, calendar, sessions, htf)
    rs, re_ = splits.research_start, splits.holdout_start
    mine_end = rs + (re_ - rs) * cfg.mine_fraction
    test_start = mine_end + pd.Timedelta(days=cfg.embargo_days)

    base = run_backtest(
        genome,
        research,
        timeframe,
        calendar=calendar,
        sessions=sessions,
        costs=costs,
        htf_bars=htf,
        start=rs,
        end=re_,
        with_context=True,
        provider=fp,
    )
    trades = base.trades
    empty = MetaReport({}, {}, [], [], mine_end, test_start, re_, len(trades))
    if len(trades) < cfg.min_base_trades:
        empty.skipped_reason = f"only {len(trades)} trades (< {cfg.min_base_trades})"
        return empty

    x = trade_features(trades)
    r = trades["r"].to_numpy(dtype=float)
    entry = pd.DatetimeIndex(trades["entry_ts"])
    is_mine = np.asarray(entry < mine_end)
    is_test = np.asarray(entry >= test_start)
    xm, rm, xt, rt = x[is_mine], r[is_mine], x[is_test], r[is_test]
    report = MetaReport(_summary(rm), _summary(rt), [], [], mine_end, test_start, re_, len(trades))
    if len(rm) < cfg.min_base_trades // 2 or len(rt) < cfg.min_oos_trades:
        report.skipped_reason = "not enough trades in the mining or testing window"
        return report

    report.importance = importance(xm.reset_index(drop=True), rm, cfg.miner.seed)
    candidates = mine(xm.reset_index(drop=True), rm, cfg.miner)

    results: list[RuleResult] = []
    for cand in candidates:
        mask = cand.rule.mask(xt)
        kept, excl = rt[mask], rt[~mask]
        gains, losses = kept[kept > 0].sum(), -kept[kept <= 0].sum()
        results.append(
            RuleResult(
                candidate=cand,
                oos_n=len(kept),
                oos_mean_r=float(kept.mean()) if len(kept) else 0.0,
                oos_win_rate=float((kept > 0).mean()) if len(kept) else 0.0,
                oos_profit_factor=float(gains / losses) if losses > 0 else None,
                oos_coverage=len(kept) / len(rt),
                oos_lift=float(kept.mean() - excl.mean()) if len(kept) and len(excl) else 0.0,
                p_value=_welch_p(kept, excl),
                oos_mask=mask,
            )
        )
    if results:
        for res, adj in zip(results, holm([x.p_value for x in results]), strict=True):
            res.p_adjusted = adj

    n_trials = len(results) + extra_trials
    for res in results:
        if res.oos_n < cfg.min_oos_trades:
            res.reasons.append(f"only {res.oos_n} OOS trades (< {cfg.min_oos_trades})")
        if res.oos_mean_r <= 0:
            res.reasons.append(f"OOS expectancy {res.oos_mean_r:+.3f}R")
        if res.p_adjusted >= cfg.alpha:
            res.reasons.append(f"lift not significant (Holm p={res.p_adjusted:.3f})")
        if res.reasons:
            continue
        refined = refine_genome(genome, res.rule)
        if refined is None:
            res.reasons.append("rule can't be expressed within the genome limits")
            continue
        res.refined = refined
        res.refined_run = run_backtest(
            refined,
            research,
            timeframe,
            calendar=calendar,
            sessions=sessions,
            costs=costs,
            htf_bars=htf,
            start=test_start,
            end=re_,
            with_context=False,
            provider=fp,
        )
        eq = res.refined_run.equity
        day = (
            calendar.trading_day(pd.DatetimeIndex(eq.index))
            if calendar
            else pd.DatetimeIndex(eq.index).normalize()
        )
        res.refined_dsr = deflated_sharpe(
            daily_returns(eq, day, costs.initial_equity).to_numpy(), max(1, n_trials)
        )
        m = res.refined_run.metrics
        if (m["n_trades"] or 0) < cfg.min_oos_trades or (m["expectancy_r"] or 0) <= 0:
            res.reasons.append(
                f"refined strategy: {m['n_trades']} trades, "
                f"{m['expectancy_r']}R in the testing window"
            )
            continue
        res.validated = True

    # Many validated candidates are restatements of one finding. Keep the most significant
    # ones whose OOS trades differ substantially.
    chosen: list[RuleResult] = []
    for res in sorted((r for r in results if r.validated), key=lambda r: r.p_value):
        if len(chosen) >= cfg.max_rules:
            break
        assert res.oos_mask is not None
        overlap = max(
            (
                float((res.oos_mask & c.oos_mask).sum() / max((res.oos_mask | c.oos_mask).sum(), 1))
                for c in chosen
            ),
            default=0.0,
        )
        if overlap <= cfg.max_overlap:
            res.selected = True
            chosen.append(res)
    report.rules = results
    return report

"""Persist Meta-Analyzer results: context rules, refined strategies, and the analysis job."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ContextRule, Job, Strategy, strategy_lineage
from app.models.enums import JobKind, JobStatus, RuleStatus, Segment, StrategyOrigin, StrategyStatus
from app.models.strategy import genome_hash
from app.services.backtests import persist_run
from engine.meta.analyzer import MetaConfig, MetaReport, RuleResult
from engine.strategy.explain import explain


def _f(v: Any) -> float | None:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


def refined_overfit_risk(res: RuleResult, cfg: MetaConfig) -> float:
    """0 robust .. 1 noise: deflated Sharpe 50%, rule significance 30%, sample size 20%."""
    dsr = res.refined_dsr or 0.0
    n = (res.refined_run.metrics["n_trades"] if res.refined_run else 0) or 0
    sig = min(1.0, res.p_adjusted / cfg.alpha)
    sample = max(0.0, 1.0 - n / (2 * cfg.min_oos_trades))
    return round(0.5 * (1 - dsr) + 0.3 * sig + 0.2 * sample, 4)


async def persist_meta(
    session: AsyncSession,
    base: Strategy,
    report: MetaReport,
    *,
    config: MetaConfig | None = None,
) -> list[ContextRule]:
    cfg = config or MetaConfig()
    rules: list[ContextRule] = []
    for res in report.rules:
        c = res.candidate
        status = RuleStatus.VALIDATED if res.selected else RuleStatus.REJECTED
        reasons = list(res.reasons) or (
            ["restates a stronger validated rule"] if res.validated and not res.selected else []
        )
        rule = ContextRule(
            strategy_id=base.id,
            name=res.rule.describe()[:128],
            conditions=[cl.to_dict() for cl in res.rule.clauses],
            description=res.rule.describe(),
            miner=res.rule.miner,
            status=status,
            is_trades=c.n,
            is_win_rate=_f(c.win_rate),
            is_profit_factor=_f(c.profit_factor),
            is_expectancy_r=_f(c.mean_r),
            oos_trades=res.oos_n,
            oos_win_rate=_f(res.oos_win_rate),
            oos_profit_factor=_f(res.oos_profit_factor),
            oos_expectancy_r=_f(res.oos_mean_r),
            coverage=_f(res.oos_coverage),
            lift=_f(res.oos_lift),
            p_value=_f(res.p_value),
            p_value_adjusted=_f(res.p_adjusted),
            details={
                **res.to_dict(),
                "reasons": reasons,
                "windows": {
                    "mine_end": report.mine_end.isoformat(),
                    "test_start": report.test_start.isoformat(),
                },
            },
        )
        session.add(rule)
        rules.append(rule)
        if not (res.selected and res.refined is not None and res.refined_run is not None):
            continue
        await session.flush()
        canonical = res.refined.canonical()
        h = genome_hash(canonical)
        refined = await session.scalar(
            select(Strategy).where(
                Strategy.genome_hash == h,
                Strategy.instrument_id == base.instrument_id,
                Strategy.timeframe == base.timeframe,
            )
        )
        if refined is None:
            refined = Strategy.from_genome(
                canonical,
                instrument_id=base.instrument_id,
                timeframe=base.timeframe,
                run_id=base.run_id,
                origin=StrategyOrigin.META,
                description=explain(res.refined),
                generation_number=base.generation_number,
                island=base.island,
            )
            session.add(refined)
            await session.flush()
            await session.execute(
                insert(strategy_lineage), [{"child_id": refined.id, "parent_id": base.id}]
            )
        m = res.refined_run.metrics
        refined.oos_sharpe = _f(m.get("sharpe"))
        refined.oos_sortino = _f(m.get("sortino"))
        refined.oos_profit_factor = _f(m.get("profit_factor"))
        refined.oos_win_rate = _f(m.get("win_rate"))
        refined.oos_expectancy_r = _f(m.get("expectancy_r"))
        refined.oos_max_drawdown = _f(m.get("max_drawdown"))
        refined.oos_trades = int(m["n_trades"])
        refined.deflated_sharpe = _f(res.refined_dsr)
        refined.overfit_risk = refined_overfit_risk(res, cfg)
        refined.fitness = base.fitness
        refined.status = StrategyStatus.VALIDATED
        result = persist_run(
            session, refined, res.refined_run, segment=Segment.VALIDATION, context_rule_id=rule.id
        )
        result.extra_metrics = {**result.extra_metrics, "meta_rule": res.to_dict()}
        rule.details = {
            **rule.details,
            "refined_strategy_id": str(refined.id),
            "refined_name": refined.name,
        }
    return rules


async def record_meta_job(session: AsyncSession, base: Strategy, report: MetaReport) -> Job:
    now = datetime.now(UTC)
    job = Job(
        kind=JobKind.META_ANALYSIS,
        status=JobStatus.SUCCEEDED,
        progress=1.0,
        params={"strategy_id": str(base.id), "strategy": base.name},
        result=report.to_dict(),
        started_at=now,
        finished_at=now,
        message=f"{len(report.rules)} rules tested, {len(report.selected)} kept",
    )
    session.add(job)
    return job

"""Persist validation outcomes and enforce the one-shot holdout."""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BacktestResult, Strategy
from app.models.enums import Segment, StrategyStatus
from app.services.backtests import persist_run
from engine.backtest.runner import BacktestRun
from engine.validation.validator import ValidationReport


class HoldoutAlreadyEvaluatedError(RuntimeError):
    """A strategy's holdout may be looked at once. A second look would turn it into
    another in-sample period."""


def _f(v: Any) -> float | None:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


async def prior_oos_validations(
    session: AsyncSession, instrument_id: Any, timeframe: Any, exclude_run: Any = None
) -> int:
    """How many strategies have already been judged on this instrument/timeframe's OOS
    windows. Every earlier look at the same out-of-sample data is a trial that the next
    Deflated Sharpe must pay for (OOS reuse)."""
    stmt = (
        select(func.count())
        .select_from(Strategy)
        .where(
            Strategy.instrument_id == instrument_id,
            Strategy.timeframe == timeframe,
            Strategy.overfit_risk.is_not(None),
        )
    )
    if exclude_run is not None:
        stmt = stmt.where((Strategy.run_id != exclude_run) | Strategy.run_id.is_(None))
    return int(await session.scalar(stmt) or 0)


async def apply_validation(
    session: AsyncSession,
    strategy: Strategy,
    report: ValidationReport,
    context: dict[str, Any] | None = None,
) -> BacktestResult:
    """Store per-fold IS/OOS results plus the stitched OOS summary, update the strategy's
    leaderboard columns and set its status to VALIDATED or REJECTED."""
    for fold in report.folds:
        persist_run(
            session,
            strategy,
            fold.is_run,
            segment=Segment.IN_SAMPLE,
            fold_index=fold.index,
            store_trades=False,
        )
        persist_run(
            session,
            strategy,
            fold.oos_run,
            segment=Segment.WALK_FORWARD,
            fold_index=fold.index,
            store_trades=True,
        )

    m = report.oos_metrics
    summary = BacktestResult(
        strategy=strategy,
        segment=Segment.VALIDATION,
        fold_index=0,
        period_start=report.folds[0].oos_run.period_start.to_pydatetime(),
        period_end=report.folds[-1].oos_run.period_end.to_pydatetime(),
        n_trades=int(m["n_trades"]),
        total_return=_f(m.get("total_return")),
        sharpe=_f(m.get("sharpe")),
        sortino=_f(m.get("sortino")),
        calmar=_f(m.get("calmar")),
        max_drawdown=_f(m.get("max_drawdown")),
        profit_factor=_f(m.get("profit_factor")),
        win_rate=_f(m.get("win_rate")),
        expectancy_r=_f(m.get("expectancy_r")),
        avg_bars_held=_f(m.get("avg_bars_held")),
        extra_metrics={"validation_report": {**report.to_dict(), **(context or {})}},
        cost_model=report.folds[0].oos_run.costs.to_dict(),
        engine_version=report.folds[0].oos_run.engine_version,
    )
    session.add(summary)

    is_sharpes = [f.is_run.metrics.get("sharpe") for f in report.folds]
    valid_is = [s for s in is_sharpes if s is not None]
    strategy.is_sharpe = sum(valid_is) / len(valid_is) if valid_is else None
    strategy.oos_sharpe = _f(m.get("sharpe"))
    strategy.oos_sortino = _f(m.get("sortino"))
    strategy.oos_profit_factor = _f(m.get("profit_factor"))
    strategy.oos_win_rate = _f(m.get("win_rate"))
    strategy.oos_expectancy_r = _f(m.get("expectancy_r"))
    strategy.oos_max_drawdown = _f(m.get("max_drawdown"))
    strategy.oos_trades = int(m["n_trades"])
    strategy.deflated_sharpe = report.dsr
    strategy.pbo = report.pbo
    strategy.overfit_risk = report.overfit_risk
    strategy.status = StrategyStatus.VALIDATED if report.passed else StrategyStatus.REJECTED
    return summary


async def record_holdout(
    session: AsyncSession, strategy: Strategy, run: BacktestRun
) -> BacktestResult:
    """Persist the single permitted holdout evaluation. Raises if one exists already."""
    existing = await session.scalar(
        select(BacktestResult.id).where(
            BacktestResult.strategy_id == strategy.id, BacktestResult.segment == Segment.HOLDOUT
        )
    )
    if existing is not None:
        raise HoldoutAlreadyEvaluatedError(f"strategy {strategy.name} already used its holdout")
    result = persist_run(session, strategy, run, segment=Segment.HOLDOUT, store_trades=True)
    try:
        await session.flush()
    except IntegrityError as exc:  # lost a race with a concurrent evaluation
        await session.rollback()
        raise HoldoutAlreadyEvaluatedError(
            f"strategy {strategy.name} already used its holdout"
        ) from exc
    return result

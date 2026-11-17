"""Strategy queries. The Leaderboard (Step 11) builds on these filters."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import SessionDep
from app.models import BacktestResult, Instrument, Strategy, Trade
from app.models.enums import StrategyStatus, Timeframe
from app.schemas.common import Page
from app.schemas.strategy import StrategyDetail, StrategySort, StrategySummary

router = APIRouter(prefix="/strategies", tags=["leaderboard"])


@router.get("", response_model=Page[StrategySummary])
async def list_strategies(
    session: SessionDep,
    symbol: str | None = Query(None, description="e.g. XAUUSD"),
    timeframe: Timeframe | None = None,
    status_: list[StrategyStatus] | None = Query(None, alias="status"),
    run_id: uuid.UUID | None = None,
    min_trades: int | None = Query(None, ge=0),
    min_win_rate: float | None = Query(None, ge=0, le=1),
    min_profit_factor: float | None = Query(None, ge=0),
    min_deflated_sharpe: float | None = Query(None, ge=0, le=1),
    max_overfit_risk: float | None = Query(None, ge=0, le=1),
    max_drawdown: float | None = Query(None, ge=0, le=1),
    sort: StrategySort = StrategySort.FITNESS,
    descending: bool = True,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Page[StrategySummary]:
    stmt = select(Strategy)
    if symbol:
        stmt = stmt.join(Strategy.instrument).where(Instrument.symbol == symbol.upper())
    if timeframe:
        stmt = stmt.where(Strategy.timeframe == timeframe)
    if status_:
        stmt = stmt.where(Strategy.status.in_(status_))
    if run_id:
        stmt = stmt.where(Strategy.run_id == run_id)
    if min_trades is not None:
        stmt = stmt.where(Strategy.oos_trades >= min_trades)
    if min_win_rate is not None:
        stmt = stmt.where(Strategy.oos_win_rate >= min_win_rate)
    if min_profit_factor is not None:
        stmt = stmt.where(Strategy.oos_profit_factor >= min_profit_factor)
    if min_deflated_sharpe is not None:
        stmt = stmt.where(Strategy.deflated_sharpe >= min_deflated_sharpe)
    if max_overfit_risk is not None:
        stmt = stmt.where(Strategy.overfit_risk <= max_overfit_risk)
    if max_drawdown is not None:
        stmt = stmt.where(Strategy.oos_max_drawdown <= max_drawdown)

    total = await session.scalar(select(func.count()).select_from(stmt.subquery())) or 0

    column = getattr(Strategy, sort.value)
    order = column.desc().nulls_last() if descending else column.asc().nulls_last()
    rows = await session.scalars(stmt.order_by(order, Strategy.id).limit(limit).offset(offset))
    return Page(
        items=[StrategySummary.model_validate(s) for s in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{strategy_id}", response_model=StrategyDetail)
async def get_strategy(strategy_id: uuid.UUID, session: SessionDep) -> StrategyDetail:
    strategy = await session.scalar(
        select(Strategy).where(Strategy.id == strategy_id).options(selectinload(Strategy.parents))
    )
    if strategy is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "strategy not found")
    summary = StrategySummary.model_validate(strategy).model_dump()
    return StrategyDetail(
        **summary,
        genome=strategy.genome,
        genome_hash=strategy.genome_hash,
        genome_version=strategy.genome_version,
        description=strategy.description,
        parent_ids=[p.id for p in strategy.parents],
    )


@router.get("/{strategy_id}/backtests")
async def strategy_backtests(strategy_id: uuid.UUID, session: SessionDep) -> list[dict[str, Any]]:
    """Every stored result: walk-forward folds, validation summary, holdout, meta-rule tests."""
    rows = await session.scalars(
        select(BacktestResult)
        .where(BacktestResult.strategy_id == strategy_id)
        .order_by(BacktestResult.segment, BacktestResult.fold_index)
    )
    return [
        {
            "id": str(r.id),
            "segment": r.segment.value,
            "fold_index": r.fold_index,
            "context_rule_id": str(r.context_rule_id) if r.context_rule_id else None,
            "period_start": r.period_start.isoformat(),
            "period_end": r.period_end.isoformat(),
            "n_trades": r.n_trades,
            "total_return": r.total_return,
            "sharpe": r.sharpe,
            "sortino": r.sortino,
            "max_drawdown": r.max_drawdown,
            "profit_factor": r.profit_factor,
            "win_rate": r.win_rate,
            "expectancy_r": r.expectancy_r,
            "extra": r.extra_metrics,
        }
        for r in rows
    ]


@router.get("/{strategy_id}/trades")
async def strategy_trades(
    strategy_id: uuid.UUID, session: SessionDep, limit: int = Query(2000, ge=1, le=20000)
) -> list[dict[str, Any]]:
    """Stored trades (out-of-sample folds, holdout, refined-rule tests) with entry context."""
    rows = await session.execute(
        select(Trade, BacktestResult.segment)
        .join(BacktestResult, Trade.backtest_id == BacktestResult.id)
        .where(Trade.strategy_id == strategy_id)
        .order_by(Trade.entry_ts)
        .limit(limit)
    )
    return [
        {
            "segment": seg.value,
            "side": t.side.value,
            "entry_ts": t.entry_ts.isoformat(),
            "exit_ts": t.exit_ts.isoformat(),
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_price": t.stop_price,
            "target_price": t.target_price,
            "pnl_r": t.pnl_r,
            "exit_reason": t.exit_reason.value,
            "bars_held": t.bars_held,
            "context": t.context,
        }
        for t, seg in rows.all()
    ]

"""Persist backtest runs (BacktestResult + Trade rows) and load inputs for ad-hoc runs."""

from __future__ import annotations

import math
import uuid
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BacktestResult, Strategy, Trade
from app.models.enums import ExitReason, Segment, Side, Timeframe
from engine.backtest.runner import BacktestRun
from engine.data.store import ParquetStore

METRIC_COLUMNS = (
    "total_return",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "profit_factor",
    "win_rate",
    "expectancy_r",
    "avg_bars_held",
    "exposure",
)


def _num(v: Any) -> float | None:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


def persist_run(
    session: AsyncSession,
    strategy: Strategy,
    run: BacktestRun,
    *,
    segment: Segment,
    fold_index: int = 0,
    context_rule_id: uuid.UUID | None = None,
    store_trades: bool = True,
) -> BacktestResult:
    m = run.metrics
    if strategy.id is None:  # Python-side default is only applied at flush
        strategy.id = uuid.uuid4()
    result = BacktestResult(
        strategy=strategy,
        context_rule_id=context_rule_id,
        segment=segment,
        fold_index=fold_index,
        period_start=run.period_start.to_pydatetime(),
        period_end=run.period_end.to_pydatetime(),
        n_trades=int(m["n_trades"]),
        extra_metrics={k: v for k, v in m.items() if k not in METRIC_COLUMNS and k != "n_trades"},
        cost_model=run.costs.to_dict(),
        engine_version=run.engine_version,
        **{k: _num(m.get(k)) for k in METRIC_COLUMNS},
    )
    if store_trades:
        records: list[dict[str, Any]] = run.trades.to_dict(orient="records")  # type: ignore[assignment]
        for row in records:
            result.trades.append(
                Trade(
                    strategy_id=strategy.id,
                    side=Side(row["side"]),
                    entry_ts=pd.Timestamp(row["entry_ts"]).to_pydatetime(),
                    exit_ts=pd.Timestamp(row["exit_ts"]).to_pydatetime(),
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    stop_price=_num(row["stop_price"]),
                    target_price=_num(row["target_price"]),
                    size=float(row["lots"]),
                    pnl=float(row["pnl"]),
                    pnl_r=_num(row["r"]),
                    mae_r=_num(row["mae_r"]),
                    mfe_r=_num(row["mfe_r"]),
                    bars_held=int(row["bars_held"]),
                    exit_reason=ExitReason(row["exit_reason"]),
                    context=row.get("context") or {},
                )
            )
    session.add(result)
    return result


def load_bars_with_warmup(
    store: ParquetStore,
    symbol: str,
    timeframe: Timeframe,
    source: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup_bars: int = 3000,
) -> pd.DataFrame:
    """Bars for [start, end) plus ``warmup_bars`` of history so indicators are settled."""
    warm_start = start - pd.Timedelta(seconds=timeframe.seconds * warmup_bars * 1.6)
    return store.read_bars(symbol, timeframe, source=source, start=warm_start, end=end)

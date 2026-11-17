from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.enums import StrategyOrigin, StrategyStatus, Timeframe


class StrategySort(StrEnum):
    """Sortable leaderboard columns (allow-list keeps ORDER BY injection-proof)."""

    FITNESS = "fitness"
    OOS_SORTINO = "oos_sortino"
    OOS_SHARPE = "oos_sharpe"
    OOS_PROFIT_FACTOR = "oos_profit_factor"
    OOS_WIN_RATE = "oos_win_rate"
    OOS_EXPECTANCY_R = "oos_expectancy_r"
    OOS_MAX_DRAWDOWN = "oos_max_drawdown"
    OOS_TRADES = "oos_trades"
    DEFLATED_SHARPE = "deflated_sharpe"
    OVERFIT_RISK = "overfit_risk"
    CREATED_AT = "created_at"


class StrategySummary(BaseModel):
    """Leaderboard row."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    symbol: str
    timeframe: Timeframe
    status: StrategyStatus
    origin: StrategyOrigin
    run_id: uuid.UUID | None
    generation_number: int | None
    fitness: float | None
    is_sharpe: float | None
    oos_sharpe: float | None
    oos_sortino: float | None
    oos_profit_factor: float | None
    oos_win_rate: float | None
    oos_expectancy_r: float | None
    oos_max_drawdown: float | None
    oos_trades: int | None
    deflated_sharpe: float | None
    pbo: float | None
    overfit_risk: float | None
    created_at: datetime


class StrategyDetail(StrategySummary):
    genome: dict[str, Any]
    genome_hash: str
    genome_version: int
    description: str | None
    parent_ids: list[uuid.UUID]

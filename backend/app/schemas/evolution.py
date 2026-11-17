from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.enums import RunStatus, Timeframe


class GenerationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    island: int
    number: int
    population_size: int
    evaluated: int
    stats: dict[str, Any]
    pareto_size: int
    diversity: float | None
    duration_ms: int | None
    created_at: datetime


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    symbol: str
    timeframe: Timeframe
    status: RunStatus
    algorithm: str
    config: dict[str, Any]
    data_splits: dict[str, Any]
    seed: int
    current_generation: int
    evaluations: int
    best_metrics: dict[str, Any]
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime

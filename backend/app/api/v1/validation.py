"""Ad-hoc validation of a genome: walk-forward, deflated Sharpe, robustness, overfit risk."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import Instrument
from app.models.enums import DataSource, Timeframe
from engine.backtest.runner import CostConfig
from engine.data.calendar import MarketCalendar
from engine.data.store import ParquetStore
from engine.strategy.explain import explain
from engine.strategy.genome import Genome
from engine.validation.splits import make_splits
from engine.validation.validator import ValidationConfig, validate_genome

router = APIRouter(tags=["validation"])


class ValidateRequest(BaseModel):
    genome: Genome
    symbol: str = "XAUUSD"
    timeframe: Timeframe = Timeframe.M15
    source: DataSource = DataSource.DUKASCOPY
    start: datetime
    end: datetime
    holdout_fraction: float = Field(0.2, ge=0.0, lt=0.9)
    n_folds: int = Field(5, ge=2, le=20)
    n_trials: int = Field(1, ge=1, le=10_000_000, description="strategies tried before this one")
    min_oos_trades: int = Field(100, ge=10, le=10_000)
    n_perturbations: int = Field(16, ge=0, le=64)

    @model_validator(mode="after")
    def _check(self) -> ValidateRequest:
        if self.timeframe is Timeframe.TICK or self.end <= self.start:
            raise ValueError("need a bar timeframe and end > start")
        return self


@router.post("/validate")
async def validate(body: ValidateRequest, request: Request, session: SessionDep) -> dict[str, Any]:
    instrument = await session.scalar(
        select(Instrument).where(Instrument.symbol == body.symbol.upper())
    )
    if instrument is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown instrument {body.symbol}")
    calendar = MarketCalendar.from_instrument_config(
        instrument.session_timezone, instrument.trading_hours
    )
    store = ParquetStore(request.app.state.settings.data_dir / "parquet")
    try:
        splits = make_splits(
            body.start, body.end, holdout_fraction=body.holdout_fraction, n_folds=body.n_folds
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    warm = pd.Timestamp(body.start) - pd.Timedelta(seconds=body.timeframe.seconds * 3000 * 1.6)
    htf_needed = {f.timeframe for f in body.genome.features() if f.timeframe}

    def work() -> dict[str, Any]:
        bars = store.read_bars(
            instrument.symbol,
            body.timeframe,
            source=body.source,
            start=warm,
            end=splits.holdout_start,
        )
        if bars.empty:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no bars in the requested window")
        htf = {
            tf: store.read_bars(
                instrument.symbol, tf, source=body.source, start=warm, end=splits.holdout_start
            )
            for tf in htf_needed
        }
        report = validate_genome(
            body.genome,
            bars,
            body.timeframe,
            splits,
            calendar=calendar,
            sessions=instrument.sessions,
            costs=CostConfig.from_instrument(instrument),
            htf_bars=htf,
            n_trials=body.n_trials,
            config=ValidationConfig(
                min_oos_trades=body.min_oos_trades, n_perturbations=body.n_perturbations
            ),
        )
        return {"explanation": explain(body.genome), "splits": splits.to_dict(), **report.to_dict()}

    return await run_in_threadpool(work)

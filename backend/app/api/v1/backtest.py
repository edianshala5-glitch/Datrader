"""Ad-hoc backtests and strategy templates (the Lab's "try a strategy" panel)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import Instrument
from app.models.enums import DataSource, Timeframe
from app.services.backtests import load_bars_with_warmup
from engine.backtest.runner import CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.data.store import ParquetStore
from engine.strategy.explain import explain
from engine.strategy.genome import Genome
from engine.strategy.templates import TEMPLATES

router = APIRouter(tags=["backtest"])

MAX_BARS = 1_500_000


class BacktestRequest(BaseModel):
    genome: Genome
    symbol: str = "XAUUSD"
    timeframe: Timeframe = Timeframe.M15
    source: DataSource = DataSource.DUKASCOPY
    start: datetime
    end: datetime
    max_trades: int = Field(500, ge=0, le=5000)
    equity_points: int = Field(1000, ge=10, le=5000)

    @model_validator(mode="after")
    def _check(self) -> BacktestRequest:
        if self.timeframe is Timeframe.TICK:
            raise ValueError("backtests run on bar timeframes")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class BacktestResponse(BaseModel):
    explanation: str
    metrics: dict[str, Any]
    signals: dict[str, int]
    period_start: datetime
    period_end: datetime
    costs: dict[str, float]
    equity: dict[str, list[float] | list[int]]  # {"time": [...], "value": [...]}
    trades: list[dict[str, Any]]
    trades_truncated: bool


@router.get("/strategy-templates")
async def list_templates() -> list[dict[str, Any]]:
    out = []
    for key, raw in TEMPLATES.items():
        g = Genome.model_validate(raw)
        out.append({"key": key, "name": g.name, "explanation": explain(g), "genome": g.canonical()})
    return out


@router.post("/backtest", response_model=BacktestResponse)
async def backtest(
    body: BacktestRequest, request: Request, session: SessionDep
) -> BacktestResponse:
    instrument = await session.scalar(
        select(Instrument).where(Instrument.symbol == body.symbol.upper())
    )
    if instrument is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown instrument {body.symbol}")
    calendar = MarketCalendar.from_instrument_config(
        instrument.session_timezone, instrument.trading_hours
    )
    sessions = instrument.sessions
    costs = CostConfig.from_instrument(instrument)
    store = ParquetStore(request.app.state.settings.data_dir / "parquet")
    start, end = pd.Timestamp(body.start), pd.Timestamp(body.end)
    htf_needed = {f.timeframe for f in body.genome.features() if f.timeframe}

    def work() -> BacktestResponse:
        bars = load_bars_with_warmup(
            store, instrument.symbol, body.timeframe, body.source, start, end
        )
        if bars.empty or bars["ts"].iloc[-1] < start:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no bars in the requested window")
        if len(bars) > MAX_BARS:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "window too large")
        htf = {
            tf: load_bars_with_warmup(store, instrument.symbol, tf, body.source, start, end, 500)
            for tf in htf_needed
        }
        try:
            run = run_backtest(
                body.genome,
                bars,
                body.timeframe,
                calendar=calendar,
                sessions=sessions,
                costs=costs,
                htf_bars=htf,
                start=start,
                end=end,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

        eq = run.equity
        step = max(1, len(eq) // body.equity_points)
        sampled = eq.iloc[::step]
        if sampled.index[-1] != eq.index[-1]:
            sampled = pd.concat([sampled, eq.iloc[-1:]])
        trades = run.trades.head(body.max_trades).copy()
        for c in ("entry_ts", "exit_ts"):
            trades[c] = trades[c].map(lambda t: t.isoformat())
        records = trades.replace({np.nan: None}).to_dict(orient="records")
        return BacktestResponse(
            explanation=explain(body.genome),
            metrics=run.metrics,
            signals=run.signals,
            period_start=run.period_start.to_pydatetime(),
            period_end=run.period_end.to_pydatetime(),
            costs=run.costs.to_dict(),
            equity={
                "time": (pd.DatetimeIndex(sampled.index).asi8 // 1_000_000_000).tolist(),
                "value": np.round(sampled.to_numpy(), 2).tolist(),
            },
            trades=records,
            trades_truncated=len(run.trades) > body.max_trades,
        )

    return await run_in_threadpool(work)

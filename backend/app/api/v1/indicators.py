"""Indicator catalogue and chart overlays (SMC markers and FVG zones)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import Instrument
from app.models.enums import DataSource, Timeframe
from engine.data.calendar import MarketCalendar
from engine.data.store import ParquetStore
from engine.indicators import REGISTRY, Inputs
from engine.indicators.kernels import confirmed_swings, fair_value_gaps
from engine.indicators.smc import structure
from engine.indicators.standard import atr_values

router = APIRouter(tags=["indicators"])

WARMUP_BARS = 500  # history before the window so structure state is established


@router.get("/indicators")
async def list_indicators() -> list[dict[str, Any]]:
    """Every indicator with parameter ranges and typed outputs (drives the strategy DSL)."""
    return [REGISTRY[name].to_dict() for name in sorted(REGISTRY)]


class Marker(BaseModel):
    time: int  # unix seconds of the bar the marker belongs to
    confirmed_time: int  # when it became knowable (== time except for swing pivots)
    kind: str
    price: float


class Zone(BaseModel):
    kind: str  # fvg_bull | fvg_bear
    start_time: int
    end_time: int | None  # None while still live
    top: float
    bottom: float


class Overlays(BaseModel):
    symbol: str
    timeframe: Timeframe
    markers: list[Marker]
    zones: list[Zone]


def _secs(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000_000)


def build_overlays(
    bars: pd.DataFrame,
    timeframe: Timeframe,
    calendar: MarketCalendar,
    window_start: pd.Timestamp,
    swing_length: int,
    fvg_min_atr: float,
) -> tuple[list[Marker], list[Zone]]:
    inp = Inputs.from_bars(bars, timeframe, calendar)
    ts = inp.ts
    sh_e, sl_e, sh_p, sl_p, sh_i, sl_i = confirmed_swings(
        inp.high, inp.low, swing_length, swing_length
    )
    st = structure(inp, swing_length, "close", 0.1)
    in_window = np.asarray(ts >= window_start)
    markers: list[Marker] = []
    for t in np.flatnonzero(in_window & (sh_e | sl_e)):
        for ev, price, piv, kind in (
            (sh_e, sh_p, sh_i, "swing_high"),
            (sl_e, sl_p, sl_i, "swing_low"),
        ):
            if ev[t]:
                markers.append(
                    Marker(
                        time=_secs(ts[int(piv[t])]),
                        confirmed_time=_secs(ts[int(t)]),
                        kind=kind,
                        price=float(price[t]),
                    )
                )
    level_hi_prev = np.concatenate([[np.nan], st[9][:-1]])
    level_lo_prev = np.concatenate([[np.nan], st[10][:-1]])
    for idx, kind, levels in (
        (1, "bos_bull", level_hi_prev),
        (3, "choch_bull", level_hi_prev),
        (5, "sweep_high", level_hi_prev),
        (2, "bos_bear", level_lo_prev),
        (4, "choch_bear", level_lo_prev),
        (6, "sweep_low", level_lo_prev),
    ):
        for t in np.flatnonzero(in_window & st[idx]):
            markers.append(
                Marker(
                    time=_secs(ts[int(t)]),
                    confirmed_time=_secs(ts[int(t)]),
                    kind=kind,
                    price=float(levels[t]),
                )
            )
    markers.sort(key=lambda m: (m.confirmed_time, m.kind))

    # end_idx is future information: fine for drawing history, never used as a feature.
    nb, ns, *_rest, end_idx = fair_value_gaps(
        inp.high, inp.low, inp.close, atr_values(inp, 14), fvg_min_atr, 64
    )
    zones: list[Zone] = []
    for t in np.flatnonzero(in_window & (nb | ns)):
        end = int(end_idx[t])
        bull = bool(nb[t])
        zones.append(
            Zone(
                kind="fvg_bull" if bull else "fvg_bear",
                start_time=_secs(ts[int(t) - 2]),
                end_time=_secs(ts[end]) if end >= 0 else None,
                top=float(inp.low[t] if bull else inp.low[t - 2]),
                bottom=float(inp.high[t - 2] if bull else inp.high[t]),
            )
        )
    return markers, zones


@router.get("/data/{symbol}/overlays", response_model=Overlays)
async def get_overlays(
    request: Request,
    symbol: str,
    session: SessionDep,
    timeframe: Timeframe = Timeframe.M15,
    source: DataSource = DataSource.DUKASCOPY,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(1000, ge=10, le=10000),
    swing_length: int = Query(5, ge=2, le=30),
    fvg_min_atr: float = Query(0.1, ge=0, le=2),
) -> Overlays:
    sym = symbol.upper()
    instrument = await session.scalar(select(Instrument).where(Instrument.symbol == sym))
    calendar = (
        MarketCalendar.from_instrument_config(instrument.session_timezone, instrument.trading_hours)
        if instrument
        else MarketCalendar.xauusd()
    )
    store = ParquetStore(request.app.state.settings.data_dir / "parquet")

    def load() -> tuple[list[Marker], list[Zone]]:
        if start is None:
            bars = store.read_bars(
                sym, timeframe, source=source, end=end, limit=limit + WARMUP_BARS, tail=True
            )
            if bars.empty:
                return [], []
            window_start = bars["ts"].iloc[max(0, len(bars) - limit)]
        else:
            window_start = pd.Timestamp(start)
            warm = window_start - pd.Timedelta(seconds=timeframe.seconds * WARMUP_BARS * 2)
            bars = store.read_bars(sym, timeframe, source=source, start=warm, end=end)
            if bars.empty:
                return [], []
        return build_overlays(bars, timeframe, calendar, window_start, swing_length, fvg_min_atr)

    markers, zones = await run_in_threadpool(load)
    return Overlays(symbol=sym, timeframe=timeframe, markers=markers, zones=zones)

"""Champions, signals and live-engine control."""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import Champion, Signal, Strategy
from app.models.enums import ChampionStatus, DataSource, SignalKind, SignalStatus, Timeframe
from app.services.champions import PromotionError, demote, evaluate_champions, promote
from app.services.live import LiveService, signal_dict
from app.services.validation import HoldoutAlreadyEvaluatedError
from engine.data.store import ParquetStore
from engine.live.feeds import BarFeed, OandaFeed, ReplayFeed
from engine.live.notify import Notifier, TelegramNotifier, WebhookNotifier

router = APIRouter(tags=["live"])


def _champion_dict(c: Champion, s: Strategy) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "strategy_id": str(s.id),
        "name": s.name,
        "description": s.description,
        "timeframe": s.timeframe.value,
        "status": c.status.value,
        "rank": c.rank,
        "promoted_at": c.promoted_at.isoformat(),
        "activated_at": c.activated_at.isoformat() if c.activated_at else None,
        "demoted_at": c.demoted_at.isoformat() if c.demoted_at else None,
        "demotion_reason": c.demotion_reason,
        "snapshot": c.promotion_snapshot,
        "live": c.live_stats,
        "context_rule_id": str(c.context_rule_id) if c.context_rule_id else None,
    }


@router.get("/champions")
async def list_champions(
    session: SessionDep, include_demoted: bool = False
) -> list[dict[str, Any]]:
    stmt = select(Champion, Strategy).join(Strategy, Champion.strategy_id == Strategy.id)
    if not include_demoted:
        stmt = stmt.where(Champion.status != ChampionStatus.DEMOTED)
    rows = (
        await session.execute(stmt.order_by(Champion.rank.asc().nulls_last(), Champion.promoted_at))
    ).all()
    return [_champion_dict(c, s) for c, s in rows]


class PromoteRequest(BaseModel):
    source: DataSource | None = None
    start: datetime | None = None
    end: datetime | None = None


def _mark_dirty(request: Request) -> None:
    live = getattr(request.app.state, "live", None)
    if live and live.get("service"):
        live["service"].dirty = True


@router.post("/champions/promote/{strategy_id}", status_code=status.HTTP_201_CREATED)
async def promote_strategy(
    strategy_id: uuid.UUID, body: PromoteRequest, request: Request, session: SessionDep
) -> dict[str, Any]:
    """Run the strategy's one-time holdout evaluation; on success it starts incubating."""
    strategy = await session.get(Strategy, strategy_id)
    if strategy is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "strategy not found")
    store = ParquetStore(request.app.state.settings.data_dir / "parquet")
    try:
        result = await promote(
            session, store, strategy, source=body.source, start=body.start, end=body.end
        )
    except HoldoutAlreadyEvaluatedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except PromotionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    await session.commit()
    if not result.passed:
        raise HTTPException(
            status.HTTP_409_CONFLICT, {"reason": result.reason, "holdout": result.holdout}
        )
    assert result.champion is not None
    _mark_dirty(request)
    return {**_champion_dict(result.champion, strategy), "holdout": result.holdout}


class DemoteRequest(BaseModel):
    reason: str = "manual"


@router.post("/champions/{champion_id}/demote")
async def demote_champion(
    champion_id: uuid.UUID, body: DemoteRequest, request: Request, session: SessionDep
) -> dict[str, Any]:
    champ = await session.get(Champion, champion_id)
    if champ is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "champion not found")
    await demote(session, champ, body.reason)
    await session.commit()
    _mark_dirty(request)
    strat = await session.get(Strategy, champ.strategy_id)
    assert strat is not None
    return _champion_dict(champ, strat)


@router.post("/champions/evaluate")
async def evaluate(request: Request, session: SessionDep) -> list[dict[str, Any]]:
    changes = await evaluate_champions(session)
    await session.commit()
    if changes:
        _mark_dirty(request)
    return changes


@router.get("/signals")
async def list_signals(
    session: SessionDep,
    status_: list[SignalStatus] | None = Query(None, alias="status"),
    kind: SignalKind | None = None,
    champion_id: uuid.UUID | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    stmt = select(Signal)
    if status_:
        stmt = stmt.where(Signal.status.in_(status_))
    if kind:
        stmt = stmt.where(Signal.kind == kind)
    if champion_id:
        stmt = stmt.where(Signal.champion_id == champion_id)
    rows = await session.scalars(
        stmt.order_by(Signal.emitted_at.desc(), Signal.bar_ts.desc()).limit(limit)
    )
    return [signal_dict(s) for s in rows]


@router.get("/signals/{signal_id}")
async def get_signal(signal_id: uuid.UUID, session: SessionDep) -> dict[str, Any]:
    sig = await session.get(Signal, signal_id)
    if sig is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "signal not found")
    return signal_dict(sig)


class LiveStartRequest(BaseModel):
    feed: Literal["replay", "oanda"] = "oanda"
    symbol: str = "XAUUSD"
    history_source: DataSource = DataSource.DUKASCOPY
    replay_start: datetime | None = None  # replay: bars after this time are "live"
    bars_per_second: float = 0.0


def notifiers_from_settings(settings: Any) -> list[Notifier]:
    out: list[Notifier] = []
    if settings.notify_webhook_url:
        out.append(WebhookNotifier(settings.notify_webhook_url))
    if settings.telegram_bot_token and settings.telegram_chat_id:
        out.append(TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id))
    return out


@router.post("/live/start", status_code=status.HTTP_202_ACCEPTED)
async def live_start(body: LiveStartRequest, request: Request) -> dict[str, Any]:
    state = request.app.state
    if getattr(state, "live", None) and not state.live["task"].done():
        raise HTTPException(status.HTTP_409_CONFLICT, "live engine already running")
    settings = state.settings
    store = ParquetStore(settings.data_dir / "parquet")
    feed: BarFeed
    if body.feed == "replay":
        if body.replay_start is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "replay needs replay_start")
        m1 = store.read_bars(body.symbol, Timeframe.M1, source=body.history_source)
        if m1.empty:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no M1 bars to replay")
        feed = ReplayFeed(m1, pd.Timestamp(body.replay_start), bars_per_second=body.bars_per_second)
    else:
        feed = OandaFeed(
            token=settings.oanda_token,
            account_id=settings.oanda_account_id,
            practice=settings.oanda_practice,
        )
    service = LiveService(
        state.sessionmaker,
        store,
        state.event_bus,
        symbol=body.symbol,
        history_source=body.history_source,
        notifiers=notifiers_from_settings(settings),
    )
    stop = threading.Event()
    task = asyncio.create_task(service.run(feed, stop), name="live-engine")
    state.live = {"task": task, "stop": stop, "service": service}
    return {"started": True, "feed": body.feed}


@router.post("/live/stop")
async def live_stop(request: Request) -> dict[str, Any]:
    live = getattr(request.app.state, "live", None)
    if not live or live["task"].done():
        raise HTTPException(status.HTTP_409_CONFLICT, "live engine is not running")
    live["stop"].set()
    return {"stopping": True}


@router.get("/live/status")
async def live_status(request: Request) -> dict[str, Any]:
    live = getattr(request.app.state, "live", None)
    if not live:
        return {"running": False}
    st = dict(live["service"].status)
    if live["task"].done() and live["task"].exception():
        st["error"] = repr(live["task"].exception())
    return st

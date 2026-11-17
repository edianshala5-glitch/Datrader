"""Champion lifecycle: validated → holdout (once) → incubating (paper) → active (top 3) → demoted.

* **Promotion** runs the single permitted holdout evaluation. Failing it rejects the strategy.
* **Incubation**: the champion paper-trades live data (signals are recorded, not pushed).
* **Activation**: after enough live trades and days with positive expectancy, the best
  ``max_active`` champions (ranked by live significance) become active and notify.
* **Demotion**: live drawdown beyond the limit, or live expectancy significantly below the
  backtest's (one-sided test), retires the strategy.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Champion, ContextRule, EvolutionRun, Instrument, Signal, Strategy
from app.models.enums import (
    ChampionStatus,
    DataSource,
    RuleStatus,
    SignalKind,
    SignalStatus,
    StrategyStatus,
)
from app.services.validation import record_holdout
from engine.backtest.runner import CostConfig
from engine.data.calendar import MarketCalendar
from engine.data.store import ParquetStore
from engine.strategy.genome import Genome
from engine.validation.splits import DataSplits, make_splits
from engine.validation.validator import evaluate_holdout

_N = NormalDist()
CLOSED_STATUSES = (SignalStatus.TARGET_HIT, SignalStatus.STOP_HIT, SignalStatus.CLOSED)


@dataclass(frozen=True)
class ChampionPolicy:
    holdout_min_trades: int = 20
    incubation_min_trades: int = 10
    incubation_min_days: float = 5.0
    max_active: int = 3
    max_drawdown_r: float = 10.0  # live cumulative-R drawdown that demotes
    degradation_p: float = 0.01
    degradation_min_trades: int = 20


class PromotionError(RuntimeError):
    pass


@dataclass
class PromotionResult:
    champion: Champion | None
    passed: bool
    reason: str
    holdout: dict[str, Any]


async def _splits_for(
    session: AsyncSession, strategy: Strategy, start: datetime | None, end: datetime | None
) -> tuple[DataSplits, DataSource]:
    run = await session.get(EvolutionRun, strategy.run_id) if strategy.run_id else None
    if run is not None:
        return DataSplits.from_dict(run.data_splits), DataSource(
            run.config.get("source", "dukascopy")
        )
    if start and end:
        return make_splits(start, end), DataSource.DUKASCOPY
    raise PromotionError("strategy has no run: pass start/end to define its holdout")


async def promote(
    session: AsyncSession,
    store: ParquetStore,
    strategy: Strategy,
    *,
    policy: ChampionPolicy | None = None,
    source: DataSource | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> PromotionResult:
    policy = policy or ChampionPolicy()
    if strategy.status is not StrategyStatus.VALIDATED:
        raise PromotionError(
            f"only validated strategies can be promoted (status: {strategy.status.value})"
        )
    splits, run_source = await _splits_for(session, strategy, start, end)
    src = source or run_source
    instrument = await session.get(Instrument, strategy.instrument_id)
    assert instrument is not None
    tf = strategy.timeframe
    genome = Genome.model_validate(strategy.genome)
    warm = splits.holdout_start - pd.Timedelta(seconds=tf.seconds * 3000 * 1.6)
    bars = store.read_bars(instrument.symbol, tf, source=src, start=warm, end=splits.holdout_end)
    if bars.empty or bars["ts"].iloc[-1] < splits.holdout_start:
        raise PromotionError("no bars cover the holdout period")
    htf = {
        f.timeframe: store.read_bars(
            instrument.symbol, f.timeframe, source=src, start=warm, end=splits.holdout_end
        )
        for f in genome.features()
        if f.timeframe
    }
    cal = MarketCalendar.from_instrument_config(
        instrument.session_timezone, instrument.trading_hours
    )
    run = evaluate_holdout(
        genome,
        bars,
        tf,
        splits,
        calendar=cal,
        sessions=instrument.sessions,
        costs=CostConfig.from_instrument(instrument),
        htf_bars=htf,
    )
    await record_holdout(session, strategy, run)  # raises if the holdout was already used
    m = run.metrics
    holdout = {
        k: m.get(k)
        for k in ("n_trades", "expectancy_r", "win_rate", "profit_factor", "sharpe", "max_drawdown")
    }
    n, exp = int(m["n_trades"]), float(m["expectancy_r"] or 0.0)
    if n < policy.holdout_min_trades or exp <= 0:
        strategy.status = StrategyStatus.REJECTED
        reason = (
            f"holdout: {n} trades, {exp:+.3f}R "
            f"(needs >= {policy.holdout_min_trades} trades and > 0R)"
        )
        return PromotionResult(None, False, reason, holdout)

    rule_id = None
    if strategy.origin.value == "meta":
        rules = await session.scalars(
            select(ContextRule).where(ContextRule.status == RuleStatus.VALIDATED)
        )
        rule_id = next(
            (r.id for r in rules if r.details.get("refined_strategy_id") == str(strategy.id)), None
        )
    champ = Champion(
        strategy_id=strategy.id,
        context_rule_id=rule_id,
        status=ChampionStatus.INCUBATING,
        promoted_at=datetime.now(UTC),
        promotion_snapshot={
            "oos_expectancy_r": strategy.oos_expectancy_r,
            "oos_trades": strategy.oos_trades,
            "oos_win_rate": strategy.oos_win_rate,
            "deflated_sharpe": strategy.deflated_sharpe,
            "overfit_risk": strategy.overfit_risk,
            "holdout": holdout,
            "holdout_expectancy_r": holdout["expectancy_r"],
        },
        live_stats={},
    )
    session.add(champ)
    strategy.status = StrategyStatus.INCUBATING
    return PromotionResult(champ, True, "holdout passed; incubating", holdout)


def live_stats(r: list[float], first: datetime | None) -> dict[str, Any]:
    if not r:
        return {"n": 0}
    arr = np.asarray(r, dtype=float)
    cum = np.cumsum(arr)
    dd = (
        float(np.max(np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum))
        if len(cum)
        else 0.0
    )
    return {
        "n": len(arr),
        "mean_r": float(arr.mean()),
        "sd_r": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "win_rate": float((arr > 0).mean()),
        "total_r": float(cum[-1]),
        "max_drawdown_r": max(dd, 0.0),
        "first_trade": first.isoformat() if first else None,
    }


async def _closed_r(
    session: AsyncSession, champion_id: uuid.UUID
) -> tuple[list[float], datetime | None]:
    rows = (
        await session.execute(
            select(Signal.pnl_r, Signal.closed_at)
            .where(
                Signal.champion_id == champion_id,
                Signal.kind == SignalKind.ENTRY,
                Signal.status.in_(CLOSED_STATUSES),
                Signal.pnl_r.is_not(None),
            )
            .order_by(Signal.closed_at)
        )
    ).all()
    return [float(r) for r, _ in rows if r is not None], (rows[0][1] if rows else None)


async def evaluate_champions(
    session: AsyncSession, policy: ChampionPolicy | None = None, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Apply demotion / activation / ranking. Returns a list of changes."""
    policy = policy or ChampionPolicy()
    now = now or datetime.now(UTC)
    champs = list(
        await session.scalars(
            select(Champion).where(
                Champion.status.in_((ChampionStatus.INCUBATING, ChampionStatus.ACTIVE))
            )
        )
    )
    changes: list[dict[str, Any]] = []
    eligible: list[tuple[float, Champion]] = []
    for c in champs:
        r, first = await _closed_r(session, c.id)
        st = live_stats(r, first)
        c.live_stats = st
        expected = c.promotion_snapshot.get("oos_expectancy_r") or 0.0
        demote = None
        if st["n"] and st["max_drawdown_r"] > policy.max_drawdown_r:
            demote = f"live drawdown {st['max_drawdown_r']:.1f}R > {policy.max_drawdown_r}R"
        elif st["n"] >= policy.degradation_min_trades and st.get("sd_r"):
            z = (st["mean_r"] - expected) / (st["sd_r"] / math.sqrt(st["n"]))
            if _N.cdf(z) < policy.degradation_p:
                demote = (
                    f"live {st['mean_r']:+.3f}R vs backtest {expected:+.3f}R (p={_N.cdf(z):.4f})"
                )
        if demote:
            c.status, c.rank, c.demoted_at, c.demotion_reason = (
                ChampionStatus.DEMOTED,
                None,
                now,
                demote,
            )
            strat = await session.get(Strategy, c.strategy_id)
            if strat:
                strat.status = StrategyStatus.RETIRED
            changes.append({"champion_id": str(c.id), "change": "demoted", "reason": demote})
            continue
        days = (now - c.promoted_at).total_seconds() / 86400
        ready = (
            st["n"] >= policy.incubation_min_trades
            and days >= policy.incubation_min_days
            and st["mean_r"] > 0
        )
        if c.status is ChampionStatus.ACTIVE or ready:
            score = st["mean_r"] * math.sqrt(st["n"]) if st["n"] else 0.0
            eligible.append((score, c))
    eligible.sort(key=lambda x: -x[0])
    for i, (_, c) in enumerate(eligible):
        strat = await session.get(Strategy, c.strategy_id)
        if i < policy.max_active:
            if c.status is not ChampionStatus.ACTIVE:
                c.status, c.activated_at = ChampionStatus.ACTIVE, now
                changes.append({"champion_id": str(c.id), "change": "activated", "rank": i + 1})
            c.rank = i + 1
            if strat:
                strat.status = StrategyStatus.CHAMPION
        elif c.status is ChampionStatus.ACTIVE:  # bumped by a better champion
            c.status, c.rank = ChampionStatus.INCUBATING, None
            if strat:
                strat.status = StrategyStatus.INCUBATING
            changes.append({"champion_id": str(c.id), "change": "bumped"})
    return changes


async def demote(session: AsyncSession, champion: Champion, reason: str) -> None:
    champion.status, champion.rank = ChampionStatus.DEMOTED, None
    champion.demoted_at, champion.demotion_reason = datetime.now(UTC), reason
    strat = await session.get(Strategy, champion.strategy_id)
    if strat:
        strat.status = StrategyStatus.RETIRED

"""Model behaviour against a real database (SQLite always, Postgres when configured)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    BacktestResult,
    Champion,
    ContextRule,
    EvolutionRun,
    Generation,
    Instrument,
    Signal,
    Strategy,
    Trade,
)
from app.models.enums import (
    ChampionStatus,
    ExitReason,
    RunStatus,
    Segment,
    Side,
    SignalKind,
    StrategyOrigin,
    StrategyStatus,
    Timeframe,
)
from app.services.seed import seed_reference_data

T0 = datetime(2024, 3, 5, 14, 30, tzinfo=UTC)  # a Tuesday, NY AM session


def make_strategy(instrument: Instrument, genome: dict, **kw) -> Strategy:  # type: ignore[type-arg]
    kw.setdefault("origin", StrategyOrigin.RANDOM)
    return Strategy.from_genome(genome, instrument=instrument, timeframe=Timeframe.M15, **kw)


async def test_seed_is_idempotent(session: AsyncSession, xauusd: Instrument) -> None:
    await seed_reference_data(session)
    await session.commit()
    assert await session.scalar(select(func.count()).select_from(Instrument)) == 1
    assert xauusd.cost_model["commission_per_lot_round_turn"] == 7.0
    assert xauusd.sessions["ny_am_kz"]["tz"] == "America/New_York"


async def test_full_pipeline_graph(session: AsyncSession, xauusd: Instrument) -> None:
    run = EvolutionRun(
        name="xau-m15-001",
        instrument=xauusd,
        timeframe=Timeframe.M15,
        status=RunStatus.RUNNING,
        config={"population": 200, "islands": 4},
        data_splits={"holdout_start": "2025-01-01T00:00:00Z"},
        seed=42,
    )
    run.generations.append(Generation(island=0, number=0, population_size=200, stats={}))
    mom = make_strategy(xauusd, {"entry": "a"}, run=run, generation_number=0)
    dad = make_strategy(xauusd, {"entry": "b"}, run=run, generation_number=0)
    child = make_strategy(
        xauusd, {"entry": "ab"}, run=run, generation_number=1, origin=StrategyOrigin.CROSSOVER
    )
    child.parents.extend([mom, dad])
    session.add_all([run, mom, dad, child])
    await session.flush()

    bt = BacktestResult(
        strategy=child,
        segment=Segment.WALK_FORWARD,
        fold_index=2,
        period_start=T0 - timedelta(days=90),
        period_end=T0,
        n_trades=1,
        sharpe=1.4,
        engine_version="0.1.0",
    )
    bt.trades.append(
        Trade(
            strategy_id=child.id,
            side=Side.LONG,
            entry_ts=T0,
            exit_ts=T0 + timedelta(minutes=45),
            entry_price=2125.40,
            exit_price=2131.90,
            stop_price=2122.10,
            target_price=2131.90,
            size=0.5,
            pnl=318.0,
            pnl_r=1.97,
            bars_held=3,
            exit_reason=ExitReason.TARGET,
            context={"weekday": "tuesday", "hour_ny": 9, "choch_bull_bars_ago": 2},
        )
    )
    rule = ContextRule(
        strategy=child,
        name="Tue NY-AM + bullish CHoCH",
        conditions=[{"feature": "weekday", "op": "in", "value": ["tuesday"]}],
        description="Only on Tuesdays 09:00-11:00 NY after a bullish CHoCH",
        miner="decision_tree",
    )
    session.add_all([bt, rule])
    await session.flush()

    champ = Champion(
        strategy=child, context_rule=rule, status=ChampionStatus.ACTIVE, rank=1, promoted_at=T0
    )
    signal = Signal(
        strategy_id=child.id,
        instrument_id=xauusd.id,
        timeframe=Timeframe.M15,
        kind=SignalKind.ENTRY,
        side=Side.LONG,
        bar_ts=T0,
        emitted_at=T0 + timedelta(seconds=2),
        entry_price=2125.40,
        rationale={"conditions": [{"label": "Bullish CHoCH", "met": True}]},
    )
    session.add_all([champ, signal])
    await session.commit()
    session.expunge_all()

    loaded = await session.scalar(
        select(Strategy)
        .where(Strategy.id == child.id)
        .options(
            selectinload(Strategy.parents),
            selectinload(Strategy.backtests).selectinload(BacktestResult.trades),
            selectinload(Strategy.context_rules),
        )
    )
    assert loaded is not None
    assert loaded.symbol == "XAUUSD"
    assert {p.name for p in loaded.parents} == {mom.name, dad.name}
    trade = loaded.backtests[0].trades[0]
    assert trade.entry_ts == T0 and trade.entry_ts.tzinfo is not None
    assert trade.context["choch_bull_bars_ago"] == 2
    assert loaded.context_rules[0].status.value == "proposed"
    assert loaded.status is StrategyStatus.CANDIDATE


async def test_deleting_strategy_cascades(session: AsyncSession, xauusd: Instrument) -> None:
    s = make_strategy(xauusd, {"g": 1})
    bt = BacktestResult(
        strategy=s,
        segment=Segment.IN_SAMPLE,
        period_start=T0,
        period_end=T0,
        engine_version="0.1.0",
    )
    session.add_all([s, bt])
    await session.flush()
    session.add(
        Trade(
            strategy_id=s.id,
            backtest_id=bt.id,
            side=Side.SHORT,
            entry_ts=T0,
            exit_ts=T0,
            entry_price=1,
            exit_price=1,
            size=1,
            pnl=0,
            bars_held=0,
            exit_reason=ExitReason.STOP,
        )
    )
    await session.commit()

    await session.delete(s)
    await session.commit()
    for model in (BacktestResult, Trade):
        assert await session.scalar(select(func.count()).select_from(model)) == 0


async def test_duplicate_genome_rejected(session: AsyncSession, xauusd: Instrument) -> None:
    session.add(make_strategy(xauusd, {"same": True}))
    await session.commit()
    session.add(make_strategy(xauusd, {"same": True}))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()
    # Same genome on another timeframe is a different strategy.
    session.add(
        Strategy.from_genome(
            {"same": True},
            instrument=xauusd,
            timeframe=Timeframe.M5,
            origin=StrategyOrigin.RANDOM,
        )
    )
    await session.commit()


async def test_signal_emission_is_idempotent(session: AsyncSession, xauusd: Instrument) -> None:
    s = make_strategy(xauusd, {"sig": 1})
    session.add(s)
    await session.flush()

    def entry() -> Signal:
        return Signal(
            strategy_id=s.id,
            instrument_id=xauusd.id,
            timeframe=Timeframe.M15,
            kind=SignalKind.ENTRY,
            side=Side.LONG,
            bar_ts=T0,
            emitted_at=T0,
        )

    session.add(entry())
    await session.commit()
    session.add(entry())  # feed reconnect replays the same bar
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_champion_rank_constraint(session: AsyncSession, xauusd: Instrument) -> None:
    s = make_strategy(xauusd, {"champ": 1})
    session.add(Champion(strategy=s, rank=4, promoted_at=T0))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_naive_datetime_rejected(session: AsyncSession, xauusd: Instrument) -> None:
    s = make_strategy(xauusd, {"tz": 1})
    session.add(Champion(strategy=s, promoted_at=datetime(2024, 1, 1, 12, 0)))
    with pytest.raises(Exception, match="naive datetime"):
        await session.commit()


async def test_persist_backtest_run(session: AsyncSession, xauusd: Instrument) -> None:
    from datetime import UTC
    from datetime import datetime as dt

    from app.services.backtests import persist_run
    from app.services.seed import XAUUSD as SPEC
    from engine.backtest.runner import CostConfig, run_backtest
    from engine.data.calendar import MarketCalendar
    from engine.data.resample import resample_bars
    from engine.data.sources.synthetic import generate_m1
    from engine.strategy.templates import template

    cal = MarketCalendar.xauusd()
    m15 = resample_bars(
        generate_m1(dt(2024, 1, 7, tzinfo=UTC), dt(2024, 3, 3, tzinfo=UTC), cal), Timeframe.M15, cal
    )
    genome = template("sweep_reversal")
    run = run_backtest(
        genome,
        m15,
        Timeframe.M15,
        calendar=cal,
        sessions=SPEC.model_dump(mode="json")["sessions"],
        costs=CostConfig.from_instrument(xauusd),
    )
    strategy = Strategy.from_genome(
        genome.canonical(), instrument=xauusd, timeframe=Timeframe.M15, origin=StrategyOrigin.SEED
    )
    session.add(strategy)
    persist_run(session, strategy, run, segment=Segment.IN_SAMPLE)
    await session.commit()

    result = await session.scalar(select(BacktestResult))
    assert result is not None and result.n_trades == run.metrics["n_trades"] > 10
    assert result.sharpe == pytest.approx(run.metrics["sharpe"])
    assert result.extra_metrics["exit_reasons"] == run.metrics["exit_reasons"]
    assert result.cost_model["commission_per_lot_rt"] == 7.0 and result.engine_version == "4.0"
    trades = (await session.scalars(select(Trade).order_by(Trade.entry_ts))).all()
    assert len(trades) == result.n_trades
    assert trades[0].context["weekday"] in range(5) and trades[0].entry_ts.tzinfo is not None
    assert sum(t.pnl for t in trades) == pytest.approx(run.trades["pnl"].sum())


async def test_validation_persistence_and_holdout_guard(
    session: AsyncSession, xauusd: Instrument
) -> None:
    from datetime import UTC
    from datetime import datetime as dt

    from app.services.seed import XAUUSD as SPEC
    from app.services.validation import (
        HoldoutAlreadyEvaluatedError,
        apply_validation,
        record_holdout,
    )
    from engine.backtest.runner import CostConfig
    from engine.data.calendar import MarketCalendar
    from engine.data.resample import resample_bars
    from engine.data.sources.synthetic import generate_m1
    from engine.strategy.templates import template
    from engine.validation.splits import make_splits
    from engine.validation.validator import ValidationConfig, evaluate_holdout, validate_genome

    cal = MarketCalendar.xauusd()
    sessions = SPEC.model_dump(mode="json")["sessions"]
    bars = resample_bars(
        generate_m1(dt(2023, 1, 1, tzinfo=UTC), dt(2023, 9, 1, tzinfo=UTC), cal), Timeframe.M15, cal
    )
    splits = make_splits("2023-01-01", "2023-09-01", n_folds=3)
    genome = template("ema_pullback")
    costs = CostConfig.from_instrument(xauusd)
    report = validate_genome(
        genome,
        bars,
        Timeframe.M15,
        splits,
        calendar=cal,
        sessions=sessions,
        costs=costs,
        config=ValidationConfig(n_perturbations=4, min_oos_trades=20),
    )
    strategy = Strategy.from_genome(
        genome.canonical(), instrument=xauusd, timeframe=Timeframe.M15, origin=StrategyOrigin.SEED
    )
    session.add(strategy)
    await apply_validation(session, strategy, report)
    await session.commit()

    assert strategy.status in (StrategyStatus.VALIDATED, StrategyStatus.REJECTED)
    assert strategy.status is (
        StrategyStatus.VALIDATED if report.passed else StrategyStatus.REJECTED
    )
    assert strategy.oos_trades == report.oos_metrics["n_trades"]
    assert strategy.deflated_sharpe == pytest.approx(report.dsr)
    assert strategy.overfit_risk == pytest.approx(report.overfit_risk)
    segs = (
        await session.scalars(
            select(BacktestResult.segment).where(BacktestResult.strategy_id == strategy.id)
        )
    ).all()
    assert segs.count(Segment.IN_SAMPLE) == 3 and segs.count(Segment.WALK_FORWARD) == 3
    summary = await session.scalar(
        select(BacktestResult).where(BacktestResult.segment == Segment.VALIDATION)
    )
    assert summary is not None and summary.extra_metrics["validation_report"]["gates"]

    hold = evaluate_holdout(
        genome, bars, Timeframe.M15, splits, calendar=cal, sessions=sessions, costs=costs
    )
    await record_holdout(session, strategy, hold)
    await session.commit()
    with pytest.raises(HoldoutAlreadyEvaluatedError):
        await record_holdout(session, strategy, hold)


async def test_database_enforces_single_holdout(session: AsyncSession, xauusd: Instrument) -> None:
    """Even code that bypasses the service check cannot store a second holdout result."""
    s = make_strategy(xauusd, {"holdout": 1})
    session.add(s)
    await session.commit()  # committed, so the deliberate failure below can't roll it back
    sid = s.id  # objects expire on rollback; keep the plain id
    for i in range(2):
        session.add(
            BacktestResult(
                strategy_id=sid,
                segment=Segment.HOLDOUT,
                fold_index=i,
                period_start=T0,
                period_end=T0,
                engine_version="4.0",
            )
        )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()
    # other segments may repeat freely
    for i in range(2):
        session.add(
            BacktestResult(
                strategy_id=sid,
                segment=Segment.WALK_FORWARD,
                fold_index=i,
                period_start=T0,
                period_end=T0,
                engine_version="4.0",
            )
        )
    await session.commit()

"""Command-line entry point: ``swarm <command>`` (or ``python -m app.cli``)."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import configure_logging

BACKEND_DIR = Path(__file__).resolve().parent.parent


def alembic_config(database_url: str | None = None):  # type: ignore[no-untyped-def]
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    if database_url:
        cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def cmd_migrate(args: argparse.Namespace) -> None:
    from alembic import command

    command.upgrade(alembic_config(get_settings().database_url), args.revision)


async def _seed() -> None:
    from app.db.session import create_engine, create_sessionmaker, session_scope
    from app.services.seed import seed_reference_data

    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        async for session in session_scope(create_sessionmaker(engine)):
            instruments = await seed_reference_data(session)
            print(f"seeded: {', '.join(i.symbol for i in instruments)}")
    finally:
        await engine.dispose()


def cmd_seed(_: argparse.Namespace) -> None:
    asyncio.run(_seed())


def _parse_date(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


async def _ingest(args: argparse.Namespace) -> None:
    from app.core.events import InMemoryEventBus
    from app.db.session import create_engine, create_sessionmaker
    from app.services.ingest import run_ingest
    from app.services.seed import XAUUSD
    from engine.data.calendar import MarketCalendar
    from engine.data.store import ParquetStore

    settings = get_settings()
    symbol = args.symbol.upper()
    calendar = MarketCalendar.from_instrument_config(
        XAUUSD.session_timezone, XAUUSD.trading_hours.model_dump(mode="json")
    )
    store = ParquetStore(settings.data_dir / "parquet")
    start, end = _parse_date(args.start), _parse_date(args.end)
    engine = create_engine(settings.database_url)
    client = None
    try:
        if args.source == "dukascopy":
            from engine.data.sources.dukascopy import DukascopyClient, DukascopySource

            client = DukascopyClient(
                settings.data_dir / "raw" / "dukascopy",
                max_concurrency=args.concurrency,
                requests_per_second=args.rps,
            )
            source: Any = DukascopySource(client, symbol, calendar, mode=args.mode)
        elif args.source == "csv":
            from engine.data.sources.csv import CsvSource

            source = CsvSource(Path(args.path), symbol, tz=args.tz, point=args.point)
        else:
            from engine.data.sources.synthetic import SyntheticSource

            source = SyntheticSource(symbol, calendar, seed=args.seed)

        result = await run_ingest(
            create_sessionmaker(engine),
            store,
            InMemoryEventBus(),
            source,
            start=start,
            end=end,
            params={k: v for k, v in vars(args).items() if k in ("mode", "path", "tz", "seed")},
        )
        q = result.quality
        print(
            f"ingested {result.m1_rows_written:,} M1 bars"
            + (f" and {result.tick_rows_written:,} ticks" if result.tick_rows_written else "")
            + f" for {symbol} ({args.source}), years {result.years}"
        )
        if q:
            print(
                f"quality score {q.get('score')} | coverage {q.get('coverage')} | "
                f"gaps {q['gaps']['count']} | spikes {q.get('spike_count')} | "
                f"median spread {q['spread']['median']}"
            )
        if client is not None:
            print(f"downloader: {client.stats}")
    finally:
        if client is not None:
            await client.aclose()
        await engine.dispose()


def cmd_ingest(args: argparse.Namespace) -> None:
    asyncio.run(_ingest(args))


def cmd_data_status(args: argparse.Namespace) -> None:
    from app.models.enums import DataSource, Timeframe
    from engine.data.store import ParquetStore

    store = ParquetStore(get_settings().data_dir / "parquet")
    for src in DataSource:
        for tf in Timeframe:
            cov = (
                store.tick_coverage(args.symbol, src)
                if tf is Timeframe.TICK
                else store.bar_coverage(args.symbol, tf, src)
            )
            if cov.rows:
                print(f"{src.value:<10} {tf.value:>5}  {cov.rows:>12,}  {cov.start} -> {cov.end}")


async def _crosscheck(args: argparse.Namespace) -> None:
    """Download one hour as ticks and as M1 candles and compare the resulting bars."""
    from datetime import timedelta

    import numpy as np

    from engine.data.resample import merge_candle_sides, ticks_to_m1
    from engine.data.sources import dukascopy as d

    hour = _parse_date(args.hour)
    scale = d.price_scale(args.symbol)
    day = hour.replace(hour=0, minute=0, second=0, microsecond=0)
    async with d.DukascopyClient(get_settings().data_dir / "raw" / "dukascopy") as c:
        ticks = d.decode_ticks(await c.fetch(d.tick_path(args.symbol, hour)), hour, scale)
        bid = d.decode_candles(
            await c.fetch(d.candle_path(args.symbol, day.date(), "BID")), day, scale
        )
        ask = d.decode_candles(
            await c.fetch(d.candle_path(args.symbol, day.date(), "ASK")), day, scale
        )
    from_ticks = ticks_to_m1(ticks).set_index("ts")
    candles = merge_candle_sides(bid, ask).set_index("ts")
    common = from_ticks.index.intersection(candles.index)
    common = common[(common >= hour) & (common < hour + timedelta(hours=1))]
    if common.empty:
        print("no overlapping minutes (market closed?)")
        return
    cols = [c for c in from_ticks.columns if c.startswith(("bid_", "ask_"))]
    diff = np.abs(from_ticks.loc[common, cols].to_numpy() - candles.loc[common, cols].to_numpy())
    print(f"{len(common)} minutes compared; max abs diff per field:")
    for col, val in zip(cols, diff.max(axis=0), strict=True):
        print(f"  {col:<10} {val:.4f}")
    verdict = "OK: candle layout matches ticks" if diff.max() < 0.01 else "MISMATCH"
    print(verdict)


def cmd_data_crosscheck(args: argparse.Namespace) -> None:
    asyncio.run(_crosscheck(args))


def cmd_backtest(args: argparse.Namespace) -> None:
    import json

    import pandas as pd

    from app.models.enums import DataSource, Timeframe
    from app.services.backtests import load_bars_with_warmup
    from app.services.seed import XAUUSD
    from engine.backtest.runner import CostConfig, run_backtest
    from engine.data.calendar import MarketCalendar
    from engine.data.store import ParquetStore
    from engine.strategy.explain import explain
    from engine.strategy.genome import Genome
    from engine.strategy.templates import template

    genome = (
        Genome.model_validate_json(Path(args.genome).read_text())
        if args.genome
        else template(args.template)
    )
    tf = Timeframe(args.timeframe)
    start, end = pd.Timestamp(_parse_date(args.start)), pd.Timestamp(_parse_date(args.end))
    store = ParquetStore(get_settings().data_dir / "parquet")
    source = DataSource(args.source)
    bars = load_bars_with_warmup(store, args.symbol.upper(), tf, source, start, end)
    if bars.empty:
        raise SystemExit(f"no {tf} bars for {args.symbol} ({source}); run `swarm ingest` first")
    htf = {
        f.timeframe: load_bars_with_warmup(
            store, args.symbol.upper(), f.timeframe, source, start, end, 500
        )
        for f in genome.features()
        if f.timeframe
    }
    cal = MarketCalendar.from_instrument_config(
        XAUUSD.session_timezone, XAUUSD.trading_hours.model_dump(mode="json")
    )
    run = run_backtest(
        genome,
        bars,
        tf,
        calendar=cal,
        sessions=XAUUSD.model_dump(mode="json")["sessions"],
        costs=CostConfig.from_instrument(XAUUSD),
        htf_bars=htf,
        start=start,
        end=end,
    )
    print(explain(genome))
    print(f"\n{run.period_start} -> {run.period_end}  ({tf}, {source})")
    m = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in run.metrics.items()}
    print(json.dumps(m, indent=2, default=str))
    if args.trades_csv:
        run.trades.drop(columns=["context"], errors="ignore").to_csv(args.trades_csv, index=False)
        print(f"trades written to {args.trades_csv}")


def cmd_validate(args: argparse.Namespace) -> None:
    import pandas as pd

    from app.models.enums import DataSource, Timeframe
    from app.services.seed import XAUUSD
    from engine.backtest.runner import CostConfig
    from engine.data.calendar import MarketCalendar
    from engine.data.store import ParquetStore
    from engine.strategy.explain import explain
    from engine.strategy.genome import Genome
    from engine.strategy.templates import template
    from engine.validation.splits import make_splits
    from engine.validation.validator import validate_genome

    genome = (
        Genome.model_validate_json(Path(args.genome).read_text())
        if args.genome
        else template(args.template)
    )
    tf, source = Timeframe(args.timeframe), DataSource(args.source)
    splits = make_splits(
        _parse_date(args.start),
        _parse_date(args.end),
        holdout_fraction=args.holdout,
        n_folds=args.folds,
    )
    store = ParquetStore(get_settings().data_dir / "parquet")
    warm = splits.research_start - pd.Timedelta(seconds=tf.seconds * 3000 * 1.6)
    bars = store.read_bars(
        args.symbol.upper(), tf, source=source, start=warm, end=splits.holdout_start
    )
    if bars.empty:
        raise SystemExit(f"no {tf} bars for {args.symbol} ({source}); run `swarm ingest` first")
    htf = {
        f.timeframe: store.read_bars(
            args.symbol.upper(), f.timeframe, source=source, start=warm, end=splits.holdout_start
        )
        for f in genome.features()
        if f.timeframe
    }
    cal = MarketCalendar.from_instrument_config(
        XAUUSD.session_timezone, XAUUSD.trading_hours.model_dump(mode="json")
    )
    rep = validate_genome(
        genome,
        bars,
        tf,
        splits,
        calendar=cal,
        sessions=XAUUSD.model_dump(mode="json")["sessions"],
        costs=CostConfig.from_instrument(XAUUSD),
        htf_bars=htf,
        n_trials=args.trials,
    )
    d = rep.to_dict()
    print(explain(genome))
    print(
        f"\nresearch {splits.research_start.date()} -> {splits.holdout_start.date()} "
        f"({len(splits.folds)} walk-forward folds); "
        f"holdout from {splits.holdout_start.date()} untouched"
    )
    print(
        f"\n{'PASSED' if d['passed'] else 'REJECTED'}   overfit risk {d['overfit_risk']:.2f}   "
        f"(0 = robust, 1 = likely noise)"
    )
    for name, g in d["gates"].items():
        print(
            f"  [{'ok' if g['passed'] else 'xx'}] {name:<16} {g['value']!s:>10}   needs {g['rule']}"
        )
    print(
        f"\n  OOS: {d['oos']['n_trades']} trades, expectancy {d['oos']['expectancy_r']}R, "
        f"PF {d['oos']['profit_factor']}, Sharpe {d['oos']['sharpe']}, "
        f"max DD {d['oos']['max_drawdown']}"
    )
    print(
        f"  walk-forward efficiency {d['wfe']}, "
        f"Monte Carlo P(loss) {d['monte_carlo'].get('prob_loss')}, "
        f"95% max DD {d['monte_carlo'].get('max_dd_p95')}"
    )
    for f in d["folds"]:
        print(
            f"  fold {f['index']}: {f['test_start'][:10]}..{f['test_end'][:10]}  "
            f"IS {f['is']['expectancy_r']}R  OOS {f['oos']['expectancy_r']}R "
            f"({f['oos']['n_trades']} trades)"
        )


async def _evolve(args: argparse.Namespace) -> None:
    import signal
    import threading
    import uuid

    from app.core.events import create_event_bus
    from app.db.session import create_engine, create_sessionmaker
    from app.models.enums import DataSource, Timeframe
    from app.services.evolution import create_run, execute_run
    from engine.data.store import ParquetStore
    from engine.evolution.engine import EvolutionConfig
    from engine.evolution.fitness import FitnessConfig

    settings = get_settings()
    engine = create_engine(settings.database_url)
    factory = create_sessionmaker(engine)
    bus = create_event_bus(
        settings.redis_url, settings.event_channel_prefix
    )  # Redis → visible in the UI
    stop = threading.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop.set)
    try:
        if args.resume:
            run_id = uuid.UUID(args.resume)
        else:
            cfg = EvolutionConfig(
                islands=args.islands,
                population=args.population,
                generations=args.generations,
                htf=tuple(args.htf or ()),
                seed=args.seed,
                fitness=FitnessConfig(min_trades=args.min_trades),
            )
            async with factory() as session:
                run = await create_run(
                    session,
                    name=args.name,
                    symbol=args.symbol,
                    timeframe=Timeframe(args.timeframe),
                    start=_parse_date(args.start),
                    end=_parse_date(args.end),
                    config=cfg,
                    source=DataSource(args.source),
                    validate_top=args.validate_top,
                )
                await session.commit()
                run_id = run.id
        print(
            f"run {run_id} (Ctrl-C pauses after the current generation; "
            f"resume with --resume {run_id})"
        )

        async def show() -> None:
            from app.core.events import Channel

            async with bus.subscribe(Channel.LAB) as events:
                async for e in events:
                    p = e.payload
                    if e.type == "generation":
                        b = p.get("best") or {}
                        m = b.get("metrics", {})
                        print(
                            f"gen {p['generation']:>3}  evals {p['evaluations']:>6}  "
                            f"{p['duration_ms']:>6}ms  "
                            f"best fitness {b.get('fitness', float('nan')):+.2f}  "
                            f"t={m.get('t_stat', 0):+.2f}  "
                            f"trades={m.get('n_trades')}"
                            + ("  [migration]" if p["migrated"] else "")
                        )
                    elif e.type == "validated":
                        print(
                            f"  {p['name']}: {'VALIDATED' if p['passed'] else 'rejected'}  "
                            f"risk {p['overfit_risk']:.2f}  DSR {p['dsr']:.3f}"
                        )
                    elif e.type in ("validating", "run_completed", "run_paused", "run_failed"):
                        print(f"[{e.type}] {p}")

        watcher = asyncio.create_task(show())
        await asyncio.sleep(0)
        status = await execute_run(
            factory,
            ParquetStore(settings.data_dir / "parquet"),
            bus,
            run_id,
            checkpoint_dir=settings.data_dir / "runs",
            workers=args.workers,
            stop=stop,
        )
        await asyncio.sleep(0.1)
        watcher.cancel()
        print(f"run {run_id}: {status.value}")
    finally:
        await bus.close()
        await engine.dispose()


def cmd_evolve(args: argparse.Namespace) -> None:
    asyncio.run(_evolve(args))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,  # our logging config applies
    )


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    parser = argparse.ArgumentParser(prog="swarm", description="Swarm Trader backend")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("migrate", help="apply database migrations")
    p.add_argument("revision", nargs="?", default="head")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("seed", help="upsert reference data (instruments)")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("serve", help="run the API server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("ingest", help="load historical data into the Parquet lake")
    p.add_argument("source", choices=["dukascopy", "csv", "synthetic"])
    p.add_argument("--symbol", default=settings.default_symbol)
    p.add_argument("--start", required=True, help="ISO date/time, UTC if no offset")
    p.add_argument("--end", required=True)
    p.add_argument(
        "--mode",
        choices=["candles", "ticks"],
        default="candles",
        help="dukascopy: M1 candles (fast) or raw ticks (exact intrabar)",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--rps", type=float, default=4.0, help="max requests per second")
    p.add_argument("--path", help="csv: file to import")
    p.add_argument("--tz", default="UTC", help="csv: timezone of file timestamps, or NY+7")
    p.add_argument("--point", type=float, default=0.01, help="csv: MT5 point size")
    p.add_argument("--seed", type=int, default=7, help="synthetic: RNG seed")
    p.set_defaults(func=cmd_ingest)

    data = sub.add_parser("data", help="inspect the data lake").add_subparsers(
        dest="data_command", required=True
    )
    p = data.add_parser("status", help="rows and date range per source/timeframe")
    p.add_argument("--symbol", default=settings.default_symbol)
    p.set_defaults(func=cmd_data_status)
    p = data.add_parser("crosscheck", help="compare Dukascopy candles vs ticks for one hour")
    p.add_argument("--symbol", default=settings.default_symbol)
    p.add_argument("--hour", default="2024-01-02T14:00", help="UTC hour start")
    p.set_defaults(func=cmd_data_crosscheck)

    p = sub.add_parser("backtest", help="backtest a template or genome JSON file")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--template", help="smc_choch_fvg | sweep_reversal | ema_pullback | divergence_bbands"
    )
    g.add_argument("--genome", help="path to a genome JSON file")
    p.add_argument("--symbol", default=settings.default_symbol)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--source", default="dukascopy")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--trades-csv", help="optional path to write the trade list")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("validate", help="walk-forward + overfitting checks for a strategy")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--template")
    g.add_argument("--genome", help="path to a genome JSON file")
    p.add_argument("--symbol", default=settings.default_symbol)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--source", default="dukascopy")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--holdout", type=float, default=0.2, help="fraction kept as locked holdout")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--trials", type=int, default=1, help="strategies tried before this one (DSR)")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("evolve", help="run the strategy swarm (GA) and validate the winners")
    p.add_argument("--resume", help="resume a paused run by id")
    p.add_argument("--name", default="xau-swarm")
    p.add_argument("--symbol", default=settings.default_symbol)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--source", default="dukascopy")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--islands", type=int, default=4)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--generations", type=int, default=25)
    p.add_argument("--htf", nargs="*", help="higher timeframes the GA may use, e.g. 1h 4h")
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--validate-top", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_evolve)

    args = parser.parse_args(argv)
    if args.command == "evolve" and not args.resume and not (args.start and args.end):
        parser.error("evolve needs --start and --end (or --resume)")
    args.func(args)


if __name__ == "__main__":
    main()

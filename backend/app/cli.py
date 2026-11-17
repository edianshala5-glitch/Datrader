"""Command-line entry point: ``swarm <command>`` (or ``python -m app.cli``)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

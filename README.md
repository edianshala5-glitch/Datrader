# Swarm Trader: XAUUSD

A self-optimizing research and signaling system for **spot gold (XAUUSD)** day trading.
A swarm of bots evolves trading strategies with genetic algorithms on years of tick
history. A Meta-Analyzer then mines the contexts in which each strategy actually works
(trading-day, NY time window, volatility regime, BOS / CHoCH / divergence), and the
top 3 out-of-sample **Champions** drive a live signal feed.

**Focus timeframes:** 5m, 15m, 30m (any timeframe from tick up is supported).
**Cost:** the whole stack uses free data and free, open-source software.

> Research tooling, not financial advice. The system emits signals only and never places orders.

## Status

| Step | Module | State |
|---|---|---|
| 1 | Backend foundation & data models | ✅ done |
| 2 | Data ingestion (Dukascopy ticks → Parquet) | next |
| 3 | Indicator engine (standard + SMC, lookahead-safe) | |
| 4 | Strategy DSL + numba backtester | |
| 5 | Validation harness (walk-forward, Deflated Sharpe, PBO) | |
| 6 | Optimization swarm (DEAP NSGA-II, islands) | |
| 7 | Meta-Analyzer (context rules) | |
| 8 | Champions + live signaling | |
| 9–12 | Next.js dashboard: Lab, Leaderboard, Live Signals | |
| 13 | Hardening | |

See [docs/PLAN.md](docs/PLAN.md) for the architecture and the full plan.

## Quick start (no Docker needed)

```bash
cd backend
uv sync                 # Python 3.12+, installs deps into .venv
uv run swarm migrate    # creates ./data/swarm.db (SQLite)
uv run swarm seed       # XAUUSD instrument definition
uv run swarm serve      # http://127.0.0.1:8000/api/docs
```

Or with the Makefile from the repo root: `make install && make dev`.

### Full stack (Postgres + Redis)

```bash
docker compose up -d --build   # api on :8000, postgres on :5432, redis on :6379
```

## Tests

```bash
make test       # SQLite + in-memory event bus
make test-all   # also PostgreSQL + Redis (needs `docker compose up -d postgres redis`)
make lint typecheck
```

`test-all` uses a dedicated `swarm_test` database. The suite refuses to run against any
database whose name doesn't end in `_test`.

## Layout

```
backend/
  app/
    core/        settings, logging, event bus (in-memory | Redis pub/sub)
    db/          async engine, declarative base, portable column types
    models/      SQLAlchemy models: instruments, datasets, jobs, runs, generations,
                 strategies (+lineage), backtests, trades, context rules, champions, signals
    schemas/     Pydantic v2 API models + validators for instrument JSON config
    api/v1/      health, instruments, runs, strategies (leaderboard), WebSocket stream
    services/    seed data (XAUUSD definition)
    cli.py       `swarm migrate | seed | serve`
  alembic/       migrations (tested for drift against the models on SQLite and Postgres)
  tests/
docker-compose.yml
```

## API (so far)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | DB + event-bus health |
| GET | `/api/v1/instruments[/{symbol}]` | instrument specs, sessions, cost model |
| GET | `/api/v1/runs`, `/runs/{id}`, `/runs/{id}/generations` | evolution runs (Lab) |
| GET | `/api/v1/strategies` | leaderboard with filters: timeframe, status, min trades/win rate/PF/DSR, max overfit risk/drawdown, sort |
| GET | `/api/v1/strategies/{id}` | genome, lineage, metrics |
| WS | `/api/v1/ws/{lab\|signals\|jobs\|system}` | live event stream |

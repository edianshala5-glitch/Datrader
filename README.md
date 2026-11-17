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
| 2 | Data ingestion (Dukascopy → Parquet lake, quality reports) | ✅ done |
| 3 | Indicator engine (standard + SMC, lookahead-safe) | ✅ done |
| 4 | Strategy DSL + numba backtester | ✅ done |
| 5 | Validation harness (walk-forward, Deflated Sharpe, PBO) | next |
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

### Loading XAUUSD history (free)

```bash
cd backend
# M1 candles with bid/ask: 2 requests per trading day. ~10 years ≈ 5,200 requests.
uv run swarm ingest dukascopy --start 2015-01-01 --end 2025-01-01
# Raw ticks (1 request per open hour, ~12x more): exact intrabar fills + spread research.
uv run swarm ingest dukascopy --mode ticks --start 2024-01-01 --end 2025-01-01
uv run swarm data status
```

The downloader is resumable: every raw file is cached under `data/raw/dukascopy/`, so
an interrupted run continues where it stopped. Dukascopy rate-limits (HTTP 429) and
may reset connections. The client paces requests (`--rps`, default 4/s), pauses all
workers together when throttled, and backs off exponentially. Run large backfills from
a home connection: shared cloud IPs are throttled much harder.

Other sources:

```bash
# Your broker's MT5 export (File → Save As in the History Center). NY-close servers: --tz NY+7
uv run swarm ingest csv --path XAUUSD_M1.csv --tz NY+7 --start 2015-01-01 --end 2026-01-01
# Synthetic XAUUSD-like data for offline development
uv run swarm ingest synthetic --start 2023-01-01 --end 2025-01-01
# Verify Dukascopy candles == ticks for one hour
uv run swarm data crosscheck --hour 2024-01-02T14:00
```

Every ingest builds 5m / 15m / 30m / 1h / 4h / 1d bars from M1. D1 and 4h are aligned to
the **17:00 New York** roll (D1 = 17:00→17:00 NY, 4h opens 17/21/01/05/09/13 NY all year).
Each run writes a quality report covering coverage, gaps, crossed quotes, spreads and bad
prints vs genuine large moves. You can see it via `GET /api/v1/data/XAUUSD/datasets`.

### Indicators

19 indicators / 82 typed outputs, all computed on mid prices:

| Category | Indicators |
|---|---|
| Trend / momentum | EMA, SMA, MACD, ADX (+DI/-DI), Donchian, RSI, Stochastic, ROC |
| Volatility / volume | ATR, Bollinger Bands, session VWAP (anchored at 17:00 NY) |
| Smart Money Concepts | swings (HH/LL), structure (BOS, CHoCH, liquidity sweeps, EQH/EQL), fair value gaps (with mitigation), order blocks, regular/hidden divergences (RSI or MACD) |
| Context | trading-day weekday, NY hour/minute, US holidays, sessions + ICT kill zones, volatility & trend regime |

- **TradingView-compatible** formulas (SMA-seeded RMA/EMA, population-stdev Bollinger).
- **No lookahead, enforced:** swing-based events are emitted at their *confirmation* bar.
  A test computes every indicator on full and truncated history and requires identical
  values. A second test proves that check fails on deliberately leaky indicators.
- **Multi-timeframe** joins only use higher-timeframe bars that were complete at decision time.
- Full set on 10 years of 5m bars (720k) in ~1.2 s, with a Parquet feature cache.
- `GET /api/v1/indicators` (catalog with param ranges and output types) and
  `GET /api/v1/data/{symbol}/overlays` (BOS/CHoCH/sweep markers, FVG zones for charts).

### Strategies & backtesting

Strategies are JSON **genomes** with condition trees (`compare`, `cross`, `event within N
bars`, `flag`, `state`, `range`) over typed indicator outputs, plus exits (ATR / swing /
percent stop, R or ATR target, ATR trailing stop, max bars, exit signal, flat before
17:00 NY). Invalid combinations (e.g. `EMA > 2000`, `RSI > 140`) are rejected at
validation time. Every genome renders as plain English.

```bash
uv run swarm backtest --template smc_choch_fvg --timeframe 5m --start 2024-01-08 --end 2024-02-01
uv run swarm backtest --genome my_strategy.json --timeframe 15m --start 2023-01-01 --end 2024-01-01
```

The execution model is deliberately conservative:
- Decisions happen at a bar's close and fill at the next open: buys at the ask, sells at the bid.
- Stops trigger on the bid low (longs) or ask high (shorts). A gap through a stop fills at the open.
- A bar that touches both stop and target counts as a stop.
- Commission, slippage and a spread floor are always charged. Position size comes from the
  risk per trade.

Integrity tests:
- Trades are identical with or without future bars (no look-ahead).
- On edge-free synthetic data, expectancy is zero without costs and negative with costs.

Speed: ~65 ms per backtest over 10 years of 5m bars once features are cached.
Templates (`smc_choch_fvg`, `sweep_reversal`, `ema_pullback`, `divergence_bbands`) seed the
GA. Untuned, they all lose after costs on real data, which is the baseline the swarm must beat.

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
    services/    seed data (XAUUSD definition), ingestion orchestration
    cli.py       `swarm migrate | seed | serve | ingest | data … | backtest`
  engine/        quant core (no web deps)
    data/        calendar (NY sessions, 17:00 roll), US holidays, Parquet+DuckDB store,
                 resampling, quality reports, sources: dukascopy | csv (MT5) | synthetic
    indicators/  registry (typed params/outputs), numba kernels, standard, SMC, context,
                 multi-timeframe alignment, feature matrices + cache
    strategy/    genome DSL (pydantic), compiler → signal arrays, explain, templates
    backtest/    numba simulator, metrics (daily Sharpe/Sortino, PF, R stats), runner
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
| GET | `/api/v1/data/{symbol}/datasets` | data catalogue + quality reports |
| GET | `/api/v1/data/{symbol}/bars` | chart bars (bid/ask/mid, any timeframe, column arrays) |
| GET | `/api/v1/data/{symbol}/overlays` | SMC markers + FVG zones for charts |
| GET | `/api/v1/indicators` | indicator catalog (params, ranges, output types) |
| POST | `/api/v1/backtest` | run a genome over a window (metrics, equity, trades) |
| GET | `/api/v1/strategy-templates` | starter genomes with explanations |
| GET | `/api/v1/jobs[/{id}]` | background job progress |
| WS | `/api/v1/ws/{lab\|signals\|jobs\|system}` | live event stream |

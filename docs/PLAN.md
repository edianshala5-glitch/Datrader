# Architecture & Project Plan

## Guiding principle: robustness over backtest beauty

Searching thousands of strategies, then slicing results by weekday × hour × market
structure, *guarantees* some spectacular backtests that are pure luck. So:

- **Three-way data split**: training / walk-forward validation / a **locked holdout**
  that a strategy is evaluated on once. Plus forward **paper incubation** before any
  strategy can emit live alerts.
- **Multiple-testing aware statistics**: the Deflated Sharpe Ratio uses the total number of
  evaluations (`evolution_runs.evaluations`). Probability of Backtest Overfitting (PBO)
  is computed via CSCV. Context rules get Holm/BH-adjusted p-values and minimum sample sizes.
- **Costs always on**: Dukascopy provides real bid/ask, so spread is modelled from data,
  with `instruments.cost_model` as a floor (commission + slippage).
- **No lookahead**: swing pivots (and therefore BOS/CHoCH/divergences) are stamped at
  the bar where they are *confirmed*. Truncation tests and batch-vs-incremental parity
  tests enforce this. Backtest and live engines share one code path.

## XAUUSD specifics

- Trades Sun 18:00 to Fri 17:00 **New York time**, with a daily 17:00–18:00 break. The
  **trading day rolls at 17:00 NY**, so a "Tuesday" bar means the session that opened Monday
  18:00 NY. All day-of-week and time-of-day context uses this convention, which stays
  correct through US/EU DST changes.
- Session windows (Asia, London, NY, COMEX hours, ICT kill zones) are stored in each
  market's own timezone.
- 1 lot = 100 oz, tick = 0.01.

## Free data plan

| Purpose | Source | Notes |
|---|---|---|
| History (primary) | **Dukascopy** public datafeed | Free bid/ask data, XAUUSD from ~2003. M1 candle files (1/day/side) or hourly tick files, LZMA `.bi5`. Resampled locally into M1→D1 bars. Rate-limited: run big backfills from a home IP. |
| History (cross-check) | HistData.com | Free M1 CSVs, for gap-filling and sanity checks. |
| Your own exports | CSV loader | e.g. MT5 history exports from your broker. |
| Live feed | OANDA v20 **practice** API | Free demo account, streaming XAU_USD prices. Feed interface is pluggable. |

Storage: Parquet partitioned `symbol/timeframe/year`, queried with DuckDB (embedded, free).
Metadata: SQLite for zero setup, Postgres in docker-compose.

## Modules

1. **Backend & data models** ✅: settings, async SQLAlchemy models, Alembic, API skeleton,
   event bus, XAUUSD seed.
2. **Data ingestion** ✅: Dukascopy downloader (resumable raw cache, global rate limit,
   shared 429 cooldown, backoff), candle mode (2 req/day) and tick mode (1 req/open hour),
   verified byte-exact against each other on real data. MT5/generic CSV import,
   synthetic generator, tick→M1→higher-TF resampling aligned to the 17:00 NY roll, quality
   reports (coverage, gaps, crossed quotes, spread distribution, bad prints vs large moves),
   `datasets` catalogue, chart-bars API.
3. **Indicator engine** ✅: registry with typed params (GA ranges) and typed outputs (DSL
   kinds). TradingView-compatible standard indicators. SMC: confirmed swings, BOS/CHoCH,
   liquidity sweeps (level stays live after a sweep, found on real NFP data), EQH/EQL, FVG
   with mitigation, order blocks, regular/hidden divergences. Context: trading-day weekday,
   NY clock, US holiday/early-close calendar, sessions + kill zones, volatility/trend regime.
   Leak-free multi-timeframe joins, Parquet feature cache, lookahead harness (proven to
   catch leaks).
4. **Strategy DSL + backtester** ✅: pydantic genome typed against the indicator registry
   (rejects meaningless rules), compiler to signal arrays with leak-free HTF features,
   numba simulator (next-open fills at bid/ask, stop-first on ambiguous bars, gap fills,
   trailing stops, session flats, costs, risk-based sizing), daily-return metrics, entry
   context per trade, persistence, explain(), templates. Integrity: truncation test and
   zero-edge random-walk test (which caught and fixed a wick artifact in the synthetic
   generator).
5. **Validation** ✅: anchored or rolling walk-forward with embargo, locked holdout (one evaluation
   per strategy, enforced by a partial unique index), PSR/DSR, Holm/BH, bootstrap CIs,
   PBO via CSCV, parameter-perturbation robustness, cost stress, Monte Carlo, an overfit-risk
   score and gates. Proven with negative, positive and lucky-winner controls (docs/VALIDATION.md).
6. **Swarm** ✅: own NSGA-II (DEAP's global creator state and list genomes fit poorly with
   validated tree genomes), island model with ring migration, typed random conditions,
   mirroring, mutation/crossover/perturbation, a declared param-dependency registry, parallel
   spawn workers, deterministic checkpoints, per-generation persistence + Lab events, and
   hall-of-fame validation with the OOS-reuse trial counter (docs/VALIDATION.md).
7. **Meta-Analyzer** ✅: direction-aligned context features that each map to an executable genome
   condition. Mining: GBM permutation importance, shallow trees, feature-diverse beam search
   ranked by the significance of kept-trade expectancy. Testing on a later window: Holm
   correction, minimum trades, refined-strategy re-backtest, DSR, distinct-rule selection.
   Rules and refined strategies are persisted with lineage, and run automatically after swarm runs.
8. **Champions & live signals** ✅: one-time holdout promotion, incubation, top-3 activation by live
   significance, demotion on drawdown or degradation. Live engine re-runs the backtest
   simulator per bar (exact parity, tested end to end and at the DB level). OANDA practice feed
   (stream + backfill + reconnect), replay feed, M1→TF aggregator, idempotent signals with
   rationale, webhook/Telegram notifiers for active champions only.
9. **Frontend foundation**: Next.js App Router, TypeScript, Tailwind (dark only), typed
   client generated from OpenAPI.
10. **Lab**: generation timeline, Pareto scatter, island grid, lineage tree.
11. **Leaderboard**: deep filters, strategy page with equity curve, trades on chart,
    weekday × hour heatmap, per-fold table.
12. **Live Signals**: feed, chart overlays (entry/SL/TP), rationale panel.
13. **Hardening**: auth, observability, CI, deployment docs.

## Design decisions

- **GA before RL**: GA output is explicit rules, which the "clear rationale" requirement
  needs. RL is an optional later experiment.
- **Enums stored as VARCHAR**: new members need no migration. Identical on SQLite/Postgres.
- **UTC everywhere in storage**: naive datetimes are rejected at the DB layer.
- **Denormalized leaderboard metrics** on `strategies`, so filtering stays fast with indexes.
  Full per-fold numbers live in `backtest_results`.
- **Trades persisted selectively**: only strategies passing the in-sample gate store trades
  in the DB (Meta-Analyzer input). Everything else stays in worker Parquet scratch.
- **Event bus is at-most-once**: durable state goes to the DB first. The bus only pushes
  live updates to the UI.

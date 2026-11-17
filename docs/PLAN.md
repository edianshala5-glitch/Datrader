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
| History (primary) | **Dukascopy** public datafeed | Free tick data with bid/ask, XAUUSD from ~2003. Hourly LZMA `.bi5` files. Resampled locally into M1→D1 bars. |
| History (cross-check) | HistData.com | Free M1 CSVs, for gap-filling and sanity checks. |
| Your own exports | CSV loader | e.g. MT5 history exports from your broker. |
| Live feed | OANDA v20 **practice** API | Free demo account, streaming XAU_USD prices. Feed interface is pluggable. |

Storage: Parquet partitioned `symbol/timeframe/year`, queried with DuckDB (embedded, free).
Metadata: SQLite for zero setup, Postgres in docker-compose.

## Modules

1. **Backend & data models** ✅: settings, async SQLAlchemy models, Alembic, API skeleton,
   event bus, XAUUSD seed.
2. **Data ingestion**: Dukascopy downloader (resumable, rate-limited, checksummed),
   tick→bar resampler aligned to the 17:00 NY roll, quality reports (gaps, spikes, spreads),
   `datasets` catalogue.
3. **Indicator engine**: registry of standard indicators (RSI, MACD, ATR, BB, EMA, VWAP…),
   SMC: confirmed swings, BOS, CHoCH, FVG, order blocks, equal highs/lows, regular/hidden
   divergences, volatility/trend regime. Multi-timeframe features without leakage.
4. **Strategy DSL + backtester**: JSON genome (entry condition tree, exits, SL/TP/trailing,
   session filters, sizing). numba event loop with next-bar-open fills and conservative
   intrabar SL/TP ordering.
5. **Validation**: walk-forward, purged/embargoed CV, holdout lock, DSR, PBO, bootstrap CIs,
   parameter-perturbation robustness → `overfit_risk`.
6. **Swarm**: DEAP NSGA-II over (OOS Sortino, max DD, trade count, fold stability), island
   model with migration, process-pool workers (Ray optional), resumable checkpoints,
   generation events → Lab.
7. **Meta-Analyzer**: per-trade context features → gradient-boosted trees for importance,
   shallow decision trees → readable rules → OOS + multiple-testing validation.
8. **Champions & live signals**: promotion criteria, incubation, demotion on drift.
   Idempotent signal emission with structured rationale.
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

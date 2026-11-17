// Shapes returned by the FastAPI backend (subset used by the dashboard).
export type Timeframe = "1m" | "5m" | "15m" | "30m" | "1h" | "4h" | "1d";

export interface Page<T> { items: T[]; total: number; limit: number; offset: number }

export interface Health {
  status: string; version: string; environment: string;
  database: { ok: boolean; detail: string | null }; event_bus: { ok: boolean; detail: string | null };
}

export interface IslandStats {
  size: number; valid: number; diversity: number; pareto_size: number;
  fitness: { min: number; mean: number; max: number } | null;
  edge_max: number | null; best_key: string | null; best_metrics: Record<string, number | null> | null;
}

export interface Run {
  id: string; name: string; symbol: string; timeframe: Timeframe; status: string; algorithm: string;
  config: Record<string, unknown>; data_splits: Record<string, unknown>; seed: number;
  current_generation: number; evaluations: number; best_metrics: Record<string, unknown>;
  started_at: string | null; finished_at: string | null; created_at: string;
}

export interface Generation {
  id: string; island: number; number: number; population_size: number; evaluated: number;
  stats: IslandStats; pareto_size: number; diversity: number | null; duration_ms: number | null; created_at: string;
}

export interface StrategySummary {
  id: string; name: string; symbol: string; timeframe: Timeframe; status: string; origin: string;
  run_id: string | null; generation_number: number | null; fitness: number | null; is_sharpe: number | null;
  oos_sharpe: number | null; oos_sortino: number | null; oos_profit_factor: number | null;
  oos_win_rate: number | null; oos_expectancy_r: number | null; oos_max_drawdown: number | null;
  oos_trades: number | null; deflated_sharpe: number | null; pbo: number | null; overfit_risk: number | null;
  created_at: string;
}

export interface StrategyDetail extends StrategySummary {
  genome: Record<string, unknown>; genome_hash: string; description: string | null; parent_ids: string[];
}

export interface Gate { value: number | string | null; passed: boolean; rule: string }

export interface ValidationReport {
  passed: boolean; overfit_risk: number; reasons: string[]; gates: Record<string, Gate>;
  components: Record<string, number>; dsr: number; n_trials: number; wfe: number;
  positive_fold_ratio: number; monte_carlo: Record<string, number>;
  folds: { index: number; test_start: string; test_end: string;
    is: Record<string, number | null>; oos: Record<string, number | null> }[];
  search_evaluations?: number; prior_oos_validations?: number;
}

export interface BacktestRow {
  id: string; segment: string; fold_index: number; context_rule_id: string | null;
  period_start: string; period_end: string; n_trades: number; sharpe: number | null;
  profit_factor: number | null; win_rate: number | null; expectancy_r: number | null;
  max_drawdown: number | null; extra: { validation_report?: ValidationReport } & Record<string, unknown>;
}

export interface TradeRow {
  segment: string; side: "long" | "short"; entry_ts: string; exit_ts: string; entry_price: number;
  exit_price: number; stop_price: number | null; target_price: number | null; pnl_r: number | null;
  exit_reason: string; bars_held: number; context: Record<string, number | boolean | null>;
}

export interface ContextRule {
  id: string; name: string; description: string; status: string; miner: string;
  is_trades: number | null; is_expectancy_r: number | null; oos_trades: number | null;
  oos_expectancy_r: number | null; oos_win_rate: number | null; coverage: number | null;
  lift: number | null; p_value_adjusted: number | null; details: Record<string, unknown>;
}

export interface Champion {
  id: string; strategy_id: string; name: string; description: string | null; timeframe: Timeframe;
  status: "incubating" | "active" | "demoted"; rank: number | null; promoted_at: string;
  activated_at: string | null; demotion_reason: string | null;
  snapshot: Record<string, unknown> & { holdout?: Record<string, number | null> };
  live: { n?: number; mean_r?: number; win_rate?: number; total_r?: number; max_drawdown_r?: number };
}

export interface RationaleCondition { label: string; met: boolean; value: number | null }

export interface SignalRow {
  id: string; champion_id: string | null; strategy_id: string; timeframe: Timeframe;
  kind: "entry" | "exit" | "cancel"; side: "long" | "short"; status: string; bar_ts: string;
  emitted_at: string; entry_price: number | null; stop_price: number | null; target_price: number | null;
  risk_reward: number | null; confidence: number | null; exit_price: number | null; pnl_r: number | null;
  rationale: { conditions?: RationaleCondition[]; filters?: RationaleCondition[]; conditions_op?: string;
    filters_op?: string; context_rule?: string | null; strategy?: string; champion_status?: string;
    reason?: string } & Record<string, unknown>;
  rationale_text: string; closed_at: string | null;
}

export interface Bars {
  symbol: string; timeframe: Timeframe; time: number[]; open: number[]; high: number[];
  low: number[]; close: number[]; volume: number[]; spread: number[];
}

export interface Overlays {
  markers: { time: number; confirmed_time: number; kind: string; price: number }[];
  zones: { kind: string; start_time: number; end_time: number | null; top: number; bottom: number }[];
}

export interface Dataset {
  id: string; source: string; timeframe: string; start_ts: string | null; end_ts: string | null;
  row_count: number; quality: Record<string, unknown>; last_ingested_at: string | null;
}

export interface LiveStatus {
  running: boolean; feed?: string; last_bar?: string | null; signals?: number; champions?: number;
  bars?: number; error?: string; history_source?: string;
}

export interface BusEvent<P = Record<string, unknown>> { channel: string; type: string; ts: string; payload: P }

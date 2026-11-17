"use client";

import Link from "next/link";
import { use, useMemo, useState } from "react";
import { CandleChart, type ChartMarker } from "@/components/charts/CandleChart";
import { EquityChart } from "@/components/charts/EquityChart";
import { WeekdayHourHeatmap } from "@/components/charts/Heatmap";
import { C } from "@/components/charts/theme";
import { Badge, Button, Card, Empty, ErrorNote, Stat, Td, Th } from "@/components/ui";
import { api } from "@/lib/api";
import { int, num, pct, signed } from "@/lib/format";
import { useFetch } from "@/lib/hooks";
import type { Bars, TradeRow } from "@/lib/types";

const GATE_LABEL: Record<string, string> = {
  oos_trades: "OOS trades", oos_expectancy: "OOS expectancy", deflated_sharpe: "Deflated Sharpe",
  positive_folds: "Positive folds", robustness: "Parameter robustness", cost_stress: "2× cost stress", pbo: "PBO",
};

export default function StrategyPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const s = useFetch(() => api.strategy(id), [id]);
  const bts = useFetch(() => api.backtests(id), [id]);
  const trades = useFetch(() => api.trades(id), [id]);
  const rules = useFetch(() => api.rules(id), [id]);
  const run = useFetch(() => (s.data?.run_id ? api.run(s.data.run_id) : Promise.resolve(null)), [s.data?.run_id]);
  const source = (run.data?.config.source as string | undefined) ?? "dukascopy";
  const [promoteMsg, setPromoteMsg] = useState<string | null>(null);
  const [promoteErr, setPromoteErr] = useState<string | null>(null);
  const report = bts.data?.find((b) => b.segment === "validation" && b.extra.validation_report)?.extra.validation_report;
  const holdout = bts.data?.find((b) => b.segment === "holdout");
  const oosTrades = useMemo(() => (trades.data ?? []).filter((t) => t.segment === "walk_forward" || t.segment === "validation"), [trades.data]);
  const equity = useMemo(() => {
    let cum = 0;
    return oosTrades.filter((t) => t.pnl_r !== null).map((t) => ({ time: Math.floor(new Date(t.exit_ts).getTime() / 1000), value: (cum += t.pnl_r!) }));
  }, [oosTrades]);
  const heat = useMemo(() => oosTrades.filter((t) => t.pnl_r !== null && typeof t.context.weekday === "number")
    .map((t) => ({ weekday: t.context.weekday as number, hour: t.context.hour_ny as number, r: t.pnl_r! })), [oosTrades]);
  const [tradeIdx, setTradeIdx] = useState(0);
  const [showRejected, setShowRejected] = useState(false);
  const recent = oosTrades.slice(-30).reverse();
  const focus: TradeRow | undefined = recent[tradeIdx];
  const bars = useFetch<Bars | null>(() => {
    if (!focus || !s.data) return Promise.resolve(null);
    const t0 = new Date(focus.entry_ts).getTime(), t1 = new Date(focus.exit_ts).getTime();
    const pad = 60 * 60 * 1000 * 8;
    return api.bars("XAUUSD", { timeframe: s.data.timeframe, source, start: new Date(t0 - pad).toISOString(),
      end: new Date(t1 + pad).toISOString(), limit: 2000 }).catch(() => null);
  }, [focus?.entry_ts, s.data?.timeframe, source]);
  const markers: ChartMarker[] = useMemo(() => focus ? [
    { time: new Date(focus.entry_ts).getTime() / 1000, position: focus.side === "long" ? "belowBar" : "aboveBar",
      color: C.accent, shape: focus.side === "long" ? "arrowUp" : "arrowDown", text: `${focus.side} @ ${focus.entry_price.toFixed(2)}` },
    { time: new Date(focus.exit_ts).getTime() / 1000, position: focus.side === "long" ? "aboveBar" : "belowBar",
      color: (focus.pnl_r ?? 0) > 0 ? C.good : C.down, shape: "circle", text: `${focus.exit_reason} ${signed(focus.pnl_r, 2, "R")}` },
  ] : [], [focus]);
  const lines = useMemo(() => focus ? [
    ...(focus.stop_price ? [{ price: focus.stop_price, color: C.down, title: "stop", dashed: true }] : []),
    ...(focus.target_price ? [{ price: focus.target_price, color: C.good, title: "target", dashed: true }] : []),
  ] : [], [focus]);

  if (s.error) return <ErrorNote error={s.error} />;
  if (!s.data) return <div className="h-40 animate-pulse rounded-xl bg-surface" />;
  const st = s.data;
  const promote = async () => {
    setPromoteErr(null); setPromoteMsg(null);
    try { const c = await api.promote(id); setPromoteMsg(`Holdout passed: now ${c.status}.`); s.reload(); bts.reload(); }
    catch (e) { setPromoteErr(e instanceof Error ? e.message : String(e)); s.reload(); }
  };
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold">{st.name}</h1><Badge status={st.status} />
            <span className="text-sm text-muted">{st.timeframe} · {st.origin}</span>
          </div>
          {st.parent_ids.length > 0 && (
            <div className="mt-1 text-xs text-muted">parents: {st.parent_ids.map((p) => (
              <Link key={p} href={`/strategies/${p}`} className="mr-2 text-ink-2 hover:text-accent">{p.slice(0, 8)}</Link>))}</div>
          )}
        </div>
        {st.status === "validated" && <Button variant="primary" onClick={promote}>Promote (run holdout once)</Button>}
      </div>
      {promoteMsg && <div className="rounded-lg border border-good/40 bg-good/10 px-3 py-2 text-sm text-good">✓ {promoteMsg}</div>}
      <ErrorNote error={promoteErr} />
      <Card title="Rules">
        <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-ink-2">{st.description ?? "—"}</pre>
      </Card>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="OOS expectancy" value={signed(st.oos_expectancy_r, 3, "R")} tone={st.oos_expectancy_r === null ? undefined : st.oos_expectancy_r > 0 ? "good" : "bad"} />
        <Stat label="OOS win rate" value={pct(st.oos_win_rate)} sub={`${int(st.oos_trades)} trades`} />
        <Stat label="Profit factor" value={num(st.oos_profit_factor)} />
        <Stat label="Max drawdown" value={pct(st.oos_max_drawdown)} />
        <Stat label="Deflated Sharpe" value={num(st.deflated_sharpe, 3)} sub={report ? `n_trials ${int(report.n_trials)}` : undefined} />
        <Stat label="Overfit risk" value={num(st.overfit_risk)} tone={st.overfit_risk === null ? undefined : st.overfit_risk < 0.3 ? "good" : st.overfit_risk < 0.6 ? "warn" : "bad"} sub="0 robust · 1 noise" />
      </div>
      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Out-of-sample equity (cumulative R)"><EquityChart points={equity} /></Card>
        <Card title="When does it work? Mean R by trading day × NY hour">
          {heat.length ? <WeekdayHourHeatmap trades={heat} /> : <Empty>No out-of-sample trades stored yet.</Empty>}
        </Card>
      </div>
      <Card title="Trades on the chart" actions={recent.length > 0 && (
        <select className="rounded border border-line bg-page px-2 py-1 text-xs" value={tradeIdx} onChange={(e) => setTradeIdx(+e.target.value)}>
          {recent.map((t, i) => <option key={t.entry_ts} value={i}>{new Date(t.entry_ts).toISOString().slice(0, 16)} · {t.side} · {signed(t.pnl_r, 2, "R")}</option>)}
        </select>)}>
        {focus ? <CandleChart bars={bars.data} markers={markers} lines={lines} /> : <Empty>No stored trades.</Empty>}
      </Card>
      {report && (
        <div className="grid gap-6 xl:grid-cols-3">
          <Card title={report.passed ? "Validation: passed" : "Validation: rejected"}>
            <ul className="space-y-1.5 text-sm">
              {Object.entries(report.gates).map(([k, g]) => (
                <li key={k} className="flex items-center gap-2">
                  <span className={g.passed ? "text-good" : "text-bad"} aria-hidden>{g.passed ? "✓" : "✕"}</span>
                  <span className="text-ink-2">{GATE_LABEL[k] ?? k}</span>
                  <span className="tabular ml-auto">{String(g.value)}</span>
                  <span className="w-28 text-right text-xs text-muted">{g.rule}</span>
                </li>
              ))}
            </ul>
            <div className="mt-3 border-t border-line pt-3 text-xs text-muted">
              Monte Carlo: P(loss) {num(report.monte_carlo.prob_loss)} · 95% max DD {pct(report.monte_carlo.max_dd_p95)} · WFE {num(report.wfe)}
              {report.search_evaluations !== undefined && <> · selected from {int(report.search_evaluations)} genomes (GA window only)</>}
            </div>
          </Card>
          <Card title="Walk-forward folds" className="xl:col-span-2">
            <table className="w-full text-sm">
              <thead className="border-b border-line"><tr><Th>Test window</Th><Th right>IS exp.</Th><Th right>OOS exp.</Th><Th right>OOS trades</Th><Th right>OOS win</Th><Th right>OOS PF</Th></tr></thead>
              <tbody className="divide-y divide-line">
                {report.folds.map((f) => (
                  <tr key={f.index}>
                    <Td className="text-ink-2">{f.test_start.slice(0, 10)} → {f.test_end.slice(0, 10)}</Td>
                    <Td right>{signed(f.is.expectancy_r, 3, "R")}</Td>
                    <Td right className={(f.oos.expectancy_r ?? 0) > 0 ? "text-good" : "text-bad"}>{signed(f.oos.expectancy_r, 3, "R")}</Td>
                    <Td right>{int(f.oos.n_trades)}</Td><Td right>{pct(f.oos.win_rate, 0)}</Td><Td right>{num(f.oos.profit_factor)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
            {holdout && (
              <div className="mt-3 rounded-lg border border-accent/30 bg-accent/5 px-3 py-2 text-sm">
                <span className="text-accent">Holdout (evaluated once):</span> {int(holdout.n_trades)} trades · {signed(holdout.expectancy_r, 3, "R")} · win {pct(holdout.win_rate, 0)} · PF {num(holdout.profit_factor)}
              </div>
            )}
          </Card>
        </div>
      )}
      <Card title="Context rules (Meta-Analyzer)" actions={(rules.data ?? []).some((r) => r.status !== "validated") && (
        <button className="text-xs text-ink-2 hover:text-ink" onClick={() => setShowRejected((v) => !v)}>
          {showRejected ? "Hide" : "Show"} {(rules.data ?? []).filter((r) => r.status !== "validated").length} rejected candidates
        </button>)}>
        {(rules.data ?? []).length > 0 && !showRejected && !(rules.data ?? []).some((r) => r.status === "validated") && (
          <p className="mb-3 text-sm text-muted">No context rule survived out-of-sample testing with the multiple-testing correction. The strategy is used unfiltered.</p>
        )}
        {rules.data?.length ? (
          <table className="w-full text-sm">
            <thead className="border-b border-line"><tr><Th>Rule</Th><Th>Status</Th><Th right>Mining</Th><Th right>Out of sample</Th><Th right>Coverage</Th><Th right>Holm p</Th></tr></thead>
            <tbody className="divide-y divide-line">
              {rules.data.filter((r) => showRejected || r.status === "validated").map((r) => (
                <tr key={r.id}>
                  <Td>{r.description}{typeof r.details.refined_strategy_id === "string" && (
                    <Link href={`/strategies/${r.details.refined_strategy_id}`} className="ml-2 text-xs text-accent">refined strategy →</Link>)}</Td>
                  <Td><Badge status={r.status} /></Td>
                  <Td right>{signed(r.is_expectancy_r, 3, "R")} <span className="text-muted">({int(r.is_trades)})</span></Td>
                  <Td right className={(r.oos_expectancy_r ?? 0) > 0 ? "text-good" : "text-bad"}>{signed(r.oos_expectancy_r, 3, "R")} <span className="text-muted">({int(r.oos_trades)})</span></Td>
                  <Td right>{pct(r.coverage, 0)}</Td>
                  <Td right>{num(r.p_value_adjusted, 3)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <Empty>No context analysis yet.</Empty>}
      </Card>
    </div>
  );
}

"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { Badge, Button, Card, Empty, ErrorNote, Field, Td, Th, inputCls } from "@/components/ui";
import { api } from "@/lib/api";
import { int, num, pct, signed } from "@/lib/format";
import { useFetch } from "@/lib/hooks";

const SORTS = [
  ["fitness", "Fitness"], ["oos_expectancy_r", "OOS expectancy"], ["oos_sortino", "OOS Sortino"],
  ["oos_win_rate", "OOS win rate"], ["oos_profit_factor", "OOS profit factor"], ["deflated_sharpe", "Deflated Sharpe"],
  ["overfit_risk", "Overfit risk"], ["oos_trades", "OOS trades"], ["created_at", "Newest"],
] as const;
const STATUSES = ["validated", "incubating", "champion", "candidate", "rejected", "retired"];

function LeaderboardInner() {
  const sp = useSearchParams();
  const router = useRouter();
  const get = (k: string, d = "") => sp.get(k) ?? d;
  const statuses = sp.getAll("status");
  const oosOnly = get("oos_only", "1") === "1";
  const params = {
    symbol: "XAUUSD", timeframe: get("timeframe") || undefined, status: statuses.length ? statuses : undefined,
    min_trades: get("min_trades") || (oosOnly ? "1" : undefined), min_win_rate: get("min_win_rate") || undefined,
    min_profit_factor: get("min_profit_factor") || undefined, max_overfit_risk: get("max_overfit_risk") || undefined,
    run_id: get("run_id") || undefined, sort: get("sort", "fitness"),
    descending: get("sort") === "overfit_risk" ? get("descending", "false") : get("descending", "true"),
    limit: 50, offset: Number(get("offset", "0")),
  };
  const res = useFetch(() => api.strategies(params as Record<string, string | number | undefined | string[]>), [sp.toString()]);
  const update = (k: string, v: string | string[] | null) => {
    const next = new URLSearchParams(sp.toString());
    next.delete(k);
    if (Array.isArray(v)) v.forEach((x) => next.append(k, x));
    else if (v) next.set(k, v);
    if (k !== "offset") next.delete("offset");
    router.replace(`/leaderboard?${next.toString()}`);
  };
  const sortBy = (k: string) => {
    if (params.sort === k) update("descending", params.descending === "true" ? "false" : "true");
    else update("sort", k);
  };
  const SortTh = ({ k, children }: { k: string; children: string }) => (
    <Th right><button className={`hover:text-ink ${params.sort === k ? "text-accent" : ""}`} onClick={() => sortBy(k)}>
      {children}{params.sort === k ? (params.descending === "true" ? " ↓" : " ↑") : ""}</button></Th>
  );
  const total = res.data?.total ?? 0;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Leaderboard</h1>
        <p className="text-sm text-muted">All metrics are out of sample (walk-forward test windows the swarm never saw).</p>
      </div>
      <Card>
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Timeframe">
            <select className={inputCls} value={get("timeframe")} onChange={(e) => update("timeframe", e.target.value)}>
              <option value="">all</option>{["5m", "15m", "30m", "1h"].map((t) => <option key={t}>{t}</option>)}
            </select>
          </Field>
          <Field label="Min OOS trades"><input className={`${inputCls} w-24`} type="number" value={get("min_trades")} onChange={(e) => update("min_trades", e.target.value)} /></Field>
          <Field label="Min win rate"><input className={`${inputCls} w-24`} type="number" step="0.05" placeholder="0.5" value={get("min_win_rate")} onChange={(e) => update("min_win_rate", e.target.value)} /></Field>
          <Field label="Min profit factor"><input className={`${inputCls} w-24`} type="number" step="0.1" placeholder="1.2" value={get("min_profit_factor")} onChange={(e) => update("min_profit_factor", e.target.value)} /></Field>
          <Field label="Max overfit risk"><input className={`${inputCls} w-24`} type="number" step="0.05" placeholder="0.3" value={get("max_overfit_risk")} onChange={(e) => update("max_overfit_risk", e.target.value)} /></Field>
          <Field label="Sort">
            <select className={inputCls} value={params.sort} onChange={(e) => update("sort", e.target.value)}>
              {SORTS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
            </select>
          </Field>
          <div className="flex flex-wrap gap-1.5">
            {STATUSES.map((s) => {
              const on = statuses.includes(s);
              return (
                <button key={s} onClick={() => update("status", on ? statuses.filter((x) => x !== s) : [...statuses, s])}
                  className={`rounded-full border px-2.5 py-1 text-xs ${on ? "border-accent/60 bg-accent/10 text-accent" : "border-line text-ink-2 hover:text-ink"}`}>
                  {on ? "✓ " : ""}{s}
                </button>
              );
            })}
          </div>
          <label className="flex items-center gap-2 text-xs text-ink-2">
            <input type="checkbox" checked={oosOnly} onChange={(e) => update("oos_only", e.target.checked ? null : "0")} className="accent-[var(--color-accent)]" />
            only out-of-sample tested
          </label>
          {sp.toString() && <Button onClick={() => router.replace("/leaderboard")}>Reset</Button>}
        </div>
      </Card>
      <ErrorNote error={res.error} />
      <Card title={`${int(total)} strategies`}>
        {res.data && res.data.items.length === 0 ? <Empty>No strategies match these filters.</Empty> : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-line">
                <tr>
                  <Th>Strategy</Th><Th>Status</Th><Th>TF</Th>
                  <SortTh k="fitness">Fitness</SortTh><SortTh k="oos_expectancy_r">OOS exp.</SortTh>
                  <SortTh k="oos_win_rate">Win</SortTh><SortTh k="oos_profit_factor">PF</SortTh>
                  <SortTh k="oos_sortino">Sortino</SortTh><SortTh k="oos_trades">Trades</SortTh>
                  <SortTh k="deflated_sharpe">DSR</SortTh><SortTh k="overfit_risk">Overfit risk</SortTh>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {(res.data?.items ?? []).map((s) => (
                  <tr key={s.id} className="cursor-pointer hover:bg-raised/60" onClick={() => router.push(`/strategies/${s.id}`)}>
                    <Td><Link href={`/strategies/${s.id}`} className="font-medium hover:text-accent">{s.name}</Link>
                      <div className="text-[11px] text-muted">{s.origin}{s.generation_number !== null ? ` · gen ${s.generation_number}` : ""}</div></Td>
                    <Td><Badge status={s.status} /></Td>
                    <Td className="text-ink-2">{s.timeframe}</Td>
                    <Td right>{num(s.fitness)}</Td>
                    <Td right className={s.oos_expectancy_r === null ? "" : s.oos_expectancy_r > 0 ? "text-good" : "text-bad"}>{signed(s.oos_expectancy_r, 3, "R")}</Td>
                    <Td right>{pct(s.oos_win_rate, 0)}</Td>
                    <Td right>{num(s.oos_profit_factor)}</Td>
                    <Td right>{num(s.oos_sortino)}</Td>
                    <Td right>{int(s.oos_trades)}</Td>
                    <Td right>{num(s.deflated_sharpe, 3)}</Td>
                    <Td right><RiskBar v={s.overfit_risk} /></Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {total > 50 && (
          <div className="mt-3 flex items-center justify-end gap-2 text-xs text-muted">
            <Button disabled={params.offset === 0} onClick={() => update("offset", String(Math.max(0, params.offset - 50)))}>← Prev</Button>
            {params.offset + 1}–{Math.min(total, params.offset + 50)} of {int(total)}
            <Button disabled={params.offset + 50 >= total} onClick={() => update("offset", String(params.offset + 50))}>Next →</Button>
          </div>
        )}
      </Card>
    </div>
  );
}

function RiskBar({ v }: { v: number | null }) {
  if (v === null) return <span className="text-muted">—</span>;
  const tone = v < 0.3 ? "bg-good" : v < 0.6 ? "bg-warn" : "bg-bad";
  return (
    <span className="inline-flex items-center gap-2">
      <span className="h-1.5 w-14 overflow-hidden rounded bg-raised"><span className={`block h-full ${tone}`} style={{ width: `${v * 100}%` }} /></span>
      <span className="tabular w-8">{v.toFixed(2)}</span>
    </span>
  );
}

export default function LeaderboardPage() {
  return <Suspense><LeaderboardInner /></Suspense>;
}

"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { CandleChart, type ChartMarker, type PriceLineSpec } from "@/components/charts/CandleChart";
import { C } from "@/components/charts/theme";
import { Badge, Card, Empty, Side } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, int, nyTime, pct, price, signed } from "@/lib/format";
import { useFetch, useStream } from "@/lib/hooks";
import type { Bars, Champion, SignalRow } from "@/lib/types";

function ChampionCard({ c }: { c: Champion }) {
  const h = c.snapshot.holdout ?? {};
  return (
    <div className={`rounded-xl border bg-surface p-4 ${c.status === "active" ? "border-accent/40" : "border-line"}`}>
      <div className="flex items-center gap-2">
        <span className="text-lg font-bold text-accent">{c.rank ? `#${c.rank}` : "–"}</span>
        <Link href={`/strategies/${c.strategy_id}`} className="font-semibold hover:text-accent">{c.name}</Link>
        <span className="text-xs text-muted">{c.timeframe}</span>
        <span className="ml-auto"><Badge status={c.status} /></span>
      </div>
      <p className="mt-2 line-clamp-2 text-xs text-ink-2">{c.description?.split("\n")[0]}</p>
      <dl className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <div><dt className="text-muted">Backtest OOS</dt><dd className="tabular">{signed(c.snapshot.oos_expectancy_r as number | null, 2, "R")}</dd></div>
        <div><dt className="text-muted">Holdout</dt><dd className="tabular">{signed(h.expectancy_r, 2, "R")}</dd></div>
        <div><dt className="text-muted">Live</dt><dd className="tabular">{int(c.live.n ?? 0)} · {signed(c.live.mean_r, 2, "R")}</dd></div>
      </dl>
      {c.status === "incubating" && <p className="mt-2 text-[11px] text-muted">Paper-trading silently until it proves itself live (≥10 trades, ≥5 days, positive).</p>}
    </div>
  );
}

function Rationale({ s }: { s: SignalRow }) {
  const r = s.rationale;
  const List = ({ items, op, title }: { items?: { label: string; met: boolean; value: number | null }[]; op?: string; title: string }) =>
    items && items.length ? (
      <div>
        <div className="mb-1 text-xs text-muted">{title}{op === "any" ? " (any of)" : ""}</div>
        <ul className="space-y-1">
          {items.map((c, i) => (
            <li key={i} className="flex items-start gap-2 text-sm">
              <span className={c.met ? "text-good" : "text-muted"} aria-hidden>{c.met ? "✓" : "○"}</span>
              <span className={c.met ? "text-ink" : "text-muted"}>{c.label}</span>
              {c.value !== null && <span className="tabular ml-auto text-xs text-ink-2">{c.value.toFixed(2)}</span>}
            </li>
          ))}
        </ul>
      </div>
    ) : null;
  return (
    <div className="space-y-3">
      <List items={r.conditions} op={r.conditions_op} title="Why this entry" />
      <List items={r.filters} op={r.filters_op} title="Filters" />
      {r.context_rule && <div className="rounded-lg border border-s3/40 bg-s3/10 px-3 py-2 text-sm"><span className="text-xs text-muted">Context rule · </span>{r.context_rule}</div>}
      <dl className="grid grid-cols-3 gap-2 text-xs">
        <div><dt className="text-muted">Entry</dt><dd className="tabular text-sm">{price(s.entry_price)}</dd></div>
        <div><dt className="text-muted">Stop</dt><dd className="tabular text-sm text-bad">{price(s.stop_price)}</dd></div>
        <div><dt className="text-muted">Target</dt><dd className="tabular text-sm text-good">{price(s.target_price)}</dd></div>
        <div><dt className="text-muted">Risk : reward</dt><dd className="tabular text-sm">{s.risk_reward ? `1 : ${s.risk_reward.toFixed(2)}` : "—"}</dd></div>
        <div><dt className="text-muted">Historical win rate</dt><dd className="tabular text-sm">{pct(s.confidence, 0)}</dd></div>
        <div><dt className="text-muted">Result</dt><dd className="tabular text-sm">{s.pnl_r !== null ? signed(s.pnl_r, 2, "R") : s.status}</dd></div>
      </dl>
      <p className="rounded-lg bg-page px-3 py-2 text-xs leading-relaxed text-ink-2">{s.rationale_text}</p>
    </div>
  );
}

export default function SignalsPage() {
  const champions = useFetch(() => api.champions(), [], 20000);
  const signals = useFetch(() => api.signals({ kind: "entry", limit: 100 }), []);
  const live = useFetch(() => api.liveStatus(), [], 10000);
  const datasets = useFetch(() => api.datasets("XAUUSD"), []);
  const dataSource = live.data?.history_source
    ?? (datasets.data?.some((d) => d.source === "dukascopy") ? "dukascopy" : datasets.data?.[0]?.source);
  const [selected, setSelected] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const connected = useStream("signals", (e) => {
    if (["entry", "fill", "exit", "expire", "exit_pending"].includes(e.type)) {
      signals.reload();
      if (e.type === "entry") { setFlash(String(e.payload.id)); setTimeout(() => setFlash(null), 4000); }
    }
    if (e.type === "champions_changed" || e.type === "live_started") champions.reload();
  });
  const list = signals.data ?? [];
  const sel = list.find((s) => s.id === selected) ?? list[0];
  const bars = useFetch<Bars | null>(() => {
    if (!sel) return Promise.resolve(null);
    const t = new Date(sel.bar_ts).getTime();
    return api.bars("XAUUSD", { timeframe: sel.timeframe, end: new Date(t + 6 * 3600e3).toISOString(), limit: 160,
      source: dataSource }).catch(() => null);
  }, [sel?.id, dataSource]);
  const markers: ChartMarker[] = useMemo(() => sel ? [{
    time: new Date(sel.bar_ts).getTime() / 1000, position: sel.side === "long" ? "belowBar" : "aboveBar",
    color: C.accent, shape: sel.side === "long" ? "arrowUp" : "arrowDown", text: sel.side.toUpperCase(),
  }] : [], [sel]);
  const lines: PriceLineSpec[] = useMemo(() => sel ? [
    ...(sel.entry_price ? [{ price: sel.entry_price, color: C.ink2, title: "entry" }] : []),
    ...(sel.stop_price ? [{ price: sel.stop_price, color: C.down, title: "stop", dashed: true }] : []),
    ...(sel.target_price ? [{ price: sel.target_price, color: C.good, title: "target", dashed: true }] : []),
  ] : [], [sel]);
  const roster = champions.data ?? [];
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Live Signals</h1>
          <p className="text-sm text-muted">
            {live.data?.running ? `Engine running on ${live.data.feed} · last bar ${live.data.last_bar ? nyTime(live.data.last_bar) : "—"} NY`
              : "Live engine is off. Start it with `swarm live` or POST /api/v1/live/start."}
            {" · "}{connected ? "streaming" : "reconnecting…"}
          </p>
        </div>
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {roster.length ? roster.map((c) => <ChampionCard key={c.id} c={c} />)
          : <div className="md:col-span-3"><Empty>No champions yet. Validated strategies are promoted from their strategy page (the holdout runs once).</Empty></div>}
      </div>
      <div className="grid gap-6 xl:grid-cols-5">
        <Card title="Signal feed" className="xl:col-span-2">
          {list.length === 0 ? <Empty>No signals yet.</Empty> : (
            <ul className="max-h-[620px] divide-y divide-line overflow-y-auto">
              {list.map((s) => (
                <li key={s.id}>
                  <button onClick={() => setSelected(s.id)}
                    className={`flex w-full items-center gap-3 px-2 py-2.5 text-left text-sm transition ${sel?.id === s.id ? "bg-raised" : "hover:bg-raised/50"} ${flash === s.id ? "ring-1 ring-accent" : ""}`}>
                    <Side side={s.side} />
                    <div className="min-w-0">
                      <div className="tabular">{price(s.entry_price)} <span className="text-xs text-muted">{s.timeframe}</span></div>
                      <div className="truncate text-[11px] text-muted">{s.rationale.strategy} · {nyTime(s.bar_ts)} NY</div>
                    </div>
                    <div className="ml-auto flex flex-col items-end gap-1">
                      <Badge status={s.status} />
                      <span className={`tabular text-[11px] ${(s.pnl_r ?? 0) > 0 ? "text-good" : s.pnl_r !== null ? "text-bad" : "text-muted"}`}>
                        {s.pnl_r !== null ? signed(s.pnl_r, 2, "R") : ago(s.emitted_at)}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>
        <div className="space-y-6 xl:col-span-3">
          <Card title={sel ? `${sel.side.toUpperCase()} · ${sel.rationale.strategy ?? ""} · ${sel.timeframe}` : "Signal"}>
            {sel ? <CandleChart bars={bars.data} markers={markers} lines={lines} height={340} /> : <Empty>Select a signal.</Empty>}
          </Card>
          {sel && <Card title="Rationale"><Rationale s={sel} /></Card>}
        </div>
      </div>
    </div>
  );
}

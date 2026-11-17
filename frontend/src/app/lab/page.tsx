"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";
import { LineChart } from "@/components/charts/LineChart";
import { Scatter } from "@/components/charts/Scatter";
import { C } from "@/components/charts/theme";
import { Badge, Button, Card, Empty, ErrorNote, Field, Stat, inputCls } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, int, num } from "@/lib/format";
import { useFetch, useStream } from "@/lib/hooks";
import type { BusEvent, Generation, Run } from "@/lib/types";

const ORIGIN_GROUP: Record<string, string> = {
  seed: "explored", random: "explored", mutation: "evolved", crossover: "evolved", migrant: "evolved", meta: "refined",
};
const GROUPS = [
  { name: "explored", color: C.series[0]! }, { name: "evolved", color: C.series[1]! }, { name: "refined", color: C.series[2]! },
];

function StartRun({ onStarted }: { onStarted: (r: Run) => void }) {
  const [form, setForm] = useState({ name: "xau-swarm", timeframe: "15m", source: "dukascopy", start: "2019-01-01",
    end: "2025-01-01", islands: 4, population: 40, generations: 25, workers: 2 });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const set = (k: string, v: string | number) => setForm((f) => ({ ...f, [k]: v }));
  const submit = async () => {
    setBusy(true); setError(null);
    try {
      onStarted(await api.startRun({ ...form, start: `${form.start}T00:00:00Z`, end: `${form.end}T00:00:00Z` }));
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Field label="Name"><input className={inputCls} value={form.name} onChange={(e) => set("name", e.target.value)} /></Field>
        <Field label="Timeframe">
          <select className={inputCls} value={form.timeframe} onChange={(e) => set("timeframe", e.target.value)}>
            {["5m", "15m", "30m", "1h"].map((t) => <option key={t}>{t}</option>)}
          </select>
        </Field>
        <Field label="Data source">
          <select className={inputCls} value={form.source} onChange={(e) => set("source", e.target.value)}>
            {["dukascopy", "csv", "synthetic"].map((t) => <option key={t}>{t}</option>)}
          </select>
        </Field>
        <Field label="From"><input type="date" className={inputCls} value={form.start} onChange={(e) => set("start", e.target.value)} /></Field>
        <Field label="To"><input type="date" className={inputCls} value={form.end} onChange={(e) => set("end", e.target.value)} /></Field>
        <Field label="Islands"><input type="number" min={1} className={inputCls} value={form.islands} onChange={(e) => set("islands", +e.target.value)} /></Field>
        <Field label="Population / island"><input type="number" min={6} className={inputCls} value={form.population} onChange={(e) => set("population", +e.target.value)} /></Field>
        <Field label="Generations"><input type="number" min={1} className={inputCls} value={form.generations} onChange={(e) => set("generations", +e.target.value)} /></Field>
        <Field label="Workers"><input type="number" min={1} className={inputCls} value={form.workers} onChange={(e) => set("workers", +e.target.value)} /></Field>
        <div className="flex items-end"><Button variant="primary" onClick={submit} disabled={busy}>{busy ? "Starting…" : "Start swarm"}</Button></div>
      </div>
      <p className="text-xs text-muted">The swarm optimises on the first training window only. The 20% holdout stays locked until promotion.</p>
      <ErrorNote error={error} />
    </div>
  );
}

function LabInner() {
  const params = useSearchParams();
  const router = useRouter();
  const runs = useFetch(() => api.runs(30), [], 8000);
  const runId = params.get("run") ?? runs.data?.items[0]?.id ?? null;
  const run = runs.data?.items.find((r) => r.id === runId) ?? null;
  const gens = useFetch(() => (runId ? api.generations(runId) : Promise.resolve([] as Generation[])), [runId]);
  const strategies = useFetch(() => (runId ? api.strategies({ run_id: runId, limit: 500, sort: "fitness" }) : Promise.resolve(null)), [runId]);
  const [feed, setFeed] = useState<BusEvent[]>([]);
  const [showForm, setShowForm] = useState(false);

  const connected = useStream("lab", (e) => {
    setFeed((f) => [e, ...f].slice(0, 60));
    if (e.payload.run_id === runId && ["generation", "validated", "run_completed", "run_paused", "meta_analyzed"].includes(e.type)) {
      gens.reload(); runs.reload();
      if (e.type !== "generation") strategies.reload();
    }
  });

  const islands = useMemo(() => {
    const by: Record<number, Generation[]> = {};
    for (const g of gens.data ?? []) (by[g.island] ??= []).push(g);
    return by;
  }, [gens.data]);
  const islandIds = Object.keys(islands).map(Number).sort((a, b) => a - b);
  const bestSeries = islandIds.slice(0, 4).map((i) => ({
    name: `Island ${i + 1}`, color: C.series[i % 4]!,
    points: islands[i]!.filter((g) => g.stats.fitness).map((g) => ({ x: g.number, y: g.stats.fitness!.max })),
  }));
  // Mean fitness = population quality. (Genome diversity is ~1.0 by construction: islands
  // de-duplicate, so it's shown in the island cards instead of as a flat line.)
  const meanSeries = islandIds.slice(0, 4).map((i) => ({
    name: `Island ${i + 1}`, color: C.series[i % 4]!,
    points: islands[i]!.filter((g) => g.stats.fitness).map((g) => ({ x: g.number, y: g.stats.fitness!.mean })),
  }));
  const latest = islandIds.map((i) => islands[i]![islands[i]!.length - 1]!);
  const scatter = (strategies.data?.items ?? []).filter((s) => s.fitness !== null && s.fitness > -1e5).map((s) => ({
    x: s.generation_number ?? 0, y: s.fitness!, group: ORIGIN_GROUP[s.origin] ?? "explored",
    label: `${s.name} (${s.status})`, href: `/strategies/${s.id}`,
  }));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Lab</h1>
          <p className="text-sm text-muted">Watch the swarm evolve. Streaming {connected ? "live" : "(reconnecting…)"}</p>
        </div>
        <div className="flex items-center gap-2">
          <select className={inputCls} value={runId ?? ""} onChange={(e) => router.push(`/lab?run=${e.target.value}`)}>
            {(runs.data?.items ?? []).map((r) => <option key={r.id} value={r.id}>{r.name} · {r.status} · {ago(r.created_at)}</option>)}
          </select>
          {run?.status === "running" && <Button variant="danger" onClick={() => api.stopRun(run.id).then(runs.reload)}>Pause</Button>}
          {run && ["paused", "failed"].includes(run.status) && <Button onClick={() => api.resumeRun(run.id).then(runs.reload)}>Resume</Button>}
          <Button variant="primary" onClick={() => setShowForm((s) => !s)}>{showForm ? "Close" : "New run"}</Button>
        </div>
      </div>
      {showForm && <Card title="Start a swarm run"><StartRun onStarted={(r) => { setShowForm(false); runs.reload(); router.push(`/lab?run=${r.id}`); }} /></Card>}
      {!run ? <Empty>No runs yet. Start one above (or `swarm evolve` from the CLI).</Empty> : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            <Stat label="Status" value={<Badge status={run.status} />} sub={`${run.timeframe} · ${run.algorithm}`} />
            <Stat label="Generation" value={`${run.current_generation} / ${String(run.config.generations ?? "?")}`} />
            <Stat label="Strategies tried" value={int(run.evaluations)} sub="unique genomes, de-duplicated" />
            <Stat label="Best fitness" value={num(run.best_metrics.fitness as number | undefined)} sub={`t = ${num(run.best_metrics.t_stat as number | undefined)} · ${int(run.best_metrics.n_trades as number | undefined)} trades`} />
            <Stat label="Validated" value={strategies.data?.items.filter((s) => ["validated", "incubating", "champion"].includes(s.status)).length ?? 0}
              sub={`${strategies.data?.items.filter((s) => s.status === "rejected").length ?? 0} rejected out of sample`} />
          </div>
          <div className="grid gap-6 xl:grid-cols-2">
            <Card title="Best fitness per island"><LineChart series={bestSeries} xLabel="generation" /></Card>
            <Card title="Mean fitness per island (population quality)"><LineChart series={meanSeries} xLabel="generation" /></Card>
          </div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {latest.map((g) => (
              <div key={g.island} className="rounded-xl border border-line bg-surface p-4">
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: C.series[g.island % 4] }} aria-hidden />
                  Island {g.island + 1}
                  <span className="ml-auto text-xs font-normal text-muted">gen {g.number}</span>
                </div>
                <dl className="grid grid-cols-2 gap-y-1 text-xs">
                  <dt className="text-muted">best / mean</dt><dd className="tabular text-right">{num(g.stats.fitness?.max)} / {num(g.stats.fitness?.mean)}</dd>
                  <dt className="text-muted">Pareto front</dt><dd className="tabular text-right">{g.pareto_size}</dd>
                  <dt className="text-muted">diversity</dt><dd className="tabular text-right">{num(g.diversity)}</dd>
                  <dt className="text-muted">trading genomes</dt><dd className="tabular text-right">{g.stats.valid} / {g.stats.size}</dd>
                </dl>
              </div>
            ))}
          </div>
          <div className="grid gap-6 xl:grid-cols-3">
            <Card title="Discovery cloud: every stored strategy by generation" className="xl:col-span-2">
              <Scatter points={scatter} groups={GROUPS} xLabel="generation born" yLabel="fitness" />
            </Card>
            <Card title="Swarm activity">
              <ul className="max-h-[300px] space-y-1.5 overflow-y-auto text-xs">
                {feed.length === 0 && <li className="text-muted">{run.status === "running" ? "Waiting for the next generation…" : "This run isn't active. Events stream here while a run is running."}</li>}
                {feed.map((e, i) => (
                  <li key={i} className="flex gap-2 text-ink-2">
                    <span className="shrink-0 text-muted">{new Date(e.ts).toLocaleTimeString()}</span>
                    <span>
                      {e.type === "generation" && `gen ${String(e.payload.generation)} · ${int(e.payload.evaluations as number)} tried · ${e.payload.migrated ? "migration" : ""}`}
                      {e.type === "validated" && <>{String(e.payload.name)}: {e.payload.passed ? <span className="text-good">✓ validated</span> : <span className="text-bad">✕ rejected</span>} (risk {num(e.payload.overfit_risk as number)})</>}
                      {e.type === "meta_analyzed" && `context rules for ${String(e.payload.strategy)}: ${(e.payload.selected as string[]).length} kept`}
                      {!["generation", "validated", "meta_analyzed"].includes(e.type) && e.type.replace("_", " ")}
                    </span>
                  </li>
                ))}
              </ul>
            </Card>
          </div>
          <Card title="Top strategies of this run" actions={<Link className="text-xs text-accent" href={`/leaderboard?run_id=${run.id}`}>Open in leaderboard →</Link>}>
            <ul className="divide-y divide-line text-sm">
              {(strategies.data?.items ?? []).slice(0, 8).map((s) => (
                <li key={s.id} className="flex items-center gap-3 py-2">
                  <Link href={`/strategies/${s.id}`} className="font-medium hover:text-accent">{s.name}</Link>
                  <Badge status={s.status} /><span className="text-xs text-muted">{s.origin} · gen {s.generation_number}</span>
                  <span className="tabular ml-auto text-xs text-ink-2">fitness {num(s.fitness)}{s.overfit_risk !== null ? ` · overfit risk ${num(s.overfit_risk)}` : ""}</span>
                </li>
              ))}
            </ul>
          </Card>
        </>
      )}
    </div>
  );
}

export default function LabPage() {
  return <Suspense><LabInner /></Suspense>;
}

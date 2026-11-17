"use client";

import Link from "next/link";
import { api } from "@/lib/api";
import { ago, int, pct, signed } from "@/lib/format";
import { useFetch } from "@/lib/hooks";
import { Badge, Card, Empty, Side, Stat } from "@/components/ui";

export default function Overview() {
  const champions = useFetch(() => api.champions(), [], 15000);
  const signals = useFetch(() => api.signals({ kind: "entry", limit: 8 }), [], 10000);
  const runs = useFetch(() => api.runs(5), [], 10000);
  const validated = useFetch(() => api.strategies({ status: ["validated", "champion", "incubating"], sort: "overfit_risk", descending: "false", limit: 5 }), []);
  const active = champions.data?.filter((c) => c.status === "active") ?? [];
  const running = runs.data?.items.filter((r) => r.status === "running") ?? [];
  const today = signals.data?.filter((s) => Date.now() - new Date(s.emitted_at).getTime() < 86400e3).length ?? 0;
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Overview</h1>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Active champions" value={`${active.length} / 3`} sub={`${champions.data?.filter((c) => c.status === "incubating").length ?? 0} incubating`} />
        <Stat label="Entry signals (24h)" value={today} />
        <Stat label="Swarm runs running" value={running.length} sub={running[0] ? `${running[0].name} · gen ${running[0].current_generation}` : "idle"} />
        <Stat label="Validated strategies" value={validated.data?.total ?? "—"} sub="passed walk-forward + DSR gates" />
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Champions" actions={<Link href="/signals" className="text-xs text-accent">Live signals →</Link>}>
          {champions.data?.length ? (
            <ul className="divide-y divide-line">
              {champions.data.map((c) => (
                <li key={c.id} className="flex items-center gap-3 py-2.5 text-sm">
                  <span className="w-6 text-center font-semibold text-accent">{c.rank ? `#${c.rank}` : "–"}</span>
                  <Link href={`/strategies/${c.strategy_id}`} className="font-medium hover:text-accent">{c.name}</Link>
                  <Badge status={c.status} />
                  <span className="ml-auto text-xs text-ink-2">live {int(c.live.n ?? 0)} trades · {signed(c.live.mean_r, 2, "R")}</span>
                </li>
              ))}
            </ul>
          ) : <Empty>No champions yet. Promote a validated strategy from the Leaderboard.</Empty>}
        </Card>
        <Card title="Latest entry signals">
          {signals.data?.length ? (
            <ul className="divide-y divide-line">
              {signals.data.map((s) => (
                <li key={s.id} className="flex items-center gap-3 py-2.5 text-sm">
                  <Side side={s.side} />
                  <span className="tabular">{s.entry_price?.toFixed(2)}</span>
                  <span className="truncate text-xs text-ink-2">{s.rationale.strategy}</span>
                  <Badge status={s.status} />
                  <span className="ml-auto text-xs text-muted">{ago(s.emitted_at)}</span>
                </li>
              ))}
            </ul>
          ) : <Empty>No signals yet. Start the live engine (`swarm live`) once champions exist.</Empty>}
        </Card>
        <Card title="Most robust strategies" actions={<Link href="/leaderboard" className="text-xs text-accent">Leaderboard →</Link>}>
          {validated.data?.items.length ? (
            <ul className="divide-y divide-line">
              {validated.data.items.map((s) => (
                <li key={s.id} className="flex items-center gap-3 py-2.5 text-sm">
                  <Link href={`/strategies/${s.id}`} className="font-medium hover:text-accent">{s.name}</Link>
                  <span className="text-xs text-muted">{s.timeframe}</span>
                  <Badge status={s.status} />
                  <span className="ml-auto text-xs text-ink-2">OOS {signed(s.oos_expectancy_r, 2, "R")} · win {pct(s.oos_win_rate, 0)} · risk {s.overfit_risk?.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          ) : <Empty>Nothing validated yet. Run the swarm from the Lab.</Empty>}
        </Card>
        <Card title="Recent swarm runs" actions={<Link href="/lab" className="text-xs text-accent">Lab →</Link>}>
          {runs.data?.items.length ? (
            <ul className="divide-y divide-line">
              {runs.data.items.map((r) => (
                <li key={r.id} className="flex items-center gap-3 py-2.5 text-sm">
                  <Link href={`/lab?run=${r.id}`} className="font-medium hover:text-accent">{r.name}</Link>
                  <Badge status={r.status} />
                  <span className="ml-auto text-xs text-ink-2">{r.timeframe} · gen {r.current_generation} · {int(r.evaluations)} strategies tried</span>
                </li>
              ))}
            </ul>
          ) : <Empty>No runs yet.</Empty>}
        </Card>
      </div>
    </div>
  );
}

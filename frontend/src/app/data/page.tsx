"use client";

import { useState } from "react";
import { CandleChart } from "@/components/charts/CandleChart";
import { C } from "@/components/charts/theme";
import { Card, Empty, Td, Th } from "@/components/ui";
import { api } from "@/lib/api";
import { int, num, pct } from "@/lib/format";
import { useFetch } from "@/lib/hooks";
import type { Overlays } from "@/lib/types";

const MARK: Record<string, { color: string; shape: "arrowUp" | "arrowDown" | "circle"; pos: "aboveBar" | "belowBar"; text: string }> = {
  bos_bull: { color: C.up, shape: "arrowUp", pos: "belowBar", text: "BOS" },
  bos_bear: { color: C.down, shape: "arrowDown", pos: "aboveBar", text: "BOS" },
  choch_bull: { color: C.up, shape: "arrowUp", pos: "belowBar", text: "CHoCH" },
  choch_bear: { color: C.down, shape: "arrowDown", pos: "aboveBar", text: "CHoCH" },
  sweep_high: { color: C.accent, shape: "circle", pos: "aboveBar", text: "" },
  sweep_low: { color: C.accent, shape: "circle", pos: "belowBar", text: "" },
};

export default function DataPage() {
  const datasets = useFetch(() => api.datasets("XAUUSD"), []);
  const sources = [...new Set((datasets.data ?? []).map((d) => d.source))];
  const [source, setSource] = useState<string | null>(null);
  const [tf, setTf] = useState("15m");
  const src = source ?? sources[0] ?? "dukascopy";
  const bars = useFetch(() => api.bars("XAUUSD", { timeframe: tf, source: src, limit: 400 }), [src, tf]);
  const ov = useFetch<Overlays | null>(() => api.overlays("XAUUSD", { timeframe: tf, source: src, limit: 400 }).catch(() => null), [src, tf]);
  const markers = (ov.data?.markers ?? []).filter((m) => MARK[m.kind]).map((m) => ({
    time: m.time, position: MARK[m.kind]!.pos, color: MARK[m.kind]!.color, shape: MARK[m.kind]!.shape, text: MARK[m.kind]!.text,
  }));
  const m1 = (datasets.data ?? []).find((d) => d.timeframe === "1m" && d.source === src);
  const q = (m1?.quality ?? {}) as Record<string, unknown>;
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Data</h1>
      <Card title="XAUUSD market structure" actions={<>
        <select className="rounded border border-line bg-page px-2 py-1 text-xs" value={src} onChange={(e) => setSource(e.target.value)}>
          {(sources.length ? sources : ["dukascopy"]).map((s) => <option key={s}>{s}</option>)}
        </select>
        <select className="rounded border border-line bg-page px-2 py-1 text-xs" value={tf} onChange={(e) => setTf(e.target.value)}>
          {["5m", "15m", "30m", "1h", "4h"].map((t) => <option key={t}>{t}</option>)}
        </select></>}>
        <CandleChart bars={bars.data} markers={markers} height={420} />
        <p className="mt-2 text-xs text-muted">Arrows: BOS / CHoCH (green bullish, red bearish) · gold dots: liquidity sweeps. Shown at the bar where they became knowable (confirmed), never earlier.</p>
      </Card>
      <div className="grid gap-6 xl:grid-cols-3">
        <Card title="Datasets" className="xl:col-span-2">
          {datasets.data?.length ? (
            <table className="w-full text-sm">
              <thead className="border-b border-line"><tr><Th>Source</Th><Th>TF</Th><Th right>Rows</Th><Th>From</Th><Th>To</Th></tr></thead>
              <tbody className="divide-y divide-line">
                {datasets.data.map((d) => (
                  <tr key={d.id}><Td>{d.source}</Td><Td>{d.timeframe}</Td><Td right>{int(d.row_count)}</Td>
                    <Td className="text-ink-2">{d.start_ts?.slice(0, 10)}</Td><Td className="text-ink-2">{d.end_ts?.slice(0, 10)}</Td></tr>
                ))}
              </tbody>
            </table>
          ) : <Empty>No data yet: `swarm ingest dukascopy --start 2019-01-01 --end 2025-01-01`</Empty>}
        </Card>
        <Card title={`M1 quality · ${src}`}>
          {m1 ? (
            <dl className="grid grid-cols-2 gap-y-1.5 text-sm">
              <dt className="text-muted">Score</dt><dd className="tabular text-right">{num(q.score as number)}</dd>
              <dt className="text-muted">Coverage</dt><dd className="tabular text-right">{pct(q.coverage as number, 2)}</dd>
              <dt className="text-muted">Gaps (non-holiday)</dt><dd className="tabular text-right">{int((q.gaps as { count?: number })?.count)}</dd>
              <dt className="text-muted">Bad prints</dt><dd className="tabular text-right">{int(q.spike_count as number)}</dd>
              <dt className="text-muted">Large real moves</dt><dd className="tabular text-right">{int(q.large_moves as number)}</dd>
              <dt className="text-muted">Median spread</dt><dd className="tabular text-right">${num((q.spread as { median?: number })?.median)}</dd>
            </dl>
          ) : <Empty>No M1 dataset.</Empty>}
        </Card>
      </div>
    </div>
  );
}

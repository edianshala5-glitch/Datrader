"use client";

import { useMemo, useState } from "react";
import { C } from "./theme";

/** End-of-line labels nudged apart so they never overlap (min 15 units). */
function endLabels(series: LineSeries[], sx: (x: number) => number, sy: (y: number) => number, top: number, bottom: number) {
  const labels = series.flatMap((s) => {
    const last = [...s.points].sort((a, b) => a.x - b.x).at(-1);
    return last ? [{ name: s.name, x: sx(last.x), y: sy(last.y) }] : [];
  }).sort((a, b) => a.y - b.y);
  const gap = 15;
  for (let i = 1; i < labels.length; i++) labels[i]!.y = Math.max(labels[i]!.y, labels[i - 1]!.y + gap);
  const overflow = (labels.at(-1)?.y ?? 0) - bottom;
  if (overflow > 0) labels.forEach((l) => (l.y -= overflow));
  for (let i = labels.length - 2; i >= 0; i--) labels[i]!.y = Math.min(labels[i]!.y, labels[i + 1]!.y - gap);
  labels.forEach((l) => (l.y = Math.max(l.y, top)));
  return labels;
}

export interface LineSeries { name: string; color: string; points: { x: number; y: number }[] }

/** Small multi-series SVG line chart: legend + direct end labels (<= 4 series), crosshair tooltip. */
export function LineChart({ series, height = 240, xLabel, yFormat = (v: number) => v.toFixed(2) }: {
  series: LineSeries[]; height?: number; xLabel?: string; yFormat?: (v: number) => string;
}) {
  const [hoverX, setHoverX] = useState<number | null>(null);
  const W = 720, H = height, pad = { l: 52, r: 92, t: 12, b: 30 };
  const all = series.flatMap((s) => s.points);
  const { x0, x1, y0, y1 } = useMemo(() => {
    if (!all.length) return { x0: 0, x1: 1, y0: 0, y1: 1 };
    const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (lo === hi) { lo -= 1; hi += 1; }
    const m = (hi - lo) * 0.08;
    return { x0: Math.min(...xs), x1: Math.max(...xs) || 1, y0: lo - m, y1: hi + m };
  }, [all]);
  if (!all.length) return <div style={{ height }} className="flex items-center justify-center text-sm text-muted">No data yet</div>;
  const sx = (x: number) => pad.l + ((x - x0) / Math.max(x1 - x0, 1)) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (y - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const yTicks = Array.from({ length: 5 }, (_, i) => y0 + ((y1 - y0) * i) / 4);
  const xs = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b);
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = xs[0]!;
    for (const x of xs) if (Math.abs(sx(x) - px) < Math.abs(sx(best) - px)) best = x;
    setHoverX(best);
  };
  return (
    <div className="relative">
      {series.length > 1 && (
        <div className="mb-2 flex flex-wrap gap-3 text-xs text-ink-2">
          {series.map((s) => (
            <span key={s.name} className="flex items-center gap-1.5">
              <span className="h-0.5 w-4 rounded" style={{ background: s.color }} aria-hidden />{s.name}
            </span>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" onMouseMove={onMove} onMouseLeave={() => setHoverX(null)}
        role="img" aria-label={series.map((s) => s.name).join(", ")}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} stroke={C.grid} strokeWidth={1} />
            <text x={pad.l - 6} y={sy(t) + 3} fill={C.muted} fontSize={13} textAnchor="end" className="tabular">{yFormat(t)}</text>
          </g>
        ))}
        {y0 < 0 && y1 > 0 && <line x1={pad.l} x2={W - pad.r} y1={sy(0)} y2={sy(0)} stroke={C.axis} strokeWidth={1} />}
        {[x0, x1].map((x) => (
          <text key={x} x={sx(x)} y={H - 8} fill={C.muted} fontSize={13} textAnchor="middle">{x}</text>
        ))}
        {xLabel && <text x={(W - pad.r + pad.l) / 2} y={H - 8} fill={C.muted} fontSize={13} textAnchor="middle">{xLabel}</text>}
        {series.map((s) => {
          const pts = [...s.points].sort((a, b) => a.x - b.x);
          return (
            <polyline key={s.name} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
              points={pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
          );
        })}
        {series.length <= 4 && endLabels(series, sx, sy, pad.t + 6, H - pad.b).map((l) => (
          <text key={l.name} x={l.x + 6} y={l.y + 4} fill={C.ink2} fontSize={13}>{l.name}</text>
        ))}
        {hoverX !== null && (
          <g>
            <line x1={sx(hoverX)} x2={sx(hoverX)} y1={pad.t} y2={H - pad.b} stroke={C.axis} strokeWidth={1} />
            {series.map((s) => {
              const p = s.points.find((q) => q.x === hoverX);
              return p ? <circle key={s.name} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={s.color} stroke={C.surface} strokeWidth={2} /> : null;
            })}
          </g>
        )}
      </svg>
      {hoverX !== null && (
        <div className="pointer-events-none absolute right-2 top-8 rounded-lg border border-line bg-raised px-3 py-2 text-xs shadow-lg">
          <div className="mb-1 text-muted">{xLabel ?? "x"} {hoverX}</div>
          {series.map((s) => {
            const p = s.points.find((q) => q.x === hoverX);
            return p ? (
              <div key={s.name} className="flex items-center gap-2 text-ink-2">
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} aria-hidden />
                {s.name}<span className="tabular ml-auto pl-3 text-ink">{yFormat(p.y)}</span>
              </div>
            ) : null;
          })}
        </div>
      )}
    </div>
  );
}

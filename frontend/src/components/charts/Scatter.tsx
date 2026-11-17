"use client";

import { useState } from "react";
import { C } from "./theme";

export interface ScatterPoint { x: number; y: number; group: string; label: string; href?: string }

/** Strategies by generation (x) and fitness (y). At most 3 groups, validated for all-pairs CVD. */
export function Scatter({ points, groups, height = 260, xLabel, yLabel }: {
  points: ScatterPoint[]; groups: { name: string; color: string }[]; height?: number; xLabel: string; yLabel: string;
}) {
  const [hover, setHover] = useState<ScatterPoint | null>(null);
  if (!points.length) return <div style={{ height }} className="flex items-center justify-center text-sm text-muted">No strategies yet</div>;
  const W = 720, H = height, pad = { l: 44, r: 16, t: 12, b: 30 };
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const x0 = Math.min(...xs) - 0.5, x1 = Math.max(...xs) + 0.5;
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const m = (y1 - y0) * 0.08; y0 -= m; y1 += m;
  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (y - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const color = Object.fromEntries(groups.map((g) => [g.name, g.color]));
  return (
    <div className="relative">
      <div className="mb-2 flex gap-3 text-xs text-ink-2">
        {groups.map((g) => (
          <span key={g.name} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: g.color }} aria-hidden />{g.name}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${yLabel} by ${xLabel}`}>
        {Array.from({ length: 5 }, (_, i) => y0 + ((y1 - y0) * i) / 4).map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} stroke={C.grid} />
            <text x={pad.l - 6} y={sy(t) + 3} fill={C.muted} fontSize={10} textAnchor="end">{t.toFixed(1)}</text>
          </g>
        ))}
        <text x={(W + pad.l) / 2} y={H - 8} fill={C.muted} fontSize={10} textAnchor="middle">{xLabel}</text>
        {points.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={hover === p ? 6 : 4} fill={color[p.group] ?? C.muted}
            fillOpacity={0.85} stroke={C.surface} strokeWidth={2} className="cursor-pointer"
            onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)}
            onClick={() => p.href && (window.location.href = p.href)} />
        ))}
      </svg>
      {hover && (
        <div className="pointer-events-none absolute right-2 top-8 rounded-lg border border-line bg-raised px-3 py-2 text-xs shadow-lg">
          <div className="text-ink">{hover.label}</div>
          <div className="text-ink-2">{hover.group} · {xLabel} {hover.x} · {yLabel} <span className="tabular">{hover.y.toFixed(2)}</span></div>
        </div>
      )}
    </div>
  );
}

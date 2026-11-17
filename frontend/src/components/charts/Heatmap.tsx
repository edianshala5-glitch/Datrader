"use client";

import { useState } from "react";
import { WEEKDAYS } from "@/lib/format";
import { C } from "./theme";

interface Cell { n: number; mean: number }

function mix(a: string, b: string, t: number) {
  const p = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [ar, ag, ab] = p(a), [br, bg, bb] = p(b);
  const c = (x: number, y: number) => Math.round(x + (y - x) * t);
  return `rgb(${c(ar!, br!)},${c(ag!, bg!)},${c(ab!, bb!)})`;
}

/** Trading-day x NY-hour mean R. Diverging: red (losing) ← grey → blue (winning).
 * Cells with fewer than `minN` trades are hatched: too few to read anything into. */
export function WeekdayHourHeatmap({ trades, minN = 5 }: {
  trades: { weekday: number; hour: number; r: number }[]; minN?: number;
}) {
  const [hover, setHover] = useState<{ d: number; h: number; cell: Cell } | null>(null);
  const grid: Cell[][] = WEEKDAYS.map(() => Array.from({ length: 24 }, () => ({ n: 0, mean: 0 })));
  for (const t of trades) {
    const cell = grid[t.weekday]?.[t.hour];
    if (!cell) continue;
    cell.mean = (cell.mean * cell.n + t.r) / (cell.n + 1);
    cell.n += 1;
  }
  const cap = Math.max(0.25, ...grid.flat().filter((c) => c.n >= minN).map((c) => Math.abs(c.mean)));
  const fill = (c: Cell) => c.n < minN ? "url(#hatch)" : c.mean >= 0
    ? mix(C.diverging.mid, C.diverging.pos, Math.min(1, c.mean / cap))
    : mix(C.diverging.mid, C.diverging.neg, Math.min(1, -c.mean / cap));
  const cw = 26, ch = 24, L = 36, T = 18;
  return (
    <div className="relative overflow-x-auto">
      <svg viewBox={`0 0 ${L + 24 * cw + 4} ${T + 5 * ch + 4}`} className="w-full min-w-[640px]" role="img"
        aria-label="Mean R by trading day and New York hour">
        <defs>
          <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill={C.surface} />
            <line x1="0" y1="0" x2="0" y2="6" stroke={C.axis} strokeWidth="2" />
          </pattern>
        </defs>
        {Array.from({ length: 24 }, (_, h) => h % 3 === 0 && (
          <text key={h} x={L + h * cw + cw / 2} y={12} fill={C.muted} fontSize={9} textAnchor="middle">{String(h).padStart(2, "0")}</text>
        ))}
        {WEEKDAYS.map((d, i) => (
          <text key={d} x={L - 6} y={T + i * ch + ch / 2 + 3} fill={C.muted} fontSize={10} textAnchor="end">{d}</text>
        ))}
        {grid.map((row, d) => row.map((cell, h) => (
          <rect key={`${d}-${h}`} x={L + h * cw + 1} y={T + d * ch + 1} width={cw - 2} height={ch - 2} rx={3}
            fill={fill(cell)} stroke={hover?.d === d && hover.h === h ? C.ink : "none"} strokeWidth={1.5}
            onMouseEnter={() => setHover({ d, h, cell })} onMouseLeave={() => setHover(null)} />
        )))}
      </svg>
      <div className="mt-2 flex items-center gap-3 text-[11px] text-muted">
        <span className="flex items-center gap-1"><span className="h-2.5 w-6 rounded" style={{ background: `linear-gradient(90deg, ${C.diverging.neg}, ${C.diverging.mid}, ${C.diverging.pos})` }} />losing → winning (mean R)</span>
        <span className="flex items-center gap-1"><svg width="14" height="10"><rect width="14" height="10" fill="url(#hatch)" /></svg>fewer than {minN} trades</span>
        <span>hours in New York time · trading day rolls at 17:00 NY</span>
      </div>
      {hover && (
        <div className="pointer-events-none absolute right-2 top-0 rounded-lg border border-line bg-raised px-3 py-2 text-xs shadow-lg">
          <div className="text-ink">{WEEKDAYS[hover.d]} {String(hover.h).padStart(2, "0")}:00–{String(hover.h + 1).padStart(2, "0")}:00 NY</div>
          <div className="text-ink-2">{hover.cell.n} trades · mean <span className="tabular text-ink">{hover.cell.n ? `${hover.cell.mean >= 0 ? "+" : ""}${hover.cell.mean.toFixed(2)}R` : "—"}</span></div>
        </div>
      )}
    </div>
  );
}

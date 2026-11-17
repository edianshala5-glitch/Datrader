"use client";

import { createChart, type Time, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useRef } from "react";
import { C, nyFormatter } from "./theme";

/** Cumulative R over time: above zero in green, below in red (baseline series). */
export function EquityChart({ points, height = 220 }: { points: { time: number; value: number }[]; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current || points.length < 2) return;
    const chart = createChart(ref.current, {
      height,
      layout: { background: { color: C.surface }, textColor: C.muted, fontSize: 11 },
      grid: { vertLines: { visible: false }, horzLines: { color: C.grid } },
      rightPriceScale: { borderColor: C.axis },
      timeScale: { borderColor: C.axis, timeVisible: false },
      localization: { timeFormatter: (t: Time) => nyFormatter(t as number), priceFormatter: (p: number) => `${p.toFixed(1)}R` },
    });
    const s = chart.addBaselineSeries({
      baseValue: { type: "price", price: 0 }, lineWidth: 2,
      topLineColor: C.good, topFillColor1: "rgba(12,163,12,0.25)", topFillColor2: "rgba(12,163,12,0.02)",
      bottomLineColor: C.down, bottomFillColor1: "rgba(230,103,103,0.02)", bottomFillColor2: "rgba(230,103,103,0.25)",
    });
    // Lightweight Charts needs strictly increasing times.
    const data: { time: UTCTimestamp; value: number }[] = [];
    for (const p of points) {
      const last = data[data.length - 1];
      if (last && p.time <= last.time) last.value = p.value;
      else data.push({ time: p.time as UTCTimestamp, value: p.value });
    }
    s.setData(data);
    chart.timeScale().fitContent();
    const ro = new ResizeObserver(([e]) => e && chart.applyOptions({ width: e.contentRect.width }));
    ro.observe(ref.current);
    return () => { ro.disconnect(); chart.remove(); };
  }, [points, height]);
  if (points.length < 2) return <div style={{ height }} className="flex items-center justify-center text-sm text-muted">Not enough trades yet</div>;
  return <div ref={ref} className="w-full" role="img" aria-label="Equity curve in cumulative R" />;
}

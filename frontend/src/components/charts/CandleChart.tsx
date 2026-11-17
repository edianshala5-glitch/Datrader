"use client";

import { CrosshairMode, LineStyle, createChart, type IChartApi, type SeriesMarker, type Time, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useRef } from "react";
import type { Bars } from "@/lib/types";
import { C, nyFormatter } from "./theme";

export interface ChartMarker { time: number; position: "aboveBar" | "belowBar" | "inBar"; color: string;
  shape: "arrowUp" | "arrowDown" | "circle" | "square"; text?: string }
export interface PriceLineSpec { price: number; color: string; title: string; dashed?: boolean }

/** OHLC candles (mid prices) with optional trade/structure markers and price lines. */
export function CandleChart({ bars, markers = [], lines = [], height = 380 }: {
  bars: Bars | null; markers?: ChartMarker[]; lines?: PriceLineSpec[]; height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  useEffect(() => {
    if (!ref.current || !bars) return;
    const chart = createChart(ref.current, {
      height,
      layout: { background: { color: C.surface }, textColor: C.muted, fontSize: 11 },
      grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
      rightPriceScale: { borderColor: C.axis },
      timeScale: { borderColor: C.axis, timeVisible: true, secondsVisible: false },
      crosshair: { mode: CrosshairMode.Normal },
      localization: { timeFormatter: (t: Time) => nyFormatter(t as number) },
    });
    chartRef.current = chart;
    const series = chart.addCandlestickSeries({
      upColor: C.up, downColor: C.down, wickUpColor: C.up, wickDownColor: C.down, borderVisible: false,
    });
    series.setData(bars.time.map((t, i) => ({
      time: t as UTCTimestamp, open: bars.open[i]!, high: bars.high[i]!, low: bars.low[i]!, close: bars.close[i]!,
    })));
    // Markers must sit on an existing bar: snap each to the bar containing it.
    const snapped: SeriesMarker<Time>[] = markers
      .map((m) => {
        let lo = 0, hi = bars.time.length - 1;
        while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (bars.time[mid]! <= m.time) lo = mid; else hi = mid - 1; }
        return { ...m, time: bars.time[lo] as UTCTimestamp };
      })
      .filter((m) => (m.time as number) >= bars.time[0]!)
      .sort((a, b) => (a.time as number) - (b.time as number));
    series.setMarkers(snapped);
    for (const l of lines) {
      series.createPriceLine({ price: l.price, color: l.color, lineWidth: 2, title: l.title,
        lineStyle: l.dashed ? LineStyle.Dashed : LineStyle.Solid, axisLabelVisible: true });
    }
    chart.timeScale().fitContent();
    const ro = new ResizeObserver(([e]) => e && chart.applyOptions({ width: e.contentRect.width }));
    ro.observe(ref.current);
    return () => { ro.disconnect(); chart.remove(); chartRef.current = null; };
  }, [bars, markers, lines, height]);
  if (!bars) return <div style={{ height }} className="animate-pulse rounded-lg bg-raised" />;
  if (!bars.time.length) return <div style={{ height }} className="flex items-center justify-center text-sm text-muted">No bars for this window</div>;
  return <div ref={ref} className="w-full" role="img" aria-label={`${bars.symbol} ${bars.timeframe} candlestick chart`} />;
}

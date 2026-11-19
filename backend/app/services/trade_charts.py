# ruff: noqa: E501, RUF001  (the embedded HTML/JS template)
"""A self-contained HTML page of a strategy's trades, for a trader to review and label.

Each card shows the zone-timeframe candles around the trade (zone, entry, stop, target,
exit), a 1-minute close-up of the entry, and buttons to label the trade. Labels are kept in
the browser and downloaded as JSON for ``swarm labels-summary``.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.models.enums import Timeframe
from engine.backtest.runner import CostConfig, run_backtest
from engine.data.calendar import MarketCalendar
from engine.strategy.compiler import FeatureProvider
from engine.strategy.genome import ZONE_PROVIDERS, Genome

NY = "America/New_York"


def _bars_payload(df: pd.DataFrame) -> list[dict[str, float]]:
    mid = {
        f: ((df[f"bid_{f}"] + df[f"ask_{f}"]) / 2).round(2).to_numpy()
        for f in ("open", "high", "low", "close")
    }
    # Chart time axis in New York wall-clock time (lightweight-charts shows UTC seconds).
    local = pd.DatetimeIndex(df["ts"]).tz_convert(NY).tz_localize(None)
    secs = (local.asi8 // 10**9).astype(np.int64)
    return [
        {
            "time": int(t),
            "open": float(o),
            "high": float(h),
            "low": float(lo),
            "close": float(c),
        }
        for t, o, h, lo, c in zip(
            secs, mid["open"], mid["high"], mid["low"], mid["close"], strict=True
        )
    ]


def _ny_secs(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).tz_convert(NY).tz_localize(None).value // 10**9)


def build_examples(
    genome: Genome,
    bars_1m: pd.DataFrame,
    htf_bars: dict[Timeframe, pd.DataFrame],
    *,
    calendar: MarketCalendar,
    sessions: dict[str, Any],
    costs: CostConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n: int = 40,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backtest ``genome`` on [start, end) and package ``n`` trades spread over the period."""
    zone = genome.zone_feature()
    if zone is None or zone.timeframe is None:
        raise ValueError("show-trades needs a zone or harmonic setup on a higher timeframe")
    zone_tf = zone.timeframe
    bars_1m = bars_1m.reset_index(drop=True)
    fp = FeatureProvider(bars_1m, Timeframe.M1, calendar, sessions, htf_bars)
    run = run_backtest(
        genome,
        bars_1m,
        Timeframe.M1,
        calendar=calendar,
        sessions=sessions,
        costs=costs,
        htf_bars=htf_bars,
        start=start,
        end=end,
        with_context=False,
        provider=fp,
    )
    trades = run.trades
    if trades.empty:
        return [], run.metrics
    ts = pd.DatetimeIndex(bars_1m["ts"])
    offset = int(ts.searchsorted(pd.Timestamp(start)))

    def level(output: str) -> np.ndarray:
        return fp.get(zone.model_copy(update={"output": output}))

    provider = ZONE_PROVIDERS[zone.indicator]
    levels = {side: (level(provider[side][1]), level(provider[side][2])) for side in provider}
    pivots: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    if zone.indicator == "harmonics":
        for side, pre in (("long", "bull"), ("short", "bear")):
            pivots[side] = [
                (name.upper(), level(f"{pre}_{name}"), level(f"{pre}_{name}_idx"))
                for name in "xabc"
            ]
    # Spread the sample over the whole period instead of taking the first n trades.
    picks = np.unique(np.linspace(0, len(trades) - 1, min(n, len(trades))).round().astype(int))
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-2, 3, size=len(picks))
    picks = np.unique(np.clip(picks + jitter, 0, len(trades) - 1))

    zbars = htf_bars[zone_tf].reset_index(drop=True)
    zts = pd.DatetimeIndex(zbars["ts"])
    out = []
    for k, t in enumerate(trades.iloc[picks].to_dict(orient="records")):
        side = str(t["side"])
        sig = int(t["signal_idx"]) + offset
        lo_arr, hi_arr = levels[side]
        z_lo, z_hi = float(lo_arr[sig]), float(hi_arr[sig])
        # When did this zone appear? Walk back while the aligned level stays the same.
        first = sig
        while first > 0 and lo_arr[first - 1] == z_lo and hi_arr[first - 1] == z_hi:
            first -= 1
        entry_ts, exit_ts = pd.Timestamp(t["entry_ts"]), pd.Timestamp(t["exit_ts"])
        zone_start = ts[first]
        a = max(0, int(zts.searchsorted(zone_start)) - 40)
        b = min(len(zbars), int(zts.searchsorted(exit_ts)) + 12)
        e = int(ts.searchsorted(entry_ts))
        m1 = bars_1m.iloc[max(0, e - 45) : min(len(bars_1m), e + 30)]
        points = []
        for label, price_arr, idx_arr in pivots.get(side, []):
            j = idx_arr[sig]
            if np.isfinite(j) and 0 <= int(j) < len(zbars):
                points.append(
                    {
                        "time": _ny_secs(zts[int(j)]),
                        "value": round(float(price_arr[sig]), 2),
                        "label": label,
                    }
                )
                a = min(a, max(0, int(j) - 10))  # keep X in view
        if points:
            points.append(
                {
                    "time": _ny_secs(entry_ts),
                    "value": round(float(t["entry_price"]), 2),
                    "label": "D",
                }
            )
        target, r = float(t["target_price"]), float(t["r"])
        out.append(
            {
                "id": f"{entry_ts.isoformat()}|{side}",
                "n": k + 1,
                "side": side,
                "entry_time_ny": entry_ts.tz_convert(NY).strftime("%a %d %b %Y %H:%M"),
                "entry": round(float(t["entry_price"]), 2),
                "stop": round(float(t["stop_price"]), 2),
                "target": None if np.isnan(target) else round(target, 2),
                "exit": round(float(t["exit_price"]), 2),
                "exit_reason": str(t["exit_reason"]),
                "r": None if np.isnan(r) else round(r, 2),
                "zone": {
                    "low": round(z_lo, 2),
                    "high": round(z_hi, 2),
                    "from": _ny_secs(zone_start),
                    "to": _ny_secs(entry_ts),
                },
                "entry_time": _ny_secs(entry_ts),
                "exit_time": _ny_secs(exit_ts),
                "zone_tf": zone_tf.value,
                "pivots": points,
                "bars": _bars_payload(zbars.iloc[a:b]),
                "bars_1m": _bars_payload(m1),
            }
        )
    return out, run.metrics


def render_page(
    title: str,
    rule: str,
    strategy: dict[str, Any],
    examples: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> str:
    data = json.dumps(
        {
            "title": title,
            "rule": rule,
            "strategy": strategy,
            "examples": examples,
            "summary": {
                "n_trades": int(metrics.get("n_trades") or 0),
                "expectancy_r": metrics.get("expectancy_r"),
                "win_rate": metrics.get("win_rate"),
                "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
        },
        default=str,
    ).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data)


_TEMPLATE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Datrader · revisione trade</title>
<style>
  :root { --page:#0d0d0d; --surface:#1a1a19; --raised:#232321; --line:#2c2c2a; --ink:#f0efec;
          --ink2:#c3c2b7; --muted:#898781; --accent:#e3b341; --good:#0ca30c; --bad:#d03b3b;
          --zone:#3987e5; color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,
         -apple-system, "Segoe UI", sans-serif; }
  header { position:sticky; top:0; z-index:5; background:rgba(13,13,13,.95);
           border-bottom:1px solid var(--line); padding:12px 16px; display:flex; flex-wrap:wrap;
           gap:12px; align-items:center; justify-content:space-between; }
  h1 { font-size:17px; margin:0; } h1 b { color:var(--accent); }
  main { max-width:1100px; margin:0 auto; padding:16px; }
  .intro { background:var(--surface); border:1px solid var(--line); border-radius:12px;
           padding:14px 16px; color:var(--ink2); }
  .intro code { color:var(--ink); white-space:pre-wrap; font-size:12px; }
  .card { background:var(--surface); border:1px solid var(--line); border-radius:12px;
          margin:16px 0; overflow:hidden; }
  .card.done { border-color:#3a3a37; }
  .head { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:baseline; padding:12px 16px;
          border-bottom:1px solid var(--line); }
  .head .t { font-weight:700; } .muted { color:var(--muted); }
  .long { color:var(--good); font-weight:700; } .short { color:var(--bad); font-weight:700; }
  .win { color:var(--good); } .loss { color:var(--bad); }
  .charts { display:grid; grid-template-columns: 2fr 1fr; gap:1px; background:var(--line); }
  .chart { background:var(--surface); height:340px; position:relative; }
  .chart .lbl { position:absolute; top:6px; left:10px; z-index:2; font-size:12px;
                color:var(--muted); pointer-events:none; }
  @media (max-width: 760px) { .charts { grid-template-columns: 1fr; } .chart { height:280px; } }
  .vote { display:flex; flex-wrap:wrap; gap:8px; padding:12px 16px; align-items:center; }
  button { font:inherit; border-radius:8px; border:1px solid var(--line); background:var(--raised);
           color:var(--ink); padding:8px 12px; cursor:pointer; }
  button:hover { border-color:#4a4a46; }
  button.on.take { background:#0d3a0d; border-color:var(--good); }
  button.on.skip { background:#3a1414; border-color:var(--bad); }
  button.on.zone { background:#3a2f0d; border-color:var(--accent); }
  input[type=text] { flex:1 1 220px; min-width:0; font:inherit; color:var(--ink); background:var(--raised);
          border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
  .primary { background:var(--accent); color:#111; border-color:var(--accent); font-weight:700; }
  .legend span { display:inline-block; margin-right:14px; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px;
        vertical-align:-1px; }
</style>
</head>
<body>
<header>
  <h1><b>Au</b> Datrader · revisione trade <span class="muted" id="sub"></span></h1>
  <div><span id="count" class="muted"></span>
    <button class="primary" id="dl">Scarica le mie risposte</button></div>
</header>
<main>
  <div class="intro" id="intro"></div>
  <div id="cards"></div>
</main>
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const D = __DATA__;
const KEY = "datrader.labels." + D.strategy.name;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} };
const C = { up:"#26a69a", down:"#ef5350", zone:"#3987e5", entry:"#f0efec", stop:"#d03b3b",
            target:"#0ca30c", grid:"#232321", text:"#898781" };
const fmtR = r => r == null ? "n/d" : (r > 0 ? "+" : "") + r.toFixed(2) + "R";
const reasons = { stop:"stop", target:"target", trailing_stop:"trailing stop",
                  session_close:"chiusura 17:00 NY", time:"tempo max", signal:"segnale uscita",
                  end_of_data:"fine periodo" };

document.getElementById("sub").textContent = "· " + D.strategy.name;
const s = D.summary;
document.getElementById("intro").innerHTML =
  `<p><b>${D.title}</b>. ${D.examples.length} trade di esempio presi da ${s.n_trades} trade del
   periodo di ricerca (risultato medio ${fmtR(s.expectancy_r)} a trade). Orari di New York.</p>
   <p class="legend"><span><i class="sw" style="background:${C.zone}"></i>zona</span>
   <span><i class="sw" style="background:${C.entry}"></i>entrata</span>
   <span><i class="sw" style="background:${C.stop}"></i>stop</span>
   <span><i class="sw" style="background:${C.target}"></i>target</span></p>
   <p><b>Alla cieca:</b> prima del voto vedi il grafico solo fino al momento dell'entrata, come dal
   vivo. Dopo il voto compaiono il resto del grafico e il risultato.</p>
   <p>Per ogni trade scegli: <b>La prenderei</b>, <b>Non la prenderei</b> oppure
   <b>Zona/pattern sbagliato</b> (non è come lo disegneresti tu), e se vuoi aggiungi una nota.
   Le risposte restano salvate in questo browser. Alla fine premi <b>Scarica le mie risposte</b>.</p>
   <p class="muted">Regola: <code>${D.rule}</code></p>`;

function chart(el, allBars, ex, oneMin, revealed) {
  el.querySelectorAll(".tv-lightweight-charts").forEach(n => n.remove());
  // Blind until voted: only what a trader saw at the moment of entry.
  const tf = oneMin ? 60 : ({ "5m": 300, "15m": 900, "30m": 1800, "1h": 3600 }[ex.zone_tf] || 900);
  const bars = revealed ? allBars : allBars.filter(b => b.time + tf <= ex.entry_time);
  const c = LightweightCharts.createChart(el, {
    layout: { background: { color: "#1a1a19" }, textColor: C.text, fontSize: 12 },
    grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
    rightPriceScale: { borderColor: "#2c2c2a" },
    timeScale: { borderColor: "#2c2c2a", timeVisible: true, secondsVisible: false },
    crosshair: { mode: 0 }, autoSize: true,
  });
  const cs = c.addCandlestickSeries({ upColor: C.up, downColor: C.down, borderVisible: false,
                                      wickUpColor: C.up, wickDownColor: C.down,
                                      priceLineVisible: false, lastValueVisible: false });
  cs.setData(bars);
  const times = bars.map(b => b.time);
  const inZone = times.filter(t => t >= ex.zone.from && t <= Math.max(ex.zone.to, ex.zone.from));
  if (inZone.length) {
    for (const p of [ex.zone.high, ex.zone.low]) {
      const z = c.addLineSeries({ color: C.zone, lineWidth: 2, priceLineVisible: false,
                                  lastValueVisible: false, crosshairMarkerVisible: false });
      z.setData(inZone.map(t => ({ time: t, value: p })));
    }
  }
  if (!oneMin && ex.pivots && ex.pivots.length) {
    const pts = ex.pivots.filter(pt => revealed || pt.label !== "D");
    const ln = c.addLineSeries({ color: "#e3b341", lineWidth: 2, priceLineVisible: false,
                                 lastValueVisible: false, crosshairMarkerVisible: false });
    const seen = new Set(); const data = [];
    for (const pt of pts) { if (!seen.has(pt.time)) { seen.add(pt.time); data.push({ time: pt.time, value: pt.value }); } }
    ln.setData(data);
    ln.setMarkers(pts.map(pt => ({ time: pt.time, position: "inBar", color: "#e3b341", shape: "circle", text: pt.label })));
  }
  cs.createPriceLine({ price: ex.entry, color: C.entry, lineWidth: 1, lineStyle: 2, title: "entry" });
  cs.createPriceLine({ price: ex.stop, color: C.stop, lineWidth: 1, lineStyle: 2, title: "stop" });
  if (ex.target != null)
    cs.createPriceLine({ price: ex.target, color: C.target, lineWidth: 1, lineStyle: 2, title: "target" });
  const snap = t => { let best = times[0];
    for (const x of times) { if (x <= t) best = x; else break; } return best; };
  const marks = [{ time: snap(ex.entry_time), position: ex.side === "long" ? "belowBar" : "aboveBar",
                   color: ex.side === "long" ? C.up : C.down,
                   shape: ex.side === "long" ? "arrowUp" : "arrowDown", text: "entry" }];
  if (revealed && (!oneMin || ex.exit_time <= times[times.length - 1]))
    marks.push({ time: snap(ex.exit_time), position: "inBar", color: "#c3c2b7", shape: "circle",
                 text: "exit" });
  marks.sort((a, b) => a.time - b.time);
  cs.setMarkers(marks);
  c.timeScale().fitContent();
}

function render() {
  const root = document.getElementById("cards");
  root.innerHTML = "";
  for (const ex of D.examples) {
    const card = document.createElement("section");
    card.className = "card";
    const res = ex.r == null ? "" : `<span class="${ex.r > 0 ? "win" : "loss"}">${fmtR(ex.r)}</span>`;
    const outcome = `esito: ${reasons[ex.exit_reason] || ex.exit_reason} ${res}`;
    card.innerHTML = `
      <div class="head"><span class="t">#${ex.n}</span>
        <span class="${ex.side}">${ex.side === "long" ? "LONG" : "SHORT"}</span>
        <span>${ex.entry_time_ny}</span>
        <span class="muted">entry ${ex.entry} · stop ${ex.stop} · target ${ex.target ?? "–"}</span>
        <span class="out"></span></div>
      <div class="charts">
        <div class="chart"><span class="lbl">${ex.zone_tf} · zona e trade</span></div>
        <div class="chart"><span class="lbl">1m · entrata</span></div></div>
      <div class="vote">
        <button data-v="take" class="take">La prenderei</button>
        <button data-v="skip" class="skip">Non la prenderei</button>
        <button data-v="zone" class="zone">Zona/pattern sbagliato</button>
        <input type="text" placeholder="Nota (facoltativa): cosa faresti diversamente?"></div>`;
    root.appendChild(card);
    const [big, small] = card.querySelectorAll(".chart");
    const out = card.querySelector(".out");
    let shown = null;
    const draw = () => {
      const revealed = !!(labels[ex.id] && labels[ex.id].choice);
      if (shown === revealed) return;
      shown = revealed;
      chart(big, ex.bars, ex, false, revealed);
      chart(small, ex.bars_1m, ex, true, revealed);
      out.innerHTML = revealed ? outcome : '<span class="muted">esito nascosto finché non voti</span>';
    };
    const btns = card.querySelectorAll("button[data-v]");
    const note = card.querySelector("input");
    const paint = () => {
      const l = labels[ex.id];
      btns.forEach(b => b.classList.toggle("on", !!l && l.choice === b.dataset.v));
      card.classList.toggle("done", !!l && !!l.choice);
      note.value = (l && l.note) || "";
      draw();
      count();
    };
    btns.forEach(b => b.addEventListener("click", () => {
      labels[ex.id] = { ...(labels[ex.id] || {}), choice: b.dataset.v, n: ex.n, side: ex.side,
                        r: ex.r, entry_time_ny: ex.entry_time_ny };
      save(); paint();
    }));
    note.addEventListener("change", () => {
      labels[ex.id] = { ...(labels[ex.id] || {}), note: note.value, n: ex.n, side: ex.side,
                        r: ex.r, entry_time_ny: ex.entry_time_ny };
      save(); paint();
    });
    paint();
  }
}

function count() {
  const done = D.examples.filter(e => labels[e.id] && labels[e.id].choice).length;
  document.getElementById("count").textContent = `${done}/${D.examples.length} valutati `;
}

document.getElementById("dl").addEventListener("click", () => {
  const out = { strategy: D.strategy, generated: D.summary.generated,
                labels: D.examples.filter(e => labels[e.id]).map(e => ({ id: e.id, ...labels[e.id] })) };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `risposte-${D.strategy.name}.json`;
  a.click();
});

if (window.LightweightCharts) render();
else document.getElementById("cards").innerHTML =
  "<p>Non riesco a caricare la libreria dei grafici: serve la connessione a internet.</p>";
</script>
</body>
</html>
"""

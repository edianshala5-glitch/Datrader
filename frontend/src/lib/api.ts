import type {
  BacktestRow, Bars, Champion, ContextRule, Dataset, Generation, Health, LiveStatus, Overlays,
  Page, Run, SignalRow, StrategyDetail, StrategySummary, TradeRow,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
export const WS_URL = API_URL.replace(/^http/, "ws");

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}/api/v1${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try { detail = (await res.json()).detail; } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const qs = (params: Record<string, string | number | boolean | undefined | null | string[]>) => {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => sp.append(k, x));
    else sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
};

export const api = {
  health: () => request<Health>("/health"),
  runs: (limit = 50) => request<Page<Run>>(`/runs${qs({ limit })}`),
  run: (id: string) => request<Run>(`/runs/${id}`),
  generations: (id: string) => request<Generation[]>(`/runs/${id}/generations`),
  startRun: (body: Record<string, unknown>) => request<Run>("/runs", { method: "POST", body: JSON.stringify(body) }),
  stopRun: (id: string) => request<Run>(`/runs/${id}/stop`, { method: "POST" }),
  resumeRun: (id: string) => request<Run>(`/runs/${id}/resume`, { method: "POST" }),
  strategies: (params: Record<string, string | number | undefined | string[]>) =>
    request<Page<StrategySummary>>(`/strategies${qs(params)}`),
  strategy: (id: string) => request<StrategyDetail>(`/strategies/${id}`),
  backtests: (id: string) => request<BacktestRow[]>(`/strategies/${id}/backtests`),
  trades: (id: string) => request<TradeRow[]>(`/strategies/${id}/trades`),
  rules: (id: string) => request<ContextRule[]>(`/strategies/${id}/rules`),
  promote: (id: string) => request<Champion>(`/champions/promote/${id}`, { method: "POST", body: "{}" }),
  champions: () => request<Champion[]>("/champions"),
  signals: (params: Record<string, string | number | undefined> = {}) =>
    request<SignalRow[]>(`/signals${qs({ limit: 100, ...params })}`),
  liveStatus: () => request<LiveStatus>("/live/status"),
  bars: (symbol: string, params: Record<string, string | number | undefined>) =>
    request<Bars>(`/data/${symbol}/bars${qs(params)}`),
  overlays: (symbol: string, params: Record<string, string | number | undefined>) =>
    request<Overlays>(`/data/${symbol}/overlays${qs(params)}`),
  datasets: (symbol: string) => request<Dataset[]>(`/data/${symbol}/datasets`),
};

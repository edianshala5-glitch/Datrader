"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { WS_URL } from "./api";
import type { BusEvent } from "./types";

/** Fetch on mount (and on `deps` change); `reload()` refetches. Optional polling. */
export function useFetch<T>(fn: () => Promise<T>, deps: unknown[] = [], pollMs?: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const reload = useCallback(() => {
    fnRef.current().then((d) => { setData(d); setError(null); })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);
  useEffect(() => {
    setLoading(true);
    reload();
    if (!pollMs) return;
    const t = setInterval(reload, pollMs);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, pollMs, reload]);
  return { data, error, loading, reload, setData };
}

/** Subscribe to a backend event channel over WebSocket, reconnecting with backoff. */
export function useStream(channel: "lab" | "signals" | "jobs" | "system", onEvent: (e: BusEvent) => void) {
  const [connected, setConnected] = useState(false);
  const cb = useRef(onEvent);
  cb.current = onEvent;
  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let attempt = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      ws = new WebSocket(`${WS_URL}/api/v1/ws/${channel}`);
      ws.onopen = () => { attempt = 0; setConnected(true); };
      ws.onmessage = (m) => { try { cb.current(JSON.parse(m.data as string) as BusEvent); } catch { /* ignore */ } };
      ws.onclose = () => {
        setConnected(false);
        if (closed) return;
        timer = setTimeout(connect, Math.min(15000, 500 * 2 ** attempt++));
      };
    };
    connect();
    return () => { closed = true; clearTimeout(timer); ws?.close(); };
  }, [channel]);
  return connected;
}

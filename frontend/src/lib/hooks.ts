"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, wsBase } from "./api";
import { getToken, requireLogin } from "./auth";
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
      // Browsers can't set headers on a WebSocket, so the token goes in the query string.
      const token = getToken();
      const q = token ? `?token=${encodeURIComponent(token)}` : "";
      ws = new WebSocket(`${wsBase()}/api/v1/ws/${channel}${q}`);
      let opened = false;
      ws.onopen = () => { opened = true; attempt = 0; setConnected(true); };
      ws.onmessage = (m) => { try { cb.current(JSON.parse(m.data as string) as BusEvent); } catch { /* ignore */ } };
      ws.onclose = () => {
        setConnected(false);
        if (closed) return;
        // A handshake refused for auth reaches the browser as a bare 1006, so ask the API.
        if (!opened) {
          api.session()
            .then((s) => { if (s.auth_enabled && !s.authenticated) requireLogin(); })
            .catch(() => { /* API down: keep retrying */ });
        }
        timer = setTimeout(connect, Math.min(15000, 500 * 2 ** attempt++));
      };
    };
    connect();
    return () => { closed = true; clearTimeout(timer); ws?.close(); };
  }, [channel]);
  return connected;
}

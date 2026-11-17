"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { api } from "@/lib/api";
import { useFetch, useStream } from "@/lib/hooks";

const NAV = [
  { href: "/", label: "Overview", icon: "◎" },
  { href: "/lab", label: "Lab", icon: "⚗" },
  { href: "/leaderboard", label: "Leaderboard", icon: "☰" },
  { href: "/signals", label: "Live Signals", icon: "⚡" },
  { href: "/data", label: "Data", icon: "▤" },
];

function Dot({ ok, label }: { ok: boolean | null; label: string }) {
  const c = ok === null ? "bg-muted" : ok ? "bg-good" : "bg-bad";
  return (
    <span className="flex items-center gap-1.5 text-xs text-ink-2">
      <span className={`h-2 w-2 rounded-full ${c}`} aria-hidden />
      {label}
      <span className="sr-only">{ok ? "ok" : "down"}</span>
    </span>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const health = useFetch(() => api.health(), [], 15000);
  const live = useFetch(() => api.liveStatus(), [], 10000);
  const wsUp = useStream("system", () => undefined);
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-52 shrink-0 flex-col border-r border-line bg-surface md:flex">
        <div className="px-5 py-5">
          <div className="text-lg font-bold tracking-tight"><span className="text-accent">Au</span> Swarm</div>
          <div className="text-[11px] text-muted">XAUUSD strategy lab</div>
        </div>
        <nav className="flex flex-col gap-0.5 px-2">
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <Link key={n.href} href={n.href}
                className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition ${
                  active ? "bg-raised text-ink" : "text-ink-2 hover:bg-raised/60 hover:text-ink"}`}>
                <span className={`w-4 text-center ${active ? "text-accent" : "text-muted"}`} aria-hidden>{n.icon}</span>
                {n.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto px-5 py-4 text-[11px] leading-relaxed text-muted">
          Research tooling, not financial advice. Signals only, no orders.
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex h-12 items-center justify-between gap-4 border-b border-line bg-page/90 px-4 backdrop-blur md:px-6">
          <nav className="flex gap-3 text-sm md:hidden">
            {NAV.map((n) => <Link key={n.href} href={n.href} className="text-ink-2">{n.icon}</Link>)}
          </nav>
          <div className="hidden text-sm text-muted md:block">XAUUSD · spot gold</div>
          <div className="flex items-center gap-4">
            <Dot ok={health.data ? health.data.status === "ok" : health.error ? false : null} label="API" />
            <Dot ok={wsUp} label="Stream" />
            <Dot ok={live.data ? !!live.data.running : null}
              label={live.data?.running ? `Live · ${live.data.feed} · ${live.data.champions ?? 0} champions` : "Live engine off"} />
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1500px] flex-1 px-4 py-6 md:px-6">{children}</main>
      </div>
    </div>
  );
}

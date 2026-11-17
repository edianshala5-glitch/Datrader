"use client";

import type { ReactNode } from "react";

export function Card({ title, actions, children, className = "" }: {
  title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-xl border border-line bg-surface ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          <div className="flex items-center gap-2">{actions}</div>
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone }: {
  label: string; value: ReactNode; sub?: ReactNode; tone?: "good" | "bad" | "warn";
}) {
  const color = tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : tone === "warn" ? "text-warn" : "text-ink";
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3">
      <div className="text-xs text-muted">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${color}`}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-2">{sub}</div>}
    </div>
  );
}

const STATUS: Record<string, { cls: string; icon: string }> = {
  validated: { cls: "border-good/40 text-good", icon: "✓" },
  champion: { cls: "border-accent/50 text-accent", icon: "★" },
  active: { cls: "border-accent/50 text-accent", icon: "★" },
  incubating: { cls: "border-s1/50 text-s1", icon: "◐" },
  candidate: { cls: "border-axis text-ink-2", icon: "·" },
  rejected: { cls: "border-bad/40 text-bad", icon: "✕" },
  retired: { cls: "border-axis text-muted", icon: "–" },
  demoted: { cls: "border-bad/40 text-bad", icon: "↓" },
  running: { cls: "border-s1/50 text-s1", icon: "●" },
  completed: { cls: "border-good/40 text-good", icon: "✓" },
  paused: { cls: "border-warn/40 text-warn", icon: "‖" },
  failed: { cls: "border-bad/40 text-bad", icon: "✕" },
  pending: { cls: "border-warn/40 text-warn", icon: "…" },
  target_hit: { cls: "border-good/40 text-good", icon: "✓" },
  stop_hit: { cls: "border-bad/40 text-bad", icon: "✕" },
  closed: { cls: "border-axis text-ink-2", icon: "■" },
  expired: { cls: "border-axis text-muted", icon: "–" },
};

/** Status always carries an icon + label, never colour alone. */
export function Badge({ status }: { status: string }) {
  const s = STATUS[status] ?? { cls: "border-axis text-ink-2", icon: "·" };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${s.cls}`}>
      <span aria-hidden>{s.icon}</span>{status.replace("_", " ")}
    </span>
  );
}

export function Side({ side }: { side: "long" | "short" }) {
  return (
    <span className={`rounded px-1.5 py-0.5 text-[11px] font-bold ${side === "long" ? "bg-good/15 text-good" : "bg-bad/15 text-bad"}`}>
      {side === "long" ? "▲ LONG" : "▼ SHORT"}
    </span>
  );
}

export function Button({ children, onClick, variant = "default", disabled, type = "button" }: {
  children: ReactNode; onClick?: () => void; variant?: "default" | "primary" | "danger";
  disabled?: boolean; type?: "button" | "submit";
}) {
  const v = variant === "primary" ? "bg-accent text-black hover:brightness-110"
    : variant === "danger" ? "border border-bad/50 text-bad hover:bg-bad/10"
    : "border border-line text-ink-2 hover:bg-raised hover:text-ink";
  return (
    <button type={type} onClick={onClick} disabled={disabled}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${v}`}>
      {children}
    </button>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-lg border border-dashed border-line px-4 py-8 text-center text-sm text-muted">{children}</div>;
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">⚠ {error}</div>;
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-muted">
      {label}
      {children}
    </label>
  );
}

export const inputCls =
  "rounded-lg border border-line bg-page px-2.5 py-1.5 text-sm text-ink outline-none focus:border-accent/60";

export function Th({ children, right }: { children?: ReactNode; right?: boolean }) {
  return <th className={`px-3 py-2 text-xs font-medium text-muted ${right ? "text-right" : "text-left"}`}>{children}</th>;
}

export function Td({ children, right, className = "" }: { children?: ReactNode; right?: boolean; className?: string }) {
  return <td className={`px-3 py-2 ${right ? "tabular text-right" : ""} ${className}`}>{children}</td>;
}

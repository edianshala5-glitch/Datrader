"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState, type FormEvent } from "react";
import { api, ApiError } from "@/lib/api";
import { setToken } from "@/lib/auth";

function safeNext(raw: string | null): string {
  // Only same-site paths: never redirect to another origin after login.
  return raw && raw.startsWith("/") && !raw.startsWith("//") && raw !== "/login" ? raw : "/";
}

function LoginForm() {
  const router = useRouter();
  const next = safeNext(useSearchParams().get("next"));
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const { token } = await api.login(password);
      setToken(token);
      router.replace(next);
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 429
          ? "Too many failed attempts. Wait a few minutes and try again."
          : err instanceof ApiError && err.status === 401
            ? "Wrong password."
            : "Can't reach the API.",
      );
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="w-full max-w-sm rounded-xl border border-line bg-surface p-6">
      <div className="mb-1 text-lg font-bold tracking-tight"><span className="text-accent">Au</span> Swarm</div>
      <p className="mb-5 text-sm text-muted">Sign in to the XAUUSD strategy lab.</p>
      <label htmlFor="password" className="mb-1.5 block text-xs text-ink-2">Password</label>
      <input
        id="password" type="password" autoComplete="current-password" autoFocus required
        value={password} onChange={(e) => setPassword(e.target.value)}
        className="w-full rounded-lg border border-line bg-raised px-3 py-2 text-sm text-ink outline-none focus:border-accent"
      />
      {error && <p role="alert" className="mt-3 text-sm text-bad">{error}</p>}
      <button type="submit" disabled={busy || !password}
        className="mt-5 w-full rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-page transition hover:brightness-110 disabled:opacity-50">
        {busy ? "Signing in…" : "Sign in"}
      </button>
    </form>
  );
}

export default function LoginPage() {
  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <Suspense fallback={null}><LoginForm /></Suspense>
    </div>
  );
}

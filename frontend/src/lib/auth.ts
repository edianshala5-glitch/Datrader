"use client";

/**
 * Session token for the single dashboard user. Kept in localStorage so a reload stays
 * logged in. Every access is guarded, because storage can be unavailable (private mode,
 * blocked site data).
 */
const KEY = "swarm.token";
export const AUTH_REQUIRED_EVENT = "swarm:auth-required";

export function getToken(): string | null {
  try { return window.localStorage.getItem(KEY); } catch { return null; }
}

export function setToken(token: string): void {
  try { window.localStorage.setItem(KEY, token); } catch { /* storage unavailable */ }
}

export function clearToken(): void {
  try { window.localStorage.removeItem(KEY); } catch { /* storage unavailable */ }
}

/** Tell the shell to send the user to /login (token missing, expired or revoked). */
export function requireLogin(): void {
  clearToken();
  window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT));
}

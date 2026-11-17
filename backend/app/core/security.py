"""Single-user authentication: scrypt password hash + HMAC-signed session tokens.

Standard library only. The password hash lives in ``SWARM_AUTH_PASSWORD_HASH``
(make one with ``swarm hash-password``). Tokens carry a fingerprint of that hash,
so changing the password logs out every session.

Hash format (``:``-separated, no ``$`` so it survives docker-compose ``.env`` files)::

    scrypt:<n>:<r>:<p>:<salt b64url>:<key b64url>
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

_N, _R, _P, _DKLEN = 2**14, 8, 1, 32
_MAXMEM = 64 * 1024 * 1024


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("use at least 10 characters")
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(
        password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN, maxmem=_MAXMEM
    )
    return f"scrypt:{_N}:{_R}:{_P}:{_b64e(salt)}:{_b64e(key)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, n, r, p, salt, key = encoded.split(":")
        if algo != "scrypt":
            return False
        expected = _b64d(key)
        got = hashlib.scrypt(
            password.encode(),
            salt=_b64d(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
            maxmem=_MAXMEM,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, expected)


def check_hash_format(encoded: str) -> None:
    """Raise ValueError if ``encoded`` is not a hash this module produced."""
    parts = encoded.split(":")
    if len(parts) != 6 or parts[0] != "scrypt" or not all(x.isdigit() for x in parts[1:4]):
        raise ValueError("not a scrypt hash from `swarm hash-password`")


@dataclass(frozen=True)
class TokenSigner:
    """Issues and checks ``<payload>.<signature>`` tokens."""

    key: bytes
    password_hash: str
    ttl_seconds: int

    def _fingerprint(self) -> str:
        return hashlib.sha256(self.password_hash.encode()).hexdigest()[:16]

    def _sign(self, body: str) -> str:
        return _b64e(hmac.new(self.key, body.encode(), hashlib.sha256).digest())

    def issue(self, now: float | None = None) -> tuple[str, int]:
        iat = int(now if now is not None else time.time())
        exp = iat + self.ttl_seconds
        body = _b64e(
            json.dumps(
                {"sub": "owner", "iat": iat, "exp": exp, "pw": self._fingerprint()},
                separators=(",", ":"),
            ).encode()
        )
        return f"{body}.{self._sign(body)}", exp

    def verify(self, token: str, now: float | None = None) -> bool:
        body, _, sig = token.partition(".")
        if not body or not sig or not hmac.compare_digest(sig, self._sign(body)):
            return False
        try:
            claims = json.loads(_b64d(body))
        except ValueError:
            return False
        current = now if now is not None else time.time()
        return (
            isinstance(claims, dict)
            and claims.get("sub") == "owner"
            and isinstance(claims.get("exp"), int)
            and claims["exp"] > current
            and hmac.compare_digest(str(claims.get("pw", "")), self._fingerprint())
        )


class LoginThrottle:
    """Per-client lockout after repeated failures (in memory; one API process)."""

    def __init__(self, max_failures: int, lockout_seconds: int) -> None:
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self._failures: dict[str, list[float]] = {}

    def _recent(self, client: str, now: float) -> list[float]:
        window = [t for t in self._failures.get(client, []) if now - t < self.lockout_seconds]
        if window:
            self._failures[client] = window
        else:
            self._failures.pop(client, None)
        return window

    def retry_after(self, client: str, now: float | None = None) -> int:
        """Seconds until ``client`` may try again (0 = allowed)."""
        now = now if now is not None else time.time()
        window = self._recent(client, now)
        if len(window) < self.max_failures:
            return 0
        return max(1, int(window[0] + self.lockout_seconds - now) + 1)

    def failed(self, client: str, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self._failures.setdefault(client, []).append(now)
        if len(self._failures) > 10_000:  # bound memory under a spray of client addresses
            for key in list(self._failures)[:5_000]:
                del self._failures[key]

    def succeeded(self, client: str) -> None:
        self._failures.pop(client, None)

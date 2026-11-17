"""Request IDs, access logs, latency metrics, security headers and a JSON 500 handler.

A pure ASGI middleware, so WebSocket upgrades and streaming responses pass through untouched.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core import metrics

logger = logging.getLogger("app.access")
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_QUIET_PATHS = frozenset({"/api/v1/health", "/metrics"})

SECURITY_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store"),
]


def current_request_id() -> str | None:
    return _request_id.get()


_TOKEN_IN_TEXT = re.compile(r"(token=)[^&\s\"']+", re.IGNORECASE)


class RedactTokensFilter(logging.Filter):
    """Masks ``token=...`` in log lines.

    Dashboard WebSockets authenticate via ``?token=``, and uvicorn logs every
    handshake with its full query string.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # malformed args: leave the record alone
            return True
        if "token=" in message.lower():
            record.msg = _TOKEN_IN_TEXT.sub(r"\1***", message)
            record.args = None
        return True


class RequestIdFilter(logging.Filter):
    """Adds ``request_id`` to records logged while a request is being handled."""

    def filter(self, record: logging.LogRecord) -> bool:
        rid = _request_id.get()
        if rid is not None and not hasattr(record, "request_id"):
            record.request_id = rid
        return True


def _route_label(scope: Scope) -> str:
    """The matched route template, e.g. ``/api/v1/strategies/{strategy_id}``.

    Never the raw path: that would make metric label cardinality unbounded.
    """
    route = scope.get("route")
    if route is None:
        return "unmatched"
    # FastAPI nests included routers; the effective context holds the full prefixed template.
    effective = (scope.get("fastapi") or {}).get("effective_route_context")
    path = getattr(effective, "path_format", None) or getattr(route, "path_format", None)
    return str(path) if path else "unmatched"


class ObservabilityMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex[:16]
        token = _request_id.set(rid)
        start = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                extra = [*SECURITY_HEADERS, (b"x-request-id", rid.encode())]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        try:
            if scope["type"] == "websocket":
                await self.app(scope, receive, send)
                return
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception:
                logger.exception(
                    "unhandled error", extra={"method": scope["method"], "path": scope["path"]}
                )
                if response_started:
                    raise
                status_code = 500
                await _send_json_500(send_wrapper, rid)
            finally:
                elapsed = time.perf_counter() - start
                route = _route_label(scope)
                metrics.HTTP_REQUESTS.labels(scope["method"], route, str(status_code)).inc()
                metrics.HTTP_LATENCY.labels(scope["method"], route).observe(elapsed)
                level = (
                    logging.WARNING
                    if status_code >= 500
                    else logging.DEBUG
                    if scope["path"] in _QUIET_PATHS or scope["method"] == "OPTIONS"
                    else logging.INFO
                )
                logger.log(
                    level,
                    "%s %s %d %.1fms",
                    scope["method"],
                    scope["path"],
                    status_code,
                    elapsed * 1000,
                    extra={
                        "method": scope["method"],
                        "path": scope["path"],
                        "route": route,
                        "status": status_code,
                        "duration_ms": round(elapsed * 1000, 2),
                    },
                )
        finally:
            _request_id.reset(token)


async def _send_json_500(send: Any, rid: str) -> None:
    body = json.dumps({"detail": "internal server error", "request_id": rid}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})

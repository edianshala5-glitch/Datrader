"""Auth dependency: every API route except health and login requires a valid token.

HTTP clients send ``Authorization: Bearer <token>``. Browsers cannot set headers on a
WebSocket, so ``/ws/*`` also accepts ``?token=``. HTTP routes never do, because query
strings end up in proxy logs.
"""

from __future__ import annotations

from fastapi import HTTPException, WebSocketException, status
from starlette.requests import HTTPConnection

from app.core.security import TokenSigner


def bearer_token(conn: HTTPConnection) -> str | None:
    header = conn.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    if conn.scope["type"] == "websocket":
        return conn.query_params.get("token") or None
    return None


def is_authenticated(conn: HTTPConnection) -> bool:
    signer: TokenSigner | None = conn.app.state.signer
    if signer is None:  # auth disabled
        return True
    token = bearer_token(conn)
    return token is not None and signer.verify(token)


async def require_auth(conn: HTTPConnection) -> None:
    if is_authenticated(conn):
        return
    if conn.scope["type"] == "websocket":
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="unauthorized")
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )

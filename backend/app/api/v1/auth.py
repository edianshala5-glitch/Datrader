"""Login for the single dashboard user."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.auth import is_authenticated
from app.core import metrics
from app.core.security import LoginThrottle, TokenSigner, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


class LoginResponse(BaseModel):
    token: str
    expires_at: datetime


class SessionInfo(BaseModel):
    auth_enabled: bool
    authenticated: bool


@router.get("/session", response_model=SessionInfo)
async def session_info(request: Request) -> SessionInfo:
    return SessionInfo(
        auth_enabled=request.app.state.signer is not None,
        authenticated=is_authenticated(request),
    )


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, request: Request) -> LoginResponse:
    signer: TokenSigner | None = request.app.state.signer
    if signer is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "authentication is disabled")
    throttle: LoginThrottle = request.app.state.login_throttle
    client = request.client.host if request.client else "unknown"
    wait = throttle.retry_after(client)
    if wait:
        metrics.LOGIN_ATTEMPTS.labels("locked").inc()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many failed attempts, try again later",
            headers={"Retry-After": str(wait)},
        )
    # scrypt is deliberately slow (~50 ms): keep it off the event loop.
    ok = await asyncio.to_thread(verify_password, body.password, signer.password_hash)
    if not ok:
        throttle.failed(client)
        metrics.LOGIN_ATTEMPTS.labels("failure").inc()
        logger.warning("login failed", extra={"client": client})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong password")
    throttle.succeeded(client)
    metrics.LOGIN_ATTEMPTS.labels("success").inc()
    token, exp = signer.issue()
    logger.info("login ok", extra={"client": client})
    return LoginResponse(token=token, expires_at=datetime.fromtimestamp(exp, UTC))

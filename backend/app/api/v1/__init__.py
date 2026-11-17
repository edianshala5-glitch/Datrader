from fastapi import APIRouter, Depends

from app.api.auth import require_auth
from app.api.v1 import (
    auth,
    backtest,
    data,
    health,
    indicators,
    instruments,
    live,
    meta,
    runs,
    strategies,
    validation,
    ws,
)

api_router = APIRouter(prefix="/api/v1")
# Public: liveness checks and the login itself.
api_router.include_router(health.router)
api_router.include_router(auth.router)

# Everything else requires a session token when auth is enabled.
protected = APIRouter(dependencies=[Depends(require_auth)])
protected.include_router(instruments.router)
protected.include_router(runs.router)
protected.include_router(strategies.router)
protected.include_router(data.router)
protected.include_router(indicators.router)
protected.include_router(backtest.router)
protected.include_router(validation.router)
protected.include_router(meta.router)
protected.include_router(live.router)
protected.include_router(ws.router)
api_router.include_router(protected)

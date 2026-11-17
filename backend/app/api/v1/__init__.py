from fastapi import APIRouter

from app.api.v1 import (
    backtest,
    data,
    health,
    indicators,
    instruments,
    meta,
    runs,
    strategies,
    validation,
    ws,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(instruments.router)
api_router.include_router(runs.router)
api_router.include_router(strategies.router)
api_router.include_router(data.router)
api_router.include_router(indicators.router)
api_router.include_router(backtest.router)
api_router.include_router(validation.router)
api_router.include_router(meta.router)
api_router.include_router(ws.router)

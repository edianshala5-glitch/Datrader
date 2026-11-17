from fastapi import APIRouter

from app.api.v1 import data, health, indicators, instruments, runs, strategies, ws

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(instruments.router)
api_router.include_router(runs.router)
api_router.include_router(strategies.router)
api_router.include_router(data.router)
api_router.include_router(indicators.router)
api_router.include_router(ws.router)

from __future__ import annotations

from pydantic import BaseModel, Field


class Page[T](BaseModel):
    items: list[T]
    total: int = Field(ge=0)
    limit: int
    offset: int


class HealthComponent(BaseModel):
    ok: bool
    detail: str | None = None


class Health(BaseModel):
    status: str  # "ok" | "degraded"
    version: str
    environment: str
    database: HealthComponent
    event_bus: HealthComponent

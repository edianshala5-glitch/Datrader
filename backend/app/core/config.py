"""Application settings, loaded from environment variables prefixed with ``SWARM_``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWARM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Swarm Trader"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # Root for Parquet market data, feature caches and equity curves.
    data_dir: Path = Path("./data")

    # SQLite works out of the box for a single machine; docker-compose switches to Postgres.
    database_url: str = "sqlite+aiosqlite:///./data/swarm.db"
    database_echo: bool = False

    # None -> in-process event bus (single process). Set to use Redis pub/sub across workers.
    redis_url: str | None = None
    event_channel_prefix: str = "swarm"

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    default_symbol: str = "XAUUSD"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()

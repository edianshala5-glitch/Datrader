"""Application settings, loaded from environment variables prefixed with ``SWARM_``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
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

    # Live feed used by autostart and by default: "capital" (Capital.com demo) or "oanda".
    live_feed: Literal["capital", "oanda"] = "capital"

    # Capital.com (free demo account). API key: Settings > API integrations (needs 2FA).
    # Use the key's custom password. These keys can trade, but this system never sends orders.
    capital_api_key: SecretStr | None = None
    capital_identifier: str | None = None  # login e-mail
    capital_password: SecretStr | None = None
    capital_demo: bool = True
    capital_epic: str = "GOLD"

    # Live feed: OANDA v20 practice account (free). The token can also be injected by a proxy.
    oanda_token: str | None = None
    oanda_account_id: str | None = None
    oanda_practice: bool = True

    # Optional alert channels (active champions only).
    notify_webhook_url: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    # Single-user auth. Enabled when a password hash is set (`swarm hash-password`).
    auth_password_hash: SecretStr | None = None
    # Signs session tokens. Required with auth in prod; in dev a random per-process key is used.
    secret_key: SecretStr | None = None
    token_ttl_hours: int = Field(24 * 7, ge=1)
    # Prod refuses to start without auth unless this is set (e.g. bound to localhost only).
    auth_disabled: bool = False
    login_max_failures: int = Field(5, ge=1)
    login_lockout_seconds: int = Field(900, ge=1)

    metrics_enabled: bool = True
    docs_enabled: bool = True

    # Start the OANDA live engine with the API and restart it after crashes.
    live_autostart: bool = False
    live_history_source: str = "dukascopy"

    @model_validator(mode="after")
    def _check_prod(self) -> Settings:
        if self.environment == "prod" and not self.auth_disabled:
            missing = [
                name
                for name, value in (
                    ("SWARM_AUTH_PASSWORD_HASH", self.auth_password_hash),
                    ("SWARM_SECRET_KEY", self.secret_key),
                )
                if value is None or not value.get_secret_value()
            ]
            placeholders = [
                name
                for name, value in (
                    ("SWARM_AUTH_PASSWORD_HASH", self.auth_password_hash),
                    ("SWARM_SECRET_KEY", self.secret_key),
                )
                if value is not None and "replace" in value.get_secret_value().lower()
            ]
            if placeholders:
                raise ValueError(f"{' and '.join(placeholders)} still hold example values")
            if missing:
                raise ValueError(
                    f"prod needs {' and '.join(missing)} (or SWARM_AUTH_DISABLED=true "
                    "when the API is reachable only from this machine)"
                )
        if self.secret_key is not None and 0 < len(self.secret_key.get_secret_value()) < 32:
            raise ValueError("SWARM_SECRET_KEY must be at least 32 characters")
        return self

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_password_hash and self.auth_password_hash.get_secret_value())

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()

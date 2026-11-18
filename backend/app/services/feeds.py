"""Builds the configured live price feed (one place for API, CLI and autostart)."""

from __future__ import annotations

from typing import Literal

from pydantic import SecretStr

from app.core.config import Settings
from engine.live.capital import CapitalAuthError, CapitalFeed
from engine.live.feeds import BarFeed, OandaFeed

LiveFeedKind = Literal["capital", "oanda"]


def check_live_feed(settings: Settings, kind: LiveFeedKind) -> None:
    """Raise CapitalAuthError early when the chosen feed has no credentials configured."""
    if kind == "capital" and not (
        settings.capital_api_key and settings.capital_identifier and settings.capital_password
    ):
        raise CapitalAuthError(
            "Capital.com feed needs SWARM_CAPITAL_API_KEY, SWARM_CAPITAL_IDENTIFIER "
            "and SWARM_CAPITAL_PASSWORD"
        )


def make_live_feed(settings: Settings, kind: LiveFeedKind) -> BarFeed:
    if kind == "capital":

        def secret(value: SecretStr | None) -> str | None:
            return value.get_secret_value() if value is not None else None

        return CapitalFeed(
            settings.capital_epic,
            api_key=secret(settings.capital_api_key),
            identifier=settings.capital_identifier,
            password=secret(settings.capital_password),
            demo=settings.capital_demo,
        )
    return OandaFeed(
        token=settings.oanda_token,
        account_id=settings.oanda_account_id,
        practice=settings.oanda_practice,
    )

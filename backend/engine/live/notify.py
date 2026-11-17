"""Outbound alert channels (optional). Only active champions notify."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    async def send(self, text: str, payload: dict[str, Any]) -> None: ...


class WebhookNotifier:
    """Generic JSON webhook. Discord reads ``content``, Slack reads ``text``."""

    def __init__(self, url: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.url = url
        self._transport = transport

    async def send(self, text: str, payload: dict[str, Any]) -> None:
        async with httpx.AsyncClient(transport=self._transport, timeout=10) as c:
            r = await c.post(self.url, json={"content": text, "text": text, "signal": payload})
            if r.status_code >= 400:
                logger.warning("webhook notify failed", extra={"status": r.status_code})


class TelegramNotifier:
    def __init__(
        self, bot_token: str, chat_id: str, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id
        self._transport = transport

    async def send(self, text: str, payload: dict[str, Any]) -> None:
        async with httpx.AsyncClient(transport=self._transport, timeout=10) as c:
            r = await c.post(self.url, json={"chat_id": self.chat_id, "text": text})
            if r.status_code >= 400:
                logger.warning("telegram notify failed", extra={"status": r.status_code})

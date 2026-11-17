"""Event bus connecting swarm workers, the signal engine and dashboard WebSockets.

Two interchangeable backends:

* ``InMemoryEventBus``: single process (dev, tests, the all-in-one `make dev`).
* ``RedisEventBus``: Redis pub/sub so separate worker processes can stream to the API.

Delivery is fire-and-forget (at-most-once). Anything that must survive a restart is
written to the database first. The bus only pushes live updates to the UI.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Channel(StrEnum):
    LAB = "lab"  # evolution progress: generations, new elites, run state
    SIGNALS = "signals"  # live entry/exit alerts from champions
    JOBS = "jobs"  # ingestion / validation / meta-analysis job progress
    SYSTEM = "system"  # worker heartbeats, feed status


class Event(BaseModel):
    channel: Channel
    type: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)


class EventBus(ABC):
    async def start(self) -> None:  # noqa: B027 - optional hook
        """Open connections. Optional."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections. Optional."""

    @abstractmethod
    async def publish(self, event: Event) -> None: ...

    @abstractmethod
    def subscribe(
        self, channel: Channel
    ) -> contextlib.AbstractAsyncContextManager[AsyncIterator[Event]]:
        """Usage: ``async with bus.subscribe(Channel.LAB) as events: async for e in events: ...``"""

    @abstractmethod
    async def ping(self) -> bool: ...


class InMemoryEventBus(EventBus):
    """Fan-out over bounded asyncio queues. A slow subscriber loses its oldest events
    instead of blocking publishers. A live dashboard only needs the latest state."""

    def __init__(self, queue_size: int = 1000) -> None:
        self._queue_size = queue_size
        self._subscribers: dict[Channel, set[asyncio.Queue[Event]]] = defaultdict(set)

    async def publish(self, event: Event) -> None:
        for queue in list(self._subscribers[event.channel]):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(event)

    @contextlib.asynccontextmanager
    async def subscribe(self, channel: Channel) -> AsyncIterator[AsyncIterator[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers[channel].add(queue)
        try:
            yield self._drain(queue)
        finally:
            self._subscribers[channel].discard(queue)

    @staticmethod
    async def _drain(queue: asyncio.Queue[Event]) -> AsyncIterator[Event]:
        while True:
            yield await queue.get()

    def subscriber_count(self, channel: Channel) -> int:
        return len(self._subscribers[channel])

    async def ping(self) -> bool:
        return True


class RedisEventBus(EventBus):
    def __init__(self, url: str, prefix: str = "swarm") -> None:
        from redis import asyncio as aioredis

        self._prefix = prefix
        self._redis = aioredis.from_url(url, decode_responses=True)

    def _key(self, channel: Channel) -> str:
        return f"{self._prefix}:{channel.value}"

    async def close(self) -> None:
        await self._redis.aclose()

    async def publish(self, event: Event) -> None:
        await self._redis.publish(self._key(event.channel), event.model_dump_json())

    @contextlib.asynccontextmanager
    async def subscribe(self, channel: Channel) -> AsyncIterator[AsyncIterator[Event]]:
        pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(self._key(channel))
        try:
            yield self._listen(pubsub)
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe()
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    @staticmethod
    async def _listen(pubsub: Any) -> AsyncIterator[Event]:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            try:
                yield Event.model_validate_json(message["data"])
            except ValueError:
                logger.warning("dropping malformed event", extra={"raw": message["data"]})

    async def ping(self) -> bool:
        from redis.exceptions import RedisError

        try:
            return bool(await self._redis.ping())
        except (RedisError, OSError):
            return False


def create_event_bus(redis_url: str | None, prefix: str = "swarm") -> EventBus:
    if redis_url:
        return RedisEventBus(redis_url, prefix=prefix)
    return InMemoryEventBus()

from __future__ import annotations

import asyncio

import pytest

from app.core.events import Channel, Event, EventBus, InMemoryEventBus, RedisEventBus
from tests.conftest import REDIS_URL, requires_redis


async def _collect(bus: EventBus, channel: Channel, n: int, ready: asyncio.Event) -> list[Event]:
    out: list[Event] = []
    async with bus.subscribe(channel) as events:
        ready.set()
        async for event in events:
            out.append(event)
            if len(out) == n:
                break
    return out


async def _fan_out_roundtrip(bus: EventBus) -> None:
    ready_a, ready_b = asyncio.Event(), asyncio.Event()
    a = asyncio.create_task(_collect(bus, Channel.LAB, 2, ready_a))
    b = asyncio.create_task(_collect(bus, Channel.LAB, 2, ready_b))
    await asyncio.wait_for(asyncio.gather(ready_a.wait(), ready_b.wait()), 2)

    await bus.publish(Event(channel=Channel.SIGNALS, type="ignored"))  # other channel
    await bus.publish(Event(channel=Channel.LAB, type="generation", payload={"n": 1}))
    await bus.publish(Event(channel=Channel.LAB, type="generation", payload={"n": 2}))

    got_a, got_b = await asyncio.wait_for(asyncio.gather(a, b), 2)
    for got in (got_a, got_b):
        assert [e.payload["n"] for e in got] == [1, 2]
        assert all(e.channel is Channel.LAB for e in got)


async def test_in_memory_fan_out() -> None:
    bus = InMemoryEventBus()
    await _fan_out_roundtrip(bus)
    assert bus.subscriber_count(Channel.LAB) == 0  # unsubscribed on exit


async def test_in_memory_slow_consumer_drops_oldest() -> None:
    bus = InMemoryEventBus(queue_size=3)
    async with bus.subscribe(Channel.LAB) as events:
        for i in range(5):
            await bus.publish(Event(channel=Channel.LAB, type="tick", payload={"i": i}))
        received = [(await anext(events)).payload["i"] for _ in range(3)]
    assert received == [2, 3, 4]


@requires_redis
async def test_redis_fan_out() -> None:
    assert REDIS_URL
    bus = RedisEventBus(REDIS_URL, prefix="swarm-test")
    try:
        assert await bus.ping()
        await _fan_out_roundtrip(bus)
    finally:
        await bus.close()


async def test_redis_ping_reports_unreachable() -> None:
    bus = RedisEventBus("redis://127.0.0.1:1/0")
    try:
        assert await bus.ping() is False
    finally:
        await bus.close()


def test_event_serialization_roundtrip() -> None:
    e = Event(channel=Channel.SIGNALS, type="entry", payload={"side": "long", "price": 2650.5})
    back = Event.model_validate_json(e.model_dump_json())
    assert back == e
    assert back.ts.tzinfo is not None


@pytest.mark.parametrize("bad", ["", "nope"])
def test_unknown_channel_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        Channel(bad)

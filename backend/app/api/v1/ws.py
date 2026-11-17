"""WebSocket stream of bus events: ``/api/v1/ws/{channel}`` (lab | signals | jobs | system)."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.core.events import Channel, EventBus

router = APIRouter(tags=["stream"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/{channel}")
async def stream(websocket: WebSocket, channel: str) -> None:
    try:
        chan = Channel(channel)
    except ValueError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="unknown channel")
        return

    bus: EventBus = websocket.app.state.event_bus

    # Subscribe before accepting so nothing published after the handshake is missed.
    async with bus.subscribe(chan) as events:
        await websocket.accept()

        async def pump() -> None:
            async for event in events:
                await websocket.send_text(event.model_dump_json())

        async def watch_disconnect() -> None:
            # Clients don't send anything, but reading is how we notice they left.
            with contextlib.suppress(WebSocketDisconnect):
                while True:
                    await websocket.receive_text()

        pump_task = asyncio.create_task(pump())
        watch_task = asyncio.create_task(watch_disconnect())
        done, pending = await asyncio.wait(
            {pump_task, watch_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                await task
        for task in done:
            if (exc := task.exception()) and not isinstance(exc, WebSocketDisconnect):
                logger.warning("websocket stream ended with error", exc_info=exc)

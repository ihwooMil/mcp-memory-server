"""Thread-safe event bus for broadcasting graph events to WebSocket clients."""

from __future__ import annotations

import asyncio
import collections
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LiveGraphEventBus:
    """Collects graph mutation events and fans them out to WebSocket queues.

    Bridge methods (sync) call ``emit()`` which safely enqueues to all
    registered async queues. Each WebSocket handler owns one queue.
    """

    def __init__(self, history_size: int = 100) -> None:
        self._queues: list[asyncio.Queue[dict[str, Any]]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._history: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=history_size
        )

    # -- WebSocket lifecycle ---------------------------------------------------

    def register(self) -> asyncio.Queue[dict[str, Any]]:
        """Create and return a new queue for a WebSocket client."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._queues.append(q)
        logger.info("Live client connected (%d total)", len(self._queues))
        return q

    def unregister(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a queue when the client disconnects."""
        if q in self._queues:
            self._queues.remove(q)
        logger.info("Live client disconnected (%d remaining)", len(self._queues))

    # -- Event emission (called from sync bridge code) -------------------------

    def emit(self, event: dict[str, Any]) -> None:
        """Broadcast *event* to every connected client (thread-safe)."""
        if event.get("type") != "ping":
            self._history.append(event)
        for q in list(self._queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Dropping event for slow client")

    @property
    def recent_events(self) -> list[dict[str, Any]]:
        """Return buffered event history for new clients."""
        return list(self._history)

    @property
    def has_clients(self) -> bool:
        return len(self._queues) > 0


# Module-level singleton — bridge and server both import this.
event_bus = LiveGraphEventBus()

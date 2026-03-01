"""Live graph visualization via WebSocket.

Provides a real-time, browser-based memory graph that updates
as memories are saved, searched, or deleted.

Usage:
    aimemory-live [--port 8765] [--db-path ./memory_db]
"""

from aimemory.live_graph.events import LiveGraphEventBus, event_bus

__all__ = ["LiveGraphEventBus", "event_bus"]

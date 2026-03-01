"""Fire-and-forget WebSocket push to the live graph server (cross-process).

The MCP bridge (separate process) calls ``notify_live_graph(event)`` to push
real-time events to the live graph dashboard.  A persistent WebSocket
connection to ``ws://<host>:<port>/event`` is lazily created and reused.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_LIVE_HOST = os.environ.get("AIMEMORY_LIVE_HOST", "127.0.0.1")
_LIVE_PORT = int(os.environ.get("AIMEMORY_LIVE_PORT", "8765"))

_ws = None
_lock = threading.Lock()


def _get_ws():
    """Return a reusable sync WebSocket connection, or None."""
    global _ws
    if _ws is not None:
        try:
            _ws.ping()
            return _ws
        except Exception:
            _ws = None

    try:
        from websockets.sync.client import connect

        _ws = connect(
            f"ws://{_LIVE_HOST}:{_LIVE_PORT}/event",
            open_timeout=1,
            close_timeout=1,
        )
        return _ws
    except Exception:
        return None


def notify_live_graph(event: dict[str, Any]) -> None:
    """Send an event to the live graph server. Silently ignores failures."""
    with _lock:
        ws = _get_ws()
        if ws is None:
            return
        try:
            ws.send(json.dumps(event))
        except Exception:
            global _ws
            _ws = None

"""Standalone WebSocket server for live memory graph visualization.

Uses the `websockets` library directly for simplicity.

Usage:
    python -m aimemory.live_graph.server [--port 8765] [--db-path ./memory_db]
    aimemory-live [--port 8765]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Pre-initialized bridge reference
_bridge: Any = None


def init_bridge(db_path: str | None = None) -> None:
    """Eagerly initialize the MemoryBridge (slow: loads embedding model)."""
    global _bridge
    from aimemory.mcp.bridge import MemoryBridge

    kwargs: dict[str, Any] = {}
    path = db_path or os.environ.get("AIMEMORY_DB_PATH")
    if path:
        kwargs["persist_directory"] = path
    if os.environ.get("AIMEMORY_GRAPH_RAG") == "1":
        kwargs["use_graph_rag"] = True

    logger.info("Loading MemoryBridge (embedding model)…")
    _bridge = MemoryBridge(**kwargs)
    logger.info("MemoryBridge ready.")


def _build_init_payload(bridge) -> dict[str, Any]:
    """Build the full graph state for newly connected clients."""
    from aimemory.visualize import CATEGORY_COLORS

    memories = bridge.store.get_all_memories(include_inactive=False)
    memory_ids = {m.memory_id for m in memories}

    nodes = []
    for m in memories:
        nodes.append({
            "id": m.memory_id,
            "label": m.content[:30] + ("…" if len(m.content) > 30 else ""),
            "content": m.content,
            "category": m.category,
            "color": CATEGORY_COLORS.get(m.category, "#999999"),
            "keywords": m.keywords,
            "access_count": m.access_count,
            "pinned": m.pinned,
            "created_at": m.created_at,
            "related_ids": m.related_ids,
        })

    edges = []
    seen_edges: set[tuple[str, str]] = set()
    for m in memories:
        for rid in m.related_ids:
            if rid in memory_ids:
                pair = (min(m.memory_id, rid), max(m.memory_id, rid))
                if pair not in seen_edges:
                    seen_edges.add(pair)
                    edges.append({"from": m.memory_id, "to": rid})

    from aimemory.live_graph.events import event_bus

    return {
        "type": "init",
        "nodes": nodes,
        "edges": edges,
        "recent_events": event_bus.recent_events,
    }


# ── HTTP handler (serves static HTML for non-WS requests) ────

async def _process_request(connection, request):
    """Intercept non-WebSocket HTTP requests to serve index.html."""
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None

    from websockets.datastructures import Headers
    from websockets.http11 import Response

    html = (STATIC_DIR / "index.html").read_bytes()
    headers = Headers()
    headers["Content-Type"] = "text/html; charset=utf-8"
    headers["Connection"] = "close"
    headers["Content-Length"] = str(len(html))
    return Response(200, "OK", headers, html)


# ── WebSocket handler ─────────────────────────────────────────

async def _ws_handler(websocket):
    """Handle a WebSocket connection — viewer (GET /) or event producer (GET /event)."""
    from aimemory.live_graph.events import event_bus

    path = websocket.request.path if websocket.request else "/"

    # ── Producer: MCP bridge sends events via /event ──
    if path == "/event":
        logger.info("Event producer connected (%s)", websocket.remote_address)
        try:
            async for raw in websocket:
                try:
                    event = json.loads(raw)
                    event_bus.emit(event)
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass
        finally:
            logger.info("Event producer disconnected")
        return

    # ── Viewer: browser clients ──
    queue = event_bus.register()
    logger.info("Viewer connected (%s)", websocket.remote_address)

    try:
        # Send full initial state
        if _bridge is not None:
            init = _build_init_payload(_bridge)
            await websocket.send(json.dumps(init))

        # Stream events
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=25.0)
                await websocket.send(json.dumps(event))
            except asyncio.TimeoutError:
                await websocket.send(json.dumps({"type": "ping"}))
    except Exception:
        pass
    finally:
        event_bus.unregister(queue)
        logger.info("Viewer disconnected")


# ── Server entry point ────────────────────────────────────────

async def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the WebSocket server with HTTP fallback for static files."""
    import websockets

    async with websockets.serve(
        _ws_handler,
        host,
        port,
        process_request=_process_request,
    ) as _server:
        logger.info("Live graph running at http://%s:%d", host, port)
        await asyncio.Future()  # Run forever


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AIMemory live graph server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--db-path", default=None,
        help="ChromaDB database path (default: env or ./memory_db)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Eagerly init bridge BEFORE starting server (model loading is slow)
    init_bridge(db_path=args.db_path)

    asyncio.run(run_server(host=args.host, port=args.port))


if __name__ == "__main__":
    main()

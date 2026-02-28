"""FastMCP server exposing AIMemory tools via MCP protocol.

Run with:
    python -m aimemory.mcp.server
    uv run -m aimemory.mcp.server
    uv run aimemory-mcp

All logging goes to stderr (never stdout) to preserve stdio JSON-RPC transport.
"""

from __future__ import annotations

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from aimemory.mcp.bridge import MemoryBridge

# Log to stderr only — stdout is reserved for MCP JSON-RPC communication
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "aimemory",
    instructions=(
        "You have access to a persistent memory system. "
        "Call auto_search at the start of every user turn to retrieve relevant memories. "
        "Call memory_save when the user shares important information."
    ),
)

# Lazy-initialized bridge (avoids loading ChromaDB at import time during testing)
_bridge: MemoryBridge | None = None


def _get_bridge() -> MemoryBridge:
    global _bridge
    if _bridge is None:
        _bridge = MemoryBridge()
    return _bridge


# ── Tool definitions ──────────────────────────────────────────────────


@mcp.tool()
async def memory_save(
    content: str,
    keywords: list[str] | None = None,
    category: str = "fact",
    related_ids: list[str] | None = None,
    immutable: bool = False,
    pinned: bool = False,
) -> str:
    """Save a new memory to the knowledge graph.

    Args:
        content: The memory content to save (Korean or English text).
        keywords: Optional list of keywords. Auto-extracted if not provided.
        category: Memory category. One of: fact, preference,
            experience, emotion, technical, core_principle.
        related_ids: Optional list of memory IDs to link as related.
        immutable: If True, memory cannot be updated or deleted.
        pinned: If True, memory is protected from the forgetting pipeline.
    """
    try:
        result = _get_bridge().save_memory(
            content=content,
            keywords=keywords,
            category=category,
            related_ids=related_ids,
            immutable=immutable,
            pinned=pinned,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_save failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def memory_search(
    query: str,
    top_k: int = 5,
    category: str | None = None,
) -> str:
    """Search memories by semantic similarity.

    Args:
        query: Search query text (Korean or English).
        top_k: Number of results to return (default: 5, max: 200).
        category: Optional category filter
            (fact/preference/experience/emotion/technical/core_principle).
    """
    try:
        top_k = max(1, min(top_k, 200))
        results = _get_bridge().search_memory(query=query, top_k=top_k, category=category)
        return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_search failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def auto_search(
    user_message: str,
    token_budget: int = 1024,
    top_k: int = 10,
) -> str:
    """Automatically retrieve and compose relevant memories for a user message.

    Searches the memory store, selects the most relevant memories, and
    composes them into a context string within the token budget using
    multi-resolution levels (full text, summary, entity triple).

    This tool should be called at the beginning of every conversation turn
    to inject relevant memory context.

    Args:
        user_message: The user's current message to find relevant memories for.
        token_budget: Maximum tokens for the composed context (default: 1024).
        top_k: Number of memories to retrieve (default: 10,
            use higher values like 100 for memory review).
    """
    try:
        token_budget = max(64, min(token_budget, 8192))
        top_k = max(1, min(top_k, 200))
        result = _get_bridge().auto_search(
            user_message=user_message,
            token_budget=token_budget,
            top_k=top_k,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("auto_search failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def memory_update(
    memory_id: str,
    content: str | None = None,
    keywords: list[str] | None = None,
) -> str:
    """Update an existing memory's content and/or keywords.

    Args:
        memory_id: The ID of the memory to update.
        content: New content text (optional, keeps existing if not provided).
        keywords: New keywords list (optional, keeps existing if not provided).
    """
    try:
        result = _get_bridge().update_memory(
            memory_id=memory_id,
            content=content,
            keywords=keywords,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_update failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def memory_delete(
    memory_id: str,
) -> str:
    """Delete a memory from the knowledge graph.

    Cannot delete immutable memories. Automatically cleans up graph edges.

    Args:
        memory_id: The ID of the memory to delete.
    """
    try:
        result = _get_bridge().delete_memory(memory_id=memory_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_delete failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def memory_get_related(
    memory_id: str,
    depth: int = 1,
) -> str:
    """Get memories related to a given memory via graph edges (BFS traversal).

    Args:
        memory_id: The starting memory ID.
        depth: How many hops to traverse (default: 1, max: 3).
    """
    try:
        depth = max(1, min(depth, 3))
        results = _get_bridge().get_related(memory_id=memory_id, depth=depth)
        return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_get_related failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def memory_pin(
    memory_id: str,
) -> str:
    """Pin a memory to protect it from the forgetting pipeline.

    Args:
        memory_id: The ID of the memory to pin.
    """
    try:
        result = _get_bridge().pin_memory(memory_id=memory_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_pin failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def memory_unpin(
    memory_id: str,
) -> str:
    """Remove pin protection from a memory, allowing it to be forgotten over time.

    Args:
        memory_id: The ID of the memory to unpin.
    """
    try:
        result = _get_bridge().unpin_memory(memory_id=memory_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_unpin failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def memory_stats() -> str:
    """Get statistics about the memory store: total count and category breakdown."""
    try:
        result = _get_bridge().get_stats()
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_stats failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def sleep_cycle_run() -> str:
    """Run the memory sleep cycle.

    Performs consolidation, resolution regeneration, forgetting,
    and checkpoint saving.

    This performs periodic memory maintenance. Recommended to run periodically
    (e.g., daily or after many conversations).
    """
    try:
        result = _get_bridge().run_sleep_cycle()
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("sleep_cycle_run failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def policy_status() -> str:
    """Get the current status of the memory policy (RL bandit model).

    Returns epsilon (exploration rate), recent action distribution,
    and total update count.
    """
    try:
        result = _get_bridge().get_policy_status()
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("policy_status failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


@mcp.tool()
async def policy_decide(
    user_message: str,
    turn_id: int = 0,
) -> str:
    """Ask the RL policy to decide what memory action to take for a user message.

    Returns the policy's decision (SAVE/SKIP/RETRIEVE) with reasoning.
    If SAVE: also returns the saved memory entry.
    If RETRIEVE: also returns retrieved memories.

    Args:
        user_message: The user's message to evaluate.
        turn_id: Optional turn identifier.
    """
    try:
        result = _get_bridge().policy_decide(
            user_message=user_message,
            turn_id=turn_id,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("policy_decide failed")
        return json.dumps(
            {"success": False, "error": f"Internal error: {type(exc).__name__}: {exc}"}
        )


def main() -> None:
    """Entry point for the MCP server."""
    logger.info("Starting AIMemory MCP server (stdio transport)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

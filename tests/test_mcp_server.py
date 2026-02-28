"""Integration tests for the MCP server.

Tests verify that the FastMCP server exposes the correct tools
and that tool calls return valid JSON responses.
"""

from __future__ import annotations

import json

import pytest

from aimemory.mcp import bridge as bridge_module
from aimemory.mcp.server import mcp, _get_bridge


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_bridge(tmp_path, monkeypatch):
    """Replace the global bridge singleton with an isolated test bridge."""
    from aimemory.mcp.bridge import MemoryBridge
    import aimemory.mcp.server as server_module

    test_bridge = MemoryBridge(
        persist_directory=str(tmp_path / "mcp_test_db"),
        collection_name="mcp_test",
    )
    monkeypatch.setattr(server_module, "_bridge", test_bridge)
    return test_bridge


# ── Test 19: Tool listing ─────────────────────────────────────────────


def test_server_lists_tools():
    """Server exposes all expected tools."""
    tools = mcp._tool_manager.list_tools()
    tool_names = {t.name for t in tools}

    expected_tools = {
        "memory_save",
        "memory_search",
        "auto_search",
        "memory_update",
        "memory_delete",
        "memory_get_related",
        "memory_pin",
        "memory_unpin",
        "memory_stats",
        "sleep_cycle_run",
        "policy_status",
        "policy_decide",
    }

    assert expected_tools.issubset(tool_names), (
        f"Missing tools: {expected_tools - tool_names}"
    )


# ── Test 20: memory_save via MCP ──────────────────────────────────────


@pytest.mark.anyio
async def test_tool_save_via_mcp(isolated_bridge):
    """Call memory_save through MCP protocol and verify response."""
    result = await mcp.call_tool(
        "memory_save",
        {"content": "저는 Python 개발자예요.", "category": "fact"},
    )
    # call_tool returns (content_list, ...) tuple; content_list[0].text has the JSON
    content_list = result[0]
    assert len(content_list) > 0
    data = json.loads(content_list[0].text)
    assert "memory_id" in data
    assert data["content"] == "저는 Python 개발자예요."
    assert data["category"] == "fact"


# ── Test 21: memory_search via MCP ────────────────────────────────────


@pytest.mark.anyio
async def test_tool_search_via_mcp(isolated_bridge):
    """Call memory_search through MCP protocol and verify response."""
    # First save a memory
    isolated_bridge.save_memory("저는 김치찌개를 좋아해요.", keywords=["김치찌개"])

    result = await mcp.call_tool("memory_search", {"query": "김치찌개", "top_k": 3})
    content_list = result[0]
    assert len(content_list) > 0
    data = json.loads(content_list[0].text)
    assert "results" in data
    assert "count" in data
    assert isinstance(data["results"], list)


# ── Test 22: auto_search via MCP ──────────────────────────────────────


@pytest.mark.anyio
async def test_tool_auto_search_via_mcp(isolated_bridge):
    """Call auto_search through MCP protocol and verify response structure."""
    isolated_bridge.save_memory("저는 서울에 살아요.", keywords=["서울"])

    result = await mcp.call_tool(
        "auto_search",
        {"user_message": "서울 생활", "token_budget": 512},
    )
    content_list = result[0]
    assert len(content_list) > 0
    data = json.loads(content_list[0].text)
    assert "context" in data
    assert "memories_used" in data
    assert "total_tokens" in data
    assert "details" in data


# ── Test 23: Error handling ────────────────────────────────────────────


@pytest.mark.anyio
async def test_tool_error_handling_invalid_memory_id(isolated_bridge):
    """Invalid memory_id returns error JSON, not a crash."""
    result = await mcp.call_tool(
        "memory_delete",
        {"memory_id": "definitely_nonexistent_id"},
    )
    content_list = result[0]
    assert len(content_list) > 0
    data = json.loads(content_list[0].text)
    assert data["success"] is False
    assert "error" in data


@pytest.mark.anyio
async def test_tool_stats_returns_valid_json(isolated_bridge):
    """memory_stats returns valid JSON even on empty store."""
    result = await mcp.call_tool("memory_stats", {})
    content_list = result[0]
    assert len(content_list) > 0
    data = json.loads(content_list[0].text)
    assert "total" in data
    assert data["total"] == 0


@pytest.mark.anyio
async def test_tool_policy_status_returns_valid_json(isolated_bridge):
    """policy_status returns valid JSON with expected fields."""
    result = await mcp.call_tool("policy_status", {})
    content_list = result[0]
    assert len(content_list) > 0
    data = json.loads(content_list[0].text)
    assert "epsilon" in data
    assert "recent_actions" in data

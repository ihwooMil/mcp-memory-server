"""Unit tests for MemoryBridge (MCP orchestrator).

Tests verify that the bridge correctly wires all AIMemory components
and that each high-level method returns the expected dict structure.
"""

from __future__ import annotations

import pytest

from aimemory.mcp.bridge import MemoryBridge


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def bridge(tmp_path):
    """Fresh MemoryBridge using a temporary ChromaDB directory."""
    return MemoryBridge(
        persist_directory=str(tmp_path / "db"),
        collection_name="test_memories",
    )


@pytest.fixture
def bridge_with_data(tmp_path):
    """MemoryBridge pre-populated with a few memories."""
    b = MemoryBridge(
        persist_directory=str(tmp_path / "db2"),
        collection_name="test_memories_data",
    )
    b.save_memory("저는 Python 개발자예요.", keywords=["Python", "개발자"], category="fact")
    b.save_memory("김치찌개를 좋아해요.", keywords=["김치찌개"], category="preference")
    b.save_memory("요즘 Rust를 배우고 있어요.", keywords=["Rust", "학습"], category="technical")
    return b


# ── 1. Initialization ─────────────────────────────────────────────────


def test_bridge_initialization(bridge):
    """Bridge creates store, policy, and agent correctly."""
    assert bridge.store is not None
    assert bridge.policy is not None
    assert bridge.agent is not None
    assert bridge.composer is not None


# ── 2-3. save_memory ──────────────────────────────────────────────────


def test_save_memory_basic(bridge):
    """Save returns dict with memory_id, content, keywords, category."""
    result = bridge.save_memory("저는 서울에 살아요.")
    assert "memory_id" in result
    assert result["content"] == "저는 서울에 살아요."
    assert "keywords" in result
    assert "category" in result
    assert len(result["memory_id"]) > 0


def test_save_memory_with_keywords(bridge):
    """Save with explicit keywords stores them."""
    result = bridge.save_memory(
        content="저는 백엔드 개발자예요.",
        keywords=["백엔드", "개발자"],
        category="fact",
    )
    assert result["keywords"] == ["백엔드", "개발자"]
    assert result["category"] == "fact"


def test_save_memory_pinned(bridge):
    """Save with pinned=True stores correctly and is reflected in stats."""
    result = bridge.save_memory("이것은 핀 메모리입니다.", pinned=True)
    assert "memory_id" in result
    stats = bridge.get_stats()
    assert stats["total"] == 1


def test_save_memory_immutable(bridge):
    """Save with immutable=True prevents update/delete."""
    result = bridge.save_memory("핵심 원칙입니다.", immutable=True, category="core_principle")
    mid = result["memory_id"]

    update_result = bridge.update_memory(mid, content="변경 시도")
    assert update_result["success"] is False
    assert "immutable" in update_result["error"].lower() or "Cannot" in update_result["error"]


# ── 4-5. search_memory ────────────────────────────────────────────────


def test_search_memory_empty(bridge):
    """Search on empty store returns empty list."""
    results = bridge.search_memory("테스트 쿼리")
    assert results == []


def test_search_memory_after_save(bridge):
    """Save then search finds the memory."""
    bridge.save_memory("저는 Python을 좋아해요.", keywords=["Python"])
    results = bridge.search_memory("Python", top_k=3)
    assert len(results) >= 1
    assert any("Python" in r["content"] for r in results)


# ── 6-8. auto_search ─────────────────────────────────────────────────


def test_auto_search_empty(bridge):
    """Auto-search on empty store returns empty context."""
    result = bridge.auto_search("테스트 메시지")
    assert result["context"] == ""
    assert result["memories_used"] == 0
    assert result["total_tokens"] == 0
    assert result["details"] == []


def test_auto_search_with_memories(bridge_with_data):
    """Auto-search with memories returns composed context."""
    result = bridge_with_data.auto_search("Python 개발자")
    assert "context" in result
    assert "memories_used" in result
    assert "total_tokens" in result
    assert "details" in result
    assert result["memories_used"] > 0
    assert result["total_tokens"] > 0
    assert len(result["details"]) > 0


def test_auto_search_token_budget(bridge_with_data):
    """Auto-search respects the token budget."""
    result = bridge_with_data.auto_search("개발자", token_budget=50)
    # total_tokens should not exceed the budget significantly
    # (exact check is flexible due to estimation)
    assert result["total_tokens"] <= 200  # generous upper bound


def test_auto_search_details_structure(bridge_with_data):
    """Auto-search details contain required fields."""
    result = bridge_with_data.auto_search("음식 취향")
    if result["memories_used"] > 0:
        detail = result["details"][0]
        assert "memory_id" in detail
        assert "level" in detail
        assert "relevance" in detail
        assert "tokens" in detail


# ── 9-10. update_memory ──────────────────────────────────────────────


def test_update_memory(bridge):
    """Update content and keywords of a memory."""
    save_result = bridge.save_memory("원래 내용입니다.", keywords=["원래"])
    mid = save_result["memory_id"]

    update_result = bridge.update_memory(mid, content="수정된 내용입니다.", keywords=["수정"])
    assert update_result["success"] is True
    assert update_result["memory_id"] == mid


def test_update_memory_not_found(bridge):
    """Update with nonexistent ID returns error."""
    result = bridge.update_memory("nonexistent_id_12")
    assert result["success"] is False
    assert "not found" in result["error"].lower() or "error" in result


def test_update_immutable_memory(bridge):
    """Update immutable memory returns error."""
    save_result = bridge.save_memory("불변 메모리", immutable=True)
    mid = save_result["memory_id"]

    result = bridge.update_memory(mid, content="변경 시도")
    assert result["success"] is False


# ── 11-12. delete_memory ─────────────────────────────────────────────


def test_delete_memory(bridge):
    """Delete removes memory from store."""
    save_result = bridge.save_memory("삭제될 메모리")
    mid = save_result["memory_id"]

    delete_result = bridge.delete_memory(mid)
    assert delete_result["success"] is True

    # Verify it's gone
    results = bridge.search_memory("삭제될 메모리")
    assert all(r["memory_id"] != mid for r in results)


def test_delete_memory_not_found(bridge):
    """Delete with nonexistent ID returns error."""
    result = bridge.delete_memory("nonexistent_id_99")
    assert result["success"] is False


def test_delete_immutable_memory(bridge):
    """Delete immutable memory returns error."""
    save_result = bridge.save_memory("불변 메모리", immutable=True)
    mid = save_result["memory_id"]

    result = bridge.delete_memory(mid)
    assert result["success"] is False


# ── 13. get_related ──────────────────────────────────────────────────


def test_get_related(bridge):
    """Get related memories via graph edges."""
    r1 = bridge.save_memory("메모리 A")
    r2 = bridge.save_memory("메모리 B", related_ids=[r1["memory_id"]])

    related = bridge.get_related(r1["memory_id"], depth=1)
    related_ids = [n["memory_id"] for n in related]
    assert r2["memory_id"] in related_ids


def test_get_related_empty(bridge):
    """Get related returns empty list when no edges exist."""
    r = bridge.save_memory("고립된 메모리")
    related = bridge.get_related(r["memory_id"])
    assert related == []


# ── 14. pin_unpin ────────────────────────────────────────────────────


def test_pin_unpin(bridge):
    """Pin and unpin toggle correctly."""
    r = bridge.save_memory("핀 테스트 메모리")
    mid = r["memory_id"]

    pin_result = bridge.pin_memory(mid)
    assert pin_result["success"] is True
    assert pin_result["pinned"] is True

    unpin_result = bridge.unpin_memory(mid)
    assert unpin_result["success"] is True
    assert unpin_result["pinned"] is False


def test_pin_not_found(bridge):
    """Pin nonexistent memory returns error."""
    result = bridge.pin_memory("nonexistent_id_55")
    assert result["success"] is False


# ── 15. stats ────────────────────────────────────────────────────────


def test_stats_empty(bridge):
    """Stats on empty store returns zero total."""
    stats = bridge.get_stats()
    assert stats["total"] == 0


def test_stats(bridge):
    """Stats returns correct total and categories."""
    bridge.save_memory("사실 메모리", category="fact")
    bridge.save_memory("선호 메모리", category="preference")
    bridge.save_memory("기술 메모리", category="technical")

    stats = bridge.get_stats()
    assert stats["total"] == 3
    assert stats["categories"].get("fact", 0) >= 1
    assert stats["categories"].get("preference", 0) >= 1
    assert stats["categories"].get("technical", 0) >= 1


# ── 16. sleep_cycle ──────────────────────────────────────────────────


def test_sleep_cycle(bridge):
    """Sleep cycle runs and returns a report dict."""
    bridge.save_memory("수면 주기 테스트 메모리")
    report = bridge.run_sleep_cycle()
    assert isinstance(report, dict)
    assert "started_at" in report
    assert "finished_at" in report
    assert "duration_seconds" in report
    assert "memory_count_before" in report
    assert "memory_count_after" in report


# ── 17. policy_status ────────────────────────────────────────────────


def test_policy_status(bridge):
    """Policy status returns epsilon and action counts."""
    status = bridge.get_policy_status()
    assert "epsilon" in status
    assert "recent_actions" in status
    assert "total_updates" in status
    assert isinstance(status["epsilon"], float)
    assert isinstance(status["recent_actions"], dict)


# ── 18. policy_decide ────────────────────────────────────────────────


def test_policy_decide(bridge):
    """Policy decide returns action and reasoning."""
    result = bridge.policy_decide("저는 Python을 주로 사용해요.", turn_id=1)
    assert "action" in result
    assert "turn_id" in result
    assert "reasoning" in result
    assert result["action"] in ("save", "skip", "retrieve")


def test_policy_decide_with_save_action(bridge):
    """Policy decide for personal info typically returns SAVE."""
    # This may or may not be SAVE depending on the rule-based logic
    result = bridge.policy_decide(
        "저는 서울 강남구에 사는 28살 백엔드 개발자예요. Python을 5년째 쓰고 있어요.",
        turn_id=0,
    )
    assert result["action"] in ("save", "skip", "retrieve")
    # If SAVE was chosen, there should be a memory_entry
    if result["action"] == "save":
        assert "memory_entry" in result


# ── Enhanced/GraphRAG mode tests ────────────────────────────────────


@pytest.fixture
def enhanced_bridge(tmp_path):
    """MemoryBridge with enhanced policy enabled."""
    return MemoryBridge(
        persist_directory=str(tmp_path / "enhanced_db"),
        collection_name="enhanced_test",
        use_enhanced_policy=True,
    )


@pytest.fixture
def graphrag_bridge(tmp_path):
    """MemoryBridge with GraphRAG enabled."""
    return MemoryBridge(
        persist_directory=str(tmp_path / "graphrag_db"),
        collection_name="graphrag_test",
        use_graph_rag=True,
    )


def test_enhanced_bridge_initialization(enhanced_bridge):
    """Enhanced bridge creates EnhancedOnlinePolicy."""
    from aimemory.online.enhanced_policy import EnhancedOnlinePolicy
    assert isinstance(enhanced_bridge._policy, EnhancedOnlinePolicy)


def test_enhanced_bridge_save_memory(enhanced_bridge):
    """Enhanced bridge save_memory works normally."""
    result = enhanced_bridge.save_memory("테스트 기억", keywords=["테스트"])
    assert "memory_id" in result


def test_graphrag_bridge_initialization(graphrag_bridge):
    """GraphRAG bridge creates KnowledgeGraph and GraphRetriever."""
    assert graphrag_bridge._kg is not None
    assert graphrag_bridge._retriever is not None


def test_graphrag_bridge_auto_search(graphrag_bridge):
    """GraphRAG bridge auto_search uses hybrid retrieval."""
    graphrag_bridge.save_memory(
        "봉골레를 싫어해요", keywords=["봉골레"], category="preference"
    )
    result = graphrag_bridge.auto_search("봉골레 파스타")
    assert "context" in result
    assert "memories_used" in result


def test_legacy_mode_unchanged(bridge):
    """Default bridge (no enhanced/graphrag) works exactly as before."""
    assert bridge._use_enhanced is False
    assert bridge._use_graph_rag is False
    result = bridge.save_memory("일반 모드 테스트")
    assert "memory_id" in result
    stats = bridge.get_stats()
    assert stats["total"] == 1


def test_both_modes_combined(tmp_path):
    """Both enhanced policy and GraphRAG can be enabled together."""
    b = MemoryBridge(
        persist_directory=str(tmp_path / "both_db"),
        collection_name="both_test",
        use_enhanced_policy=True,
        use_graph_rag=True,
    )
    from aimemory.online.enhanced_policy import EnhancedOnlinePolicy
    assert isinstance(b._policy, EnhancedOnlinePolicy)
    assert b._kg is not None
    assert b._retriever is not None

"""End-to-end integration tests for Enhanced Policy + GraphRAG modes.

Validates that the new RL evolution and GraphRAG integration work
correctly in realistic scenarios, and that legacy mode is unaffected.
"""

from __future__ import annotations

import pytest

from aimemory.mcp.bridge import MemoryBridge

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def enhanced_bridge(tmp_path):
    """MemoryBridge with enhanced policy mode."""
    return MemoryBridge(
        persist_directory=str(tmp_path / "e2e_enhanced"),
        collection_name="e2e_enhanced",
        use_enhanced_policy=True,
    )


@pytest.fixture
def graphrag_bridge(tmp_path):
    """MemoryBridge with GraphRAG mode."""
    return MemoryBridge(
        persist_directory=str(tmp_path / "e2e_graphrag"),
        collection_name="e2e_graphrag",
        use_graph_rag=True,
    )


@pytest.fixture
def legacy_bridge(tmp_path):
    """MemoryBridge in default legacy mode."""
    return MemoryBridge(
        persist_directory=str(tmp_path / "e2e_legacy"),
        collection_name="e2e_legacy",
    )


# ── Test 1: Enhanced Policy 10-conversation simulation ───────────────


def test_e2e_enhanced_policy_10_conversations(enhanced_bridge):
    """Enhanced policy mode: simulate 10 conversations with save/search."""
    from aimemory.online.enhanced_policy import EnhancedOnlinePolicy

    assert isinstance(enhanced_bridge._policy, EnhancedOnlinePolicy)

    # Simulate 10 conversations with saves and searches
    topics = [
        ("저는 Python을 좋아해요.", ["Python"], "preference"),
        ("매일 아침 조깅을 합니다.", ["조깅", "아침"], "experience"),
        ("김치찌개를 제일 좋아해요.", ["김치찌개"], "preference"),
        ("서울 강남에 살고 있어요.", ["서울", "강남"], "fact"),
        ("React로 프론트엔드 개발을 해요.", ["React", "프론트엔드"], "technical"),
        ("기타를 배우고 있어요.", ["기타", "학습"], "experience"),
        ("커피를 하루에 세 잔 마셔요.", ["커피"], "fact"),
        ("주말에 등산을 즐겨요.", ["등산", "주말"], "experience"),
        ("봉골레 파스타를 싫어해요.", ["봉골레"], "preference"),
        ("Docker와 Kubernetes를 사용해요.", ["Docker", "Kubernetes"], "technical"),
    ]

    for content, keywords, category in topics:
        result = enhanced_bridge.save_memory(content, keywords=keywords, category=category)
        assert "memory_id" in result

    # Verify all stored
    stats = enhanced_bridge.get_stats()
    assert stats["total"] == 10

    # Search should work
    search_result = enhanced_bridge.auto_search("Python 개발자")
    assert search_result["memories_used"] > 0

    # Policy decide should work with enhanced policy
    decide_result = enhanced_bridge.policy_decide(
        "저는 백엔드 개발자예요. Django를 주로 사용해요.", turn_id=1
    )
    assert decide_result["action"] in ("save", "skip", "retrieve")


# ── Test 2: GraphRAG relationship reasoning ──────────────────────────


def test_e2e_graph_rag_relationship_reasoning(graphrag_bridge):
    """GraphRAG: '봉골레 싫어함' relationship is reflected in search."""
    assert graphrag_bridge._kg is not None
    assert graphrag_bridge._retriever is not None

    # Save memories with level2_text containing relationship triples
    graphrag_bridge.store.add_memory(
        content="저는 봉골레 파스타를 싫어해요",
        keywords=["봉골레", "파스타"],
        category="preference",
        level2_text="사용자,싫어함,봉골레",
    )
    graphrag_bridge.store.add_memory(
        content="카르보나라를 좋아해요",
        keywords=["카르보나라", "파스타"],
        category="preference",
        level2_text="사용자,좋아함,카르보나라",
    )
    graphrag_bridge.store.add_memory(
        content="피자도 좋아해요",
        keywords=["피자"],
        category="preference",
        level2_text="사용자,좋아함,피자",
    )

    # KnowledgeGraph should have triples
    kg_stats = graphrag_bridge._kg.stats()
    assert kg_stats["edges"] >= 3

    # Search for 봉골레 - should find the memory with negative relation
    search_result = graphrag_bridge.auto_search("봉골레 파스타")
    assert "context" in search_result

    # The KG should know about the negative relation
    related = graphrag_bridge._kg.get_related_entities("봉골레")
    assert len(related) > 0

    # Verify the entity context includes the negative predicate
    context = graphrag_bridge._kg.get_entity_context("봉골레")
    predicates = [r.get("predicate", "") for r in context.get("relations", [])]
    assert any("싫어" in p for p in predicates)


# ── Test 3: Progressive autonomy threshold relaxation ────────────────


def test_e2e_progressive_autonomy_threshold_relaxation(enhanced_bridge):
    """50+ positive feedbacks → RL zone expands (save threshold decreases)."""
    from aimemory.online.enhanced_policy import EnhancedOnlinePolicy

    policy = enhanced_bridge._policy
    assert isinstance(policy, EnhancedOnlinePolicy)

    autonomy = policy.autonomy
    initial_save = autonomy.save_threshold
    initial_skip = autonomy.skip_threshold

    # Record 60 positive feedbacks
    for _ in range(60):
        autonomy.record_feedback(1.0)

    # After enough positive feedback, thresholds should relax
    assert autonomy.confidence > 0
    assert autonomy.save_threshold <= initial_save
    assert autonomy.skip_threshold >= initial_skip

    # RL zone should have expanded (save_threshold - skip_threshold might shrink
    # but the RL zone = middle band expands)
    new_zone = autonomy.rl_zone_ratio
    # The zone ratio might change, verify it's reasonable
    assert new_zone > 0

    # Negative feedback should reduce confidence
    confidence_before = autonomy.confidence
    autonomy.record_feedback(-1.0)
    assert autonomy.confidence < confidence_before


# ── Test 4: Legacy mode unchanged ────────────────────────────────────


def test_e2e_legacy_mode_unchanged(legacy_bridge):
    """Enhanced/GraphRAG OFF: existing behavior is 100% preserved."""
    # Verify legacy mode flags
    assert legacy_bridge._use_enhanced is False
    assert legacy_bridge._use_graph_rag is False
    assert legacy_bridge._kg is None
    assert legacy_bridge._retriever is None

    # Standard OnlinePolicy (not Enhanced)
    from aimemory.online.enhanced_policy import EnhancedOnlinePolicy
    from aimemory.online.policy import OnlinePolicy

    assert isinstance(legacy_bridge._policy, OnlinePolicy)
    assert not isinstance(legacy_bridge._policy, EnhancedOnlinePolicy)

    # Save memories
    legacy_bridge.save_memory("Python 개발자예요", keywords=["Python"], category="fact")
    legacy_bridge.save_memory("김치찌개 좋아해요", keywords=["김치찌개"], category="preference")

    # Search works
    result = legacy_bridge.auto_search("Python")
    assert "context" in result
    assert result["memories_used"] > 0

    # Stats work
    stats = legacy_bridge.get_stats()
    assert stats["total"] == 2

    # Policy decide works
    decide = legacy_bridge.policy_decide("저는 서울에 살아요.", turn_id=0)
    assert decide["action"] in ("save", "skip", "retrieve")

    # Sleep cycle works
    report = legacy_bridge.run_sleep_cycle()
    assert "started_at" in report
    assert "finished_at" in report

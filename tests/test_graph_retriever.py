"""Tests for GraphRetriever (hybrid vector + graph retrieval)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aimemory.memory.graph_retriever import GraphRetriever
from aimemory.memory.graph_store import MemoryNode
from aimemory.memory.knowledge_graph import KnowledgeGraph


def _make_node(
    memory_id: str,
    content: str,
    similarity_score: float = 0.8,
    keywords: list[str] | None = None,
    level2_text: str = "",
) -> MemoryNode:
    return MemoryNode(
        memory_id=memory_id,
        content=content,
        keywords=keywords or [],
        similarity_score=similarity_score,
        level2_text=level2_text,
    )


@pytest.fixture
def kg() -> KnowledgeGraph:
    return KnowledgeGraph()


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.get_all_memories.return_value = []
    return store


# ── Vector-only retrieval ──────────────────────────────────────


def test_vector_only_retrieval(mock_store: MagicMock, kg: KnowledgeGraph) -> None:
    """When KG is empty, results come from vector search only."""
    nodes = [
        _make_node("mem1", "파이썬 좋아합니다", similarity_score=0.9),
        _make_node("mem2", "자바 사용합니다", similarity_score=0.7),
    ]
    mock_store.search.return_value = nodes

    retriever = GraphRetriever(mock_store, kg)
    results = retriever.retrieve("파이썬", top_k=5, final_k=2)

    assert len(results) == 2
    # mem1 should rank higher (higher vector score, no graph signal)
    assert results[0].memory_id == "mem1"
    assert results[1].memory_id == "mem2"


# ── Graph-only retrieval ───────────────────────────────────────


def test_graph_only_retrieval(mock_store: MagicMock, kg: KnowledgeGraph) -> None:
    """Vector returns nothing, but graph has relevant memory IDs → returns results."""
    node = _make_node("mem_graph", "봉골레 파스타 싫어함", similarity_score=0.5)
    mock_store.search.return_value = []  # No vector results
    mock_store.get_all_memories.return_value = [node]

    # Add triple to KG
    kg.add_triple("사용자", "싫어함", "봉골레", "mem_graph")

    retriever = GraphRetriever(mock_store, kg)
    results = retriever.retrieve("봉골레", top_k=5, final_k=5)

    assert len(results) == 1
    assert results[0].memory_id == "mem_graph"


# ── Hybrid score fusion ────────────────────────────────────────


def test_hybrid_score_fusion(mock_store: MagicMock, kg: KnowledgeGraph) -> None:
    """Both vector and graph contribute to final scores."""
    node_a = _make_node("mem_a", "파이썬 좋아함", similarity_score=0.5)
    node_b = _make_node("mem_b", "자바 사용", similarity_score=0.9)
    mock_store.search.return_value = [node_a, node_b]

    # KG connects 파이썬 to mem_a → graph_score boost for mem_a
    kg.add_triple("사용자", "선호함", "파이썬", "mem_a")

    retriever = GraphRetriever(mock_store, kg, vector_weight=0.6, graph_weight=0.4)
    results = retriever.retrieve("파이썬", top_k=5, final_k=2)

    assert len(results) == 2
    # mem_a: vector=0.5*0.6 + graph≥0.5*0.4 = 0.30 + 0.20 = 0.50
    # mem_b: vector=0.9*0.6 + graph=0.0     = 0.54
    # mem_b should still win due to high vector score
    scores = {r.memory_id: r.similarity_score for r in results}
    assert scores["mem_b"] > scores["mem_a"]


# ── Negative relation handling ─────────────────────────────────


def test_negative_relation_handling(mock_store: MagicMock, kg: KnowledgeGraph) -> None:
    """'싫어함' predicate still returns memory with graph score."""
    node = _make_node("mem_neg", "봉골레 파스타 싫어함", similarity_score=0.4)
    mock_store.search.return_value = [node]

    kg.add_triple("사용자", "싫어함", "봉골레", "mem_neg")

    retriever = GraphRetriever(mock_store, kg)
    results = retriever.retrieve("봉골레", top_k=5, final_k=5)

    assert len(results) >= 1
    mem_ids = [r.memory_id for r in results]
    assert "mem_neg" in mem_ids

    # Verify the node got a graph score (final score > pure vector score)
    result_node = next(r for r in results if r.memory_id == "mem_neg")
    # vector contribution: 0.4*0.6 = 0.24; graph contribution > 0
    assert result_node.similarity_score > 0.24


# ── Empty KG fallback ─────────────────────────────────────────


def test_empty_kg_fallback(mock_store: MagicMock, kg: KnowledgeGraph) -> None:
    """Empty KG → behaves like pure vector search."""
    nodes = [
        _make_node("mem1", "리액트 컴포넌트", similarity_score=0.85),
        _make_node("mem2", "타입스크립트 타입", similarity_score=0.65),
    ]
    mock_store.search.return_value = nodes

    retriever = GraphRetriever(mock_store, kg)
    results = retriever.retrieve("리액트", top_k=5, final_k=2)

    assert len(results) == 2
    # With empty KG, order is determined purely by vector score
    assert results[0].memory_id == "mem1"
    # Final score == vector_score * vector_weight (graph_score = 0)
    assert abs(results[0].similarity_score - 0.85 * 0.6) < 1e-6


# ── Entity extraction ──────────────────────────────────────────


def test_extract_entities_korean(mock_store: MagicMock, kg: KnowledgeGraph) -> None:
    """Korean text produces entity list of 2+ char Korean tokens."""
    retriever = GraphRetriever(mock_store, kg)
    entities = retriever.extract_entities("사용자가 봉골레를 싫어합니다")

    # extractor captures full eojeol tokens (noun + particle), not bare nouns
    assert len(entities) > 0
    # All extracted entities should be 2+ char Korean sequences
    for e in entities:
        if all("\uac00" <= c <= "\ud7a3" for c in e):
            assert len(e) >= 2
    # The tokens containing "사용자" and "봉골레" should be present
    assert any("사용자" in e for e in entities)
    assert any("봉골레" in e for e in entities)


def test_extract_entities_tech(mock_store: MagicMock, kg: KnowledgeGraph) -> None:
    """Tech terms extracted from English text."""
    retriever = GraphRetriever(mock_store, kg)
    entities = retriever.extract_entities("I use Python and PyTorch for ML projects")

    assert any(e.lower() == "python" for e in entities)
    assert any(e.lower() == "pytorch" for e in entities)


# ── Deduplication ──────────────────────────────────────────────


def test_deduplicated_results(mock_store: MagicMock, kg: KnowledgeGraph) -> None:
    """No duplicate memory IDs in results."""
    node = _make_node("mem_dup", "중복 테스트", similarity_score=0.75)
    mock_store.search.return_value = [node]
    # Graph also returns same memory id
    mock_store.get_all_memories.return_value = [node]
    kg.add_triple("사용자", "테스트함", "중복", "mem_dup")

    retriever = GraphRetriever(mock_store, kg)
    results = retriever.retrieve("중복 테스트", top_k=5, final_k=10)

    memory_ids = [r.memory_id for r in results]
    assert len(memory_ids) == len(set(memory_ids)), "Duplicate memory IDs found in results"

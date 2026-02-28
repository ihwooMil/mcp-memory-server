"""Tests for KnowledgeGraph core component."""

from __future__ import annotations

from unittest.mock import MagicMock

from aimemory.memory.graph_store import MemoryNode
from aimemory.memory.knowledge_graph import KnowledgeGraph


def make_node(memory_id: str, level2_text: str, content: str = "test") -> MemoryNode:
    return MemoryNode(memory_id=memory_id, content=content, level2_text=level2_text)


# ── Basic triple operations ───────────────────────────────────────────────────


def test_add_triple_basic():
    kg = KnowledgeGraph()
    kg.add_triple("사용자", "싫어함", "봉골레", memory_id="mem1")

    triples = kg.get_related_entities("사용자", depth=1)
    assert len(triples) == 1
    assert triples[0] == ("사용자", "싫어함", "봉골레")


def test_add_from_memory_parses_level2_text():
    kg = KnowledgeGraph()
    node = make_node("mem1", level2_text="사용자,좋아함,커피")
    count = kg.add_from_memory(node)

    assert count == 1
    triples = kg.get_related_entities("사용자", depth=1)
    assert ("사용자", "좋아함", "커피") in triples


def test_add_from_memory_multiple_triples_semicolon():
    kg = KnowledgeGraph()
    node = make_node("mem1", level2_text="A,loves,B;C,hates,D")
    count = kg.add_from_memory(node)

    assert count == 2
    assert ("A", "loves", "B") in kg.get_related_entities("A", depth=1)
    assert ("C", "hates", "D") in kg.get_related_entities("C", depth=1)


def test_add_from_memory_multiple_triples_newline():
    kg = KnowledgeGraph()
    node = make_node("mem1", level2_text="X,likes,Y\nZ,knows,W")
    count = kg.add_from_memory(node)

    assert count == 2
    assert ("X", "likes", "Y") in kg.get_related_entities("X", depth=1)
    assert ("Z", "knows", "W") in kg.get_related_entities("Z", depth=1)


def test_add_from_memory_empty_level2():
    kg = KnowledgeGraph()
    node = make_node("mem1", level2_text="")
    count = kg.add_from_memory(node)

    assert count == 0
    assert kg.stats()["nodes"] == 0


# ── Depth traversal ───────────────────────────────────────────────────────────


def test_get_related_entities_depth():
    kg = KnowledgeGraph()
    # Chain: A -> B -> C
    kg.add_triple("A", "r1", "B", memory_id="m1")
    kg.add_triple("B", "r2", "C", memory_id="m2")

    depth1 = kg.get_related_entities("A", depth=1)
    depth2 = kg.get_related_entities("A", depth=2)

    # At depth=1, only A->B edge visible from A
    {t[0] for t in depth1}
    objects_depth1 = {t[2] for t in depth1}
    assert "C" not in objects_depth1

    # At depth=2, B->C should also be reachable
    objects_depth2 = {t[2] for t in depth2}
    assert "C" in objects_depth2


# ── Memory ID lookups ─────────────────────────────────────────────────────────


def test_get_memory_ids_for_entity():
    kg = KnowledgeGraph()
    kg.add_triple("사용자", "좋아함", "커피", memory_id="mem_a")
    kg.add_triple("사용자", "싫어함", "봉골레", memory_id="mem_b")

    ids = kg.get_memory_ids_for_entity("사용자")
    assert ids == {"mem_a", "mem_b"}


def test_get_memory_ids_for_entity_as_object():
    kg = KnowledgeGraph()
    kg.add_triple("A", "likes", "커피", memory_id="mem_x")

    ids = kg.get_memory_ids_for_entity("커피")
    assert "mem_x" in ids


def test_get_memory_ids_for_unknown_entity():
    kg = KnowledgeGraph()
    ids = kg.get_memory_ids_for_entity("ghost")
    assert ids == set()


# ── Path queries ──────────────────────────────────────────────────────────────


def test_query_path_exists():
    kg = KnowledgeGraph()
    kg.add_triple("Alice", "knows", "Bob", memory_id="m1")
    kg.add_triple("Bob", "works_with", "Carol", memory_id="m2")

    path = kg.query_path("Alice", "Carol")
    assert path is not None
    assert len(path) == 2
    # First hop Alice->Bob
    assert path[0][0] == "Alice"
    assert path[0][2] == "Bob"


def test_query_path_none():
    kg = KnowledgeGraph()
    kg.add_triple("Alice", "knows", "Bob", memory_id="m1")
    kg.add_triple("Carol", "likes", "Dave", memory_id="m2")

    path = kg.query_path("Alice", "Carol")
    assert path is None


def test_query_path_missing_node():
    kg = KnowledgeGraph()
    kg.add_triple("A", "r", "B", memory_id="m1")

    assert kg.query_path("A", "Z") is None
    assert kg.query_path("Z", "A") is None


# ── Entity context ────────────────────────────────────────────────────────────


def test_get_entity_context():
    kg = KnowledgeGraph()
    kg.add_triple("사용자", "좋아함", "커피", memory_id="mem1")
    kg.add_triple("친구", "선물한", "사용자", memory_id="mem2")

    ctx = kg.get_entity_context("사용자")
    assert ctx["entity"] == "사용자"
    assert ctx["neighbor_count"] == 2  # 커피 and 친구

    directions = {r["direction"] for r in ctx["relations"]}
    assert "outgoing" in directions
    assert "incoming" in directions


def test_get_entity_context_unknown():
    kg = KnowledgeGraph()
    ctx = kg.get_entity_context("nobody")
    assert ctx == {"entity": "nobody", "relations": [], "neighbor_count": 0}


# ── Remove triples ────────────────────────────────────────────────────────────


def test_remove_triples_by_memory():
    kg = KnowledgeGraph()
    kg.add_triple("사용자", "좋아함", "커피", memory_id="mem1")
    kg.add_triple("사용자", "싫어함", "봉골레", memory_id="mem2")

    removed = kg.remove_triples_by_memory("mem1")
    assert removed == 1

    # 커피 edge gone, 봉골레 edge still present
    triples = kg.get_related_entities("사용자", depth=1)
    predicates = {t[1] for t in triples}
    assert "좋아함" not in predicates
    assert "싫어함" in predicates


def test_remove_triples_cleans_isolated_nodes():
    kg = KnowledgeGraph()
    kg.add_triple("A", "r", "B", memory_id="only_mem")
    kg.remove_triples_by_memory("only_mem")

    s = kg.stats()
    assert s["nodes"] == 0
    assert s["edges"] == 0


# ── Stats ─────────────────────────────────────────────────────────────────────


def test_stats():
    kg = KnowledgeGraph()
    assert kg.stats() == {"nodes": 0, "edges": 0, "components": 0}

    kg.add_triple("A", "r1", "B", memory_id="m1")
    kg.add_triple("C", "r2", "D", memory_id="m2")

    s = kg.stats()
    assert s["nodes"] == 4
    assert s["edges"] == 2
    assert s["components"] == 2


# ── Rebuild from store ────────────────────────────────────────────────────────


def test_rebuild_from_store():
    node1 = make_node("mem1", level2_text="A,loves,B")
    node2 = make_node("mem2", level2_text="C,knows,D;E,hates,F")
    node3 = make_node("mem3", level2_text="")  # no triples

    mock_store = MagicMock()
    mock_store.get_all_memories.return_value = [node1, node2, node3]

    kg = KnowledgeGraph()
    # Add some prior state to ensure rebuild clears it
    kg.add_triple("old", "r", "data", memory_id="stale")

    total = kg.rebuild_from_store(mock_store)

    assert total == 3  # 1 + 2 + 0
    mock_store.get_all_memories.assert_called_once_with(include_inactive=True)

    s = kg.stats()
    assert s["edges"] == 3
    # Stale data should be gone
    assert kg.get_related_entities("old", depth=1) == []

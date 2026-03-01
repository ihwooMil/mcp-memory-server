"""Tests for the memory graph visualization module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aimemory.visualize import _compute_node_size, _truncate_content

# ── TestNodeSizing ────────────────────────────────────────────────────


class TestNodeSizing:
    def test_min_size_at_zero_access(self):
        assert _compute_node_size(0, 100) == 15

    def test_max_size_at_max_access(self):
        assert _compute_node_size(100, 100) == 50

    def test_mid_range(self):
        size = _compute_node_size(50, 100)
        assert 15 < size < 50

    def test_zero_max_access(self):
        assert _compute_node_size(0, 0) == 15

    def test_zero_max_with_nonzero_count(self):
        assert _compute_node_size(5, 0) == 15


# ── TestTruncateContent ──────────────────────────────────────────────


class TestTruncateContent:
    def test_short_content_unchanged(self):
        assert _truncate_content("hello") == "hello"

    def test_long_content_truncated(self):
        result = _truncate_content("a" * 50, max_chars=30)
        assert len(result) == 31  # 30 chars + ellipsis
        assert result.endswith("…")

    def test_exact_length_unchanged(self):
        text = "a" * 30
        assert _truncate_content(text, max_chars=30) == text

    def test_korean_content(self):
        text = "한국어 테스트 문장입니다 이것은 긴 문장입니다"
        result = _truncate_content(text, max_chars=10)
        assert result.endswith("…")
        # First 10 characters + ellipsis
        assert result == text[:10] + "…"


# ── Tests requiring pyvis ────────────────────────────────────────────

pyvis_installed = True
try:
    import pyvis  # noqa: F401
except ImportError:
    pyvis_installed = False

skip_no_pyvis = pytest.mark.skipif(not pyvis_installed, reason="pyvis not installed")


@skip_no_pyvis
class TestBuildGraph:
    def test_empty_graph(self, tmp_path):
        from aimemory.visualize import build_graph

        store = MagicMock()
        store.get_all_memories.return_value = []

        out = build_graph(store, kg=None, output_path=tmp_path / "empty.html")
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "No memories" in content

    def test_single_node(self, tmp_path):
        from aimemory.memory.graph_store import MemoryNode
        from aimemory.visualize import build_graph

        node = MemoryNode(
            memory_id="mem1",
            content="테스트 메모리",
            keywords=["테스트"],
            category="fact",
            access_count=5,
            created_at="2026-01-01T00:00:00",
            active=True,
        )
        store = MagicMock()
        store.get_all_memories.return_value = [node]

        out = build_graph(store, kg=None, output_path=tmp_path / "single.html")
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "mem1" in content

    def test_with_edges(self, tmp_path):
        from aimemory.memory.graph_store import MemoryNode
        from aimemory.visualize import build_graph

        node1 = MemoryNode(
            memory_id="mem1",
            content="First memory",
            category="fact",
            related_ids=["mem2"],
            access_count=3,
            active=True,
        )
        node2 = MemoryNode(
            memory_id="mem2",
            content="Second memory",
            category="preference",
            access_count=1,
            active=True,
        )
        store = MagicMock()
        store.get_all_memories.return_value = [node1, node2]

        out = build_graph(store, kg=None, output_path=tmp_path / "edges.html")
        assert Path(out).exists()

    def test_with_knowledge_graph(self, tmp_path):
        import networkx as nx

        from aimemory.memory.graph_store import MemoryNode
        from aimemory.visualize import build_graph

        node = MemoryNode(
            memory_id="mem1",
            content="Python developer",
            category="technical",
            access_count=2,
            active=True,
        )
        store = MagicMock()
        store.get_all_memories.return_value = [node]

        kg = MagicMock()
        graph = nx.DiGraph()
        graph.add_edge("Python", "programming", predicate="is_a", memory_id="mem1")
        kg._graph = graph
        kg.stats.return_value = {"nodes": 2, "edges": 1, "components": 1}

        out = build_graph(store, kg=kg, output_path=tmp_path / "kg.html")
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "Python" in content

    def test_dangling_related_ids_skipped(self, tmp_path):
        """Edges pointing to non-existent nodes should be skipped."""
        from aimemory.memory.graph_store import MemoryNode
        from aimemory.visualize import build_graph

        node = MemoryNode(
            memory_id="mem1",
            content="Test",
            category="fact",
            related_ids=["nonexistent"],
            access_count=1,
            active=True,
        )
        store = MagicMock()
        store.get_all_memories.return_value = [node]

        # Should not raise
        out = build_graph(store, kg=None, output_path=tmp_path / "dangling.html")
        assert Path(out).exists()

    def test_pinned_memory_has_gold_border(self, tmp_path):
        from aimemory.memory.graph_store import MemoryNode
        from aimemory.visualize import build_graph

        node = MemoryNode(
            memory_id="mem_pinned",
            content="Pinned memory",
            category="fact",
            access_count=1,
            active=True,
            pinned=True,
        )
        store = MagicMock()
        store.get_all_memories.return_value = [node]

        out = build_graph(store, kg=None, output_path=tmp_path / "pinned.html")
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "FFD700" in content  # gold border color


@skip_no_pyvis
class TestVisualizeFromBridge:
    def test_returns_success_dict(self, tmp_path):
        from aimemory.memory.graph_store import MemoryNode
        from aimemory.visualize import visualize_from_bridge

        node = MemoryNode(
            memory_id="mem1",
            content="Test",
            category="fact",
            access_count=1,
            active=True,
        )
        bridge = MagicMock()
        bridge.store.get_all_memories.return_value = [node]
        bridge._kg = None

        result = visualize_from_bridge(
            bridge=bridge,
            output_path=tmp_path / "bridge_test.html",
        )
        assert result["success"] is True
        assert "file_path" in result
        assert result["node_count"] == 1
        assert Path(result["file_path"]).exists()

    def test_file_actually_created(self, tmp_path):
        from aimemory.visualize import visualize_from_bridge

        bridge = MagicMock()
        bridge.store.get_all_memories.return_value = []
        bridge._kg = None

        result = visualize_from_bridge(
            bridge=bridge,
            output_path=tmp_path / "created.html",
        )
        assert result["success"] is True
        assert Path(result["file_path"]).exists()


class TestMissingPyvis:
    def test_import_error_returns_error_dict(self):
        from aimemory.visualize import visualize_from_bridge

        bridge = MagicMock()

        with patch.dict(sys.modules, {"pyvis": None, "pyvis.network": None}):
            with patch("aimemory.visualize._check_pyvis", side_effect=ImportError("no pyvis")):
                result = visualize_from_bridge(bridge=bridge)
                assert result["success"] is False
                assert "error" in result


# ── TestBuildNodeTitle ───────────────────────────────────────────────


class TestBuildNodeTitle:
    def test_basic_title(self):
        from aimemory.memory.graph_store import MemoryNode
        from aimemory.visualize import _build_node_title

        node = MemoryNode(
            memory_id="test_id",
            content="Test content",
            keywords=["kw1", "kw2"],
            category="fact",
            access_count=3,
            created_at="2026-01-01",
            active=True,
            pinned=False,
            immutable=False,
        )
        title = _build_node_title(node)
        assert "test_id" in title
        assert "Test content" in title
        assert "kw1" in title
        assert "fact" in title

    def test_title_with_related_and_levels(self):
        from aimemory.memory.graph_store import MemoryNode
        from aimemory.visualize import _build_node_title

        node = MemoryNode(
            memory_id="test_id",
            content="Content",
            related_ids=["rel1", "rel2"],
            level1_text="Summary text",
            level2_text="entity, predicate, object",
            active=True,
        )
        title = _build_node_title(node)
        assert "rel1" in title
        assert "Summary text" in title
        assert "entity, predicate, object" in title

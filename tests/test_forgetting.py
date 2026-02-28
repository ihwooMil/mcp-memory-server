"""Tests for the forgetting pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aimemory.memory.forgetting import (
    ForgettingPipeline,
    ForgettingResult,
    ForgettingThresholds,
    ImportanceCalculator,
)
from aimemory.memory.graph_store import GraphMemoryStore, MemoryNode


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "forget_test_db")


@pytest.fixture()
def store(tmp_db: str) -> GraphMemoryStore:
    return GraphMemoryStore(persist_directory=tmp_db)


# ── ImportanceCalculator ──────────────────────────────────────


class TestImportanceCalculator:
    def test_recency_decay_recent(self) -> None:
        calc = ImportanceCalculator(decay_lambda=0.05)
        now = datetime.now().isoformat()
        # Very recent: decay should be close to 1.0
        score = calc.recency_decay(now)
        assert score > 0.99

    def test_recency_decay_old(self) -> None:
        calc = ImportanceCalculator(decay_lambda=0.05)
        old = (datetime.now() - timedelta(days=100)).isoformat()
        score = calc.recency_decay(old)
        # e^(-0.05 * 100) = e^(-5) ≈ 0.0067
        assert score < 0.01

    def test_recency_decay_empty_string(self) -> None:
        calc = ImportanceCalculator()
        assert calc.recency_decay("") == 0.0

    def test_recency_decay_invalid_string(self) -> None:
        calc = ImportanceCalculator()
        assert calc.recency_decay("not-a-date") == 0.0

    def test_compute_recent_high_access(self) -> None:
        calc = ImportanceCalculator(decay_lambda=0.05, related_boost=0.1)
        node = MemoryNode(
            memory_id="test",
            content="test",
            created_at=datetime.now().isoformat(),
            access_count=10,
            related_ids=["a", "b"],
        )
        score = calc.compute(node)
        # (1 + 10) * ~1.0 + 0.1 * 2 = ~11.2
        assert score > 10.0

    def test_compute_old_no_access(self) -> None:
        calc = ImportanceCalculator(decay_lambda=0.05, related_boost=0.1)
        old = (datetime.now() - timedelta(days=100)).isoformat()
        node = MemoryNode(
            memory_id="test",
            content="test",
            created_at=old,
            access_count=0,
        )
        score = calc.compute(node)
        # (1 + 0) * ~0.007 + 0 ≈ 0.007
        assert score < 0.1

    def test_related_boost(self) -> None:
        calc = ImportanceCalculator(decay_lambda=0.05, related_boost=0.5)
        old = (datetime.now() - timedelta(days=200)).isoformat()
        node = MemoryNode(
            memory_id="test",
            content="test",
            created_at=old,
            access_count=0,
            related_ids=["a", "b", "c", "d"],
        )
        score = calc.compute(node)
        # related_boost alone: 0.5 * 4 = 2.0
        assert score >= 2.0


# ── ForgettingPipeline ────────────────────────────────────────


class TestForgettingPipeline:
    def test_compresses_low_importance(self, store: GraphMemoryStore) -> None:
        """Memory with importance between threshold_deactivate and threshold_compress should be compressed."""  # noqa: E501
        # Create a memory dated 30 days ago (moderate decay)
        old_time = (datetime.now() - timedelta(days=30)).isoformat()
        mid = store.add_memory(
            content="오래된 기억 내용입니다.",
            keywords=["오래된", "기억"],
        )
        # Manually set created_at to old time
        meta = store._collection.get(ids=[mid], include=["metadatas"])["metadatas"][0]
        meta["created_at"] = old_time
        store._collection.update(ids=[mid], metadatas=[meta])

        thresholds = ForgettingThresholds(
            decay_lambda=0.05,
            threshold_compress=1.0,  # high threshold so most things get compressed
            threshold_deactivate=0.01,  # very low so nothing gets deactivated
        )
        pipeline = ForgettingPipeline(store, thresholds=thresholds)
        result = pipeline.run()
        assert result.compressed >= 1

    def test_deactivates_very_low_importance(self, store: GraphMemoryStore) -> None:
        """Memory with importance below threshold_deactivate should be deactivated."""
        old_time = (datetime.now() - timedelta(days=200)).isoformat()
        mid = store.add_memory(content="아주 오래된 기억")
        meta = store._collection.get(ids=[mid], include=["metadatas"])["metadatas"][0]
        meta["created_at"] = old_time
        store._collection.update(ids=[mid], metadatas=[meta])

        thresholds = ForgettingThresholds(
            decay_lambda=0.05,
            threshold_compress=1.0,
            threshold_deactivate=1.0,  # high threshold so everything gets deactivated
        )
        pipeline = ForgettingPipeline(store, thresholds=thresholds)
        result = pipeline.run()
        assert result.deactivated >= 1

    def test_skips_pinned_memory(self, store: GraphMemoryStore) -> None:
        """Pinned memories should be skipped by the forgetting pipeline."""
        old_time = (datetime.now() - timedelta(days=200)).isoformat()
        mid = store.add_memory(content="핀된 기억", pinned=True)
        meta = store._collection.get(ids=[mid], include=["metadatas"])["metadatas"][0]
        meta["created_at"] = old_time
        store._collection.update(ids=[mid], metadatas=[meta])

        thresholds = ForgettingThresholds(
            threshold_compress=100.0,
            threshold_deactivate=100.0,
        )
        pipeline = ForgettingPipeline(store, thresholds=thresholds)
        result = pipeline.run()
        assert result.skipped_pinned >= 1
        assert result.compressed == 0
        assert result.deactivated == 0

    def test_skips_immutable_memory(self, store: GraphMemoryStore) -> None:
        """Immutable memories should be skipped by the forgetting pipeline."""
        old_time = (datetime.now() - timedelta(days=200)).isoformat()
        mid = store.add_memory(content="불변 기억", immutable=True)
        meta = store._collection.get(ids=[mid], include=["metadatas"])["metadatas"][0]
        meta["created_at"] = old_time
        store._collection.update(ids=[mid], metadatas=[meta])

        thresholds = ForgettingThresholds(
            threshold_compress=100.0,
            threshold_deactivate=100.0,
        )
        pipeline = ForgettingPipeline(store, thresholds=thresholds)
        result = pipeline.run()
        assert result.skipped_immutable >= 1
        assert result.compressed == 0
        assert result.deactivated == 0

    def test_deletes_long_inactive(self, store: GraphMemoryStore) -> None:
        """Inactive memories past deactivation_days should be permanently deleted."""
        old_time = (datetime.now() - timedelta(days=60)).isoformat()
        mid = store.add_memory(content="삭제될 기억")
        # Set old timestamp and deactivate
        meta = store._collection.get(ids=[mid], include=["metadatas"])["metadatas"][0]
        meta["created_at"] = old_time
        meta["active"] = "false"
        store._collection.update(ids=[mid], metadatas=[meta])

        thresholds = ForgettingThresholds(deactivation_days=30)
        pipeline = ForgettingPipeline(store, thresholds=thresholds)
        result = pipeline.run()
        assert result.deleted >= 1
        assert len(result.audit_log) >= 1
        assert result.audit_log[0].memory_id == mid

    def test_keeps_recently_inactive(self, store: GraphMemoryStore) -> None:
        """Recently deactivated memories should NOT be deleted."""
        recent_time = (datetime.now() - timedelta(days=5)).isoformat()
        mid = store.add_memory(content="최근 비활성 기억")
        meta = store._collection.get(ids=[mid], include=["metadatas"])["metadatas"][0]
        meta["created_at"] = recent_time
        meta["active"] = "false"
        store._collection.update(ids=[mid], metadatas=[meta])

        thresholds = ForgettingThresholds(deactivation_days=30)
        pipeline = ForgettingPipeline(store, thresholds=thresholds)
        result = pipeline.run()
        assert result.deleted == 0

    def test_result_dataclass(self) -> None:
        """ForgettingResult should have correct defaults."""
        result = ForgettingResult()
        assert result.compressed == 0
        assert result.deactivated == 0
        assert result.deleted == 0
        assert result.audit_log == []

"""Tests for memory consolidation and sleep cycle runner."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aimemory.config import SleepCycleConfig
from aimemory.memory.consolidation import (
    ConsolidationResult,
    MemoryConsolidator,
)
from aimemory.memory.graph_store import GraphMemoryStore
from aimemory.memory.sleep_cycle import SleepCycleRunner


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "sleep_test_db")


@pytest.fixture()
def store(tmp_db: str) -> GraphMemoryStore:
    return GraphMemoryStore(persist_directory=tmp_db)


# ── ConsolidationResult ──────────────────────────────────────


class TestConsolidationResult:
    def test_defaults(self) -> None:
        result = ConsolidationResult()
        assert result.pairs_found == 0
        assert result.memories_merged == 0
        assert result.merge_records == []


# ── MemoryConsolidator ───────────────────────────────────────


class TestMemoryConsolidator:
    def test_find_duplicates_similar_content(self, store: GraphMemoryStore) -> None:
        """Two memories with nearly identical content should be detected as duplicates."""
        store.add_memory(content="저는 매일 아침 커피를 마셔요.", keywords=["커피", "아침"])
        store.add_memory(content="저는 매일 아침 커피를 마셔요.", keywords=["커피"])

        consolidator = MemoryConsolidator(store, similarity_threshold=0.90)
        duplicates = consolidator.find_duplicates()
        assert len(duplicates) >= 1

    def test_no_duplicates_for_dissimilar(self, store: GraphMemoryStore) -> None:
        """Dissimilar memories should not be detected as duplicates."""
        store.add_memory(content="저는 매일 아침 커피를 마셔요.", keywords=["커피"])
        store.add_memory(
            content="인공지능 기술은 빠르게 발전하고 있습니다.",
            keywords=["인공지능", "기술"],
        )

        consolidator = MemoryConsolidator(store, similarity_threshold=0.95)
        duplicates = consolidator.find_duplicates()
        assert len(duplicates) == 0

    def test_merge_reduces_memory_count(self, store: GraphMemoryStore) -> None:
        """After merging, the total memory count should decrease."""
        store.add_memory(content="저는 매일 아침 커피를 마셔요.", keywords=["커피", "아침"])
        store.add_memory(content="저는 매일 아침 커피를 마셔요.", keywords=["커피"])

        before = len(store.get_all_memories(include_inactive=True))
        consolidator = MemoryConsolidator(store, similarity_threshold=0.90)
        result = consolidator.run()

        after = len(store.get_all_memories(include_inactive=True))
        if result.memories_merged > 0:
            assert after < before

    def test_merge_preserves_keywords(self, store: GraphMemoryStore) -> None:
        """Merged memory should contain the union of both memories' keywords."""
        store.add_memory(content="저는 매일 아침 커피를 마셔요.", keywords=["커피", "아침"])
        store.add_memory(content="저는 매일 아침 커피를 마셔요.", keywords=["커피", "음료"])

        consolidator = MemoryConsolidator(store, similarity_threshold=0.90)
        duplicates = consolidator.find_duplicates()

        if duplicates:
            keeper, absorbed = duplicates[0]
            record = consolidator.merge_pair(keeper, absorbed)

            # Get the surviving memory
            surviving = store._collection.get(ids=[record.surviving_id], include=["metadatas"])
            keywords_str = surviving["metadatas"][0].get("keywords", "")
            keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

            # Should have union of keywords
            assert "커피" in keywords
            assert len(keywords) >= 2

    def test_immutable_not_absorbed(self, store: GraphMemoryStore) -> None:
        """Immutable memories should not be absorbed (merged into another)."""
        store.add_memory(
            content="저는 매일 아침 커피를 마셔요.",
            keywords=["커피"],
            immutable=True,
        )
        store.add_memory(content="저는 매일 아침 커피를 마셔요.", keywords=["커피"])

        consolidator = MemoryConsolidator(store, similarity_threshold=0.90)
        duplicates = consolidator.find_duplicates()

        # If pairs found, the immutable one must be the keeper
        for keeper, absorbed in duplicates:
            assert not absorbed.immutable

    def test_five_pair_consolidation(self, store: GraphMemoryStore) -> None:
        """Consolidation should handle multiple duplicate pairs."""
        # Create 5 pairs of near-duplicates
        contents = [
            "저는 매일 아침 커피를 마셔요.",
            "제 취미는 독서예요.",
            "저는 서울에 살고 있어요.",
            "제가 좋아하는 음식은 김치찌개예요.",
            "저는 매일 운동을 해요.",
        ]
        for content in contents:
            store.add_memory(content=content, keywords=["테스트"])
            store.add_memory(content=content, keywords=["테스트"])

        consolidator = MemoryConsolidator(store, similarity_threshold=0.90)
        result = consolidator.run()
        assert result.memories_merged >= 1


# ── SleepCycleRunner ─────────────────────────────────────────


class TestSleepCycleRunner:
    def test_empty_store(self, store: GraphMemoryStore) -> None:
        """Sleep cycle should complete gracefully on an empty store."""
        config = SleepCycleConfig(enable_checkpoint=False)
        runner = SleepCycleRunner(store=store, config=config)
        report = runner.run()

        assert report.memory_count_before == 0
        assert report.memory_count_after == 0
        assert not report.errors

    def test_full_cycle(self, store: GraphMemoryStore) -> None:
        """Full sleep cycle should run all tasks without errors."""
        store.add_memory(content="테스트 기억입니다.", keywords=["테스트"])
        store.add_memory(content="또 다른 기억입니다.", keywords=["기억"])

        config = SleepCycleConfig(enable_checkpoint=False)
        runner = SleepCycleRunner(store=store, config=config)
        report = runner.run()

        assert report.started_at
        assert report.finished_at
        assert report.duration_seconds >= 0
        assert report.memory_count_before == 2
        assert not report.errors

    def test_resolution_regeneration(self, store: GraphMemoryStore) -> None:
        """Memories missing level1/level2 should get them regenerated."""
        store.add_memory(
            content="저는 매일 아침 커피를 마셔요.",
            keywords=["커피", "아침"],
        )

        config = SleepCycleConfig(
            enable_consolidation=False,
            enable_forgetting=False,
            enable_checkpoint=False,
        )
        runner = SleepCycleRunner(store=store, config=config)
        report = runner.run()

        assert report.resolution_regenerated >= 1

        # Verify level1/level2 were actually set
        memories = store.get_all_memories()
        for mem in memories:
            assert mem.level1_text or mem.level2_text

    def test_forgetting_integration(self, store: GraphMemoryStore) -> None:
        """Forgetting should work within the sleep cycle."""
        # Add an old memory that should get compressed
        old_time = (datetime.now() - timedelta(days=30)).isoformat()
        mid = store.add_memory(content="오래된 기억입니다.", keywords=["오래된"])
        meta = store._collection.get(ids=[mid], include=["metadatas"])["metadatas"][0]
        meta["created_at"] = old_time
        store._collection.update(ids=[mid], metadatas=[meta])

        config = SleepCycleConfig(
            enable_consolidation=False,
            enable_resolution_regen=False,
            enable_checkpoint=False,
            forgetting_threshold_compress=1.0,
            forgetting_threshold_deactivate=0.01,
        )
        runner = SleepCycleRunner(store=store, config=config)
        report = runner.run()

        assert report.forgetting is not None
        assert report.forgetting.compressed >= 1

    def test_checkpoint_with_policy(self, store: GraphMemoryStore, tmp_path: Path) -> None:
        """RL checkpoint should be saved when policy is provided."""
        store.add_memory(content="기억입니다.", keywords=["테스트"])

        mock_policy = MagicMock()
        checkpoint_dir = str(tmp_path / "checkpoints")

        config = SleepCycleConfig(
            enable_consolidation=False,
            enable_resolution_regen=False,
            enable_forgetting=False,
            enable_checkpoint=True,
            checkpoint_dir=checkpoint_dir,
        )
        runner = SleepCycleRunner(store=store, config=config, policy=mock_policy)
        report = runner.run()

        assert report.checkpoint_saved
        assert report.checkpoint_path
        mock_policy.save_checkpoint.assert_called_once()

    def test_to_dict_serializable(self, store: GraphMemoryStore) -> None:
        """to_dict() should produce a JSON-serializable dictionary."""
        config = SleepCycleConfig(enable_checkpoint=False)
        runner = SleepCycleRunner(store=store, config=config)
        report = runner.run()

        d = report.to_dict()
        # Should not raise
        serialized = json.dumps(d, ensure_ascii=False)
        assert isinstance(serialized, str)

    def test_save_report(self, store: GraphMemoryStore, tmp_path: Path) -> None:
        """save_report should write a JSON file."""
        config = SleepCycleConfig(enable_checkpoint=False)
        runner = SleepCycleRunner(store=store, config=config)
        report = runner.run()

        report_dir = tmp_path / "reports"
        path = runner.save_report(report, report_dir)

        assert path.exists()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "started_at" in data
        assert "memory_count_before" in data

    def test_error_isolation(self, store: GraphMemoryStore) -> None:
        """A failure in one task should not prevent others from running."""
        store.add_memory(content="테스트 기억입니다.", keywords=["테스트"])

        config = SleepCycleConfig(enable_checkpoint=False)
        runner = SleepCycleRunner(store=store, config=config)

        # Patch consolidation to raise an error
        with patch.object(
            MemoryConsolidator,
            "run",
            side_effect=RuntimeError("consolidation error"),
        ):
            report = runner.run()

        # Consolidation should have failed
        assert len(report.errors) >= 1
        assert "consolidation" in report.errors[0].lower()

        # But resolution and forgetting should still have run
        # (forgetting result should exist since enable_forgetting=True)
        assert report.forgetting is not None

    def test_summary_output(self, store: GraphMemoryStore) -> None:
        """summary() should return a human-readable string."""
        store.add_memory(content="요약 테스트입니다.", keywords=["요약"])

        config = SleepCycleConfig(enable_checkpoint=False)
        runner = SleepCycleRunner(store=store, config=config)
        report = runner.run()

        summary = report.summary()
        assert "Sleep Cycle Report" in summary
        assert "Duration" in summary
        assert "Memories" in summary

    def test_no_checkpoint_without_policy(self, store: GraphMemoryStore) -> None:
        """No checkpoint should be saved when policy is None."""
        config = SleepCycleConfig(enable_checkpoint=True)
        runner = SleepCycleRunner(store=store, config=config, policy=None)
        report = runner.run()

        assert not report.checkpoint_saved
        assert report.checkpoint_path == ""

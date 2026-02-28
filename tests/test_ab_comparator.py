"""A/B 비교 프레임워크 테스트."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aimemory.memory.graph_store import MemoryNode
from aimemory.online.ab_comparator import ABComparator, ABReport, ABResult
from aimemory.online.reranker import ReRankPolicy, ReRanker


# ─── 헬퍼 함수 ──────────────────────────────────────────────────────


def _make_node(
    memory_id: str,
    similarity_score: float = 0.8,
    content: str = "테스트 내용",
    keywords: list[str] | None = None,
    category: str = "fact",
) -> MemoryNode:
    return MemoryNode(
        memory_id=memory_id,
        content=content,
        keywords=keywords or ["테스트"],
        category=category,
        similarity_score=similarity_score,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_candidates(n: int = 10) -> list[MemoryNode]:
    return [
        _make_node(
            memory_id=f"mem_{i:03d}",
            similarity_score=1.0 - i * 0.05,
            content=f"메모리 내용 {i}",
        )
        for i in range(n)
    ]


# ─── TestABComparator ────────────────────────────────────────────────


class TestABComparator:
    def setup_method(self):
        # 결정론적 테스트를 위해 epsilon=0 사용
        self.reranker = ReRanker(
            policy=ReRankPolicy(epsilon=0.0, select_k=3),
            enabled=True,
        )
        self.comparator = ABComparator(reranker=self.reranker)

    def test_compare_returns_ab_result(self):
        """compare()는 올바른 필드를 가진 ABResult를 반환해야 합니다."""
        candidates = _make_candidates(10)
        result = self.comparator.compare("Python 개발", candidates, select_k=3)

        assert isinstance(result, ABResult)
        assert result.query == "Python 개발"
        assert len(result.baseline_ids) == 3
        assert len(result.reranked_ids) == 3
        assert isinstance(result.overlap_count, int)
        assert 0 <= result.overlap_count <= 3
        assert isinstance(result.baseline_avg_sim, float)
        assert isinstance(result.reranked_avg_sim, float)
        assert isinstance(result.position_changes, list)

    def test_baseline_is_similarity_order(self):
        """baseline_ids는 ChromaDB 유사도 순서의 top-k여야 합니다."""
        candidates = _make_candidates(10)
        result = self.comparator.compare("테스트", candidates, select_k=3)

        expected_baseline = [candidates[0].memory_id, candidates[1].memory_id, candidates[2].memory_id]
        assert result.baseline_ids == expected_baseline

    def test_overlap_count_correct(self):
        """overlap_count는 기준선과 리랭크 결과의 교집합 수와 일치해야 합니다."""
        candidates = _make_candidates(10)
        result = self.comparator.compare("테스트", candidates, select_k=3)

        expected_overlap = len(set(result.baseline_ids) & set(result.reranked_ids))
        assert result.overlap_count == expected_overlap

    def test_position_changes_computed(self):
        """position_changes는 순위 델타를 반영해야 합니다."""
        candidates = _make_candidates(10)
        result = self.comparator.compare("테스트", candidates, select_k=3)

        # 리랭크 결과의 수만큼 position_changes가 있어야 함
        assert len(result.position_changes) == len(result.reranked_ids)
        # 모두 정수여야 함
        assert all(isinstance(d, int) for d in result.position_changes)

    def test_run_batch_aggregates(self):
        """run_batch()는 올바른 집계 수치를 가진 ABReport를 반환해야 합니다."""
        queries_and_candidates = [
            ("Python 개발", _make_candidates(10)),
            ("머신러닝 모델", _make_candidates(8)),
            ("Django 웹 프레임워크", _make_candidates(10)),
        ]

        report = self.comparator.run_batch(queries_and_candidates, select_k=3)

        assert isinstance(report, ABReport)
        assert report.total_queries == 3
        assert len(report.results) == 3
        assert 0.0 <= report.avg_overlap <= 1.0
        assert report.avg_position_change >= 0.0
        assert report.rerank_latency_ms_p50 >= 0.0
        assert report.rerank_latency_ms_p95 >= report.rerank_latency_ms_p50

    def test_run_batch_empty(self):
        """빈 배치에 대해 run_batch()는 기본 ABReport를 반환해야 합니다."""
        report = self.comparator.run_batch([], select_k=3)
        assert report.total_queries == 0
        assert report.avg_overlap == 0.0

    def test_summary_returns_string(self):
        """ABReport.summary()는 문자열을 반환해야 합니다."""
        queries_and_candidates = [("테스트", _make_candidates(5))]
        report = self.comparator.run_batch(queries_and_candidates, select_k=3)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_latency_recorded(self):
        """각 ABResult에 latency_ms가 기록되어야 합니다."""
        candidates = _make_candidates(10)
        result = self.comparator.compare("테스트", candidates)
        assert result.latency_ms >= 0.0

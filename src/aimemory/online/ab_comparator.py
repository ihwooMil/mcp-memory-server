"""A/B comparison framework for evaluating re-ranker quality.

리랭커와 ChromaDB 기준선(baseline) 결과를 비교하는 프레임워크입니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from aimemory.memory.graph_store import MemoryNode
from aimemory.online.reranker import ReRanker


@dataclass
class ABResult:
    """단일 A/B 비교 결과."""

    query: str
    baseline_ids: list[str]        # ChromaDB 유사도 순서 top-k ID
    reranked_ids: list[str]        # 리랭커 순서 top-k ID
    overlap_count: int             # 두 결과 ID 교집합 수
    baseline_avg_sim: float        # 기준선 top-k의 평균 유사도
    reranked_avg_sim: float        # 리랭크된 top-k의 평균 유사도
    reranked_avg_score: float      # 리랭커 점수 평균 (사용 가능 시)
    position_changes: list[int]    # 각 리랭크 결과의 순위 변화 (기준선 대비)
    latency_ms: float = 0.0        # rerank() 호출 지연 시간 (ms)


@dataclass
class ABReport:
    """배치 A/B 비교 집계 보고서."""

    total_queries: int
    avg_overlap: float             # 기준선과 리랭크 결과의 평균 겹침 비율
    avg_position_change: float     # 평균 절대 순위 변화량
    rerank_latency_ms_p50: float   # 리랭킹 지연 시간 p50 (ms)
    rerank_latency_ms_p95: float   # 리랭킹 지연 시간 p95 (ms)
    results: list[ABResult] = field(default_factory=list)

    def summary(self) -> str:
        """사람이 읽을 수 있는 요약 문자열을 반환합니다."""
        overlap_pct = self.avg_overlap * 100
        return (
            f"A/B 비교 보고서 | 총 쿼리: {self.total_queries} | "
            f"평균 겹침: {overlap_pct:.1f}% | "
            f"평균 순위 변화: {self.avg_position_change:.2f} | "
            f"지연 p50: {self.rerank_latency_ms_p50:.2f}ms | "
            f"지연 p95: {self.rerank_latency_ms_p95:.2f}ms"
        )


class ABComparator:
    """리랭크된 결과와 ChromaDB 기준선 순서를 비교합니다."""

    def __init__(self, reranker: ReRanker) -> None:
        self._reranker = reranker

    def compare(
        self,
        query: str,
        candidates: list[MemoryNode],
        select_k: int = 3,
    ) -> ABResult:
        """단일 A/B 비교를 실행합니다.

        Args:
            query: 검색 쿼리.
            candidates: ChromaDB top-K 후보 목록 (유사도 내림차순).
            select_k: 선택할 결과 수.

        Returns:
            기준선과 리랭크 결과 비교를 담은 ABResult.
        """
        # 기준선: ChromaDB 유사도 순서 그대로 top-k
        baseline_nodes = candidates[:select_k]
        baseline_ids = [n.memory_id for n in baseline_nodes]
        baseline_avg_sim = float(
            np.mean([n.similarity_score or 0.0 for n in baseline_nodes])
            if baseline_nodes else 0.0
        )

        # 리랭크: 정책으로 재정렬
        start_time = time.perf_counter()
        reranked_nodes = self._reranker.rerank(query=query, candidates=candidates)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        reranked_ids = [n.memory_id for n in reranked_nodes]
        reranked_avg_sim = float(
            np.mean([n.similarity_score or 0.0 for n in reranked_nodes])
            if reranked_nodes else 0.0
        )

        # 겹침 수
        overlap_count = len(set(baseline_ids) & set(reranked_ids))

        # 순위 변화: 리랭크 결과 각각의 기준선 대비 순위 델타
        baseline_rank_map = {mid: i for i, mid in enumerate(baseline_ids)}
        position_changes: list[int] = []
        for new_rank, mid in enumerate(reranked_ids):
            if mid in baseline_rank_map:
                delta = baseline_rank_map[mid] - new_rank
            else:
                # 기준선에 없는 경우 (후보 목록의 후반부에서 선택됨)
                # 기준선 외부로부터 이동한 것으로 표현: 최대 변화량 추정
                original_rank = next(
                    (i for i, n in enumerate(candidates) if n.memory_id == mid),
                    len(candidates),
                )
                delta = original_rank - new_rank
            position_changes.append(delta)

        # 리랭크 점수 평균 (리랭커의 내부 상태에서 추출 불가하므로 0.0으로 기록)
        reranked_avg_score = 0.0

        return ABResult(
            query=query,
            baseline_ids=baseline_ids,
            reranked_ids=reranked_ids,
            overlap_count=overlap_count,
            baseline_avg_sim=baseline_avg_sim,
            reranked_avg_sim=reranked_avg_sim,
            reranked_avg_score=reranked_avg_score,
            position_changes=position_changes,
            latency_ms=latency_ms,
        )

    def run_batch(
        self,
        queries_and_candidates: list[tuple[str, list[MemoryNode]]],
        select_k: int = 3,
    ) -> ABReport:
        """배치 쿼리에 대해 A/B 비교를 실행합니다.

        Args:
            queries_and_candidates: (쿼리, 후보 목록) 튜플의 목록.
            select_k: 각 쿼리에서 선택할 결과 수.

        Returns:
            집계된 메트릭을 담은 ABReport.
        """
        results: list[ABResult] = []

        for query, candidates in queries_and_candidates:
            result = self.compare(query, candidates, select_k=select_k)
            results.append(result)

        if not results:
            return ABReport(
                total_queries=0,
                avg_overlap=0.0,
                avg_position_change=0.0,
                rerank_latency_ms_p50=0.0,
                rerank_latency_ms_p95=0.0,
                results=[],
            )

        # 집계 메트릭 계산
        overlaps = [r.overlap_count / max(len(r.baseline_ids), 1) for r in results]
        avg_overlap = float(np.mean(overlaps))

        all_position_changes = [
            abs(delta)
            for r in results
            for delta in r.position_changes
        ]
        avg_position_change = float(np.mean(all_position_changes)) if all_position_changes else 0.0

        latencies = [r.latency_ms for r in results]
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))

        return ABReport(
            total_queries=len(results),
            avg_overlap=avg_overlap,
            avg_position_change=avg_position_change,
            rerank_latency_ms_p50=p50,
            rerank_latency_ms_p95=p95,
            results=results,
        )

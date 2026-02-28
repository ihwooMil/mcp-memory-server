"""RL Re-ranker 유닛 및 통합 테스트."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aimemory.memory.graph_store import MemoryNode
from aimemory.online.reranker import (
    RERANK_FEATURE_DIM,
    ReRanker,
    ReRankFeatureExtractor,
    ReRankPolicy,
)

# ─── 헬퍼 함수 ──────────────────────────────────────────────────────


def _make_node(
    memory_id: str = "mem_001",
    content: str = "Python 개발 경험이 있습니다",
    keywords: list[str] | None = None,
    category: str = "technical",
    related_ids: list[str] | None = None,
    similarity_score: float = 0.8,
    access_count: int = 0,
    level1_text: str = "",
    level2_text: str = "",
    created_at: str = "",
) -> MemoryNode:
    if created_at == "":
        created_at = datetime.now(timezone.utc).isoformat()
    return MemoryNode(
        memory_id=memory_id,
        content=content,
        keywords=keywords or ["Python", "개발"],
        category=category,
        related_ids=related_ids or [],
        similarity_score=similarity_score,
        access_count=access_count,
        level1_text=level1_text,
        level2_text=level2_text,
        created_at=created_at,
    )


def _make_candidates(n: int = 5) -> list[MemoryNode]:
    """n개의 테스트용 MemoryNode 후보를 생성합니다."""
    return [
        _make_node(
            memory_id=f"mem_{i:03d}",
            content=f"테스트 메모리 내용 {i}",
            keywords=[f"키워드{i}", "테스트"],
            similarity_score=1.0 - i * 0.05,
            access_count=i,
        )
        for i in range(n)
    ]


# ─── TestReRankFeatureExtractor ──────────────────────────────────────


class TestReRankFeatureExtractor:
    def setup_method(self):
        self.extractor = ReRankFeatureExtractor()

    def test_output_shape(self):
        """extract()는 (n, 8) shape의 ndarray를 반환해야 합니다."""
        candidates = _make_candidates(5)
        features = self.extractor.extract("Python 개발 경험", candidates)
        assert features.shape == (5, RERANK_FEATURE_DIM)
        assert features.dtype == np.float32

    def test_chroma_similarity_passthrough(self):
        """feature[0]은 node.similarity_score와 동일해야 합니다."""
        node = _make_node(similarity_score=0.75)
        features = self.extractor.extract("테스트 쿼리", [node])
        assert features[0, 0] == pytest.approx(0.75)

    def test_keyword_overlap_computation(self):
        """쿼리 키워드와 메모리 키워드가 겹칠 때 올바른 비율을 계산해야 합니다."""
        node = _make_node(keywords=["Python", "Django", "개발"])
        # 쿼리 키워드 2개 중 1개 겹침
        features = self.extractor.extract(
            "Python 서버",
            [node],
            query_keywords=["Python", "서버"],
        )
        # overlap = 1 / max(2, 1) = 0.5
        assert features[0, 1] == pytest.approx(0.5)

    def test_category_match_detection(self):
        """기술 쿼리 + technical 카테고리 메모리 → category_match = 1.0."""
        node = _make_node(category="technical")
        # PyTorch는 기술 키워드 → inferred_category = "technical"
        features = self.extractor.extract("PyTorch로 모델을 만들었어요", [node])
        assert features[0, 2] == pytest.approx(1.0)

    def test_category_mismatch(self):
        """기술 쿼리 + preference 카테고리 메모리 → category_match = 0.0."""
        node = _make_node(category="preference")
        features = self.extractor.extract("PyTorch로 모델을 만들었어요", [node])
        assert features[0, 2] == pytest.approx(0.0)

    def test_recency_decay_recent(self):
        """최근 생성 메모리 → 높은 recency 점수."""
        recent_node = _make_node(created_at=datetime.now(timezone.utc).isoformat())
        features = self.extractor.extract("테스트", [recent_node])
        # 방금 생성 → exp(-0.05 * 0) ≈ 1.0
        assert features[0, 3] > 0.9

    def test_recency_decay_old(self):
        """오래된 메모리 → 카테고리별 감쇠 적용 (기본 λ=0.01)."""
        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        old_node = _make_node(created_at=old_date.isoformat())
        features = self.extractor.extract("테스트", [old_node])
        # exp(-0.01 * 100) ≈ 0.368 (category=fact → λ=0.005 → ~0.607)
        # fact의 λ=0.005이므로 exp(-0.005*100) ≈ 0.607
        assert features[0, 3] < 0.8  # 100일 지나면 확실히 1.0보다 낮음
        assert features[0, 3] > 0.3  # 하지만 fact이므로 급격하게 떨어지지 않음

    def test_empty_candidates(self):
        """빈 후보 목록 → shape (0, 8) ndarray 반환."""
        features = self.extractor.extract("테스트", [])
        assert features.shape == (0, RERANK_FEATURE_DIM)

    def test_access_frequency_log_scaled(self):
        """access_count가 높을수록 log1p로 스케일된 feature[4]가 커야 합니다."""
        node_low = _make_node(access_count=0)
        node_high = _make_node(access_count=10)
        feat_low = self.extractor.extract("테스트", [node_low])
        feat_high = self.extractor.extract("테스트", [node_high])
        assert feat_high[0, 4] > feat_low[0, 4]

    def test_has_related_flag(self):
        """related_ids가 있으면 feature[6] = 1.0, 없으면 0.0."""
        node_with = _make_node(related_ids=["mem_other"])
        node_without = _make_node(related_ids=[])
        feat_with = self.extractor.extract("테스트", [node_with])
        feat_without = self.extractor.extract("테스트", [node_without])
        assert feat_with[0, 6] == pytest.approx(1.0)
        assert feat_without[0, 6] == pytest.approx(0.0)

    def test_resolution_available(self):
        """resolution levels 수에 따라 feature[7]이 달라야 합니다."""
        node_full = _make_node(level1_text="요약", level2_text="엔티티")
        node_none = _make_node(level1_text="", level2_text="")
        feat_full = self.extractor.extract("테스트", [node_full])
        feat_none = self.extractor.extract("테스트", [node_none])
        # node_full: content + level1 + level2 = 3/3 = 1.0
        # node_none: content만 = 1/3 ≈ 0.333
        assert feat_full[0, 7] == pytest.approx(1.0)
        assert feat_none[0, 7] == pytest.approx(1 / 3, abs=0.01)


# ─── TestReRankPolicy ────────────────────────────────────────────────


class TestReRankPolicy:
    def setup_method(self):
        self.policy = ReRankPolicy(
            feature_dim=RERANK_FEATURE_DIM,
            hidden_dim=32,
            lr=0.005,
            epsilon=0.0,  # 기본적으로 greedy
            select_k=3,
        )

    def _make_features(self, n: int = 10) -> np.ndarray:
        return np.random.randn(n, RERANK_FEATURE_DIM).astype(np.float32)

    def test_rank_returns_valid_indices(self):
        """반환된 인덱스는 [0, n) 범위 내에 있어야 합니다."""
        features = self._make_features(10)
        indices = self.policy.rank(features)
        assert all(0 <= i < 10 for i in indices)

    def test_rank_returns_k_results(self):
        """정확히 select_k개의 인덱스를 반환해야 합니다."""
        features = self._make_features(10)
        indices = self.policy.rank(features)
        assert len(indices) == self.policy.select_k

    def test_rank_fewer_candidates_than_k(self):
        """후보 수가 select_k보다 적으면 모두 반환해야 합니다."""
        features = self._make_features(2)
        indices = self.policy.rank(features)
        assert len(indices) == 2

    def test_greedy_deterministic(self):
        """epsilon=0이면 동일한 특징에 대해 항상 동일한 순위를 반환해야 합니다."""
        self.policy.epsilon = 0.0
        features = self._make_features(10)
        results = [tuple(self.policy.rank(features)) for _ in range(5)]
        assert len(set(results)) == 1

    def test_exploration_randomizes(self):
        """epsilon=1이면 선택이 무작위화되어야 합니다."""
        self.policy.epsilon = 1.0
        features = self._make_features(10)
        # 여러 번 호출해 다른 결과가 나와야 함
        all_selections = [tuple(self.policy.rank(features)) for _ in range(20)]
        unique = set(all_selections)
        assert len(unique) > 1

    def test_update_returns_loss(self):
        """update()는 0 이상의 float 손실값을 반환해야 합니다."""
        features = self._make_features(10)
        loss = self.policy.update(features, [0, 1, 2], reward=1.0)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_update_improves_ranking(self):
        """반복 업데이트 후 선택된 후보의 점수가 올라가야 합니다."""
        self.policy.epsilon = 0.0
        features = np.zeros((5, RERANK_FEATURE_DIM), dtype=np.float32)
        features[0, 0] = 1.0  # 첫 번째 후보에 강한 신호

        # 첫 번째 후보를 선택하도록 반복 학습
        for _ in range(100):
            self.policy.update(features, selected_indices=[0], reward=1.0)

        indices = self.policy.rank(features)
        # 학습 후 첫 번째 후보가 상위에 위치해야 함
        assert 0 in indices

    def test_get_set_parameters_roundtrip(self):
        """파라미터 직렬화/역직렬화가 완전히 일치해야 합니다."""
        params_before = self.policy.get_parameters()
        assert isinstance(params_before, np.ndarray)
        assert params_before.ndim == 1

        new_params = params_before + 0.1
        self.policy.set_parameters(new_params)
        params_after = self.policy.get_parameters()
        np.testing.assert_allclose(params_after, new_params, atol=1e-6)

    def test_parameter_count(self):
        """MLP 구조에 맞는 파라미터 수를 확인합니다."""
        params = self.policy.get_parameters()
        # Linear(8, 32): 8*32 + 32 = 288
        # Linear(32, 1): 32*1 + 1 = 33
        # 총: 321
        expected = RERANK_FEATURE_DIM * 32 + 32 + 32 * 1 + 1
        assert len(params) == expected


# ─── TestReRanker ────────────────────────────────────────────────────


class TestReRanker:
    def setup_method(self):
        self.reranker = ReRanker(
            policy=ReRankPolicy(epsilon=0.0, select_k=3),
            enabled=True,
        )

    def test_rerank_returns_top_k_nodes(self):
        """rerank()는 MemoryNode 목록을 반환하며 길이가 select_k여야 합니다."""
        candidates = _make_candidates(10)
        results = self.reranker.rerank("Python 개발", candidates)
        assert len(results) == 3
        assert all(isinstance(n, MemoryNode) for n in results)

    def test_rerank_disabled_passthrough(self):
        """비활성화 시 원래 순서의 처음 3개를 반환해야 합니다."""
        self.reranker.enabled = False
        candidates = _make_candidates(10)
        results = self.reranker.rerank("Python 개발", candidates)
        assert len(results) == 3
        # 원래 순서 유지
        assert results[0].memory_id == candidates[0].memory_id
        assert results[1].memory_id == candidates[1].memory_id
        assert results[2].memory_id == candidates[2].memory_id

    def test_rerank_latency_under_budget(self):
        """wall-clock 지연 시간이 max_latency_ms 이내여야 합니다."""
        import time

        candidates = _make_candidates(10)
        start = time.perf_counter()
        self.reranker.rerank("Python 개발", candidates)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert elapsed_ms < self.reranker.max_latency_ms

    def test_update_from_feedback_returns_loss(self):
        """rerank() 후 update_from_feedback()은 float 손실값을 반환해야 합니다."""
        candidates = _make_candidates(10)
        self.reranker.rerank("Python 개발", candidates)
        loss = self.reranker.update_from_feedback(reward=1.0)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_no_update_without_rerank(self):
        """rerank() 호출 없이 update_from_feedback()은 None을 반환해야 합니다."""
        reranker = ReRanker()
        result = reranker.update_from_feedback(reward=1.0)
        assert result is None

    def test_has_pending_state(self):
        """rerank() 후 has_pending_state는 True, update 후 False여야 합니다."""
        candidates = _make_candidates(10)
        assert not self.reranker.has_pending_state
        self.reranker.rerank("Python 개발", candidates)
        assert self.reranker.has_pending_state
        self.reranker.update_from_feedback(reward=1.0)
        assert not self.reranker.has_pending_state

    def test_rerank_too_few_candidates(self):
        """후보가 select_k보다 적으면 그대로 반환해야 합니다."""
        candidates = _make_candidates(2)
        results = self.reranker.rerank("Python 개발", candidates)
        assert len(results) == 2

    def test_save_load_checkpoint(self):
        """체크포인트 저장/로드 후 파라미터가 일치해야 합니다."""
        candidates = _make_candidates(10)
        # 업데이트로 모델 변경
        self.reranker.rerank("Python 개발", candidates)
        self.reranker.update_from_feedback(reward=1.0)
        params_before = self.reranker._policy.get_parameters()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "reranker.pt"
            self.reranker.save_checkpoint(ckpt_path)

            new_reranker = ReRanker()
            new_reranker.load_checkpoint(ckpt_path)
            params_after = new_reranker._policy.get_parameters()

        np.testing.assert_allclose(params_after, params_before, atol=1e-6)


# ─── TestReRankerIntegration ─────────────────────────────────────────


class TestReRankerIntegration:
    """MemoryPolicyAgent와의 통합 테스트."""

    def _make_agent_with_reranker(self, reranker=None):
        from aimemory.online.policy import MemoryPolicyAgent, OnlinePolicy

        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"total": 5}
        mock_store.add_memory.return_value = "mem_abc"
        # search 결과: 10개의 MemoryNode 반환
        mock_store.search.return_value = _make_candidates(10)

        mock_feedback = MagicMock()
        policy = OnlinePolicy(epsilon=0.0)

        if reranker is None:
            reranker = ReRanker(
                policy=ReRankPolicy(epsilon=0.0, select_k=3),
                enabled=True,
            )

        agent = MemoryPolicyAgent(
            graph_store=mock_store,
            policy=policy,
            feedback_detector=mock_feedback,
            reranker=reranker,
        )
        return agent, mock_store, mock_feedback

    def test_policy_agent_with_reranker(self):
        """MemoryPolicyAgent가 _execute_retrieve에서 reranker를 사용해야 합니다."""
        from aimemory.schemas import Role, Turn

        agent, mock_store, _ = self._make_agent_with_reranker()

        # 질문 형태의 쿼리 → RETRIEVE 경로
        turn = Turn(turn_id=1, role=Role.USER, content="Python에 대해 알려주세요?")
        decision = agent.decide(turn, [])

        if decision.action.name == "RETRIEVE":
            # top_k=10으로 검색해야 함 (리랭커 사용 시)
            call_args = mock_store.search.call_args
            if call_args:
                assert (
                    call_args[1].get("top_k", call_args[0][1] if len(call_args[0]) > 1 else 3) == 10
                )

    def test_feedback_updates_reranker(self):
        """RETRIEVE 후 process_feedback()이 reranker를 업데이트해야 합니다."""
        from aimemory.reward.feedback_detector import FeedbackSignal, FeedbackType
        from aimemory.schemas import Role, Turn

        reranker = ReRanker(
            policy=ReRankPolicy(epsilon=0.0, select_k=3),
            enabled=True,
        )
        agent, mock_store, mock_feedback = self._make_agent_with_reranker(reranker)

        # RETRIEVE 액션 강제 실행
        agent._execute_retrieve(Turn(turn_id=1, role=Role.USER, content="Python?"))
        # 리랭커가 pending 상태인지 확인
        assert reranker.has_pending_state

        # RETRIEVE 액션으로 설정
        from aimemory.online.policy import ACTION_INDEX
        from aimemory.schemas import MemoryActionType

        agent._last_action_id = ACTION_INDEX[MemoryActionType.RETRIEVE]
        agent._recent_actions.append(MemoryActionType.RETRIEVE)

        # 긍정 피드백 신호
        mock_feedback.detect.return_value = FeedbackSignal(
            signal_type=FeedbackType.MEMORY_CORRECT,
            reward_value=1.0,
            confidence=0.9,
            matched_pattern="test",
        )

        feedback_turn = Turn(turn_id=2, role=Role.USER, content="맞아요 잘 기억하시네요")
        signal, reward = agent.process_feedback(feedback_turn, [])

        # 업데이트 후 pending 상태가 해제되어야 함
        assert not reranker.has_pending_state
        assert reward == pytest.approx(1.0)


# ─── TestGraphFeatures ───────────────────────────────────────────────


class TestGraphFeatures:
    """Graph-aware feature tests."""

    def test_feature_dim_without_kg(self):
        """Without KG, feature dim is 8."""
        ext = ReRankFeatureExtractor()
        assert ext.feature_dim == 8

    def test_feature_dim_with_kg(self):
        """With KG, feature dim is 11."""
        from aimemory.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        ext = ReRankFeatureExtractor(kg=kg)
        assert ext.feature_dim == 11

    def test_graph_features_shape(self):
        """With KG, extract produces (n, 11) array."""
        from aimemory.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple("사용자", "좋아함", "Python", "mem_001")
        ext = ReRankFeatureExtractor(kg=kg)
        candidates = _make_candidates(3)
        features = ext.extract("Python 개발", candidates)
        assert features.shape == (3, 11)

    def test_reranker_with_kg(self):
        """ReRanker initializes correctly with KG."""
        from aimemory.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        reranker = ReRanker(kg=kg, enabled=True)
        candidates = _make_candidates(10)
        results = reranker.rerank("Python 개발", candidates)
        assert len(results) > 0

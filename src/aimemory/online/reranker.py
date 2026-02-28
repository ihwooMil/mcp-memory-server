"""RL Re-ranker: learns to re-order ChromaDB search results from user feedback.

ChromaDB top-K 검색 결과를 사용자 피드백으로 학습한 MLP 정책으로 리랭킹합니다.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aimemory.memory.graph_store import MemoryNode

if TYPE_CHECKING:
    from aimemory.memory.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

RERANK_FEATURE_DIM = 8
RERANK_GRAPH_FEATURE_DIM = 11

# 기술 키워드 패턴 (쿼리 인텐트 분류용)
_TECH_PATTERN = re.compile(
    r"(?<![가-힣a-zA-Z_])(?:"
    r"Python|Java(?:Script)?|TypeScript|Rust|Go|C\+\+|Ruby|Swift|Kotlin|Dart|"
    r"React|Vue|Angular|Next\.js|Django|Flask|FastAPI|Spring|Rails|"
    r"Docker|Kubernetes|k8s|AWS|GCP|Azure|Linux|Ubuntu|"
    r"MySQL|PostgreSQL|SQLite|Redis|MongoDB|"
    r"Git|GitHub|GitLab|CI/CD|DevOps|MLOps|"
    r"pandas|numpy|scipy|sklearn|TensorFlow|PyTorch|Keras|"
    r"LLM|GPT|Claude|Gemini|머신러닝|딥러닝|인공지능|AI|"
    r"API|REST|GraphQL|WebSocket|gRPC|"
    r"알고리즘|자료구조|데이터베이스|클라우드|마이크로서비스"
    r")(?![a-zA-Z_])",
    re.IGNORECASE,
)

# 감정/개인 키워드 패턴
_PERSONAL_PATTERN = re.compile(
    r"저는|제가|나는|내가|우리|가족|친구|학교|회사|취미|좋아|싫어|느낌|감정|기분|행복|슬픔|화남"
)

# 선호도 키워드 패턴
_PREFERENCE_PATTERN = re.compile(
    r"좋아|싫어|선호|취미|즐겨|주로|자주|항상|절대"
)


class ReRankFeatureExtractor:
    """후보 메모리 노드별 8차원 (또는 KG 사용 시 11차원) 특징 벡터를 추출합니다.

    Features (8-dim, always):
        0: chroma_similarity    - ChromaDB 코사인 유사도 (원본 순위 신호 보존)
        1: keyword_overlap      - 쿼리 키워드와 메모리 키워드의 겹치는 비율
        2: category_match       - 쿼리 인텐트와 메모리 카테고리 일치 여부
        3: recency              - 생성 이후 경과 시간에 대한 지수 감쇠
        4: access_frequency     - log1p(접근 횟수)
        5: content_length_ratio - 메모리 내용 길이 / 쿼리 길이 (상한 3.0)
        6: has_related          - 그래프 엣지(연관 메모리) 존재 여부
        7: resolution_available - 사용 가능한 해상도 레벨 수 (0-2 정규화)

    Additional features (indices 8-10, only when KG is provided):
        8: graph_connection_count - number of KG connections between query entities and this memory
        9: graph_hop_distance     - normalized distance (1.0=direct, 0.5=2-hop, 0.0=not found)
        10: has_negative_relation - 1.0 if a negative predicate exists for this memory's entities
    """

    _NEGATIVE_PREDICATES = frozenset({
        "싫어함", "부정", "반대", "혐오", "거부", "기피", "부담",
        "dislikes", "hates", "opposes", "avoids", "refuses",
    })

    # 카테고리별 시간 감쇠 계수 — 항구적 정보는 느리게, 일시적 정보는 빠르게
    _CATEGORY_DECAY: dict[str, float] = {
        "core_principle": 0.001,   # 반감기 ~693일: 거의 감쇠 없음
        "fact": 0.005,             # 반감기 ~139일
        "preference": 0.005,       # 반감기 ~139일
        "technical": 0.01,         # 반감기 ~69일
        "emotion": 0.03,           # 반감기 ~23일
        "experience": 0.02,        # 반감기 ~35일
    }

    def __init__(self, decay_lambda: float = 0.01, kg: KnowledgeGraph | None = None) -> None:
        self._decay_lambda = decay_lambda  # 기본 감쇠 계수 (카테고리 없을 때 사용)
        self._kg = kg

    @property
    def feature_dim(self) -> int:
        """Returns 8 if no KG is attached, 11 if KG is present."""
        return RERANK_GRAPH_FEATURE_DIM if self._kg is not None else RERANK_FEATURE_DIM

    def extract(
        self,
        query: str,
        candidates: list[MemoryNode],
        query_keywords: list[str] | None = None,
    ) -> np.ndarray:
        """모든 후보에 대한 특징 행렬을 추출합니다.

        Args:
            query: 검색 쿼리 텍스트.
            candidates: ChromaDB 검색 결과 MemoryNode 목록.
            query_keywords: 사전 추출된 쿼리 키워드 (없으면 자동 추출).

        Returns:
            shape (len(candidates), feature_dim)의 np.ndarray.
            KG가 없으면 (n, 8), KG가 있으면 (n, 11).
        """
        dim = self.feature_dim
        if not candidates:
            return np.zeros((0, dim), dtype=np.float32)

        if query_keywords is None:
            query_keywords = self._extract_query_keywords(query)

        features = np.zeros((len(candidates), dim), dtype=np.float32)
        for i, candidate in enumerate(candidates):
            features[i] = self._extract_single(query, candidate, query_keywords)

        return features

    def _extract_single(
        self,
        query: str,
        candidate: MemoryNode,
        query_keywords: list[str],
    ) -> np.ndarray:
        """단일 (쿼리, 후보) 쌍의 특징 벡터를 추출합니다."""
        feat = np.zeros(self.feature_dim, dtype=np.float32)

        # 0: chroma_similarity - 원본 ChromaDB 유사도 점수
        feat[0] = float(candidate.similarity_score or 0.0)

        # 1: keyword_overlap - 쿼리 키워드와 메모리 키워드 교집합 비율
        if query_keywords:
            mem_keywords_lower = {kw.lower() for kw in candidate.keywords}
            query_keywords_lower = {kw.lower() for kw in query_keywords}
            overlap = len(query_keywords_lower & mem_keywords_lower)
            feat[1] = overlap / max(len(query_keywords_lower), 1)
        else:
            feat[1] = 0.0

        # 2: category_match - 쿼리 인텐트와 메모리 카테고리 일치 여부
        inferred_category = self._infer_category(query)
        feat[2] = 1.0 if inferred_category == candidate.category else 0.0

        # 3: recency - 생성 시간으로부터 카테고리별 지수 감쇠
        category_lambda = self._CATEGORY_DECAY.get(candidate.category, self._decay_lambda)
        feat[3] = self._compute_recency(candidate.created_at, decay_lambda=category_lambda)

        # 4: access_frequency - 접근 횟수 log 스케일
        feat[4] = float(math.log1p(candidate.access_count))

        # 5: content_length_ratio - 메모리 내용 길이 / 쿼리 길이 (0~1 정규화)
        query_len = max(len(query), 1)
        ratio = len(candidate.content) / query_len
        feat[5] = min(ratio, 3.0) / 3.0

        # 6: has_related - 연관 메모리(그래프 엣지) 존재 여부
        feat[6] = 1.0 if candidate.related_ids else 0.0

        # 7: resolution_available - 사용 가능한 해상도 레벨 수 (0~3 → 0~1 정규화)
        resolution_count = sum([
            bool(candidate.content),
            bool(candidate.level1_text),
            bool(candidate.level2_text),
        ])
        feat[7] = resolution_count / 3.0

        # 8-10: KG 기반 그래프 피처 (KG가 있을 때만)
        if self._kg is not None:
            feat[8], feat[9], feat[10] = self._extract_graph_features(
                query_keywords, candidate
            )

        return feat

    def _extract_graph_features(
        self,
        query_keywords: list[str],
        candidate: MemoryNode,
    ) -> tuple[float, float, float]:
        """KG 기반 그래프 피처 3개를 추출합니다.

        Returns:
            (graph_connection_count, graph_hop_distance, has_negative_relation)
        """
        kg = self._kg
        assert kg is not None

        # 후보 메모리와 연결된 메모리 ID 집합 수집 (쿼리 엔티티 기준)
        connection_count = 0
        min_hop = None  # 1 = direct, 2 = 2-hop
        has_negative = False

        for keyword in query_keywords:
            # 쿼리 엔티티가 KG에 있는지 확인
            ctx = kg.get_entity_context(keyword)
            if not ctx["relations"]:
                continue

            for rel in ctx["relations"]:
                rel_memory_id = rel.get("memory_id", "")
                if rel_memory_id == candidate.memory_id:
                    connection_count += 1
                    if min_hop is None or min_hop > 1:
                        min_hop = 1

                # 부정적 관계 확인
                predicate = rel.get("predicate", "").lower()
                if predicate in self._NEGATIVE_PREDICATES:
                    # 이 관계가 candidate와 연결된 엔티티와 관련된 경우 체크
                    has_negative = True

            # 2-hop: 쿼리 키워드에서 2홉 이내 관계 중 candidate memory_id가 있는지
            if min_hop != 1:
                related = kg.get_related_entities(keyword, depth=2)
                for _subj, _pred, _obj in related:
                    mem_ids_subj = kg.get_memory_ids_for_entity(_subj)
                    mem_ids_obj = kg.get_memory_ids_for_entity(_obj)
                    if candidate.memory_id in (mem_ids_subj | mem_ids_obj):
                        connection_count += 1
                        if min_hop is None or min_hop > 2:
                            min_hop = 2

        # graph_hop_distance: 1.0=direct, 0.5=2-hop, 0.0=not found
        if min_hop == 1:
            graph_hop_distance = 1.0
        elif min_hop == 2:
            graph_hop_distance = 0.5
        else:
            graph_hop_distance = 0.0

        # graph_connection_count: 0~1 정규화 (상한 10개)
        normalized_connection_count = min(connection_count, 10) / 10.0

        return (
            float(normalized_connection_count),
            float(graph_hop_distance),
            float(has_negative),
        )

    def _compute_recency(self, created_at: str, decay_lambda: float | None = None) -> float:
        """생성 시간으로부터 경과 일수를 계산하여 지수 감쇠를 적용합니다."""
        if not created_at:
            return 0.5  # 생성 시간 정보 없을 때 중립값

        lam = decay_lambda if decay_lambda is not None else self._decay_lambda

        try:
            from datetime import datetime, timezone
            created_dt = datetime.fromisoformat(created_at)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_elapsed = max((now - created_dt).total_seconds() / 86400.0, 0.0)
            return float(math.exp(-lam * days_elapsed))
        except (ValueError, TypeError):
            return 0.5

    @staticmethod
    def _infer_category(query: str) -> str:
        """쿼리 텍스트에서 인텐트 카테고리를 추론합니다.

        Returns:
            'technical', 'preference', 'fact', 'experience', 'emotion' 중 하나.
        """
        if _TECH_PATTERN.search(query):
            return "technical"
        if _PREFERENCE_PATTERN.search(query):
            return "preference"
        if re.search(r"기분|감정|슬프|화나|행복|외로|불안|설레|걱정|힘들|무서", query):
            return "emotion"
        if re.search(r"경험|했어|해봤|갔었|먹었|봤어|여행|생활|일상", query):
            return "experience"
        return "fact"

    @staticmethod
    def _extract_query_keywords(query: str) -> list[str]:
        """쿼리에서 기술 키워드 + 따옴표 문자열을 추출합니다."""
        tech_matches = _TECH_PATTERN.findall(query)
        quoted_matches = re.findall(r"['\"]([^'\"]{2,30})['\"]", query)
        # 2글자 이상 한글/영문 단어
        word_matches = re.findall(r"[가-힣a-zA-Z]{2,}", query)

        # 중복 제거 후 반환
        all_keywords = list(dict.fromkeys(tech_matches + quoted_matches + word_matches))
        return all_keywords


# ─── MLP 모델 ────────────────────────────────────────────────────────


class _ReRankMLP(nn.Module):
    """포인트와이즈 점수화를 위한 소형 MLP."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── ReRankPolicy ────────────────────────────────────────────────────


class ReRankPolicy:
    """엡실론-그리디 탐색을 포함한 포인트와이즈 RL 리랭킹 정책.

    각 후보를 독립적으로 점수화하고, 점수 내림차순으로 top-k를 선택합니다.
    사용자 피드백으로부터 온라인 SGD 업데이트를 수행합니다.
    """

    def __init__(
        self,
        feature_dim: int = RERANK_FEATURE_DIM,
        hidden_dim: int = 32,
        lr: float = 0.005,
        epsilon: float = 0.15,
        select_k: int = 3,
    ) -> None:
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.select_k = select_k

        self._model = _ReRankMLP(feature_dim, hidden_dim)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._rng = np.random.default_rng()

    def rank(self, features_batch: np.ndarray) -> list[int]:
        """후보들을 점수화하고 top-k 인덱스를 내림차순으로 반환합니다.

        Args:
            features_batch: shape (n_candidates, feature_dim).

        Returns:
            top-k 후보 인덱스 목록 (점수 내림차순).
        """
        n = len(features_batch)
        if n == 0:
            return []

        k = min(self.select_k, n)

        # 엡실론-그리디 탐색
        if self._rng.random() < self.epsilon:
            indices = self._rng.choice(n, size=k, replace=False).tolist()
            return [int(i) for i in indices]

        with torch.no_grad():
            x = torch.from_numpy(features_batch).float()
            scores = self._model(x).squeeze(-1)  # shape (n,)
            top_k_indices = scores.argsort(descending=True)[:k].tolist()

        return [int(i) for i in top_k_indices]

    def update(
        self,
        features_batch: np.ndarray,
        selected_indices: list[int],
        reward: float,
    ) -> float:
        """관측된 피드백으로 정책을 업데이트합니다.

        선택된 후보는 관측된 보상을 타겟으로, 나머지는 0을 타겟으로 사용합니다.

        Args:
            features_batch: 리랭킹 호출 시의 특징 행렬.
            selected_indices: 선택된 후보 인덱스 목록.
            reward: FeedbackDetector에서 받은 보상값.

        Returns:
            손실값.
        """
        self._model.train()
        x = torch.from_numpy(features_batch).float()
        scores = self._model(x).squeeze(-1)

        # 타겟: 선택된 후보는 보상값, 나머지는 0
        targets = torch.zeros(len(scores))
        for idx in selected_indices:
            if 0 <= idx < len(targets):
                targets[idx] = reward

        loss = F.mse_loss(scores, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return float(loss.item())

    def get_parameters(self) -> np.ndarray:
        """평탄화된 모델 파라미터 반환 (gossip 프로토콜 호환)."""
        params = []
        for p in self._model.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_parameters(self, params: np.ndarray) -> None:
        """평탄화된 numpy 배열로 모델 파라미터를 설정합니다 (gossip 프로토콜 호환)."""
        offset = 0
        for p in self._model.parameters():
            numel = p.data.numel()
            chunk = params[offset: offset + numel]
            p.data.copy_(torch.from_numpy(chunk.reshape(p.data.shape)).float())
            offset += numel

    def save_checkpoint(self, path: str | Path) -> None:
        """모델 및 옵티마이저 상태를 파일로 저장합니다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "epsilon": self.epsilon,
                "feature_dim": self.feature_dim,
                "hidden_dim": self.hidden_dim,
                "select_k": self.select_k,
            },
            str(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """파일에서 모델 및 옵티마이저 상태를 로드합니다."""
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.select_k = checkpoint.get("select_k", self.select_k)


# ─── ReRanker (메인 파사드) ──────────────────────────────────────────


class ReRanker:
    """특징 추출 + 정책을 결합한 리랭킹 파사드 클래스.

    Usage:
        reranker = ReRanker()
        top3 = reranker.rerank(query, candidates)
        # ... 사용자 피드백 수신 ...
        reranker.update_from_feedback(reward)
    """

    def __init__(
        self,
        feature_extractor: Optional[ReRankFeatureExtractor] = None,
        policy: Optional[ReRankPolicy] = None,
        max_latency_ms: float = 20.0,
        enabled: bool = True,
        kg: Optional[KnowledgeGraph] = None,
    ) -> None:
        self._extractor = feature_extractor or ReRankFeatureExtractor(kg=kg)
        feat_dim = self._extractor.feature_dim
        self._policy = policy or ReRankPolicy(feature_dim=feat_dim)
        self.max_latency_ms = max_latency_ms
        self.enabled = enabled

        # 마지막 rerank() 호출 상태 저장 (update_from_feedback()에서 사용)
        self._last_features: Optional[np.ndarray] = None
        self._last_selected_indices: Optional[list[int]] = None

    def rerank(
        self,
        query: str,
        candidates: list[MemoryNode],
        query_keywords: list[str] | None = None,
    ) -> list[MemoryNode]:
        """후보를 리랭킹하여 top-k MemoryNode 목록을 반환합니다.

        비활성화 상태이거나 후보가 너무 적으면 ChromaDB 순서를 그대로 반환합니다.
        지연 시간이 예산을 초과하면 경고 후 ChromaDB 순서로 폴백합니다.
        후속 update_from_feedback() 호출을 위해 내부 상태를 저장합니다.

        Args:
            query: 검색 쿼리 텍스트.
            candidates: ChromaDB 검색 결과 (최대 10개).
            query_keywords: 사전 추출된 키워드 (선택 사항).

        Returns:
            학습된 정책으로 재정렬된 top-k MemoryNode 목록.
        """
        # 내부 상태 초기화
        self._last_features = None
        self._last_selected_indices = None

        select_k = self._policy.select_k

        # 비활성화 또는 후보 부족 시 폴백
        if not self.enabled or len(candidates) < select_k:
            return candidates[:select_k]

        start_time = time.perf_counter()

        # 특징 추출
        features = self._extractor.extract(query, candidates, query_keywords)

        # 정책으로 top-k 인덱스 선택
        selected_indices = self._policy.rank(features)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 지연 시간 초과 시 ChromaDB 순서로 폴백
        if elapsed_ms > self.max_latency_ms:
            logger.warning(
                "리랭킹 지연 시간 초과: %.2fms > %.2fms, ChromaDB 순서로 폴백합니다.",
                elapsed_ms,
                self.max_latency_ms,
            )
            return candidates[:select_k]

        # 내부 상태 저장 (피드백 업데이트용)
        self._last_features = features
        self._last_selected_indices = selected_indices

        # 선택된 인덱스 순서대로 MemoryNode 반환
        return [candidates[i] for i in selected_indices]

    def update_from_feedback(self, reward: float) -> Optional[float]:
        """저장된 상태를 사용하여 마지막 rerank() 호출에 대한 정책을 업데이트합니다.

        Args:
            reward: FeedbackDetector에서 받은 보상값.

        Returns:
            손실값, 또는 저장된 상태가 없으면 None.
        """
        if self._last_features is None or self._last_selected_indices is None:
            return None

        loss = self._policy.update(
            self._last_features,
            self._last_selected_indices,
            reward,
        )

        # 상태 초기화 (중복 업데이트 방지)
        self._last_features = None
        self._last_selected_indices = None

        return loss

    @property
    def has_pending_state(self) -> bool:
        """rerank()가 호출되었고 아직 update가 이루어지지 않은 경우 True."""
        return self._last_features is not None

    def save_checkpoint(self, path: str | Path) -> None:
        """정책 체크포인트를 저장합니다."""
        self._policy.save_checkpoint(path)

    def load_checkpoint(self, path: str | Path) -> None:
        """정책 체크포인트를 로드합니다."""
        self._policy.load_checkpoint(path)

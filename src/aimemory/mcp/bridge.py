"""MemoryBridge: singleton orchestrator wiring all AIMemory components for MCP tools."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

from aimemory.config import MCPServerConfig
from aimemory.live_graph.notify import notify_live_graph
from aimemory.memory.composer import ContextComposer
from aimemory.memory.graph_store import GraphMemoryStore, ImmutableMemoryError, MemoryNode
from aimemory.memory.sleep_cycle import SleepCycleRunner
from aimemory.online.policy import MemoryPolicyAgent, OnlinePolicy, StateEncoder
from aimemory.online.reranker import ReRanker
from aimemory.reward.feedback_detector import FeedbackDetector
from aimemory.schemas import MemoryActionType, Role, Turn

logger = logging.getLogger(__name__)


def _node_to_dict(node: MemoryNode, include_similarity: bool = True) -> dict[str, Any]:
    """Convert a MemoryNode to a JSON-serializable dict."""
    d: dict[str, Any] = {
        "memory_id": node.memory_id,
        "content": node.content,
        "keywords": node.keywords,
        "category": node.category,
        "related_ids": node.related_ids,
        "created_at": node.created_at,
        "immutable": node.immutable,
        "conversation_id": node.conversation_id,
        "access_count": node.access_count,
        "level1_text": node.level1_text,
        "level2_text": node.level2_text,
        "active": node.active,
        "pinned": node.pinned,
    }
    if include_similarity and node.similarity_score is not None:
        d["similarity"] = round(node.similarity_score, 4)
    return d


class MemoryBridge:
    """Singleton orchestrator that wires all AIMemory components for MCP tool handlers.

    Initializes and holds references to:
    - GraphMemoryStore (ChromaDB backend)
    - ContextComposer (multi-resolution context builder)
    - OnlinePolicy (MLP bandit)
    - MemoryPolicyAgent (policy + store integration)
    - FeedbackDetector (feedback signal detection)
    - SleepCycleRunner (periodic maintenance)
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        token_budget: int | None = None,
        top_k: int | None = None,
        policy_checkpoint: str | None = None,
        config: MCPServerConfig | None = None,
        use_enhanced_policy: bool | None = None,
        use_graph_rag: bool | None = None,
    ) -> None:
        cfg = config or MCPServerConfig()

        # Environment variable overrides
        self._persist_directory = (
            persist_directory or os.environ.get("AIMEMORY_DB_PATH") or cfg.persist_directory
        )
        self._collection_name = (
            collection_name or os.environ.get("AIMEMORY_COLLECTION") or cfg.collection_name
        )
        self._embedding_model = (
            embedding_model or os.environ.get("AIMEMORY_EMBEDDING_MODEL") or cfg.embedding_model
        )
        self._token_budget = token_budget or int(
            os.environ.get("AIMEMORY_TOKEN_BUDGET", cfg.token_budget)
        )
        self._top_k = top_k or cfg.top_k
        self._reranker_pool_size = int(
            os.environ.get("AIMEMORY_RERANKER_POOL", cfg.reranker_pool_size)
        )
        self._min_relevance = float(os.environ.get("AIMEMORY_MIN_RELEVANCE", cfg.min_relevance))
        self._policy_checkpoint = (
            policy_checkpoint
            or os.environ.get("AIMEMORY_POLICY_CHECKPOINT")
            or cfg.policy_checkpoint
        )

        # Enhanced/GraphRAG mode flags (param > env var > config)
        self._use_enhanced = (
            use_enhanced_policy
            if use_enhanced_policy is not None
            else (os.environ.get("AIMEMORY_ENHANCED_POLICY") == "1")
        ) or cfg.use_enhanced_policy
        self._use_graph_rag = (
            use_graph_rag
            if use_graph_rag is not None
            else (os.environ.get("AIMEMORY_GRAPH_RAG") == "1")
        ) or cfg.use_graph_rag

        logger.info(
            "Initializing MemoryBridge: db=%s, collection=%s, enhanced=%s, graphrag=%s",
            self._persist_directory,
            self._collection_name,
            self._use_enhanced,
            self._use_graph_rag,
        )

        # KnowledgeGraph (optional)
        self._kg = None
        if self._use_graph_rag:
            from aimemory.memory.knowledge_graph import KnowledgeGraph

            self._kg = KnowledgeGraph()

        # Initialize core components
        self._store = GraphMemoryStore(
            persist_directory=self._persist_directory,
            collection_name=self._collection_name,
            embedding_model=self._embedding_model,
            knowledge_graph=self._kg,
        )

        # Policy: enhanced or standard
        if self._use_enhanced:
            from aimemory.online.autonomy import ProgressiveAutonomy
            from aimemory.online.enhanced_encoder import EnhancedStateEncoder
            from aimemory.online.enhanced_policy import EnhancedOnlinePolicy
            from aimemory.online.replay_buffer import ReplayBuffer

            encoder = EnhancedStateEncoder()
            encoder.set_embedding_fn(self._store._embedding_fn)
            replay = ReplayBuffer()
            autonomy = ProgressiveAutonomy()
            self._policy = EnhancedOnlinePolicy(encoder, replay, autonomy)
        else:
            self._policy = OnlinePolicy()

        if self._policy_checkpoint:
            try:
                self._policy.load_checkpoint(self._policy_checkpoint)
                logger.info("Loaded policy checkpoint from %s", self._policy_checkpoint)
            except Exception as exc:
                logger.warning("Failed to load policy checkpoint: %s", exc)

        self._feedback_detector = FeedbackDetector()

        # ReRanker with optional graph features
        self._reranker = ReRanker(enabled=True, kg=self._kg if self._use_graph_rag else None)

        # 최근 검색 기록 — 같은 기억이 반복 검색되는 것을 방지
        self._recent_retrievals: dict[str, int] = {}
        self._retrieval_turn_count: int = 0

        # GraphRetriever (optional)
        self._retriever = None
        if self._kg is not None:
            from aimemory.memory.graph_retriever import GraphRetriever

            self._retriever = GraphRetriever(self._store, self._kg)

        self._agent = MemoryPolicyAgent(
            graph_store=self._store,
            policy=self._policy,
            feedback_detector=self._feedback_detector,
            encoder=StateEncoder(),
            reranker=self._reranker,
        )

        self._sleep_runner = SleepCycleRunner(
            store=self._store,
            policy=self._policy,
        )

        # Track recent policy actions for status reporting
        self._recent_actions: dict[str, int] = {
            "save": 0,
            "skip": 0,
            "retrieve": 0,
        }
        self._total_policy_updates = 0

    # ── Properties ────────────────────────────────────────────────

    @property
    def store(self) -> GraphMemoryStore:
        return self._store

    @property
    def composer(self) -> ContextComposer:
        return ContextComposer(token_budget=self._token_budget, top_k=self._top_k)

    @property
    def policy(self) -> OnlinePolicy:
        return self._policy

    @property
    def agent(self) -> MemoryPolicyAgent:
        return self._agent

    # ── High-level operations ─────────────────────────────────────

    def save_memory(
        self,
        content: str,
        keywords: list[str] | None = None,
        category: str = "fact",
        related_ids: list[str] | None = None,
        immutable: bool = False,
        pinned: bool = False,
    ) -> dict[str, Any]:
        """Save a new memory to the graph store. Returns a dict with memory info."""
        memory_id = self._store.add_memory(
            content=content,
            keywords=keywords,
            category=category,
            related_ids=related_ids,
            immutable=immutable,
            pinned=pinned,
        )

        # Fetch auto-linked related_ids from the stored memory
        stored = self._store._collection.get(ids=[memory_id], include=["metadatas"])
        auto_related: list[str] = []
        if stored["ids"]:
            from aimemory.memory.graph_store import _parse_csv

            auto_related = _parse_csv(stored["metadatas"][0].get("related_ids", ""))

        logger.info(
            "Saved memory %s (category=%s, auto_linked=%d)",
            memory_id, category, len(auto_related),
        )

        # Emit live graph event
        from aimemory.visualize import CATEGORY_COLORS

        notify_live_graph({
            "type": "node_add",
            "node": {
                "id": memory_id,
                "content": content,
                "label": content[:30] + ("…" if len(content) > 30 else ""),
                "category": category,
                "color": CATEGORY_COLORS.get(category, "#999999"),
                "keywords": keywords or [],
                "access_count": 0,
                "pinned": pinned,
                "created_at": "",
                "related_ids": auto_related,
            },
        })

        return {
            "memory_id": memory_id,
            "content": content,
            "keywords": keywords or [],
            "category": category,
            "related_ids": auto_related,
        }

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories by semantic similarity. Returns list of node dicts."""
        nodes = self._store.search(
            query_text=query,
            top_k=top_k,
            category_filter=category,
        )
        result = [_node_to_dict(n) for n in nodes]

        # Emit live graph event
        if result:
            notify_live_graph({
                "type": "search_highlight",
                "query": query,
                "results": [
                    {"memory_id": r["memory_id"], "similarity": r.get("similarity")}
                    for r in result
                ],
            })

        return result

    def auto_search(
        self,
        user_message: str,
        token_budget: int | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Search for relevant memories and compose a context string.

        Returns dict with context string, memory count, token count, and details.
        """
        import random

        budget = token_budget or self._token_budget
        effective_top_k = top_k or self._top_k

        # 10턴마다 최근 검색 기록 리셋 (세션 내 지속적 감점 방지)
        self._retrieval_turn_count += 1
        if self._retrieval_turn_count >= 10:
            self._recent_retrievals.clear()
            self._retrieval_turn_count = 0

        # Phase 1: 넓은 후보풀 — top_k의 3배 또는 reranker_pool_size 중 큰 값
        pool_size = max(
            self._reranker_pool_size if self._reranker.enabled else effective_top_k,
            effective_top_k * 3,
        )
        if self._retriever is not None:
            results = self._retriever.retrieve(
                user_message,
                top_k=pool_size,
                final_k=pool_size,
                track_access=False,
            )
        else:
            results = self._store.search(user_message, top_k=pool_size, track_access=False)

        if not results:
            return {
                "context": "",
                "memories_used": 0,
                "total_tokens": 0,
                "details": [],
            }

        # Phase 2: ReRanker — 점수 리스코어링 (자르지 않고 전체 결과 유지)
        if self._reranker.enabled:
            features = self._reranker._extractor.extract(user_message, results)
            if len(features) > 0:
                import torch

                with torch.no_grad():
                    x = torch.from_numpy(features).float()
                    scores = self._reranker._policy._model(x).squeeze(-1).numpy()
                # RL 점수를 similarity_score에 블렌딩 (원본 70% + RL 30%)
                for i, node in enumerate(results):
                    original = node.similarity_score or 0.0
                    rl_score = float(scores[i]) if i < len(scores) else 0.0
                    # RL score를 0~1로 정규화
                    rl_normalized = 1.0 / (1.0 + math.exp(-rl_score))
                    node.similarity_score = 0.7 * original + 0.3 * rl_normalized

        # Phase 3: 최근 검색 감점 — 같은 기억이 반복 검색되지 않도록
        for node in results:
            if node.similarity_score is not None:
                mid = node.memory_id
                if mid in self._recent_retrievals:
                    count = self._recent_retrievals[mid]
                    # 최근 검색 횟수에 따라 감점 (1회: -5%, 2회: -10%, ...)
                    penalty = min(count * 0.05, 0.25)  # 최대 25% 감점
                    node.similarity_score *= 1.0 - penalty

        # Phase 4: 노이즈 — 검색 결과에 약간의 변동성 부여
        for node in results:
            if node.similarity_score is not None:
                noise = random.gauss(0, 0.02)  # σ=0.02 (±4% 범위 내)
                node.similarity_score = max(0.0, node.similarity_score + noise)

        # Phase 5: 안전장치 — relevance 최소 기준 적용
        results = [
            r
            for r in results
            if r.similarity_score is not None and r.similarity_score >= self._min_relevance
        ]

        if not results:
            return {
                "context": "",
                "memories_used": 0,
                "total_tokens": 0,
                "details": [],
            }

        # Phase 6: Composer (MMR) — 다양성 보장하며 top_k개 선택
        composer = ContextComposer(token_budget=budget, top_k=effective_top_k)
        composed = composer.compose(results)
        context_str = composer.format_context(composed)

        # 최근 검색 기록 업데이트
        for cm in composed:
            self._recent_retrievals[cm.memory_id] = self._recent_retrievals.get(cm.memory_id, 0) + 1

        details = [
            {
                "memory_id": cm.memory_id,
                "level": cm.level,
                "relevance": round(cm.relevance, 4),
                "tokens": cm.tokens,
            }
            for cm in composed
        ]

        return {
            "context": context_str,
            "memories_used": len(composed),
            "total_tokens": sum(cm.tokens for cm in composed),
            "details": details,
        }

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update an existing memory. Returns success status."""
        try:
            success = self._store.update_memory(
                memory_id=memory_id,
                content=content,
                keywords=keywords,
            )
            if not success:
                return {"success": False, "error": f"Memory not found: {memory_id}"}
            return {"success": True, "memory_id": memory_id}
        except ImmutableMemoryError as exc:
            return {"success": False, "error": str(exc)}

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory node. Returns success status."""
        try:
            success = self._store.delete_memory(memory_id=memory_id)
            if not success:
                return {"success": False, "error": f"Memory not found: {memory_id}"}
            logger.info("Deleted memory %s", memory_id)
            notify_live_graph({"type": "node_remove", "memory_id": memory_id})
            return {"success": True, "memory_id": memory_id}
        except ImmutableMemoryError as exc:
            return {"success": False, "error": str(exc)}

    def get_related(
        self,
        memory_id: str,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Get related memories via BFS graph traversal."""
        nodes = self._store.get_related(memory_id=memory_id, depth=depth)
        return [_node_to_dict(n, include_similarity=False) for n in nodes]

    def pin_memory(self, memory_id: str) -> dict[str, Any]:
        """Pin a memory to protect it from forgetting."""
        success = self._store.pin_memory(memory_id=memory_id)
        if not success:
            return {"success": False, "error": f"Memory not found: {memory_id}"}
        return {"success": True, "memory_id": memory_id, "pinned": True}

    def unpin_memory(self, memory_id: str) -> dict[str, Any]:
        """Remove pin protection from a memory."""
        success = self._store.unpin_memory(memory_id=memory_id)
        if not success:
            return {"success": False, "error": f"Memory not found: {memory_id}"}
        return {"success": True, "memory_id": memory_id, "pinned": False}

    def get_stats(self) -> dict[str, Any]:
        """Return memory store statistics."""
        return self._store.get_stats()

    def run_sleep_cycle(self) -> dict[str, Any]:
        """Run the memory sleep cycle and return the report as a dict."""
        logger.info("Starting sleep cycle...")
        report = self._sleep_runner.run()
        logger.info("Sleep cycle complete: %s", report.summary())
        return report.to_dict()

    def get_policy_status(self) -> dict[str, Any]:
        """Return current policy status."""
        return {
            "epsilon": self._policy.epsilon,
            "recent_actions": dict(self._recent_actions),
            "total_updates": self._total_policy_updates,
        }

    def policy_decide(
        self,
        user_message: str,
        turn_id: int = 0,
    ) -> dict[str, Any]:
        """Ask the RL policy to decide what memory action to take.

        Returns the action (SAVE/SKIP/RETRIEVE) with reasoning and any
        resulting memory entries or retrieved memories.
        """
        turn = Turn(turn_id=turn_id, role=Role.USER, content=user_message)
        decision = self._agent.decide(turn=turn, recent_turns=[], turn_position=0.0)

        action_str = (
            decision.action.value if hasattr(decision.action, "value") else str(decision.action)
        )
        result: dict[str, Any] = {
            "action": action_str,
            "turn_id": turn_id,
            "reasoning": decision.reasoning or "",
        }

        # Update action counter
        action_key = action_str.lower()
        if action_key in self._recent_actions:
            self._recent_actions[action_key] += 1

        if decision.action == MemoryActionType.SAVE and decision.memory_entry:
            entry = decision.memory_entry
            result["memory_entry"] = {
                "memory_id": entry.memory_id,
                "content": entry.content,
                "keywords": entry.keywords,
                "category": entry.category,
            }

        if decision.action == MemoryActionType.RETRIEVE and decision.retrieved_memories:
            result["retrieved_memories"] = [
                {
                    "memory_id": m.memory_id,
                    "content": m.content,
                    "keywords": m.keywords,
                    "category": m.category,
                }
                for m in decision.retrieved_memories
            ]

        return result

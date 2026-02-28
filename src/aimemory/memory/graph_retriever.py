"""Hybrid retrieval: ChromaDB vector search + KnowledgeGraph traversal."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from aimemory.i18n import get_patterns

if TYPE_CHECKING:
    from aimemory.memory.graph_store import GraphMemoryStore, MemoryNode
    from aimemory.memory.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Hybrid retrieval combining vector similarity and graph traversal.

    1. ChromaDB vector search → top candidates with cosine similarity
    2. KG entity extraction from query → graph traversal → related memory_ids
    3. Score fusion: vector_score * weight + graph_score * weight
    4. Deduplicate and rank
    """

    def __init__(
        self,
        store: GraphMemoryStore,
        kg: KnowledgeGraph,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        lang: str = "ko",
    ) -> None:
        self._store = store
        self._kg = kg
        self._vector_weight = vector_weight
        self._graph_weight = graph_weight
        self._lp = get_patterns(lang)
        self._word_re = re.compile(self._lp.word_extraction_pattern)

    def retrieve(self, query: str, top_k: int = 20, final_k: int = 5) -> list[MemoryNode]:
        """Hybrid retrieval combining vector and graph signals."""
        # 1. Vector search
        vector_results = self._store.search(query, top_k=top_k)

        # Build score map: memory_id → score components
        score_map: dict[str, dict] = {}
        for node in vector_results:
            score_map[node.memory_id] = {
                "node": node,
                "vector_score": node.similarity_score or 0.0,
                "graph_score": 0.0,
            }

        # 2. Graph traversal
        entities = self.extract_entities(query)
        graph_memory_ids: set[str] = set()

        for entity in entities:
            related_ids = self._kg.get_memory_ids_for_entity(entity)
            graph_memory_ids.update(related_ids)

        # Add graph-found memories that weren't in vector results
        if graph_memory_ids:
            missing_ids = graph_memory_ids - score_map.keys()
            if missing_ids:
                all_mems = self._store.get_all_memories(include_inactive=False)
                node_map = {n.memory_id: n for n in all_mems}
                for mid in missing_ids:
                    if mid in node_map:
                        score_map[mid] = {
                            "node": node_map[mid],
                            "vector_score": 0.0,
                            "graph_score": 0.0,
                        }

        # Calculate graph scores
        for mid, entry in score_map.items():
            graph_score = 0.0

            for entity in entities:
                mids = self._kg.get_memory_ids_for_entity(entity)
                if mid in mids:
                    graph_score += 0.5  # Base connection bonus

                    # Check for negative relations
                    context = self._kg.get_entity_context(entity)
                    for rel in context.get("relations", []):
                        pred = rel.get("predicate", "")
                        if rel.get("memory_id") == mid and self._is_negative_predicate(pred):
                            graph_score += 0.3  # Still valuable for negative context

            entry["graph_score"] = min(graph_score, 1.0)

        # 3. Score fusion
        ranked = []
        for mid, entry in score_map.items():
            final_score = (
                entry["vector_score"] * self._vector_weight
                + entry["graph_score"] * self._graph_weight
            )
            entry["final_score"] = final_score
            ranked.append(entry)

        # Sort by final score descending
        ranked.sort(key=lambda x: x["final_score"], reverse=True)

        # 4. Return top results with updated similarity scores
        results = []
        for entry in ranked[:final_k]:
            node = entry["node"]
            node.similarity_score = entry["final_score"]
            results.append(node)

        return results

    def extract_entities(self, text: str) -> list[str]:
        """Extract entities from query text for graph lookup."""
        entities: list[str] = []

        # Language-specific word extraction
        words = self._word_re.findall(text)
        entities.extend(words)

        # English proper nouns / tech terms (always included)
        tech_words = re.findall(
            r"(?<![a-zA-Z])(?:Python|Java|React|Django|Docker|Rust|TypeScript|Go|"
            r"FastAPI|Flask|Spring|Rails|Vue|Angular|Next\.js|"
            r"MySQL|PostgreSQL|Redis|MongoDB|AWS|GCP|Azure|"
            r"PyTorch|TensorFlow|Keras|Claude|GPT)(?![a-zA-Z])",
            text, re.IGNORECASE,
        )
        entities.extend(tech_words)

        # Deduplicate preserving order
        return list(dict.fromkeys(entities))

    def _is_negative_predicate(self, predicate: str) -> bool:
        """Check if a predicate indicates a negative relation."""
        return any(neg in predicate for neg in self._lp.negative_predicates)

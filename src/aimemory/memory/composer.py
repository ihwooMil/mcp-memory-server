"""Context composer: selects optimal resolution level per memory within a token budget.

Uses MMR (Maximal Marginal Relevance) to balance relevance and diversity,
then greedily packs memories choosing the highest resolution level that fits.
"""

from __future__ import annotations

from dataclasses import dataclass

from aimemory.memory.graph_store import MemoryNode
from aimemory.memory.resolution import estimate_tokens


@dataclass
class ComposedMemory:
    """A memory selected for context with its chosen resolution level."""

    memory_id: str
    text: str
    level: int  # 0=full, 1=summary, 2=triple
    relevance: float
    tokens: int


def _keyword_similarity(a: MemoryNode, b: MemoryNode) -> float:
    """Compute Jaccard similarity between two memories' keyword sets."""
    kw_a = {k.lower() for k in a.keywords} if a.keywords else set()
    kw_b = {k.lower() for k in b.keywords} if b.keywords else set()
    if not kw_a and not kw_b:
        # Fallback: category match
        return 1.0 if a.category == b.category else 0.0
    if not kw_a or not kw_b:
        return 0.0
    return len(kw_a & kw_b) / len(kw_a | kw_b)


class ContextComposer:
    """Composes context from memories within a token budget.

    Strategy (MMR):
    1. Iteratively pick the memory that maximises:
       MMR = λ * relevance - (1-λ) * max_sim(selected)
    2. For each picked memory, try Level 0 first, then L1, then L2.
    3. Stop when budget is exhausted or top_k memories are selected.

    λ=1.0 behaves like pure relevance (greedy). λ=0.5 is maximum diversity.
    Default λ=0.7 balances relevance with diversity.
    """

    def __init__(
        self,
        token_budget: int = 1024,
        top_k: int = 10,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.token_budget = token_budget
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda

    def compose(self, memories: list[MemoryNode]) -> list[ComposedMemory]:
        """Select memories with MMR diversity and resolution levels within the token budget."""
        if not memories:
            return []

        composed: list[ComposedMemory] = []
        remaining_budget = self.token_budget

        # Candidate pool (indices into memories list)
        candidates = list(range(len(memories)))
        selected_nodes: list[MemoryNode] = []

        while candidates and len(composed) < self.top_k and remaining_budget > 0:
            best_idx = -1
            best_mmr = float("-inf")

            for idx in candidates:
                node = memories[idx]
                relevance = node.similarity_score if node.similarity_score is not None else 0.0

                # Max similarity to already-selected memories
                if selected_nodes:
                    max_sim = max(_keyword_similarity(node, sel) for sel in selected_nodes)
                else:
                    max_sim = 0.0

                mmr = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx < 0:
                break

            candidates.remove(best_idx)
            node = memories[best_idx]
            relevance = node.similarity_score if node.similarity_score is not None else 0.0

            # Try levels from highest resolution (0) to lowest (2)
            level_candidates = self._get_level_candidates(node)
            selected = None

            for level, text, tokens in level_candidates:
                if tokens <= remaining_budget:
                    selected = ComposedMemory(
                        memory_id=node.memory_id,
                        text=text,
                        level=level,
                        relevance=relevance,
                        tokens=tokens,
                    )
                    break

            if selected is not None:
                composed.append(selected)
                selected_nodes.append(node)
                remaining_budget -= selected.tokens

        return composed

    def format_context(self, composed: list[ComposedMemory]) -> str:
        """Format composed memories into a context string.

        Uses [L0], [L1], [L2] tags to indicate resolution level.
        """
        if not composed:
            return ""

        level_tags = {0: "[L0]", 1: "[L1]", 2: "[L2]"}
        parts: list[str] = []

        for cm in composed:
            tag = level_tags.get(cm.level, "[L0]")
            parts.append(f"{tag} {cm.text}")

        return "\n".join(parts)

    @staticmethod
    def _get_level_candidates(
        node: MemoryNode,
    ) -> list[tuple[int, str, int]]:
        """Get (level, text, tokens) candidates from highest to lowest resolution."""
        candidates: list[tuple[int, str, int]] = []

        # Level 0: full text
        if node.content:
            candidates.append((0, node.content, estimate_tokens(node.content)))

        # Level 1: summary
        if node.level1_text:
            candidates.append((1, node.level1_text, estimate_tokens(node.level1_text)))

        # Level 2: entity triple
        if node.level2_text:
            candidates.append((2, node.level2_text, estimate_tokens(node.level2_text)))

        return candidates

"""Context composer: selects optimal resolution level per memory within a token budget.

Uses a greedy algorithm to pack the most relevant memories into a fixed
token budget, choosing the highest resolution level that fits.
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


class ContextComposer:
    """Composes context from memories within a token budget.

    Strategy:
    1. Sort memories by relevance (similarity_score) descending
    2. For each memory, try Level 0 first, then Level 1, then Level 2
    3. Pick the highest resolution that fits within remaining budget
    4. Stop when budget is exhausted or top_k memories are selected
    """

    def __init__(self, token_budget: int = 1024, top_k: int = 10) -> None:
        self.token_budget = token_budget
        self.top_k = top_k

    def compose(self, memories: list[MemoryNode]) -> list[ComposedMemory]:
        """Select memories and resolution levels within the token budget.

        Args:
            memories: List of MemoryNode, typically from search results
                     (should have similarity_score set).

        Returns:
            List of ComposedMemory in relevance order.
        """
        # Sort by relevance descending
        sorted_memories = sorted(
            memories,
            key=lambda m: m.similarity_score if m.similarity_score is not None else 0.0,
            reverse=True,
        )

        composed: list[ComposedMemory] = []
        remaining_budget = self.token_budget

        for node in sorted_memories[: self.top_k]:
            relevance = node.similarity_score if node.similarity_score is not None else 0.0

            # Try levels from highest resolution (0) to lowest (2)
            candidates = self._get_level_candidates(node)
            selected = None

            for level, text, tokens in candidates:
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
                remaining_budget -= selected.tokens

            if remaining_budget <= 0:
                break

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

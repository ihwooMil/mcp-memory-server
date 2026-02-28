"""Memory consolidation: duplicate detection and merging.

Finds semantically similar memories and merges them to reduce redundancy.
The memory with higher access_count survives; keywords and related_ids
are merged as a union; the longer content is kept.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from aimemory.memory.graph_store import GraphMemoryStore, MemoryNode

logger = logging.getLogger(__name__)


@dataclass
class MergeRecord:
    """Audit record for a single merge operation."""

    surviving_id: str
    absorbed_ids: list[str]
    original_contents: dict[str, str]  # memory_id -> content before merge
    merged_at: str = ""

    def __post_init__(self) -> None:
        if not self.merged_at:
            self.merged_at = datetime.now().isoformat()


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    pairs_found: int = 0
    memories_merged: int = 0
    merge_records: list[MergeRecord] = field(default_factory=list)


class MemoryConsolidator:
    """Detects and merges duplicate memories based on semantic similarity.

    Uses the store's search() method to find similar memories above a
    threshold, then merges pairs by keeping the higher-access-count memory
    and absorbing the other.
    """

    def __init__(
        self,
        store: GraphMemoryStore,
        similarity_threshold: float = 0.92,
        max_pairs_per_run: int = 50,
    ) -> None:
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.max_pairs_per_run = max_pairs_per_run

    def find_duplicates(self) -> list[tuple[MemoryNode, MemoryNode]]:
        """Find pairs of active memories with similarity >= threshold.

        Immutable memories are excluded from being absorbed (but can be keepers).
        Returns a list of (keeper, absorbed) tuples.
        """
        all_memories = self.store.get_all_memories(include_inactive=False)
        seen_pairs: set[tuple[str, str]] = set()
        duplicates: list[tuple[MemoryNode, MemoryNode]] = []

        for node in all_memories:
            if len(duplicates) >= self.max_pairs_per_run:
                break

            # Search for similar memories using this node's content
            similar = self.store.search(node.content, top_k=5)

            for candidate in similar:
                if candidate.memory_id == node.memory_id:
                    continue

                sim = candidate.similarity_score
                if sim is None or sim < self.similarity_threshold:
                    continue

                # Normalize pair ordering to avoid duplicates
                pair_key = tuple(sorted([node.memory_id, candidate.memory_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Determine keeper vs absorbed
                keeper, absorbed = self._decide_keeper(node, candidate)

                # Cannot absorb immutable memories
                if absorbed.immutable:
                    continue

                duplicates.append((keeper, absorbed))

                if len(duplicates) >= self.max_pairs_per_run:
                    break

        return duplicates

    def merge_pair(self, keeper: MemoryNode, absorbed: MemoryNode) -> MergeRecord:
        """Merge absorbed memory into keeper.

        - Keeps the longer content
        - Unions keywords and related_ids
        - Sums access_counts
        - Deletes the absorbed memory
        """
        original_contents = {
            keeper.memory_id: keeper.content,
            absorbed.memory_id: absorbed.content,
        }

        # Choose longer content
        merged_content = (
            keeper.content
            if len(keeper.content) >= len(absorbed.content)
            else absorbed.content
        )

        # Union keywords
        merged_keywords = list(dict.fromkeys(keeper.keywords + absorbed.keywords))

        # Union related_ids (exclude both self-references)
        excluded = {keeper.memory_id, absorbed.memory_id}
        merged_related = list(dict.fromkeys(
            rid for rid in keeper.related_ids + absorbed.related_ids
            if rid not in excluded
        ))

        # Update keeper with merged data
        self.store.update_memory(
            keeper.memory_id,
            content=merged_content,
            keywords=merged_keywords,
        )

        # Update access_count and related_ids in metadata
        existing = self.store._collection.get(
            ids=[keeper.memory_id], include=["metadatas"]
        )
        if existing["ids"]:
            meta = existing["metadatas"][0]
            meta["access_count"] = keeper.access_count + absorbed.access_count
            meta["related_ids"] = ",".join(merged_related)
            self.store._collection.update(ids=[keeper.memory_id], metadatas=[meta])

        # Delete absorbed memory
        self.store.delete_memory(absorbed.memory_id)

        record = MergeRecord(
            surviving_id=keeper.memory_id,
            absorbed_ids=[absorbed.memory_id],
            original_contents=original_contents,
        )
        logger.info(
            "Merged memory %s into %s (sim >= %.2f)",
            absorbed.memory_id,
            keeper.memory_id,
            self.similarity_threshold,
        )
        return record

    def run(self) -> ConsolidationResult:
        """Run full consolidation: find duplicates then merge."""
        duplicates = self.find_duplicates()
        result = ConsolidationResult(pairs_found=len(duplicates))

        for keeper, absorbed in duplicates:
            try:
                record = self.merge_pair(keeper, absorbed)
                result.merge_records.append(record)
                result.memories_merged += 1
            except Exception:
                logger.exception(
                    "Failed to merge %s into %s",
                    absorbed.memory_id,
                    keeper.memory_id,
                )

        return result

    @staticmethod
    def _decide_keeper(a: MemoryNode, b: MemoryNode) -> tuple[MemoryNode, MemoryNode]:
        """Decide which memory survives. Higher access_count wins; ties broken by content length."""
        if a.access_count > b.access_count:
            return a, b
        if b.access_count > a.access_count:
            return b, a
        # Tie-break: longer content survives
        if len(a.content) >= len(b.content):
            return a, b
        return b, a

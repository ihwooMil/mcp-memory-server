"""Forgetting pipeline: importance-based memory decay, compression, and deactivation.

Implements a three-stage forgetting process:
1. Compress: Low-importance memories are compressed to Level 2 (entity triple)
2. Deactivate: Very low-importance memories are marked inactive (excluded from search)
3. Delete: Long-inactive memories are permanently removed with audit logging
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

from aimemory.memory.graph_store import GraphMemoryStore, MemoryNode

logger = logging.getLogger(__name__)


@dataclass
class ForgettingThresholds:
    """Thresholds for the forgetting pipeline."""

    decay_lambda: float = 0.05
    threshold_compress: float = 0.3
    threshold_deactivate: float = 0.1
    deactivation_days: int = 30
    related_boost: float = 0.1


@dataclass
class AuditEntry:
    """Audit log entry for a permanently deleted memory."""

    memory_id: str
    content_preview: str
    deleted_at: str
    reason: str


class ImportanceCalculator:
    """Calculates importance scores for memory nodes."""

    def __init__(self, decay_lambda: float = 0.05, related_boost: float = 0.1) -> None:
        self.decay_lambda = decay_lambda
        self.related_boost = related_boost

    def recency_decay(self, created_at: str) -> float:
        """Calculate recency decay: e^(-lambda * days_since_creation).

        Returns 1.0 for very recent memories, approaching 0 for old ones.
        """
        if not created_at:
            return 0.0
        try:
            created = datetime.fromisoformat(created_at)
            now = datetime.now()
            days = (now - created).total_seconds() / 86400.0
            return math.exp(-self.decay_lambda * max(0, days))
        except (ValueError, TypeError):
            return 0.0

    def compute(self, node: MemoryNode) -> float:
        """Compute overall importance score for a memory node.

        Formula: (1 + access_count) * recency_decay + related_boost * len(related_ids)
        """
        decay = self.recency_decay(node.created_at)
        access_factor = 1 + node.access_count
        relation_bonus = self.related_boost * len(node.related_ids)
        return access_factor * decay + relation_bonus


@dataclass
class ForgettingResult:
    """Result of a forgetting pipeline run."""

    compressed: int = 0
    deactivated: int = 0
    deleted: int = 0
    skipped_pinned: int = 0
    skipped_immutable: int = 0
    audit_log: list[AuditEntry] = field(default_factory=list)


class ForgettingPipeline:
    """Runs the forgetting pipeline on a memory store."""

    def __init__(
        self,
        store: GraphMemoryStore,
        calculator: ImportanceCalculator | None = None,
        thresholds: ForgettingThresholds | None = None,
    ) -> None:
        self.store = store
        self.thresholds = thresholds or ForgettingThresholds()
        self.calculator = calculator or ImportanceCalculator(
            decay_lambda=self.thresholds.decay_lambda,
            related_boost=self.thresholds.related_boost,
        )

    def run(self) -> ForgettingResult:
        """Execute the forgetting pipeline.

        Returns a ForgettingResult with counts of compressed, deactivated,
        and deleted memories plus an audit log.
        """
        result = ForgettingResult()

        # Get all memories including inactive for deletion check
        all_memories = self.store.get_all_memories(include_inactive=True)

        for node in all_memories:
            # Skip protected memories
            if node.pinned:
                result.skipped_pinned += 1
                continue
            if node.immutable:
                result.skipped_immutable += 1
                continue

            importance = self.calculator.compute(node)

            if node.active:
                # Stage 1: Compress low-importance active memories
                if importance < self.thresholds.threshold_compress:
                    if importance < self.thresholds.threshold_deactivate:
                        # Stage 2: Deactivate very low-importance memories
                        self.store.deactivate_memory(node.memory_id)
                        result.deactivated += 1
                        logger.info(
                            "Deactivated memory %s (importance=%.3f)",
                            node.memory_id,
                            importance,
                        )
                    else:
                        # Compress to Level 2
                        level2 = node.level2_text or f"{','.join(node.keywords[:3])}"
                        self.store.compress_memory(node.memory_id, level2)
                        result.compressed += 1
                        logger.info(
                            "Compressed memory %s (importance=%.3f)",
                            node.memory_id,
                            importance,
                        )
            else:
                # Stage 3: Delete long-inactive memories
                if self._days_since_creation(node) > self.thresholds.deactivation_days:
                    content_preview = node.content[:50]
                    self.store.delete_memory(node.memory_id)
                    result.deleted += 1
                    result.audit_log.append(
                        AuditEntry(
                            memory_id=node.memory_id,
                            content_preview=content_preview,
                            deleted_at=datetime.now().isoformat(),
                            reason=f"Inactive for >{self.thresholds.deactivation_days} days",
                        )
                    )
                    logger.info(
                        "Permanently deleted memory %s (inactive >%d days)",
                        node.memory_id,
                        self.thresholds.deactivation_days,
                    )

        return result

    @staticmethod
    def _days_since_creation(node: MemoryNode) -> float:
        """Calculate days since the memory was created."""
        if not node.created_at:
            return 0.0
        try:
            created = datetime.fromisoformat(node.created_at)
            return (datetime.now() - created).total_seconds() / 86400.0
        except (ValueError, TypeError):
            return 0.0

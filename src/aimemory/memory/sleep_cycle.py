"""Sleep cycle runner: periodic memory maintenance orchestrator.

Runs four sequential tasks:
1. Memory consolidation (duplicate detection & merging)
2. Multi-resolution regeneration (fill missing level1/level2 texts)
3. Forgetting pipeline (decay, compress, deactivate, delete)
4. RL checkpoint saving
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from aimemory.config import SleepCycleConfig
from aimemory.memory.consolidation import (
    ConsolidationResult,
    MemoryConsolidator,
)
from aimemory.memory.forgetting import (
    ForgettingPipeline,
    ForgettingResult,
    ForgettingThresholds,
    ImportanceCalculator,
)
from aimemory.memory.graph_store import GraphMemoryStore
from aimemory.memory.resolution import generate_all_levels

logger = logging.getLogger(__name__)


@dataclass
class SleepCycleReport:
    """Report from a complete sleep cycle run."""

    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    memory_count_before: int = 0
    memory_count_after: int = 0
    consolidation: ConsolidationResult | None = None
    resolution_regenerated: int = 0
    forgetting: ForgettingResult | None = None
    checkpoint_saved: bool = False
    checkpoint_path: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a JSON-compatible dictionary."""
        result: dict[str, Any] = {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "memory_count_before": self.memory_count_before,
            "memory_count_after": self.memory_count_after,
            "resolution_regenerated": self.resolution_regenerated,
            "checkpoint_saved": self.checkpoint_saved,
            "checkpoint_path": self.checkpoint_path,
            "errors": self.errors,
        }

        if self.consolidation is not None:
            result["consolidation"] = {
                "pairs_found": self.consolidation.pairs_found,
                "memories_merged": self.consolidation.memories_merged,
                "merge_records": [
                    {
                        "surviving_id": r.surviving_id,
                        "absorbed_ids": r.absorbed_ids,
                        "merged_at": r.merged_at,
                    }
                    for r in self.consolidation.merge_records
                ],
            }

        if self.forgetting is not None:
            result["forgetting"] = {
                "compressed": self.forgetting.compressed,
                "deactivated": self.forgetting.deactivated,
                "deleted": self.forgetting.deleted,
                "skipped_pinned": self.forgetting.skipped_pinned,
                "skipped_immutable": self.forgetting.skipped_immutable,
                "audit_log": [
                    {
                        "memory_id": e.memory_id,
                        "content_preview": e.content_preview,
                        "deleted_at": e.deleted_at,
                        "reason": e.reason,
                    }
                    for e in self.forgetting.audit_log
                ],
            }

        return result

    def summary(self) -> str:
        """Human-readable summary of the sleep cycle."""
        lines = [
            f"Sleep Cycle Report ({self.started_at})",
            f"  Duration: {self.duration_seconds:.1f}s",
            f"  Memories: {self.memory_count_before} → {self.memory_count_after}",
        ]

        if self.consolidation is not None:
            lines.append(
                f"  Consolidation: {self.consolidation.pairs_found} pairs found, "
                f"{self.consolidation.memories_merged} merged"
            )

        if self.resolution_regenerated > 0:
            lines.append(f"  Resolution regenerated: {self.resolution_regenerated}")

        if self.forgetting is not None:
            lines.append(
                f"  Forgetting: {self.forgetting.compressed} compressed, "
                f"{self.forgetting.deactivated} deactivated, "
                f"{self.forgetting.deleted} deleted"
            )

        if self.checkpoint_saved:
            lines.append(f"  Checkpoint: {self.checkpoint_path}")

        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for err in self.errors:
                lines.append(f"    - {err}")

        return "\n".join(lines)


class SleepCycleRunner:
    """Orchestrates periodic memory maintenance tasks.

    Each task is isolated with try/except so that a failure in one
    does not prevent the others from running.
    """

    def __init__(
        self,
        store: GraphMemoryStore,
        config: SleepCycleConfig | None = None,
        policy: Any = None,
    ) -> None:
        self.store = store
        self.config = config or SleepCycleConfig()
        self.policy = policy

    def run(self) -> SleepCycleReport:
        """Execute the full sleep cycle."""
        report = SleepCycleReport()
        report.started_at = datetime.now().isoformat()
        start_time = time.time()

        # Count memories before
        report.memory_count_before = len(
            self.store.get_all_memories(include_inactive=True)
        )

        # Task 1: Memory consolidation
        if self.config.enable_consolidation:
            try:
                consolidator = MemoryConsolidator(
                    store=self.store,
                    similarity_threshold=self.config.consolidation_threshold,
                    max_pairs_per_run=self.config.max_consolidation_pairs,
                )
                report.consolidation = consolidator.run()
                logger.info(
                    "Consolidation complete: %d merged",
                    report.consolidation.memories_merged,
                )
            except Exception as exc:
                msg = f"Consolidation failed: {exc}"
                report.errors.append(msg)
                logger.exception(msg)

        # Task 2: Multi-resolution regeneration
        if self.config.enable_resolution_regen:
            try:
                report.resolution_regenerated = self._regenerate_resolutions()
                logger.info(
                    "Resolution regeneration: %d memories updated",
                    report.resolution_regenerated,
                )
            except Exception as exc:
                msg = f"Resolution regeneration failed: {exc}"
                report.errors.append(msg)
                logger.exception(msg)

        # Task 3: Forgetting pipeline
        if self.config.enable_forgetting:
            try:
                thresholds = ForgettingThresholds(
                    decay_lambda=self.config.forgetting_decay_lambda,
                    threshold_compress=self.config.forgetting_threshold_compress,
                    threshold_deactivate=self.config.forgetting_threshold_deactivate,
                    deactivation_days=self.config.forgetting_deactivation_days,
                    related_boost=self.config.forgetting_related_boost,
                )
                calculator = ImportanceCalculator(
                    decay_lambda=thresholds.decay_lambda,
                    related_boost=thresholds.related_boost,
                )
                pipeline = ForgettingPipeline(
                    store=self.store,
                    calculator=calculator,
                    thresholds=thresholds,
                )
                report.forgetting = pipeline.run()
                logger.info(
                    "Forgetting complete: %d compressed, %d deactivated, %d deleted",
                    report.forgetting.compressed,
                    report.forgetting.deactivated,
                    report.forgetting.deleted,
                )
            except Exception as exc:
                msg = f"Forgetting failed: {exc}"
                report.errors.append(msg)
                logger.exception(msg)

        # Task 4: RL checkpoint
        if self.config.enable_checkpoint and self.policy is not None:
            try:
                checkpoint_dir = Path(self.config.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = checkpoint_dir / f"policy_{timestamp}.pt"
                self.policy.save_checkpoint(checkpoint_path)
                report.checkpoint_saved = True
                report.checkpoint_path = str(checkpoint_path)
                logger.info("Checkpoint saved: %s", checkpoint_path)
            except Exception as exc:
                msg = f"Checkpoint save failed: {exc}"
                report.errors.append(msg)
                logger.exception(msg)

        # Count memories after
        report.memory_count_after = len(
            self.store.get_all_memories(include_inactive=True)
        )

        report.finished_at = datetime.now().isoformat()
        report.duration_seconds = time.time() - start_time

        return report

    def save_report(self, report: SleepCycleReport, output_dir: str | Path) -> Path:
        """Save a JSON report to the output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"sleep_cycle_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Report saved: %s", path)
        return path

    def _regenerate_resolutions(self) -> int:
        """Regenerate level1/level2 text for memories missing them."""
        all_memories = self.store.get_all_memories(include_inactive=False)
        regenerated = 0

        for node in all_memories:
            if node.level1_text and node.level2_text:
                continue

            levels = generate_all_levels(node.content, node.keywords)

            # Update metadata with generated levels
            existing = self.store._collection.get(
                ids=[node.memory_id], include=["metadatas"]
            )
            if not existing["ids"]:
                continue

            meta = existing["metadatas"][0]
            if not node.level1_text:
                meta["level1_text"] = levels.level1
            if not node.level2_text:
                meta["level2_text"] = levels.level2
            self.store._collection.update(ids=[node.memory_id], metadatas=[meta])
            regenerated += 1

        return regenerated

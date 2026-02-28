#!/usr/bin/env python3
"""Script 03: Compute heuristic rewards on raw episodes.

Reads JSONL episode files, computes reward signals for each memory decision,
and writes enriched JSONL files with reward annotations.

Usage:
    uv run scripts/03_compute_rewards.py [OPTIONS]

Options:
    --input-dir PATH    Directory with raw episode JSONL files (default: data/raw/episodes)
    --output-dir PATH   Directory for reward-annotated JSONL files (default: data/raw/episodes)
    --in-place          Overwrite input files with reward annotations
    --workers INT       Number of parallel workers (default: 1)
    --dry-run           Validate setup without computing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aimemory.config import AppConfig, DataPaths, RewardConfig
from aimemory.schemas import Episode, RewardBreakdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("03_compute_rewards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute heuristic rewards on raw episode JSONL files"
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_reward_calculator(reward_config: RewardConfig):
    """Load the real reward calculator if available, otherwise use mock."""
    try:
        from aimemory.reward.calculator import RewardCalculator  # type: ignore[import]

        logger.info("Using real RewardCalculator from aimemory.reward.calculator")
        return RewardCalculator(reward_config)
    except ImportError:
        logger.warning("aimemory.reward.calculator not available; using mock reward calculator")
        return MockRewardCalculator(reward_config)


class MockRewardCalculator:
    """Placeholder reward calculator used when the real module is not yet available.

    Computes simple heuristic rewards based on keyword reappearance and efficiency.
    Replace with the real RewardCalculator once aimemory.reward is implemented.
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def compute_episode_rewards(self, episode: Episode) -> dict[int, RewardBreakdown]:
        """Compute a RewardBreakdown for each decision turn in the episode.

        Returns:
            Mapping of turn_id → RewardBreakdown.
        """
        # Collect all keywords mentioned across turns for reappearance check
        all_turn_text = " ".join(t.content for t in episode.turns).lower()

        rewards: dict[int, RewardBreakdown] = {}
        saved_keywords: list[str] = []

        for decision in sorted(episode.memory_decisions, key=lambda d: d.turn_id):
            rb = RewardBreakdown()

            # R1: keyword reappearance
            if saved_keywords:
                reappearance_count = sum(1 for kw in saved_keywords if kw.lower() in all_turn_text)
                rb.r1_keyword_reappearance = min(1.0, reappearance_count * 0.1)

            # R3: efficiency (shorter saved content = better)
            if decision.memory_entry is not None and decision.memory_entry.content:
                content_len = len(decision.memory_entry.content.split())
                rb.r3_efficiency = max(0.0, 1.0 - content_len / 50.0)

            # R4: retrieval relevance (mock: reward retrieval actions)
            from aimemory.schemas import MemoryActionType

            if decision.action == MemoryActionType.RETRIEVE:
                rb.r4_retrieval_relevance = 0.5 * len(decision.retrieved_memories)

            # R7: info density (mock: non-empty saves score higher)
            if decision.memory_entry is not None:
                rb.r7_info_density = 0.3

            # Update saved keywords
            if decision.memory_entry is not None:
                saved_keywords.extend(decision.memory_entry.keywords)

            # Compute weighted total
            rb.compute_total(self.config.weights)
            rewards[decision.turn_id] = rb

        return rewards


def compute_rewards_for_file(
    input_path: Path,
    output_path: Path,
    calculator,
) -> tuple[int, int]:
    """Process one JSONL file, computing rewards for each episode.

    Returns:
        Tuple of (episodes_processed, errors).
    """
    processed = 0
    errors = 0

    with (
        open(input_path, encoding="utf-8") as in_f,
        open(output_path, "w", encoding="utf-8") as out_f,
    ):
        for line_num, line in enumerate(in_f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                episode = Episode.model_validate(data)

                # Compute rewards
                reward_map = calculator.compute_episode_rewards(episode)

                # Annotate episode data with reward breakdown per decision
                reward_annotations: dict[str, Any] = {}
                for turn_id, rb in reward_map.items():
                    reward_annotations[str(turn_id)] = rb.model_dump()

                data["_reward_annotations"] = reward_annotations
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                processed += 1

            except Exception as exc:
                logger.error("Error on line %d of %s: %s", line_num, input_path, exc)
                errors += 1

    return processed, errors


def main() -> None:
    args = parse_args()
    config = AppConfig()
    paths = DataPaths()

    input_dir: Path = args.input_dir or paths.raw_episodes
    output_dir: Path = args.output_dir or paths.raw_episodes

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    jsonl_files = sorted(input_dir.glob("episodes_*.jsonl"))
    if not jsonl_files:
        logger.warning("No episode JSONL files found in %s", input_dir)
        return

    logger.info("Found %d JSONL files to process", len(jsonl_files))

    if args.dry_run:
        logger.info("[DRY RUN] Would process: %s", [f.name for f in jsonl_files])
        return

    calculator = load_reward_calculator(config.reward)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_errors = 0

    for jsonl_file in jsonl_files:
        if args.in_place:
            # Write to a temp file first, then replace the original
            tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".jsonl", dir=str(jsonl_file.parent))
            os.close(tmp_fd)
            tmp_path = Path(tmp_path_str)
            out_path = tmp_path
        else:
            out_path = output_dir / jsonl_file.name.replace(".jsonl", "_rewarded.jsonl")
            tmp_path = None

        logger.info("Processing %s → %s", jsonl_file.name, out_path.name)
        n_processed, n_errors = compute_rewards_for_file(jsonl_file, out_path, calculator)
        total_processed += n_processed
        total_errors += n_errors

        if tmp_path is not None and n_processed > 0:
            shutil.move(str(tmp_path), str(jsonl_file))
        elif tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

        logger.info(
            "  %s: %d episodes processed, %d errors",
            jsonl_file.name,
            n_processed,
            n_errors,
        )

    logger.info(
        "Done: %d total episodes processed, %d errors",
        total_processed,
        total_errors,
    )


if __name__ == "__main__":
    main()

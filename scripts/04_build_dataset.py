#!/usr/bin/env python3
"""Script 04: Build Parquet dataset with train/val/test splits.

Reads reward-annotated JSONL episode files, converts to SARTriples,
splits into train/val/test, and saves as Parquet files.

Usage:
    uv run scripts/04_build_dataset.py [OPTIONS]

Options:
    --input-dir PATH    Directory with episode JSONL files (default: data/raw/episodes)
    --output-dir PATH   Directory for Parquet split files (default: data/splits)
    --seed INT          Random seed (default: 42)
    --no-rewards        Skip reward annotations (use zero rewards)
    --dry-run           Validate setup without building
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyarrow as pa
import pyarrow.parquet as pq

from aimemory.config import AppConfig, DataPaths
from aimemory.dataset.builder import EpisodeBuilder
from aimemory.dataset.splitter import EpisodeSplitter
from aimemory.schemas import Episode, RewardBreakdown, SARTriple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("04_build_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Parquet dataset from episode JSONL files"
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rewards", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_episodes_from_dir(input_dir: Path, use_rewards: bool = True) -> tuple[list[Episode], dict[str, dict[int, RewardBreakdown]]]:
    """Load all episodes and their reward annotations from JSONL files.

    Returns:
        Tuple of (episodes, reward_maps) where reward_maps maps
        episode_id → {turn_id → RewardBreakdown}.
    """
    episodes: list[Episode] = []
    reward_maps: dict[str, dict[int, RewardBreakdown]] = {}

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    logger.info("Loading from %d JSONL files in %s", len(jsonl_files), input_dir)

    for jsonl_file in jsonl_files:
        with open(jsonl_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    episode = Episode.model_validate(data)
                    episodes.append(episode)

                    # Extract reward annotations if present
                    if use_rewards and "_reward_annotations" in data:
                        ep_rewards: dict[int, RewardBreakdown] = {}
                        for turn_id_str, rb_data in data["_reward_annotations"].items():
                            rb = RewardBreakdown.model_validate(rb_data)
                            ep_rewards[int(turn_id_str)] = rb
                        if ep_rewards:
                            reward_maps[episode.episode_id] = ep_rewards

                except Exception as exc:
                    logger.error(
                        "Parse error line %d in %s: %s", line_num, jsonl_file.name, exc
                    )

    logger.info(
        "Loaded %d episodes, %d with reward annotations",
        len(episodes),
        len(reward_maps),
    )
    return episodes, reward_maps


def rows_to_parquet(rows: list[dict], output_path: Path) -> None:
    """Write a list of dicts to a Parquet file using PyArrow."""
    if not rows:
        logger.warning("No rows to write to %s", output_path)
        return

    # Build PyArrow table from rows
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Wrote %d rows to %s", len(rows), output_path)


def main() -> None:
    args = parse_args()
    config = AppConfig()
    paths = DataPaths()

    input_dir: Path = args.input_dir or paths.raw_episodes
    output_dir: Path = args.output_dir or paths.splits

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    if args.dry_run:
        logger.info("[DRY RUN] Would build dataset from %s → %s", input_dir, output_dir)
        return

    # Load episodes
    episodes, reward_maps = load_episodes_from_dir(
        input_dir, use_rewards=not args.no_rewards
    )

    if not episodes:
        logger.error("No episodes found in %s", input_dir)
        sys.exit(1)

    # Build SARTriples
    builder = EpisodeBuilder(config.dataset)
    logger.info("Building SARTriples from %d episodes...", len(episodes))
    triples = builder.build_from_episodes(episodes, reward_maps=reward_maps)
    logger.info("Built %d SARTriples total", len(triples))

    # Split
    dataset_config = config.dataset
    dataset_config.random_seed = args.seed
    splitter = EpisodeSplitter(dataset_config)
    split_result = splitter.split(episodes, triples)
    logger.info("Split summary: %s", split_result.summary())

    # Convert to Parquet rows
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_triples in [
        ("train", split_result.train),
        ("val", split_result.val),
        ("test", split_result.test),
    ]:
        if not split_triples:
            logger.warning("Split '%s' has no triples, skipping", split_name)
            continue

        rows = builder.triples_to_parquet_rows(split_triples)
        out_path = output_dir / f"{split_name}.parquet"
        rows_to_parquet(rows, out_path)

    # Write split metadata
    meta = {
        "total_episodes": len(episodes),
        "total_triples": len(triples),
        "split_summary": split_result.summary(),
        "config": {
            "context_window": config.dataset.context_window,
            "train_ratio": config.dataset.train_ratio,
            "val_ratio": config.dataset.val_ratio,
            "test_ratio": config.dataset.test_ratio,
            "random_seed": args.seed,
        },
    }
    meta_path = output_dir / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Wrote split metadata to %s", meta_path)
    logger.info("Dataset build complete.")


if __name__ == "__main__":
    main()

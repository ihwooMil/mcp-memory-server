#!/usr/bin/env python3
"""Script 02: Generate synthetic episodes via self-play simulation.

Usage:
    uv run scripts/02_generate_episodes.py [OPTIONS]

Options:
    --num-episodes INT      Number of episodes to generate (default: 1000)
    --output-dir PATH       Output directory for JSONL files (default: data/raw/episodes)
    --checkpoint-interval INT  Flush every N episodes (default: 10)
    --resume                Resume from latest checkpoint
    --model STR             Ollama model name (default: from config)
    --seed INT              Random seed (default: 42)
    --dry-run               Validate setup without generating
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aimemory.config import AppConfig, DataPaths
from aimemory.selfplay.engine import SelfPlayEngine
from aimemory.selfplay.llm_client import is_korean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("02_generate_episodes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic self-play episodes"
    )
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_existing_episodes(output_dir: Path) -> int:
    """Count episodes already generated in JSONL files."""
    count = 0
    for jsonl_file in output_dir.glob("*.jsonl"):
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as exc:
            logger.warning("Could not read %s: %s", jsonl_file, exc)
    return count


def main() -> None:
    args = parse_args()
    config = AppConfig()

    if args.model:
        config.ollama.model = args.model

    # Resolve output directory
    paths = DataPaths()
    output_dir: Path = args.output_dir or paths.raw_episodes
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: count existing episodes
    existing = 0
    if args.resume:
        existing = count_existing_episodes(output_dir)
        logger.info("Resuming: found %d existing episodes", existing)

    remaining = args.num_episodes - existing
    if remaining <= 0:
        logger.info("Already have %d episodes (target: %d). Nothing to do.", existing, args.num_episodes)
        return

    logger.info(
        "Generating %d episodes (model: %s, seed: %d)",
        remaining,
        config.ollama.model,
        args.seed,
    )

    if args.dry_run:
        logger.info("[DRY RUN] Setup looks good. Exiting without generating.")
        return

    # Create engine ONCE and reuse for all episodes
    engine = SelfPlayEngine(config=config, seed=args.seed)
    topic_plan = engine.scenario_mgr.round_robin_topics(remaining)

    # Verify Ollama is available
    if not engine.user_client.is_available():
        logger.error("Ollama server not reachable at %s", config.ollama.base_url)
        sys.exit(1)
    logger.info("Ollama connection OK (model: %s)", config.ollama.model)

    # Open output file for appending
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"episodes_{timestamp}.jsonl"
    logger.info("Writing episodes to %s", out_path)

    try:
        from tqdm import tqdm
        progress = tqdm(total=remaining, desc="Episodes", unit="ep")
    except ImportError:
        progress = None
        logger.info("tqdm not available; install for progress bar")

    generated = 0
    total_korean = 0
    total_turns = 0
    total_saves = 0
    total_retrieves = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as out_f:
        for ep_num in range(existing, args.num_episodes):
            ep_start = time.time()
            try:
                topic_idx = ep_num - existing
                topic = topic_plan[topic_idx] if topic_idx < len(topic_plan) else engine.scenario_mgr.random_topic()
                scenario = topic.scenario_type
                episode = engine.run_episode(scenario, episode_index=ep_num, topic=topic)

                out_f.write(episode.model_dump_json() + "\n")
                generated += 1

                # Stats
                n_turns = len(episode.turns)
                korean = sum(1 for t in episode.turns if is_korean_text(t.content))
                saves = episode.num_saves
                retrieves = episode.num_retrieves
                total_turns += n_turns
                total_korean += korean
                total_saves += saves
                total_retrieves += retrieves

                ep_time = time.time() - ep_start
                topic = episode.metadata.get("topic_name", "?")

                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        kr=f"{korean}/{n_turns}",
                        saves=saves,
                        time=f"{ep_time:.0f}s",
                        topic=topic[:8],
                    )
                elif generated % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = generated / elapsed * 3600 if elapsed > 0 else 0
                    logger.info(
                        "Progress: %d/%d episodes (%.0f ep/hr) | Korean: %d/%d (%.0f%%) | Saves: %d",
                        generated, remaining, rate,
                        total_korean, total_turns,
                        total_korean / total_turns * 100 if total_turns else 0,
                        total_saves,
                    )

                # Checkpoint flush
                if generated % args.checkpoint_interval == 0:
                    out_f.flush()

            except KeyboardInterrupt:
                logger.info("Interrupted at episode %d. Progress saved.", ep_num)
                out_f.flush()
                break
            except Exception as exc:
                logger.error("Episode %d failed: %s", ep_num, exc)
                continue

    if progress is not None:
        progress.close()

    elapsed = time.time() - start_time
    rate = generated / elapsed * 3600 if elapsed > 0 else 0

    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info("  Episodes: %d generated in %.1f min (%.0f ep/hr)", generated, elapsed / 60, rate)
    logger.info("  Korean: %d/%d turns (%.1f%%)",
                total_korean, total_turns,
                total_korean / total_turns * 100 if total_turns else 0)
    logger.info("  Memory: %d saves, %d retrieves", total_saves, total_retrieves)
    logger.info("  Output: %s", out_path)


if __name__ == "__main__":
    main()

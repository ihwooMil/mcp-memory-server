#!/usr/bin/env python3
"""Script 07: Run the sleep cycle (periodic memory maintenance).

Usage:
    uv run scripts/07_sleep_cycle.py [OPTIONS]

Options:
    --db-path PATH           ChromaDB persistence directory (default: ./memory_db)
    --checkpoint-path PATH   Policy checkpoint to load (optional)
    --report-dir PATH        Output directory for reports (default: data/reports/sleep_cycle)
    --no-consolidation       Skip memory consolidation
    --no-forgetting          Skip forgetting pipeline
    --no-resolution          Skip multi-resolution regeneration
    --no-checkpoint          Skip RL checkpoint saving
    --dry-run                Validate setup without running
    --verbose                Enable debug logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aimemory.config import AppConfig, SleepCycleConfig
from aimemory.memory.graph_store import GraphMemoryStore
from aimemory.memory.sleep_cycle import SleepCycleRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("07_sleep_cycle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sleep cycle (periodic memory maintenance)")
    parser.add_argument("--db-path", type=str, default="./memory_db")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--report-dir",
        type=str,
        default="data/reports/sleep_cycle",
    )
    parser.add_argument("--no-consolidation", action="store_true")
    parser.add_argument("--no-forgetting", action="store_true")
    parser.add_argument("--no-resolution", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = AppConfig()
    sleep_config = SleepCycleConfig(
        enable_consolidation=not args.no_consolidation,
        enable_resolution_regen=not args.no_resolution,
        enable_forgetting=not args.no_forgetting,
        enable_checkpoint=not args.no_checkpoint,
        report_dir=args.report_dir,
    )

    logger.info("Sleep cycle configuration:")
    logger.info("  DB path: %s", args.db_path)
    logger.info("  Consolidation: %s", sleep_config.enable_consolidation)
    logger.info("  Resolution regen: %s", sleep_config.enable_resolution_regen)
    logger.info("  Forgetting: %s", sleep_config.enable_forgetting)
    logger.info("  Checkpoint: %s", sleep_config.enable_checkpoint)

    if args.dry_run:
        logger.info("[DRY RUN] Setup looks good. Exiting without running.")
        return

    # Initialize store
    store = GraphMemoryStore(persist_directory=args.db_path)
    stats = store.get_stats()
    logger.info("Memory store: %d total memories", stats["total"])

    # Optionally load policy for checkpointing
    policy = None
    if args.checkpoint_path and not args.no_checkpoint:
        try:
            from aimemory.online.policy import OnlinePolicy

            policy = OnlinePolicy(
                feature_dim=config.online_policy.feature_dim,
                n_actions=config.online_policy.n_actions,
                lr=config.online_policy.lr,
                epsilon=config.online_policy.epsilon,
            )
            policy.load_checkpoint(args.checkpoint_path)
            logger.info("Loaded policy checkpoint: %s", args.checkpoint_path)
        except Exception as exc:
            logger.warning("Could not load policy: %s", exc)

    # Run sleep cycle
    runner = SleepCycleRunner(store=store, config=sleep_config, policy=policy)
    report = runner.run()

    # Save report
    report_path = runner.save_report(report, sleep_config.report_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info(report.summary())
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()

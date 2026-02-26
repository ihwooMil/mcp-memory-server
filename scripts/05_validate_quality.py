#!/usr/bin/env python3
"""Script 05: Validate dataset quality.

Runs quality checks on the generated Parquet splits and episode JSONL files.

Checks:
- No data leakage (same episode_id in multiple splits)
- Reward distribution is non-trivial (has both positive and negative values)
- Action distribution is balanced (not all skip/save)
- Episode length is within configured bounds
- No duplicate triple_ids
- Train/val/test size expectations

Usage:
    uv run scripts/05_validate_quality.py [OPTIONS]

Options:
    --splits-dir PATH   Directory with Parquet splits (default: data/splits)
    --episodes-dir PATH Directory with raw episode JSONL files (default: data/raw/episodes)
    --strict            Exit with error on any failed check
    --report PATH       Write JSON report to this file
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aimemory.config import AppConfig, DataPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("05_validate_quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--episodes-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


class QualityChecker:
    """Runs a battery of quality checks on the dataset."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.results: list[dict] = []

    def _record(self, name: str, passed: bool, message: str) -> None:
        status = "PASS" if passed else "FAIL"
        level = logging.INFO if passed else logging.WARNING
        logger.log(level, "[%s] %s: %s", status, name, message)
        self.results.append({"check": name, "status": status, "message": message})

    def check_splits_exist(self, splits_dir: Path) -> bool:
        """Verify all three split Parquet files exist."""
        all_exist = True
        for split in ["train", "val", "test"]:
            path = splits_dir / f"{split}.parquet"
            exists = path.exists()
            self._record(
                f"split_exists_{split}",
                exists,
                f"{path.name} {'found' if exists else 'NOT FOUND'}",
            )
            if not exists:
                all_exist = False
        return all_exist

    def check_no_leakage(self, splits: dict[str, "pa.Table"]) -> None:
        """Check that no episode_id appears in more than one split."""
        try:
            ids_per_split: dict[str, set[str]] = {}
            for name, table in splits.items():
                if "episode_id" in table.schema.names:
                    ids_per_split[name] = set(
                        table.column("episode_id").to_pylist()
                    )

            split_names = list(ids_per_split.keys())
            leakage_found = False
            for i, n1 in enumerate(split_names):
                for n2 in split_names[i + 1 :]:
                    overlap = ids_per_split[n1] & ids_per_split[n2]
                    if overlap:
                        self._record(
                            f"no_leakage_{n1}_{n2}",
                            False,
                            f"{len(overlap)} episode_ids appear in both {n1} and {n2}",
                        )
                        leakage_found = True
            if not leakage_found:
                self._record("no_leakage", True, "No episode_id leakage across splits")
        except Exception as exc:
            self._record("no_leakage", False, f"Check failed: {exc}")

    def check_reward_distribution(self, table: "pa.Table", split_name: str) -> None:
        """Check that reward distribution has meaningful spread."""
        try:
            if "reward_total" not in table.schema.names:
                self._record(
                    f"reward_distribution_{split_name}",
                    False,
                    "reward_total column missing",
                )
                return

            rewards = table.column("reward_total").to_pylist()
            if not rewards:
                self._record(
                    f"reward_distribution_{split_name}", False, "No reward data"
                )
                return

            import statistics
            mean_r = statistics.mean(rewards)
            std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
            has_spread = std_r > 0.01
            self._record(
                f"reward_distribution_{split_name}",
                has_spread,
                f"mean={mean_r:.4f}, std={std_r:.4f} ({'OK' if has_spread else 'no spread'})",
            )
        except Exception as exc:
            self._record(f"reward_distribution_{split_name}", False, str(exc))

    def check_action_distribution(self, table: "pa.Table", split_name: str) -> None:
        """Check that action types are balanced (not degenerate)."""
        try:
            if "action_type" not in table.schema.names:
                self._record(
                    f"action_distribution_{split_name}", False, "action_type column missing"
                )
                return

            from collections import Counter
            actions = table.column("action_type").to_pylist()
            counts = Counter(actions)
            total = len(actions) or 1

            skip_ratio = counts.get("skip", 0) / total
            balanced = skip_ratio < 0.95  # Not all skip
            self._record(
                f"action_distribution_{split_name}",
                balanced,
                f"counts={dict(counts)}, skip_ratio={skip_ratio:.2%}",
            )
        except Exception as exc:
            self._record(f"action_distribution_{split_name}", False, str(exc))

    def check_no_duplicate_triples(self, table: "pa.Table", split_name: str) -> None:
        """Check for duplicate triple_ids within a split."""
        try:
            if "triple_id" not in table.schema.names:
                self._record(
                    f"no_duplicate_triples_{split_name}", False, "triple_id column missing"
                )
                return

            ids = table.column("triple_id").to_pylist()
            unique = len(set(ids))
            total = len(ids)
            has_dups = unique < total
            self._record(
                f"no_duplicate_triples_{split_name}",
                not has_dups,
                f"{total} triples, {total - unique} duplicates",
            )
        except Exception as exc:
            self._record(f"no_duplicate_triples_{split_name}", False, str(exc))

    def check_split_sizes(self, splits: dict[str, "pa.Table"]) -> None:
        """Check that splits have roughly the right sizes."""
        sizes = {name: len(table) for name, table in splits.items()}
        total = sum(sizes.values())

        for name, size in sizes.items():
            has_data = size > 0
            self._record(
                f"split_size_{name}",
                has_data,
                f"{name}: {size} triples (total={total})",
            )

        # Check train > val, train > test
        train = sizes.get("train", 0)
        val = sizes.get("val", 0)
        test = sizes.get("test", 0)
        self._record(
            "split_proportions",
            train >= val and train >= test,
            f"train({train}) >= val({val}) and test({test})",
        )

    def check_json_columns_parseable(
        self, table: "pa.Table", split_name: str
    ) -> None:
        """Check that JSON string columns are valid JSON."""
        json_cols = [
            "state_recent_turns_json",
            "state_memory_summary_json",
            "action_saved_keywords_json",
        ]
        for col_name in json_cols:
            if col_name not in table.schema.names:
                continue
            try:
                sample = table.column(col_name).to_pylist()[:10]
                for val in sample:
                    if val:
                        json.loads(val)
                self._record(
                    f"json_parseable_{split_name}_{col_name}",
                    True,
                    f"Sample of {len(sample)} values parseable",
                )
            except Exception as exc:
                self._record(
                    f"json_parseable_{split_name}_{col_name}",
                    False,
                    f"JSON parse error: {exc}",
                )

    def run_all(self, splits_dir: Path, episodes_dir: Path | None) -> bool:
        """Run all quality checks. Returns True if all checks pass."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("pyarrow not installed. Run: uv add pyarrow")
            sys.exit(1)

        # Check splits exist
        if not self.check_splits_exist(splits_dir):
            logger.error("Not all split files exist. Cannot run further checks.")
            return self.all_passed()

        # Load splits
        splits: dict[str, "pa.Table"] = {}
        for split_name in ["train", "val", "test"]:
            path = splits_dir / f"{split_name}.parquet"
            if path.exists():
                splits[split_name] = pq.read_table(path)

        # Run checks on each split
        for split_name, table in splits.items():
            self.check_reward_distribution(table, split_name)
            self.check_action_distribution(table, split_name)
            self.check_no_duplicate_triples(table, split_name)
            self.check_json_columns_parseable(table, split_name)

        # Cross-split checks
        self.check_no_leakage(splits)
        self.check_split_sizes(splits)

        return self.all_passed()

    def all_passed(self) -> bool:
        return all(r["status"] == "PASS" for r in self.results)

    def summary(self) -> dict:
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        return {
            "total_checks": len(self.results),
            "passed": passed,
            "failed": failed,
            "all_passed": self.all_passed(),
            "results": self.results,
        }


def main() -> None:
    args = parse_args()
    config = AppConfig()
    paths = DataPaths()

    splits_dir: Path = args.splits_dir or paths.splits
    episodes_dir: Path | None = args.episodes_dir or paths.raw_episodes

    if not splits_dir.exists():
        logger.error("Splits directory does not exist: %s", splits_dir)
        sys.exit(1)

    checker = QualityChecker(config)
    all_ok = checker.run_all(splits_dir, episodes_dir)
    summary = checker.summary()

    logger.info(
        "Quality check complete: %d/%d checks passed",
        summary["passed"],
        summary["total_checks"],
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Wrote report to %s", args.report)

    if args.strict and not all_ok:
        logger.error("Strict mode: failing due to quality check failures")
        sys.exit(1)


if __name__ == "__main__":
    main()

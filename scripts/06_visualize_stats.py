#!/usr/bin/env python3
"""Script 06: Visualize dataset statistics with matplotlib.

Generates plots for:
- Reward total distribution (histogram + KDE)
- Per-component reward distributions
- Action type distribution (bar chart)
- Episode length distribution
- Per-scenario breakdowns

Usage:
    uv run scripts/06_visualize_stats.py [OPTIONS]

Options:
    --splits-dir PATH   Directory with Parquet splits (default: data/splits)
    --episodes-dir PATH Directory with episode JSONL files (default: data/raw/episodes)
    --output-dir PATH   Directory for output figures (default: data/plots)
    --format STR        Figure format: png, pdf, svg (default: png)
    --no-show           Do not display interactive plots
    --dpi INT           Figure DPI (default: 150)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aimemory.config import AppConfig, DataPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("06_visualize_stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dataset statistics")
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--episodes-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_splits(splits_dir: Path) -> dict[str, "pa.Table"]:
    """Load all available Parquet split files."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow not installed. Run: uv add pyarrow")
        sys.exit(1)

    splits = {}
    for name in ["train", "val", "test"]:
        path = splits_dir / f"{name}.parquet"
        if path.exists():
            splits[name] = pq.read_table(path)
            logger.info("Loaded %s: %d rows", name, len(splits[name]))
        else:
            logger.warning("Split file not found: %s", path)
    return splits


def load_episodes_from_dir(episodes_dir: Path) -> list[dict]:
    """Load episode dicts from JSONL files."""
    episodes = []
    for jsonl_file in sorted(episodes_dir.glob("*.jsonl")):
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        episodes.append(json.loads(line))
        except Exception as exc:
            logger.warning("Could not read %s: %s", jsonl_file, exc)
    return episodes


def save_or_show(fig, output_dir: Path | None, name: str, fmt: str, dpi: int, show: bool) -> None:
    """Save figure to file and optionally display it."""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure: %s", path)
    if show:
        import matplotlib.pyplot as plt

        plt.show()


def plot_reward_distribution(
    splits: dict[str, list[float]],
    output_dir: Path | None,
    fmt: str,
    dpi: int,
    show: bool,
) -> None:
    """Plot histogram of total reward distribution per split."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4), sharey=False)
    if len(splits) == 1:
        axes = [axes]

    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

    for ax, (split_name, rewards) in zip(axes, splits.items()):
        color = colors.get(split_name, "#999999")
        ax.hist(rewards, bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.set_title(f"{split_name.capitalize()} Reward Distribution")
        ax.set_xlabel("Total Reward")
        ax.set_ylabel("Count")
        if rewards:
            import statistics

            mean_r = statistics.mean(rewards)
            ax.axvline(mean_r, color="red", linestyle="--", label=f"mean={mean_r:.3f}")
            ax.legend()

    fig.suptitle("Reward Distribution by Split", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_or_show(fig, output_dir, "01_reward_distribution", fmt, dpi, show)
    plt.close(fig)


def plot_action_distribution(
    splits: dict[str, dict[str, int]],
    output_dir: Path | None,
    fmt: str,
    dpi: int,
    show: bool,
) -> None:
    """Plot action type distribution bar chart per split."""
    import matplotlib.pyplot as plt

    action_types = ["save", "skip", "retrieve"]
    colors = {"save": "#4C72B0", "skip": "#DD8452", "retrieve": "#55A868"}

    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4))
    if len(splits) == 1:
        axes = [axes]

    for ax, (split_name, counts) in zip(axes, splits.items()):
        values = [counts.get(a, 0) for a in action_types]
        bars = ax.bar(action_types, values, color=[colors[a] for a in action_types])
        ax.set_title(f"{split_name.capitalize()} Action Distribution")
        ax.set_xlabel("Action Type")
        ax.set_ylabel("Count")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Action Distribution by Split", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_or_show(fig, output_dir, "02_action_distribution", fmt, dpi, show)
    plt.close(fig)


def plot_episode_length_distribution(
    lengths: list[int],
    output_dir: Path | None,
    fmt: str,
    dpi: int,
    show: bool,
) -> None:
    """Plot histogram of episode lengths."""
    import matplotlib.pyplot as plt

    if not lengths:
        logger.warning("No episode lengths to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=20, color="#4C72B0", alpha=0.75, edgecolor="white")
    import statistics

    mean_len = statistics.mean(lengths)
    ax.axvline(mean_len, color="red", linestyle="--", label=f"mean={mean_len:.1f}")
    ax.set_xlabel("Episode Length (turns)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")
    ax.legend()
    plt.tight_layout()
    save_or_show(fig, output_dir, "03_episode_length", fmt, dpi, show)
    plt.close(fig)


def plot_per_scenario_rewards(
    scenario_rewards: dict[str, list[float]],
    output_dir: Path | None,
    fmt: str,
    dpi: int,
    show: bool,
) -> None:
    """Plot per-scenario reward distributions as box plots."""
    import matplotlib.pyplot as plt

    if not scenario_rewards:
        logger.warning("No scenario reward data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    scenario_names = sorted(scenario_rewards.keys())
    data = [scenario_rewards[s] for s in scenario_names]

    bp = ax.boxplot(data, labels=scenario_names, patch_artist=True, notch=False)

    # Color boxes
    import itertools

    color_cycle = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    for patch, color in zip(bp["boxes"], itertools.cycle(color_cycle)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Distribution by Scenario")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_or_show(fig, output_dir, "04_scenario_rewards", fmt, dpi, show)
    plt.close(fig)


def plot_reward_components(
    component_data: dict[str, list[float]],
    output_dir: Path | None,
    fmt: str,
    dpi: int,
    show: bool,
) -> None:
    """Plot mean reward per component as a horizontal bar chart."""
    import statistics

    import matplotlib.pyplot as plt

    if not component_data:
        return

    components = list(component_data.keys())
    means = [statistics.mean(vals) if vals else 0.0 for vals in component_data.values()]
    stds = [statistics.stdev(vals) if len(vals) > 1 else 0.0 for vals in component_data.values()]

    fig, ax = plt.subplots(figsize=(8, max(4, len(components) * 0.5)))
    y_pos = range(len(components))
    ax.barh(list(y_pos), means, xerr=stds, color="#4C72B0", alpha=0.75, ecolor="gray")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(components)
    ax.set_xlabel("Mean Reward ± Std")
    ax.set_title("Mean Reward per Component")
    plt.tight_layout()
    save_or_show(fig, output_dir, "05_reward_components", fmt, dpi, show)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _config = AppConfig()  # noqa: F841
    paths = DataPaths()

    splits_dir: Path = args.splits_dir or paths.splits
    episodes_dir: Path = args.episodes_dir or paths.raw_episodes
    output_dir: Path | None = args.output_dir or (paths.root / "plots")
    show = not args.no_show

    try:
        import matplotlib

        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        logger.error("matplotlib not installed. Run: uv add matplotlib")
        sys.exit(1)

    # Load splits
    if not splits_dir.exists():
        logger.error("Splits directory does not exist: %s", splits_dir)
        sys.exit(1)

    splits = load_splits(splits_dir)
    if not splits:
        logger.error("No split files found in %s", splits_dir)
        sys.exit(1)

    # Extract reward data per split
    reward_by_split: dict[str, list[float]] = {}
    action_by_split: dict[str, dict[str, int]] = {}
    for split_name, table in splits.items():
        if "reward_total" in table.schema.names:
            reward_by_split[split_name] = table.column("reward_total").to_pylist()
        if "action_type" in table.schema.names:
            from collections import Counter

            action_by_split[split_name] = dict(Counter(table.column("action_type").to_pylist()))

    # Reward component data (from train split)
    component_data: dict[str, list[float]] = {}
    if "train" in splits:
        train_table = splits["train"]
        reward_cols = [c for c in train_table.schema.names if c.startswith("reward_r")]
        for col in reward_cols:
            component_data[col.replace("reward_", "")] = train_table.column(col).to_pylist()

    # Episode lengths from JSONL files
    ep_lengths: list[int] = []
    scenario_rewards: dict[str, list[float]] = {}

    if episodes_dir and episodes_dir.exists():
        episodes = load_episodes_from_dir(episodes_dir)
        for ep in episodes:
            turns = ep.get("turns", [])
            ep_lengths.append(len(turns))
            scenario = ep.get("scenario", "unknown")
            # Aggregate rewards from annotations if available
            annotations = ep.get("_reward_annotations", {})
            if annotations:
                ep_total = sum(v.get("total", 0.0) for v in annotations.values())
                scenario_rewards.setdefault(scenario, []).append(ep_total)

    # Generate plots
    logger.info("Generating plots...")

    if reward_by_split:
        plot_reward_distribution(reward_by_split, output_dir, args.format, args.dpi, show)

    if action_by_split:
        plot_action_distribution(action_by_split, output_dir, args.format, args.dpi, show)

    if ep_lengths:
        plot_episode_length_distribution(ep_lengths, output_dir, args.format, args.dpi, show)

    if scenario_rewards:
        plot_per_scenario_rewards(scenario_rewards, output_dir, args.format, args.dpi, show)

    if component_data:
        plot_reward_components(component_data, output_dir, args.format, args.dpi, show)

    logger.info("All plots generated successfully.")


if __name__ == "__main__":
    main()

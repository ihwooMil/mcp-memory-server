"""Convergence benchmark: individual vs P2P learning speed.

Compares:
- 3 individual OnlinePolicy instances (no communication)
- 3 GossipNode-connected OnlinePolicy instances (P2P gossip)

Output:
- data/plots/convergence_benchmark.png  (if matplotlib available)
- data/plots/convergence_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from aimemory.online.gossip import GossipNode, InMemoryTransport
from aimemory.online.policy import FEATURE_DIM, OnlinePolicy

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_steps: int, n_partitions: int, seed: int = 42
) -> list[list[tuple[np.ndarray, int, float]]]:
    """Generate synthetic (features, action, reward) tuples partitioned across nodes.

    Each partition has a slightly different reward signal to simulate heterogeneous data.
    """
    rng = np.random.default_rng(seed)
    partitions: list[list[tuple[np.ndarray, int, float]]] = [[] for _ in range(n_partitions)]

    for step in range(n_steps):
        node_idx = step % n_partitions
        features = rng.normal(0, 1, FEATURE_DIM).astype(np.float32)
        # Best action for node i is action i (simulates heterogeneous tasks)
        best_action = node_idx
        action = rng.integers(0, 3)
        reward = 1.0 if action == best_action else -0.1
        partitions[node_idx].append((features, int(action), float(reward)))

    return partitions


def _make_policy(seed_params: np.ndarray) -> OnlinePolicy:
    """Create a fresh OnlinePolicy initialized to seed_params."""
    policy = OnlinePolicy(feature_dim=FEATURE_DIM, epsilon=0.1)
    policy.set_parameters(seed_params.copy())
    return policy


def run_individual(
    partitions: list[list[tuple[np.ndarray, int, float]]],
    seed_params: np.ndarray,
) -> list[list[float]]:
    """Train individual policies without communication.

    Returns list of cumulative reward histories, one per node.
    """
    n_nodes = len(partitions)
    policies = [_make_policy(seed_params) for _ in range(n_nodes)]
    reward_histories: list[list[float]] = [[] for _ in range(n_nodes)]

    max_steps = max(len(p) for p in partitions)
    for step in range(max_steps):
        for i, (policy, partition) in enumerate(zip(policies, partitions)):
            if step < len(partition):
                features, action, reward = partition[step]
                policy.update(features, action_id=action, reward=reward)
                reward_histories[i].append(reward)

    return reward_histories


def run_p2p(
    partitions: list[list[tuple[np.ndarray, int, float]]],
    seed_params: np.ndarray,
    gossip_interval: int = 10,
) -> list[list[float]]:
    """Train P2P-connected policies with gossip.

    Returns list of cumulative reward histories, one per node.
    """
    n_nodes = len(partitions)
    bus: dict[str, list] = {}
    policies = [_make_policy(seed_params) for _ in range(n_nodes)]
    node_ids = [f"node_{i}" for i in range(n_nodes)]

    nodes = []
    for node_id, policy in zip(node_ids, policies):
        transport = InMemoryTransport(node_id, bus)
        node = GossipNode(
            node_id=node_id,
            policy=policy,
            transport=transport,
            max_norm=1.0,
            gossip_interval=gossip_interval,
            dp_enabled=False,
        )
        nodes.append(node)

    # Fully connected peers
    for i, node in enumerate(nodes):
        for j, peer_id in enumerate(node_ids):
            if j != i:
                node.register_peer(peer_id, node._transport)

    reward_histories: list[list[float]] = [[] for _ in range(n_nodes)]
    max_steps = max(len(p) for p in partitions)

    for step in range(max_steps):
        for i, (node, partition) in enumerate(zip(nodes, partitions)):
            if step < len(partition):
                features, action, reward = partition[step]
                node._policy.update(features, action_id=action, reward=reward)
                reward_histories[i].append(reward)
                node.step()

    return reward_histories


def _moving_average(values: list[float], window: int = 50) -> list[float]:
    """Compute moving average over a window."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start : i + 1]) / (i - start + 1))
    return result


def plot_convergence(
    individual_rewards: list[list[float]],
    p2p_rewards: list[list[float]],
    output_path: Path,
) -> None:
    """Plot convergence curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Individual
    ax = axes[0]
    ax.set_title("Individual Learning (No Communication)")
    for i, rewards in enumerate(individual_rewards):
        smoothed = _moving_average(rewards, window=50)
        ax.plot(smoothed, label=f"Node {i}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # P2P
    ax = axes[1]
    ax.set_title("P2P Gossip Learning")
    for i, rewards in enumerate(p2p_rewards):
        smoothed = _moving_average(rewards, window=50)
        ax.plot(smoothed, label=f"Node {i}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close()
    logger.info("Plot saved to %s", output_path)


def compute_stats(rewards: list[list[float]], window: int = 100) -> dict:
    """Compute summary statistics over last `window` steps."""
    stats = {}
    for i, node_rewards in enumerate(rewards):
        last = node_rewards[-window:] if len(node_rewards) >= window else node_rewards
        stats[f"node_{i}"] = {
            "mean_reward_last_{window}": float(np.mean(last)) if last else 0.0,
            "total_steps": len(node_rewards),
        }
    all_last = []
    for node_rewards in rewards:
        all_last.extend(node_rewards[-window:] if len(node_rewards) >= window else node_rewards)
    stats["overall_mean"] = float(np.mean(all_last)) if all_last else 0.0
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Convergence benchmark: individual vs P2P")
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--nodes", type=int, default=3, help="Number of nodes")
    parser.add_argument("--gossip-interval", type=int, default=10, help="Gossip interval (steps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/plots", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic data: %d steps, %d nodes", args.steps, args.nodes)
    partitions = generate_synthetic_data(args.steps, args.nodes, seed=args.seed)

    # Shared initial parameters
    seed_policy = OnlinePolicy(feature_dim=FEATURE_DIM, epsilon=0.1)
    seed_params = seed_policy.get_parameters()

    logger.info("Running individual learning baseline...")
    individual_rewards = run_individual(partitions, seed_params)

    logger.info("Running P2P gossip learning (interval=%d)...", args.gossip_interval)
    p2p_rewards = run_p2p(partitions, seed_params, gossip_interval=args.gossip_interval)

    # Compute and save stats
    individual_stats = compute_stats(individual_rewards)
    p2p_stats = compute_stats(p2p_rewards)
    results = {
        "config": {
            "steps": args.steps,
            "nodes": args.nodes,
            "gossip_interval": args.gossip_interval,
            "seed": args.seed,
        },
        "individual": individual_stats,
        "p2p": p2p_stats,
    }

    stats_path = output_dir / "convergence_stats.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Stats saved to %s", stats_path)

    # Print summary
    print("\n=== Convergence Benchmark Results ===")
    print(f"Individual overall mean reward: {individual_stats['overall_mean']:.4f}")
    print(f"P2P overall mean reward:        {p2p_stats['overall_mean']:.4f}")

    # Plot
    plot_convergence(individual_rewards, p2p_rewards, output_dir / "convergence_benchmark.png")


if __name__ == "__main__":
    main()

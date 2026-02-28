"""Dataset statistics computation for the AI Memory System.

Provides:
- Reward distribution stats (mean, std, percentiles)
- Action distribution (save/skip/retrieve counts)
- Episode length distribution
- Per-scenario breakdowns
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from aimemory.schemas import Episode, MemoryActionType, SARTriple

logger = logging.getLogger(__name__)


def _percentile(data: list[float], p: float) -> float:
    """Compute p-th percentile of a list using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


@dataclass
class DistributionStats:
    """Statistics for a numerical distribution."""

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    p25: float = 0.0
    median: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    max: float = 0.0

    @classmethod
    def from_values(cls, values: list[float]) -> "DistributionStats":
        if not values:
            return cls()
        return cls(
            count=len(values),
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            min=min(values),
            p25=_percentile(values, 25),
            median=_percentile(values, 50),
            p75=_percentile(values, 75),
            p90=_percentile(values, 90),
            p95=_percentile(values, 95),
            max=max(values),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "p25": round(self.p25, 4),
            "median": round(self.median, 4),
            "p75": round(self.p75, 4),
            "p90": round(self.p90, 4),
            "p95": round(self.p95, 4),
            "max": round(self.max, 4),
        }


@dataclass
class ActionDistribution:
    """Counts and ratios for each memory action type."""

    save: int = 0
    skip: int = 0
    retrieve: int = 0

    @property
    def total(self) -> int:
        return self.save + self.skip + self.retrieve

    def ratios(self) -> dict[str, float]:
        t = self.total or 1
        return {
            "save": round(self.save / t, 4),
            "skip": round(self.skip / t, 4),
            "retrieve": round(self.retrieve / t, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {"save": self.save, "skip": self.skip, "retrieve": self.retrieve},
            "total": self.total,
            "ratios": self.ratios(),
        }


@dataclass
class ScenarioBreakdown:
    """Per-scenario statistics."""

    scenario: str
    episode_count: int = 0
    triple_count: int = 0
    reward_stats: DistributionStats = field(default_factory=DistributionStats)
    action_dist: ActionDistribution = field(default_factory=ActionDistribution)
    episode_length_stats: DistributionStats = field(default_factory=DistributionStats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "episode_count": self.episode_count,
            "triple_count": self.triple_count,
            "reward_stats": self.reward_stats.to_dict(),
            "action_distribution": self.action_dist.to_dict(),
            "episode_length_stats": self.episode_length_stats.to_dict(),
        }


@dataclass
class DatasetStats:
    """Full dataset statistics summary."""

    total_episodes: int = 0
    total_triples: int = 0
    reward_stats: DistributionStats = field(default_factory=DistributionStats)
    reward_component_stats: dict[str, DistributionStats] = field(default_factory=dict)
    action_distribution: ActionDistribution = field(default_factory=ActionDistribution)
    episode_length_stats: DistributionStats = field(default_factory=DistributionStats)
    scenario_breakdowns: dict[str, ScenarioBreakdown] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "total_triples": self.total_triples,
            "reward_stats": self.reward_stats.to_dict(),
            "reward_component_stats": {
                k: v.to_dict() for k, v in self.reward_component_stats.items()
            },
            "action_distribution": self.action_distribution.to_dict(),
            "episode_length_stats": self.episode_length_stats.to_dict(),
            "scenario_breakdowns": {k: v.to_dict() for k, v in self.scenario_breakdowns.items()},
        }


class StatsComputer:
    """Computes statistics from episodes and SARTriples."""

    REWARD_COMPONENTS = [
        "r1_keyword_reappearance",
        "r2_repeated_question_penalty",
        "r3_efficiency",
        "r4_retrieval_relevance",
        "r5_speech_act_weight",
        "r6_self_reference",
        "r7_info_density",
        "r8_preference_constraint",
        "r9_emotional_salience",
        "r10_topic_boundary",
        "r11_user_feedback",
    ]

    def compute(
        self,
        episodes: list[Episode],
        triples: list[SARTriple],
    ) -> DatasetStats:
        """Compute full dataset statistics.

        Args:
            episodes: All episodes in the dataset.
            triples: All SARTriples in the dataset.

        Returns:
            DatasetStats containing all computed statistics.
        """
        stats = DatasetStats(
            total_episodes=len(episodes),
            total_triples=len(triples),
        )

        # Episode length distribution
        ep_lengths = [len(ep.turns) for ep in episodes]
        stats.episode_length_stats = DistributionStats.from_values([float(x) for x in ep_lengths])

        # Reward distribution (total)
        total_rewards = [t.reward.total for t in triples]
        stats.reward_stats = DistributionStats.from_values(total_rewards)

        # Per-component reward stats
        for comp in self.REWARD_COMPONENTS:
            values = [getattr(t.reward, comp) for t in triples]
            stats.reward_component_stats[comp] = DistributionStats.from_values(values)

        # Action distribution
        action_counts = Counter(t.action.action_type for t in triples)
        stats.action_distribution = ActionDistribution(
            save=action_counts.get(MemoryActionType.SAVE, 0),
            skip=action_counts.get(MemoryActionType.SKIP, 0),
            retrieve=action_counts.get(MemoryActionType.RETRIEVE, 0),
        )

        # Per-scenario breakdowns
        ep_by_scenario: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            ep_by_scenario[ep.scenario.value].append(ep)

        # Map episode_id → scenario for triple lookup
        ep_to_scenario: dict[str, str] = {ep.episode_id: ep.scenario.value for ep in episodes}
        triples_by_scenario: dict[str, list[SARTriple]] = defaultdict(list)
        for t in triples:
            scenario = ep_to_scenario.get(t.episode_id, "unknown")
            triples_by_scenario[scenario].append(t)

        for scenario_name, scenario_eps in ep_by_scenario.items():
            scenario_triples = triples_by_scenario.get(scenario_name, [])
            breakdown = self._compute_scenario_breakdown(
                scenario_name, scenario_eps, scenario_triples
            )
            stats.scenario_breakdowns[scenario_name] = breakdown

        return stats

    def _compute_scenario_breakdown(
        self,
        scenario: str,
        episodes: list[Episode],
        triples: list[SARTriple],
    ) -> ScenarioBreakdown:
        ep_lengths = [float(len(ep.turns)) for ep in episodes]
        total_rewards = [t.reward.total for t in triples]
        action_counts = Counter(t.action.action_type for t in triples)

        return ScenarioBreakdown(
            scenario=scenario,
            episode_count=len(episodes),
            triple_count=len(triples),
            reward_stats=DistributionStats.from_values(total_rewards),
            action_dist=ActionDistribution(
                save=action_counts.get(MemoryActionType.SAVE, 0),
                skip=action_counts.get(MemoryActionType.SKIP, 0),
                retrieve=action_counts.get(MemoryActionType.RETRIEVE, 0),
            ),
            episode_length_stats=DistributionStats.from_values(ep_lengths),
        )

    def compare_splits(
        self,
        train_triples: list[SARTriple],
        val_triples: list[SARTriple],
        test_triples: list[SARTriple],
    ) -> dict[str, Any]:
        """Compare reward and action distributions across splits."""
        result: dict[str, Any] = {}
        for name, triples in [
            ("train", train_triples),
            ("val", val_triples),
            ("test", test_triples),
        ]:
            rewards = [t.reward.total for t in triples]
            action_counts = Counter(t.action.action_type for t in triples)
            total_actions = len(triples) or 1
            result[name] = {
                "count": len(triples),
                "reward_stats": DistributionStats.from_values(rewards).to_dict(),
                "action_ratios": {
                    "save": round(action_counts.get(MemoryActionType.SAVE, 0) / total_actions, 4),
                    "skip": round(action_counts.get(MemoryActionType.SKIP, 0) / total_actions, 4),
                    "retrieve": round(
                        action_counts.get(MemoryActionType.RETRIEVE, 0) / total_actions,
                        4,
                    ),
                },
            }
        return result

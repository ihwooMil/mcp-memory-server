"""Episode-level stratified splitting for train/val/test sets.

Splits 800 train / 100 val / 100 test by episode_id (no data leakage).
Stratifies by scenario type to maintain class balance across splits.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar

from aimemory.config import DatasetConfig
from aimemory.schemas import Episode, SARTriple

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SplitResult:
    """Container for train/val/test split results."""

    train: list[SARTriple]
    val: list[SARTriple]
    test: list[SARTriple]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "total": self.total,
        }


@dataclass
class EpisodeSplitResult:
    """Container for episode-level split results."""

    train: list[str]
    val: list[str]
    test: list[str]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


class EpisodeSplitter:
    """Performs stratified episode-level splitting.

    Splits at the episode level to avoid data leakage (all SARTriples
    from the same episode go into the same split).
    """

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def split_episode_ids(
        self,
        episodes: list[Episode],
    ) -> EpisodeSplitResult:
        """Split episode IDs into train/val/test sets.

        Uses stratified sampling by scenario type to maintain class balance.
        Split sizes are determined purely by config ratios (default 80/10/10).

        Args:
            episodes: List of all episodes to split.

        Returns:
            EpisodeSplitResult with episode_id lists for each split.
        """
        rng = random.Random(self.config.random_seed)

        # Group episodes by scenario type
        by_scenario: dict[str, list[str]] = defaultdict(list)
        for ep in episodes:
            by_scenario[ep.scenario.value].append(ep.episode_id)

        total = len(episodes)
        if total == 0:
            return EpisodeSplitResult(train=[], val=[], test=[])

        # Determine target counts from ratios
        n_train = int(total * self.config.train_ratio)
        n_val = int(total * self.config.val_ratio)
        n_test = total - n_train - n_val

        logger.info(
            "Splitting %d episodes → train=%d, val=%d, test=%d",
            total,
            n_train,
            n_val,
            n_test,
        )

        train_ids: list[str] = []
        val_ids: list[str] = []
        test_ids: list[str] = []

        # Stratified split: allocate proportionally from each scenario
        scenario_names = sorted(by_scenario.keys())
        for scenario in scenario_names:
            ids = by_scenario[scenario]
            rng.shuffle(ids)
            n = len(ids)

            # Proportional allocation
            s_train = max(1, round(n * self.config.train_ratio)) if n >= 3 else n
            s_val = max(0, round(n * self.config.val_ratio)) if n >= 3 else 0
            s_test = n - s_train - s_val
            s_test = max(0, s_test)

            train_ids.extend(ids[:s_train])
            val_ids.extend(ids[s_train : s_train + s_val])
            test_ids.extend(ids[s_train + s_val :])

        # Shuffle final lists to avoid scenario ordering bias
        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        rng.shuffle(test_ids)

        # Trim to target counts (handles rounding across scenarios)
        train_ids = train_ids[:n_train]
        val_ids = val_ids[:n_val]

        return EpisodeSplitResult(train=train_ids, val=val_ids, test=test_ids)

    def split_triples(
        self,
        triples: list[SARTriple],
        episode_split: EpisodeSplitResult,
    ) -> SplitResult:
        """Assign SARTriples to splits based on episode_id membership.

        Args:
            triples: All SARTriples to distribute.
            episode_split: Pre-computed episode ID split.

        Returns:
            SplitResult with SARTriple lists for each split.
        """
        train_set = set(episode_split.train)
        val_set = set(episode_split.val)
        test_set = set(episode_split.test)

        train: list[SARTriple] = []
        val: list[SARTriple] = []
        test: list[SARTriple] = []
        unassigned = 0

        for triple in triples:
            eid = triple.episode_id
            if eid in train_set:
                train.append(triple)
            elif eid in val_set:
                val.append(triple)
            elif eid in test_set:
                test.append(triple)
            else:
                unassigned += 1

        if unassigned:
            logger.warning("%d triples had unassigned episode_id", unassigned)

        result = SplitResult(train=train, val=val, test=test)
        logger.info("Triple split: %s", result.summary())
        return result

    def split(
        self,
        episodes: list[Episode],
        triples: list[SARTriple],
    ) -> SplitResult:
        """Full split pipeline: stratify episodes, then assign triples.

        Args:
            episodes: All episodes (needed for stratification by scenario).
            triples: All SARTriples to distribute.

        Returns:
            SplitResult containing train/val/test SARTriple lists.
        """
        episode_split = self.split_episode_ids(episodes)
        return self.split_triples(triples, episode_split)

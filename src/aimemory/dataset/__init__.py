"""Dataset module for the AI Memory System.

Provides Episode→SARTriple conversion, train/val/test splitting, and statistics.
"""

from aimemory.dataset.builder import EpisodeBuilder
from aimemory.dataset.splitter import EpisodeSplitter, SplitResult
from aimemory.dataset.stats import DatasetStats, StatsComputer

__all__ = [
    "EpisodeBuilder",
    "EpisodeSplitter",
    "SplitResult",
    "DatasetStats",
    "StatsComputer",
]

"""Progressive autonomy: confidence-based threshold relaxation for RL policy."""

from __future__ import annotations

import json
from pathlib import Path


class ProgressiveAutonomy:
    """Confidence-based threshold relaxation.

    As the RL policy accumulates positive feedback, rule-based thresholds relax
    to give RL more decision-making authority.

    - confidence = Σ(positive) - decay_rate * Σ(negative)
    - When confidence >= confidence_threshold: save_threshold ↓, skip_threshold ↑
    - rl_zone_ratio: initially ~60%, can grow up to ~90%
    - Negative feedback significantly reduces confidence (safety mechanism)
    """

    def __init__(
        self,
        initial_save: float = 0.7,
        initial_skip: float = 0.1,
        min_save: float = 0.3,
        max_skip: float = 0.4,
        confidence_threshold: int = 50,
        decay_rate: float = 0.01,
    ) -> None:
        self._initial_save = initial_save
        self._initial_skip = initial_skip
        self._min_save = min_save
        self._max_skip = max_skip
        self._confidence_threshold = confidence_threshold
        self._decay_rate = decay_rate

        self._confidence: float = 0.0
        self._positive_count: int = 0
        self._negative_count: int = 0

    @property
    def save_threshold(self) -> float:
        """Current SAVE threshold. Decreases as confidence grows."""
        if self._confidence < self._confidence_threshold:
            return self._initial_save
        # Linear interpolation: initial_save → min_save as confidence grows beyond threshold
        progress = min(
            (self._confidence - self._confidence_threshold) / self._confidence_threshold, 1.0
        )
        return self._initial_save - progress * (self._initial_save - self._min_save)

    @property
    def skip_threshold(self) -> float:
        """Current SKIP threshold. Increases as confidence grows."""
        if self._confidence < self._confidence_threshold:
            return self._initial_skip
        progress = min(
            (self._confidence - self._confidence_threshold) / self._confidence_threshold, 1.0
        )
        return self._initial_skip + progress * (self._max_skip - self._initial_skip)

    @property
    def rl_zone_ratio(self) -> float:
        """Fraction of decisions handled by RL (vs rules). Initially ~60%, max ~90%."""
        # RL zone = 1 - save_threshold + skip_threshold (the middle band)
        # When initial: 1 - 0.7 + 0.1 = 0.4... no, it's save_threshold - skip_threshold = rule zone
        # RL zone = 1 - (save_threshold - skip_threshold) ... no
        # Actually: rule zone covers [0, skip_threshold] and [save_threshold, 1.0]
        # So RL zone = save_threshold - skip_threshold
        return self.save_threshold - self.skip_threshold

    @property
    def confidence(self) -> float:
        return self._confidence

    def record_feedback(self, reward: float) -> None:
        """Record feedback and update confidence.

        Positive rewards increment confidence.
        Negative rewards decrement confidence more aggressively.
        """
        if reward > 0:
            self._positive_count += 1
            self._confidence += reward
        elif reward < 0:
            self._negative_count += 1
            self._confidence += reward / self._decay_rate  # Amplified negative impact
            # Floor at 0
            self._confidence = max(self._confidence, 0.0)

    def save(self, path: str) -> None:
        """Save autonomy state to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "confidence": self._confidence,
            "positive_count": self._positive_count,
            "negative_count": self._negative_count,
            "initial_save": self._initial_save,
            "initial_skip": self._initial_skip,
            "min_save": self._min_save,
            "max_skip": self._max_skip,
            "confidence_threshold": self._confidence_threshold,
            "decay_rate": self._decay_rate,
        }
        p.write_text(json.dumps(data))

    def load(self, path: str) -> None:
        """Load autonomy state from JSON file."""
        data = json.loads(Path(path).read_text())
        self._confidence = data["confidence"]
        self._positive_count = data["positive_count"]
        self._negative_count = data["negative_count"]
        self._initial_save = data.get("initial_save", self._initial_save)
        self._initial_skip = data.get("initial_skip", self._initial_skip)
        self._min_save = data.get("min_save", self._min_save)
        self._max_skip = data.get("max_skip", self._max_skip)
        self._confidence_threshold = data.get("confidence_threshold", self._confidence_threshold)
        self._decay_rate = data.get("decay_rate", self._decay_rate)

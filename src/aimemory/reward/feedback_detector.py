"""Feedback detector for memory action evaluation.

Detects implicit/explicit user feedback about memory accuracy from dialogue
turns. Supports multiple languages via i18n patterns. Handles morphological
variations and provides context-aware classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from aimemory.i18n import get_patterns
from aimemory.schemas import MemoryActionType, Role, Turn

# ─── Feedback taxonomy ───


class FeedbackType(str, Enum):
    """Types of user feedback about memory operations."""

    MEMORY_CORRECT = "memory_correct"  # +1.0: user confirms memory is correct
    MEMORY_USEFUL = "memory_useful"  # +0.7: user finds recalled info helpful
    MEMORY_FAILURE = "memory_failure"  # -1.0: user says memory is wrong/missing
    MEMORY_ERROR = "memory_error"  # -1.5: user corrects factual memory error
    REPEATED_QUESTION = "repeated_question"  # -0.8: agent asked the same thing again
    NEUTRAL = "neutral"  # 0.0: no memory-related feedback detected


_FEEDBACK_REWARDS: dict[FeedbackType, float] = {
    FeedbackType.MEMORY_CORRECT: 1.0,
    FeedbackType.MEMORY_USEFUL: 0.7,
    FeedbackType.MEMORY_FAILURE: -1.0,
    FeedbackType.MEMORY_ERROR: -1.5,
    FeedbackType.REPEATED_QUESTION: -0.8,
    FeedbackType.NEUTRAL: 0.0,
}


@dataclass(frozen=True)
class FeedbackSignal:
    """Result of feedback detection on a dialogue turn."""

    signal_type: FeedbackType
    reward_value: float
    confidence: float  # 0.0–1.0
    matched_pattern: str  # the regex/pattern that triggered detection


# ─── Character n-gram based similarity ───


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character n-grams from text (whitespace removed)."""
    text = re.sub(r"\s+", "", text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ─── Main detector ───


class FeedbackDetector:
    """Detects user feedback about memory operations from dialogue.

    Context-aware: only classifies feedback as memory-related when the
    last action was RETRIEVE or SAVE. During general conversation (SKIP),
    positive/negative expressions are treated as NEUTRAL.
    """

    def __init__(
        self,
        repeated_question_threshold: float = 0.45,
        ngram_size: int = 2,
        lang: str = "ko",
    ) -> None:
        self._repeated_threshold = repeated_question_threshold
        self._ngram_size = ngram_size
        lp = get_patterns(lang)
        self._positive_patterns = [
            (re.compile(pat), conf, name) for pat, conf, name in lp.positive_feedback
        ]
        self._negative_patterns = [
            (re.compile(pat), conf, name) for pat, conf, name in lp.negative_feedback
        ]
        self._useful_patterns = [
            (re.compile(pat), conf, name) for pat, conf, name in lp.useful_feedback
        ]
        self._correction_names = lp.correction_names

    def detect(
        self,
        current_turn: Turn,
        previous_turns: list[Turn],
        last_action: MemoryActionType,
    ) -> FeedbackSignal:
        """Detect feedback signal from the current turn."""
        text = current_turn.content

        # 1. Check for repeated questions
        repeated_signal = self._detect_repeated_question(
            current_turn,
            previous_turns,
        )
        if repeated_signal is not None:
            return repeated_signal

        # 2. Memory-related feedback: only relevant after RETRIEVE or SAVE
        is_memory_context = last_action in (
            MemoryActionType.RETRIEVE,
            MemoryActionType.SAVE,
        )

        if is_memory_context:
            # Check negative patterns first (more specific, higher priority)
            for pattern, confidence, name in self._negative_patterns:
                if pattern.search(text):
                    if self._is_correction_pattern(name):
                        return FeedbackSignal(
                            signal_type=FeedbackType.MEMORY_ERROR,
                            reward_value=_FEEDBACK_REWARDS[FeedbackType.MEMORY_ERROR],
                            confidence=confidence,
                            matched_pattern=name,
                        )
                    return FeedbackSignal(
                        signal_type=FeedbackType.MEMORY_FAILURE,
                        reward_value=_FEEDBACK_REWARDS[FeedbackType.MEMORY_FAILURE],
                        confidence=confidence,
                        matched_pattern=name,
                    )

            # Check positive patterns
            for pattern, confidence, name in self._positive_patterns:
                if pattern.search(text):
                    return FeedbackSignal(
                        signal_type=FeedbackType.MEMORY_CORRECT,
                        reward_value=_FEEDBACK_REWARDS[FeedbackType.MEMORY_CORRECT],
                        confidence=confidence,
                        matched_pattern=name,
                    )

            # Check useful patterns
            for pattern, confidence, name in self._useful_patterns:
                if pattern.search(text):
                    return FeedbackSignal(
                        signal_type=FeedbackType.MEMORY_USEFUL,
                        reward_value=_FEEDBACK_REWARDS[FeedbackType.MEMORY_USEFUL],
                        confidence=confidence,
                        matched_pattern=name,
                    )

        # 3. No memory-related feedback detected
        return FeedbackSignal(
            signal_type=FeedbackType.NEUTRAL,
            reward_value=_FEEDBACK_REWARDS[FeedbackType.NEUTRAL],
            confidence=1.0,
            matched_pattern="",
        )

    def _detect_repeated_question(
        self,
        current_turn: Turn,
        previous_turns: list[Turn],
    ) -> FeedbackSignal | None:
        """Detect if the current assistant turn repeats a previous question."""
        if current_turn.role != Role.ASSISTANT:
            return None

        if not previous_turns:
            return None

        current_ngrams = _char_ngrams(current_turn.content, self._ngram_size)
        if not current_ngrams:
            return None

        for prev_turn in previous_turns:
            if prev_turn.role != Role.ASSISTANT:
                continue
            if prev_turn.turn_id == current_turn.turn_id:
                continue

            prev_ngrams = _char_ngrams(prev_turn.content, self._ngram_size)
            similarity = _jaccard_similarity(current_ngrams, prev_ngrams)

            if similarity > self._repeated_threshold:
                return FeedbackSignal(
                    signal_type=FeedbackType.REPEATED_QUESTION,
                    reward_value=_FEEDBACK_REWARDS[FeedbackType.REPEATED_QUESTION],
                    confidence=min(similarity, 1.0),
                    matched_pattern=f"jaccard={similarity:.2f}",
                )

        return None

    def _is_correction_pattern(self, pattern_name: str) -> bool:
        """Determine if a negative pattern indicates factual correction (ERROR)
        vs. memory forgetting (FAILURE)."""
        return pattern_name in self._correction_names

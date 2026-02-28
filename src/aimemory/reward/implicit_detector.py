"""Implicit reward detection from conversation flow patterns."""

from __future__ import annotations

import re

from aimemory.i18n import get_patterns
from aimemory.schemas import Role, Turn


class ImplicitRewardDetector:
    """Detects implicit rewards from conversation flow.

    Complements FeedbackDetector's explicit patterns with implicit signals:
    1. Conversation continuation: user continues 2+ turns after memory usage → +0.3
    2. Topic expansion: memory-related keywords reappear in subsequent turns → +0.2
    3. Short dismissive response: user gives brief dismissive reply after memory → -0.1
    """

    def __init__(self, lang: str = "ko") -> None:
        lp = get_patterns(lang)
        self._short_dismissive = re.compile(lp.dismissive_pattern)
        self._word_re = re.compile(lp.word_extraction_pattern)

    def detect(self, turns: list[Turn], memory_used: list[str]) -> float:
        """Detect implicit reward from conversation flow."""
        if not turns or not memory_used:
            return 0.0

        reward = 0.0

        # 1. Check for short dismissive response (first user turn after memory)
        first_user_turns = [t for t in turns if t.role == Role.USER]
        if first_user_turns:
            first_user = first_user_turns[0]
            if self._short_dismissive.match(first_user.content.strip()):
                reward -= 0.1
                return reward  # Short response → no continuation/expansion check

        # 2. Conversation continuation: count user turns after memory usage
        user_turn_count = sum(1 for t in turns if t.role == Role.USER)
        if user_turn_count >= 2:
            reward += 0.3

        # 3. Topic expansion: memory keywords reappear in subsequent turns
        memory_keywords = self._extract_keywords(memory_used)
        if memory_keywords and len(first_user_turns) >= 2:
            subsequent_text = " ".join(t.content for t in first_user_turns[1:])
            reappeared = sum(1 for kw in memory_keywords if kw in subsequent_text)
            if reappeared > 0:
                reward += 0.2

        return reward

    def _extract_keywords(self, memory_contents: list[str]) -> list[str]:
        """Extract meaningful keywords from memory content strings."""
        keywords: list[str] = []
        for content in memory_contents:
            words = self._word_re.findall(content)
            tech_words = re.findall(
                r"(?<![a-zA-Z])(?:Python|Java|React|Django|Docker|Rust|TypeScript)(?![a-zA-Z])",
                content, re.IGNORECASE,
            )
            keywords.extend(words)
            keywords.extend(tech_words)
        return list(dict.fromkeys(keywords))

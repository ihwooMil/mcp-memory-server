"""Multilingual pattern registry for AIMemory.

Provides language-specific patterns for feedback detection, importance scoring,
entity extraction, and text resolution. Default language is Korean (ko).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguagePatterns:
    """Language-specific patterns used across the AIMemory pipeline."""

    # ── feedback_detector ──
    positive_feedback: list[tuple[str, float, str]] = field(default_factory=list)
    negative_feedback: list[tuple[str, float, str]] = field(default_factory=list)
    useful_feedback: list[tuple[str, float, str]] = field(default_factory=list)
    correction_names: frozenset[str] = field(default_factory=frozenset)

    # ── policy / StateEncoder ──
    question_pattern: str = ""
    personal_info_patterns: list[str] = field(default_factory=list)
    preference_patterns: list[str] = field(default_factory=list)
    tech_keywords: str = ""
    emotion_keywords: str = ""

    # ── policy / MemoryPolicyAgent ──
    discourse_markers: list[str] = field(default_factory=list)

    # ── resolution ──
    subject_markers: str = ""
    predicate_patterns: str = ""
    verb_endings: str = ""
    verb_endings_fallback: str = ""
    subject_strip: str = ""

    # ── graph_retriever ──
    word_extraction_pattern: str = ""
    negative_predicates: list[str] = field(default_factory=list)

    # ── implicit_detector ──
    dismissive_pattern: str = ""

    # ── korean_patterns equivalents ──
    sentence_endings: list[str] = field(default_factory=list)
    first_person: list[str] = field(default_factory=list)
    emphasis: list[str] = field(default_factory=list)
    preference_expressions: list[str] = field(default_factory=list)
    constraint_expressions: list[str] = field(default_factory=list)
    request_patterns: list[str] = field(default_factory=list)
    positive_feedback_simple: list[str] = field(default_factory=list)
    negative_feedback_simple: list[str] = field(default_factory=list)
    clarification_patterns: list[str] = field(default_factory=list)
    elaboration_patterns: list[str] = field(default_factory=list)
    fact_utterance_patterns: list[str] = field(default_factory=list)
    preference_utterance_patterns: list[str] = field(default_factory=list)
    experience_utterance_patterns: list[str] = field(default_factory=list)

    # ── token estimation ──
    chars_per_token: float = 2.5

    # ── MeCab POS tags (Korean only, empty for other langs) ──
    substantive_pos_tags: frozenset[str] = field(default_factory=frozenset)
    proper_noun_tag: str = ""
    common_noun_tags: frozenset[str] = field(default_factory=frozenset)


_REGISTRY: dict[str, LanguagePatterns] = {}


def register(lang: str, patterns: LanguagePatterns) -> None:
    """Register patterns for a language code."""
    _REGISTRY[lang] = patterns


def get_patterns(lang: str = "ko") -> LanguagePatterns:
    """Return patterns for the given language, defaulting to Korean."""
    if lang not in _REGISTRY:
        # Lazy-load built-in language modules
        if lang == "ko":
            from aimemory.i18n import ko as _  # noqa: F401
        elif lang == "en":
            from aimemory.i18n import en as _  # noqa: F401
    return _REGISTRY.get(lang, _REGISTRY.get("ko", LanguagePatterns()))


def available_languages() -> list[str]:
    """Return list of registered language codes."""
    # Ensure built-ins are loaded
    for lang in ("ko", "en"):
        if lang not in _REGISTRY:
            get_patterns(lang)
    return sorted(_REGISTRY.keys())

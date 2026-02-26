"""Reward signal computation functions for the AI Memory System.

Each function computes a single reward signal (R1-R11) and returns a float score.
All signals are normalized to approximately [-1, 1] or [0, 1] unless noted.
"""

from __future__ import annotations

import re
from collections import Counter

from aimemory.schemas import Action, MemoryEntry, State, Turn

from .korean_patterns import (
    CLARIFICATION_PATTERNS,
    COMMON_NOUN_TAGS,
    CONSTRAINT_EXPRESSIONS,
    DISCOURSE_MARKERS,
    ELABORATION_PATTERNS,
    EMPHASIS_EXPRESSIONS,
    EXPERIENCE_UTTERANCE_PATTERNS,
    FIRST_PERSON_EXPRESSIONS,
    NEGATIVE_FEEDBACK,
    POSITIVE_FEEDBACK,
    PREFERENCE_EXPRESSIONS,
    PREFERENCE_UTTERANCE_PATTERNS,
    PROPER_NOUN_TAG,
    REQUEST_PATTERNS,
    SENTENCE_FINAL_ENDINGS,
    SUBSTANTIVE_POS_TAGS,
)

# MeCab with Korean dictionary (mecab-ko-dic)
try:
    import MeCab
    import mecab_ko_dic

    _MECAB_TAGGER = MeCab.Tagger(f"-d {mecab_ko_dic.DICDIR}")
    MECAB_AVAILABLE = True
except Exception:
    _MECAB_TAGGER = None
    MECAB_AVAILABLE = False


def _parse_mecab(text: str) -> list[tuple[str, str]]:
    """Parse text with MeCab (mecab-ko-dic) and return (surface, pos) pairs.

    mecab-ko-dic output format: surface\tPOS,*,*,...
    POS tags follow the Korean tagset:
      NNG (일반명사), NNP (고유명사), NNB (의존명사), NR (수사),
      NP (대명사), VV (동사), VA (형용사), MAG (부사),
      JK* (조사), E* (어미), SF (마침표), SL (외래어), SH (한자), ...

    Returns an empty list if MeCab is unavailable.
    """
    if not MECAB_AVAILABLE or _MECAB_TAGGER is None:
        return []
    result = _MECAB_TAGGER.parse(text)
    tokens: list[tuple[str, str]] = []
    for line in result.splitlines():
        if line == "EOS" or not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        surface = parts[0]
        # POS is the first field in comma-separated feature string
        pos = parts[1].split(",")[0]
        tokens.append((surface, pos))
    return tokens


def _is_substantive_token(surface: str, pos: str) -> bool:
    """Return True if a (surface, pos) pair is a substantive morpheme."""
    if pos in SUBSTANTIVE_POS_TAGS:
        return True
    # ASCII-only alphabetic tokens not tagged as noun by unidic may still be
    # foreign proper nouns (e.g. 'Python', 'pandas') — treat as SL
    if surface.isascii() and surface.isalpha() and len(surface) > 1:
        return True
    return False


def _extract_keywords_from_text(text: str) -> list[str]:
    """Extract noun-type tokens from text using MeCab (with fallback)."""
    tokens = _parse_mecab(text)
    if tokens:
        return [surface for surface, pos in tokens if _is_substantive_token(surface, pos)]
    # Fallback: split on spaces and keep tokens ≥ 2 chars
    return [w for w in text.split() if len(w) >= 2]


def compute_r1_keyword_reappearance(
    state: State,
    action: Action,
    future_turns: list[Turn],
    proper_noun_multiplier: float = 3.0,
    common_noun_multiplier: float = 0.3,
) -> float:
    """R1 (+) Keyword reappearance reward.

    Rewards saving memory when the saved keywords reappear in future turns.
    Proper nouns are weighted higher (3x) than common nouns (0.3x).

    Args:
        state: Current RL state.
        action: Memory action taken (should be SAVE).
        future_turns: Turns that follow this action.
        proper_noun_multiplier: Weight multiplier for proper nouns.
        common_noun_multiplier: Weight multiplier for common nouns.

    Returns:
        Accumulated score from keyword reappearances.
    """
    if not action.saved_keywords or not future_turns:
        return 0.0

    future_text = " ".join(t.content for t in future_turns)
    score = 0.0
    for keyword in action.saved_keywords:
        if keyword in future_text:
            # Determine if proper noun via MeCab
            tokens = _parse_mecab(keyword)
            if tokens and tokens[0][1] == PROPER_NOUN_TAG:
                score += proper_noun_multiplier
            else:
                score += common_noun_multiplier

    # Normalize by keyword count to keep in [0, 3]
    return score / max(len(action.saved_keywords), 1)


def compute_r2_repeated_question_penalty(
    current_turn: Turn,
    history_turns: list[Turn],
) -> float:
    """R2 (-) Repeated question penalty.

    Penalizes when the same question is asked again, unless the re-question
    is a legitimate clarification, elaboration request, or time-gap follow-up.

    Args:
        current_turn: The turn being evaluated.
        history_turns: Previous turns in the conversation.

    Returns:
        Negative score (0.0 = no penalty, -1.0 = full penalty).
    """
    text = current_turn.content

    # Check if this is a clarification or elaboration (excluded from penalty)
    for pattern in CLARIFICATION_PATTERNS + ELABORATION_PATTERNS:
        if pattern in text:
            return 0.0

    if not history_turns:
        return 0.0

    # Check for semantic similarity by counting shared keywords
    current_keywords = set(_extract_keywords_from_text(text))
    if not current_keywords:
        # Fallback: simple character overlap
        current_words = set(text.split())
        for turn in history_turns:
            if turn.role == current_turn.role:
                prev_words = set(turn.content.split())
                overlap = len(current_words & prev_words) / max(len(current_words), 1)
                if overlap > 0.6:
                    return -1.0
        return 0.0

    for turn in history_turns:
        if turn.role != current_turn.role:
            continue
        prev_keywords = set(_extract_keywords_from_text(turn.content))
        if not prev_keywords:
            continue
        overlap = len(current_keywords & prev_keywords) / max(len(current_keywords), 1)
        if overlap > 0.6:
            return -1.0

    return 0.0


def compute_r3_efficiency(
    original_content: str,
    compressed_content: str,
) -> float:
    """R3 (+) Compression efficiency reward.

    Rewards memories that are both compressed (shorter) and still preserve
    substantive morpheme density (information content).

    compression_ratio = len(compressed) / len(original)
    density_preservation = density(compressed) / max(density(original), 1e-6)
    score = (1 - compression_ratio) * density_preservation

    Args:
        original_content: The original turn text.
        compressed_content: The memory-compressed version.

    Returns:
        Score in [0, 1].
    """
    if not original_content or not compressed_content:
        return 0.0

    orig_len = len(original_content)
    comp_len = len(compressed_content)

    if orig_len == 0:
        return 0.0

    compression_ratio = comp_len / orig_len

    def _substantive_density(text: str) -> float:
        tokens = _parse_mecab(text)
        if not tokens:
            words = text.split()
            substantive_words = [w for w in words if len(w) >= 2]
            return min(len(substantive_words) / max(len(words), 1), 1.0)
        substantive_count = sum(
            1 for surface, pos in tokens if _is_substantive_token(surface, pos)
        )
        return substantive_count / max(len(tokens), 1)

    orig_density = _substantive_density(original_content)
    comp_density = _substantive_density(compressed_content)

    density_preservation = comp_density / max(orig_density, 1e-6)
    density_preservation = min(density_preservation, 1.0)

    # Keyword preservation: what fraction of original keywords survived compression
    original_keywords = set(_extract_keywords_from_text(original_content))
    if original_keywords:
        compressed_keywords = set(_extract_keywords_from_text(compressed_content))
        keyword_preservation = len(original_keywords & compressed_keywords) / len(original_keywords)
    else:
        keyword_preservation = 1.0

    score = (1.0 - compression_ratio) * density_preservation * keyword_preservation
    return max(0.0, score)


def compute_r4_retrieval_relevance(
    retrieved_memories: list[MemoryEntry],
    current_context: str,
) -> float:
    """R4 (+) Retrieval relevance reward.

    Rewards retrievals where the retrieved memory keywords overlap
    significantly with the current turn context.

    Args:
        retrieved_memories: Memory entries that were retrieved.
        current_context: Text of the current turn/context.

    Returns:
        Score in [0, 1].
    """
    if not retrieved_memories or not current_context:
        return 0.0

    context_keywords = set(_extract_keywords_from_text(current_context))
    if not context_keywords:
        context_keywords = set(current_context.split())

    def _keyword_in_context(keyword: str, ctx_keywords: set[str], ctx_text: str) -> bool:
        """Check if keyword appears in context (exact or substring for Korean/mixed)."""
        if keyword in ctx_keywords:
            return True
        # For Korean text where morphological boundaries are not split, check
        # if the keyword appears as a substring in the raw context text
        if keyword in ctx_text:
            return True
        return False

    total_overlap = 0.0
    for memory in retrieved_memories:
        memory_keywords = list(memory.keywords)
        if not memory_keywords:
            memory_keywords = _extract_keywords_from_text(memory.content)
        if not memory_keywords:
            continue
        matched = sum(
            1 for kw in memory_keywords
            if _keyword_in_context(kw, context_keywords, current_context)
        )
        overlap = matched / max(len(memory_keywords), 1)
        total_overlap += overlap

    return min(total_overlap / max(len(retrieved_memories), 1), 1.0)


def compute_r5_speech_act_weight(text: str) -> float:
    """R5 (+) Speech act weight via sentence-final ending patterns.

    Counts how many sentences in the text end with Korean sentence-final
    endings that indicate complete speech acts.

    Args:
        text: The utterance text to analyze.

    Returns:
        Normalized score in [0, 1].
    """
    if not text:
        return 0.0

    sentences = re.split(r"[.!?。！？\n]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    matched = 0
    for sentence in sentences:
        for ending in SENTENCE_FINAL_ENDINGS:
            if sentence.endswith(ending):
                matched += 1
                break

    return matched / len(sentences)


def compute_r6_self_reference(text: str) -> float:
    """R6 (+) Self-reference reward.

    Detects utterances that contain a first-person expression combined
    with a fact, preference, or experience statement — indicating the
    user is sharing personal information worth remembering.

    Uses graduated scoring based on keyword novelty to differentiate
    information-rich self-references from minimal ones.

    Args:
        text: The user turn text to analyze.

    Returns:
        Score in {0.0, 0.3, 0.5, 0.7, 1.0}.
    """
    if not text:
        return 0.0

    has_first_person = any(expr in text for expr in FIRST_PERSON_EXPRESSIONS)
    if not has_first_person:
        return 0.0

    all_utterance_patterns = (
        PREFERENCE_UTTERANCE_PATTERNS
        + EXPERIENCE_UTTERANCE_PATTERNS
    )
    has_utterance = any(pat in text for pat in all_utterance_patterns)

    # Count substantive keywords for graduated scoring
    keywords = _extract_keywords_from_text(text)
    keyword_count = len(keywords)

    if has_utterance:
        # Graduated: more unique keywords = higher score
        if keyword_count >= 4:
            return 1.0
        elif keyword_count >= 2:
            return 0.7
        else:
            return 0.3

    # Softer signal: first-person + sentence-final ending
    r5 = compute_r5_speech_act_weight(text)
    if r5 > 0.5:
        if keyword_count >= 3:
            return 0.5
        return 0.3

    return 0.0


def compute_r7_info_density(text: str) -> float:
    """R7 (+) Information density via substantive morpheme ratio (MeCab).

    The ratio of substantive morphemes (nouns, numerals, proper nouns)
    to total morphemes in the text.

    Args:
        text: The text to analyze.

    Returns:
        Density score in [0, 1].
    """
    if not text:
        return 0.0

    tokens = _parse_mecab(text)
    if not tokens:
        words = text.split()
        substantive_words = [w for w in words if len(w) >= 2]
        return min(len(substantive_words) / max(len(words), 1), 1.0)

    substantive_count = sum(
        1 for surface, pos in tokens if _is_substantive_token(surface, pos)
    )
    return substantive_count / len(tokens)


def compute_r8_preference_constraint(text: str) -> float:
    """R8 (+) Preference/constraint expression detection.

    Detects patterns like "항상", "절대", "~로 해주세요" indicating strong
    user preferences or constraints that should be remembered.

    Args:
        text: The user turn text to analyze.

    Returns:
        Score in [0, 1] (higher = more preference/constraint signals).
    """
    if not text:
        return 0.0

    score = 0.0
    matched = 0

    for expr in PREFERENCE_EXPRESSIONS:
        if expr in text:
            matched += 1

    for expr in CONSTRAINT_EXPRESSIONS:
        if expr in text:
            matched += 1

    for pat in REQUEST_PATTERNS:
        if pat in text:
            matched += 1

    # Each match contributes; cap at 1.0
    score = min(matched * 0.33, 1.0)
    return score


def compute_r9_emotional_salience(text: str) -> float:
    """R9 (+) Emotional salience reward.

    Rewards utterances that combine emphasis expressions (진짜, 너무, 완전...)
    with substantive factual content (high info density).

    Args:
        text: The text to analyze.

    Returns:
        Score in [0, 1].
    """
    if not text:
        return 0.0

    has_emphasis = any(expr in text for expr in EMPHASIS_EXPRESSIONS)
    if not has_emphasis:
        return 0.0

    info_density = compute_r7_info_density(text)
    # Only reward if there's also substantive content
    if info_density < 0.1:
        return 0.0

    return min(info_density * 1.5, 1.0)


def compute_r10_topic_boundary(
    current_turn: Turn,
    previous_summary: str | None = None,
) -> float:
    """R10 (+) Topic boundary detection via discourse markers.

    When a discourse marker is detected, it signals a topic shift.
    The reward encourages saving a summary of the previous topic
    before the boundary.

    Args:
        current_turn: The current turn being evaluated.
        previous_summary: Whether a previous topic summary exists.

    Returns:
        Score (1.0 if topic boundary detected and summary available,
               0.5 if boundary detected but no prior summary,
               0.0 otherwise).
    """
    text = current_turn.content

    has_discourse_marker = any(marker in text for marker in DISCOURSE_MARKERS)
    if not has_discourse_marker:
        return 0.0

    if previous_summary:
        return 1.0
    return 0.5


def compute_r11_user_feedback(text: str) -> float:
    """R11 (+/-) User feedback signal.

    Detects direct positive or negative feedback from the user about
    the assistant's memory/recall quality.

    - Positive feedback (맞아요, 정확해요, ...) → +1.0
    - Negative feedback (아니 그게 아니라, ...) → -1.0
    - No feedback detected → 0.0

    Args:
        text: The user turn text to analyze.

    Returns:
        Score in {-1.0, 0.0, +1.0}.
    """
    if not text:
        return 0.0

    for phrase in NEGATIVE_FEEDBACK:
        if phrase in text:
            return -1.0

    for phrase in POSITIVE_FEEDBACK:
        if phrase in text:
            return 1.0

    return 0.0

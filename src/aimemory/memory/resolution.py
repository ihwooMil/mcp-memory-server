"""Multi-resolution text generation for memory nodes.

Provides three levels of text resolution:
- Level 0: Full original text
- Level 1: Keyword-focused summary (1-2 sentences, ~100 chars)
- Level 2: Entity triple (subject, predicate, object)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MultiResolutionText:
    """Container for all resolution levels of a memory text."""

    level0: str  # original full text
    level1: str  # keyword-focused summary
    level2: str  # entity triple


def generate_level1(text: str, keywords: list[str] | None = None) -> str:
    """Generate Level 1 summary: keyword-containing sentences (max 100 chars).

    Extracts sentences that contain any of the given keywords.
    Falls back to first sentence if no keyword match.
    """
    if not text.strip():
        return ""

    keywords = keywords or []
    # Split into sentences (Korean-aware)
    sentences = re.split(r"(?<=[.!?。])\s*", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:100]

    if not keywords:
        return sentences[0][:100]

    # Find sentences containing keywords
    matched: list[str] = []
    for sent in sentences:
        for kw in keywords:
            if kw in sent:
                matched.append(sent)
                break

    if not matched:
        return sentences[0][:100]

    # Combine up to 2 sentences, max 100 chars
    result = matched[0]
    if len(matched) > 1:
        candidate = f"{matched[0]} {matched[1]}"
        if len(candidate) <= 100:
            result = candidate

    return result[:100]


def generate_level2(text: str, keywords: list[str] | None = None) -> str:
    """Generate Level 2 entity triple: (subject, predicate, object).

    Uses heuristic extraction for Korean text.
    Returns format: "subject,predicate,object"
    """
    if not text.strip():
        return ""

    keywords = keywords or []

    # Try to extract subject from keywords or text patterns
    subject = ""
    predicate = ""
    obj = ""

    # Korean subject markers
    subj_patterns = [
        r"([\w]+(?:는|은|이|가))",  # noun+subject marker
    ]

    # Korean predicate patterns (verb/adjective endings)
    pred_patterns = [
        r"([\w]+(?:를|을|에서|에|으로|로|와|과|하고))\s*([\w]+(?:합니다|해요|했어요|입니다|이에요|예요|좋아|싫어|있어|없어|먹어|마셔))",
    ]

    # Try keyword-based extraction first
    if keywords:
        subject = keywords[0]
        if len(keywords) > 1:
            obj = keywords[1]

    # Try pattern-based extraction
    for pattern in subj_patterns:
        match = re.search(pattern, text)
        if match:
            subject = subject or match.group(1)
            break

    # Extract predicate and object from text
    for pattern in pred_patterns:
        match = re.search(pattern, text)
        if match:
            obj = obj or match.group(1)
            predicate = predicate or match.group(2)
            break

    # Fallback: simple split-based extraction
    if not predicate:
        # Try to find a verb ending
        verb_match = re.search(
            r"([\w]+(?:합니다|해요|했어요|입니다|이에요|예요|있어요|없어요|좋아해요|싫어해요|먹어요|마셔요|좋아합니다))",
            text,
        )
        if verb_match:
            predicate = verb_match.group(1)

    if not subject:
        subject = keywords[0] if keywords else text.split()[0] if text.split() else ""
    if not predicate:
        # Last resort: take main verb concept
        words = text.split()
        predicate = words[-1] if words else ""
    if not obj:
        obj = keywords[1] if len(keywords) > 1 else ""

    # Clean markers from subject
    subject = re.sub(r"(는|은|이|가)$", "", subject)

    return f"{subject},{predicate},{obj}"


def generate_all_levels(
    text: str, keywords: list[str] | None = None
) -> MultiResolutionText:
    """Generate all resolution levels for a text."""
    return MultiResolutionText(
        level0=text,
        level1=generate_level1(text, keywords),
        level2=generate_level2(text, keywords),
    )


def estimate_tokens(text: str) -> int:
    """Estimate token count for Korean text.

    Uses len/2.5 approximation for Korean characters.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 2.5))

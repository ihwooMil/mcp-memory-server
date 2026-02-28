"""Multi-resolution text generation for memory nodes.

Provides three levels of text resolution:
- Level 0: Full original text
- Level 1: Keyword-focused summary (1-2 sentences, ~100 chars)
- Level 2: Entity triple (subject, predicate, object)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aimemory.i18n import get_patterns


@dataclass
class MultiResolutionText:
    """Container for all resolution levels of a memory text."""

    level0: str  # original full text
    level1: str  # keyword-focused summary
    level2: str  # entity triple


def generate_level1(text: str, keywords: list[str] | None = None, lang: str = "ko") -> str:
    """Generate Level 1 summary: keyword-containing sentences (max 100 chars)."""
    if not text.strip():
        return ""

    keywords = keywords or []
    # Split into sentences
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


def generate_level2(text: str, keywords: list[str] | None = None, lang: str = "ko") -> str:
    """Generate Level 2 entity triple: (subject, predicate, object).

    Returns format: "subject,predicate,object"
    """
    if not text.strip():
        return ""

    keywords = keywords or []
    lp = get_patterns(lang)

    subject = ""
    predicate = ""
    obj = ""

    subj_patterns = [lp.subject_markers]
    pred_patterns = [lp.predicate_patterns]

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
            if match.lastindex and match.lastindex >= 2:
                predicate = predicate or match.group(2)
            break

    # Fallback: verb ending search
    if not predicate and lp.verb_endings:
        verb_match = re.search(lp.verb_endings, text)
        if verb_match:
            predicate = verb_match.group(1)

    if not subject:
        subject = keywords[0] if keywords else text.split()[0] if text.split() else ""
    if not predicate:
        words = text.split()
        predicate = words[-1] if words else ""
    if not obj:
        obj = keywords[1] if len(keywords) > 1 else ""

    # Clean markers from subject
    if lp.subject_strip:
        subject = re.sub(lp.subject_strip, "", subject)

    return f"{subject},{predicate},{obj}"


def generate_all_levels(
    text: str,
    keywords: list[str] | None = None,
    lang: str = "ko",
) -> MultiResolutionText:
    """Generate all resolution levels for a text."""
    return MultiResolutionText(
        level0=text,
        level1=generate_level1(text, keywords, lang=lang),
        level2=generate_level2(text, keywords, lang=lang),
    )


def estimate_tokens(text: str, lang: str = "ko") -> int:
    """Estimate token count for text based on language."""
    if not text:
        return 0
    lp = get_patterns(lang)
    return max(1, int(len(text) / lp.chars_per_token))

"""Rule-based memory decision agent.

Handles keyword/entity extraction and save/skip/retrieve decisions
using regex-based heuristics and probabilistic rules.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field

from aimemory.reward.korean_patterns import DISCOURSE_MARKERS
from aimemory.schemas import (
    MemoryActionType,
    MemoryDecision,
    MemoryEntry,
    Role,
    Turn,
)

logger = logging.getLogger(__name__)


# ─── Keyword extraction patterns ───

# Korean stopwords — common meaningless words to filter out
_KOREAN_STOPWORDS: set[str] = {
    "것", "수", "때", "거", "게", "줄", "데", "말", "점", "중",
    "건", "뭐", "저", "제", "내", "그", "이", "더", "안", "좀",
    "걸", "곳",
}

# Korean proper nouns / technical terms heuristics
_TECH_KEYWORDS = re.compile(
    r"(?<![가-힣a-zA-Z_])(?:"
    r"Python|Java(?:Script)?|TypeScript|Rust|Go|C\+\+|Ruby|Swift|Kotlin|Dart|"
    r"React|Vue|Angular|Next\.js|Django|Flask|FastAPI|Spring|Rails|"
    r"Docker|Kubernetes|k8s|AWS|GCP|Azure|Linux|Ubuntu|macOS|Windows|"
    r"MySQL|PostgreSQL|SQLite|Redis|MongoDB|Elasticsearch|"
    r"Git|GitHub|GitLab|CI/CD|DevOps|MLOps|"
    r"pandas|numpy|scipy|sklearn|scikit-learn|TensorFlow|PyTorch|Keras|"
    r"LLM|GPT|Claude|Gemini|머신러닝|딥러닝|인공지능|AI|"
    r"API|REST|GraphQL|WebSocket|gRPC|"
    r"알고리즘|자료구조|데이터베이스|클라우드|마이크로서비스"
    r")(?![a-zA-Z_])",
    re.IGNORECASE,
)

# Korean personal info patterns (name-like nouns, preferences)
_PREFERENCE_PATTERNS = [
    re.compile(r"좋아(?:해요|합니다|하는|함)"),
    re.compile(r"싫어(?:해요|합니다|하는|함)"),
    re.compile(r"선호(?:해요|합니다|하는|함|하다)"),
    re.compile(r"취미(?:가|는|로)?"),
    re.compile(r"주로\s+\S+"),
    re.compile(r"즐겨\s*\S+"),
]

_PERSONAL_INFO_PATTERNS = [
    re.compile(r"저는?\s+(.{2,20})(?:이에요|입니다|예요|인데|거든요|이라서)"),
    re.compile(r"제\s+(?:이름|나이|직업|전공|회사|팀|프로젝트)"),
    re.compile(r"살고\s*있(?:어요|습니다)"),
    re.compile(r"다니고\s*있(?:어요|습니다)"),
    re.compile(r"일하고\s*있(?:어요|습니다)"),
]

_QUESTION_PATTERN = re.compile(r"[?？]|(?:인가요|나요|을까요|ㄹ까요|어요\?|습니까)")
_PARAPHRASE_PATTERN = re.compile(
    r"하시는군요|이시군요|이시네요|하셨군요|좋아하시|관심이\s*있으시"
)
_EMOTION_KEYWORDS = re.compile(
    r"기쁘|슬프|화나|무서|불안|설레|걱정|힘들|어렵|좋아|싫어|즐거|행복|우울|피곤|신나"
)


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from Korean text using MeCab + regex."""
    keywords: list[str] = []

    # 1) Technical terms (English or Korean tech vocab)
    for match in _TECH_KEYWORDS.finditer(text):
        kw = match.group(0)
        if kw not in keywords:
            keywords.append(kw)

    # 2) Quoted strings often contain important terms
    for match in re.finditer(r"['\"]([^'\"]{2,30})['\"]", text):
        kw = match.group(1).strip()
        if kw and kw not in keywords:
            keywords.append(kw)

    # 3) MeCab-based noun extraction (primary method for Korean)
    try:
        import MeCab
        import mecab_ko_dic
        tagger = MeCab.Tagger(f"-d {mecab_ko_dic.DICDIR}")
        result = tagger.parse(text)
        for line in result.splitlines():
            if line == "EOS" or not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            surface = parts[0]
            pos = parts[1].split(",")[0]
            # NNG=일반명사, NNP=고유명사, SL=외래어
            if pos in ("NNG", "NNP", "SL") and len(surface) >= 2:
                if surface not in keywords and surface not in _KOREAN_STOPWORDS:
                    keywords.append(surface)
    except Exception:
        # Fallback: Korean compound nouns via particle regex
        for match in re.finditer(
            r"[\uAC00-\uD7A3]{2,6}(?=을|를|이|가|은|는|도|만|에서|으로|로|와|과|에|의)",
            text,
        ):
            kw = match.group(0)
            if len(kw) >= 2 and kw not in keywords and kw not in _KOREAN_STOPWORDS:
                keywords.append(kw)

    return keywords[:10]  # cap at 10 keywords


def _extract_korean_nouns(text: str) -> list[str]:
    """Fallback keyword extraction: extract Korean nouns via MeCab if available,
    otherwise extract 2+ char Korean words from whitespace split."""
    try:
        import MeCab
        import mecab_ko_dic
        tagger = MeCab.Tagger(f"-d {mecab_ko_dic.DICDIR}")
        result = tagger.parse(text)
        nouns = []
        for line in result.splitlines():
            if line == "EOS" or not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            surface = parts[0]
            pos = parts[1].split(",")[0]
            if pos in ("NNG", "NNP", "SL") and len(surface) >= 2:
                if surface not in nouns and surface not in _KOREAN_STOPWORDS:
                    nouns.append(surface)
        return nouns[:10]
    except Exception:
        pass
    # Pure regex fallback: Korean words 2+ chars
    nouns = []
    for match in re.finditer(r"[\uAC00-\uD7A3]{2,}", text):
        kw = match.group(0)
        if kw not in nouns and kw not in _KOREAN_STOPWORDS:
            nouns.append(kw)
    return nouns[:10]


def _sentence_summary(text: str, keywords: list[str]) -> str:
    """Summarize text by keeping keyword-containing sentences, cut at sentence boundary."""
    # Split into sentences on Korean/English sentence endings
    sentences = re.split(r"(?<=[.!?。！？])\s*", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:150].strip()

    keyword_set = {kw.lower() for kw in keywords}

    # Find sentences containing at least one keyword
    keyword_sentences = [
        s for s in sentences
        if any(kw in s.lower() for kw in keyword_set)
    ]

    if not keyword_sentences:
        # No keyword sentences — use first sentence
        result = sentences[0]
        if len(result) > 150:
            result = result[:150].rstrip()
        return result

    # Keep up to 2 keyword-containing sentences
    selected = keyword_sentences[:2]
    result = " ".join(selected)

    if len(result) > 150:
        # Keep only the first selected sentence
        result = selected[0]
        if len(result) > 150:
            result = result[:150].rstrip()

    return result


def classify_category(text: str, keywords: list[str]) -> str:
    """Classify the memory category based on content heuristics."""
    tech_hits = len(_TECH_KEYWORDS.findall(text))
    pref_hits = sum(1 for p in _PREFERENCE_PATTERNS if p.search(text))
    personal_hits = sum(1 for p in _PERSONAL_INFO_PATTERNS if p.search(text))

    if tech_hits >= 1:
        return "technical"
    if pref_hits >= 1:
        return "preference"
    if personal_hits >= 1:
        return "personal"
    return "general"


@dataclass
class MemoryStore:
    """In-memory store with simple keyword-based retrieval."""

    entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def retrieve_relevant(self, query_keywords: list[str], top_k: int = 3) -> list[MemoryEntry]:
        """Return up to top_k entries that share keywords with query."""
        if not query_keywords or not self.entries:
            return []

        query_set = {kw.lower() for kw in query_keywords}
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self.entries:
            entry_set = {kw.lower() for kw in entry.keywords}
            overlap = len(query_set & entry_set)
            if overlap > 0:
                scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def __len__(self) -> int:
        return len(self.entries)


class MemoryAgent:
    """Rule-based memory decision agent.

    At each turn, decides whether to:
    - SAVE: store information from the turn into memory
    - RETRIEVE: look up relevant prior memories
    - SKIP: take no memory action
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def decide(
        self,
        turn: Turn,
        memory_store: MemoryStore,
        recent_turns: list[Turn],
    ) -> MemoryDecision:
        """Make a memory decision for the given turn."""
        # A2: Check if assistant turn paraphrases user info — if so, SAVE it
        if turn.role != Role.USER:
            has_paraphrase = bool(_PARAPHRASE_PATTERN.search(turn.content))
            if has_paraphrase:
                asst_keywords = extract_keywords(turn.content)
                if asst_keywords:
                    category = classify_category(turn.content, asst_keywords)
                    summary = _sentence_summary(turn.content, asst_keywords)
                    entry = MemoryEntry(
                        content=summary,
                        source_turn_id=turn.turn_id,
                        keywords=asst_keywords,
                        category=category,
                    )
                    return MemoryDecision(
                        turn_id=turn.turn_id,
                        action=MemoryActionType.SAVE,
                        memory_entry=entry,
                        reasoning="어시스턴트 발화에서 사용자 정보 패러프레이즈 감지",
                    )
            return MemoryDecision(
                turn_id=turn.turn_id,
                action=MemoryActionType.SKIP,
                reasoning="어시스턴트 발화 - 저장/검색 불필요",
            )

        keywords = extract_keywords(turn.content)
        is_question = bool(_QUESTION_PATTERN.search(turn.content))
        has_emotion = bool(_EMOTION_KEYWORDS.search(turn.content))
        has_personal = any(p.search(turn.content) for p in _PERSONAL_INFO_PATTERNS)
        has_preference = any(p.search(turn.content) for p in _PREFERENCE_PATTERNS)
        has_tech = bool(_TECH_KEYWORDS.search(turn.content))

        # ── Retrieve decision ──────────────────────────────────────────
        # A3: RETRIEVE when: question, OR keyword overlap ≥ 2, OR discourse marker
        has_discourse_marker = any(marker in turn.content for marker in DISCOURSE_MARKERS)
        has_keyword_overlap = False
        if keywords and len(memory_store) > 0:
            query_set = {kw.lower() for kw in keywords}
            for entry in memory_store.entries:
                entry_set = {kw.lower() for kw in entry.keywords}
                if len(query_set & entry_set) >= 1:
                    has_keyword_overlap = True
                    break

        if (is_question or has_keyword_overlap or has_discourse_marker) and keywords and len(memory_store) > 0:
            retrieved = memory_store.retrieve_relevant(keywords)
            if retrieved:
                return MemoryDecision(
                    turn_id=turn.turn_id,
                    action=MemoryActionType.RETRIEVE,
                    retrieved_memories=retrieved,
                    reasoning=f"질문 감지 + 관련 기억 {len(retrieved)}건 검색됨 (키워드: {keywords[:3]})",
                )

        # ── Save decision ──────────────────────────────────────────────
        # Save if: personal info / preference / technical content with keywords
        should_save = False
        save_reason = ""

        if has_personal and not is_question:
            should_save = True
            save_reason = "개인 정보 발화 감지"
        elif has_preference and not is_question:
            should_save = True
            save_reason = "선호도/취향 정보 감지"
        elif has_tech and len(keywords) >= 1 and not is_question:
            should_save = True
            save_reason = f"기술 관련 정보 감지 (키워드: {keywords[:3]})"
        elif has_emotion and not is_question:
            # Probabilistic: save 85% of emotional statements
            if self._rng.random() < 0.85:
                should_save = True
                save_reason = "감정/경험 발화 감지"
        elif keywords and not is_question:
            # Probabilistic: save 60% of other content with keywords
            if self._rng.random() < 0.6:
                should_save = True
                save_reason = "키워드 포함 발화 (확률적 저장)"

        if should_save:
            # If no keywords from regex, extract from Korean morphemes as fallback
            if not keywords:
                keywords = _extract_korean_nouns(turn.content)

            if keywords:
                category = classify_category(turn.content, keywords)
                # Summarize content at sentence boundary (A1)
                summary = _sentence_summary(turn.content, keywords)

                entry = MemoryEntry(
                    content=summary,
                    source_turn_id=turn.turn_id,
                    keywords=keywords,
                    category=category,
                )
                return MemoryDecision(
                    turn_id=turn.turn_id,
                    action=MemoryActionType.SAVE,
                    memory_entry=entry,
                    reasoning=save_reason,
                )

        return MemoryDecision(
            turn_id=turn.turn_id,
            action=MemoryActionType.SKIP,
            reasoning="저장/검색 조건 미충족",
        )

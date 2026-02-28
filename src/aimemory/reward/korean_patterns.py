"""Korean language pattern dictionaries for reward signal computation."""

from __future__ import annotations

# ─── 종결어미 (Sentence-final endings) ───
# Indicates speech act completion; used for R5
SENTENCE_FINAL_ENDINGS: list[str] = [
    "ㅂ니다",
    "습니다",
    "입니다",
    "합니다",
    "아요",
    "어요",
    "여요",
    "거든요",
    "잖아요",
    "할게요",
    "할께요",
    "해주세요",
    "해줘요",
    "해줘",
    "할래요",
    "할래",
    "네요",
    "군요",
    "구나",
    "겠어요",
    "겠습니다",
    "이에요",
    "예요",
    "이야",
    "이라",
    "이죠",
    "죠",
]

# ─── 담화 표지 (Discourse markers) ───
# Signals topic boundary; used for R10
DISCOURSE_MARKERS: list[str] = [
    "그건 그렇고",
    "아 맞다",
    "그나저나",
    "근데",
    "그런데",
    "참",
    "아참",
    "아 그리고",
    "그리고",
    "그래서",
    "그러면",
    "그러니까",
    "아 근데",
    "아무튼",
    "어쨌든",
    "얘기가 나온 김에",
    "그 얘기는 됐고",
    "다른 얘기인데",
    "주제를 바꿔서",
]

# ─── 1인칭 표현 (First-person expressions) ───
# Used for R6 (self-reference detection)
FIRST_PERSON_EXPRESSIONS: list[str] = [
    "저는",
    "저도",
    "저한테",
    "저에게",
    "저의",
    "제가",
    "제",
    "나는",
    "나도",
    "나한테",
    "나에게",
    "내가",
    "내",
    "나야",
    "저야",
]

# ─── 강조 표현 (Emphasis expressions) ───
# Used for R9 (emotional salience)
EMPHASIS_EXPRESSIONS: list[str] = [
    "진짜",
    "정말",
    "너무",
    "완전",
    "엄청",
    "굉장히",
    "매우",
    "아주",
    "되게",
    "엄청나게",
    "진심으로",
    "절대적으로",
    "확실히",
    "특히",
    "정말로",
    "진짜로",
]

# ─── 선호/제약 표현 (Preference and constraint expressions) ───
# Used for R8 (preference constraint detection)
PREFERENCE_EXPRESSIONS: list[str] = [
    "항상",
    "언제나",
    "매번",
    "반드시",
]

CONSTRAINT_EXPRESSIONS: list[str] = [
    "절대",
    "절대로",
    "결코",
    "절대 안",
    "절대로 안",
    "하지 마",
    "하지 마세요",
    "하지 말아요",
]

REQUEST_PATTERNS: list[str] = [
    "로 해주세요",
    "으로 해주세요",
    "로 해줘",
    "으로 해줘",
    "로 부탁해요",
    "로 해주시면",
    "게 해주세요",
    "게 해줘",
    "해주세요",
    "해줘요",
    "해줘",
    "부탁해요",
    "부탁드려요",
    "해주시겠어요",
    "해주실 수 있나요",
]

# ─── 긍정 피드백 (Positive feedback) ───
# Used for R11 (positive user feedback)
POSITIVE_FEEDBACK: list[str] = [
    "맞아요",
    "맞아",
    "맞습니다",
    "맞네요",
    "정확해요",
    "정확합니다",
    "잘 이해하셨네요",
    "그렇죠",
    "그렇습니다",
    "네 맞아요",
    "네 맞습니다",
    "딱 맞아요",
    "완벽해요",
    "좋아요",
    "좋습니다",
    "감사해요",
    "고마워요",
    "잘 기억하시네요",
    "기억해주셨네요",
]

# ─── 부정 피드백 (Negative feedback) ───
# Used for R11 (negative user feedback)
NEGATIVE_FEEDBACK: list[str] = [
    "아니 그게 아니라",
    "아니요 그게 아니라",
    "그게 아니라",
    "아닌데요",
    "아닙니다",
    "아니에요",
    "틀렸어요",
    "틀렸습니다",
    "잘못 이해하셨어요",
    "제가 말한 건 그게 아니라",
    "제 말은 그게 아니라",
    "다시 말씀드리면",
    "그건 제 말이 아니에요",
    "착각하신 것 같아요",
    "기억이 잘못되셨네요",
]

# ─── 재질문 제외 패턴 (Clarification/elaboration re-question exclusions) ───
# Used for R2 (exclude legitimate re-questions from penalty)
CLARIFICATION_PATTERNS: list[str] = [
    "무슨 말인지",
    "어떤 의미인지",
    "좀 더 설명해",
    "자세히 말해",
    "구체적으로 말해",
    "예를 들어",
    "다시 말해줘",
    "다시 설명해줘",
    "이해가 안 돼",
    "이해가 되지 않아",
    "잘 모르겠어",
    "무슨 뜻이에요",
    "어떤 뜻이에요",
]

ELABORATION_PATTERNS: list[str] = [
    "더 구체적으로",
    "더 자세히",
    "좀 더 알려줘",
    "계속 말해줘",
    "이어서 말해줘",
    "더 말해줘",
    "추가로",
    "그리고 또",
]

# ─── 사실/선호/경험 발화 패턴 (Fact/preference/experience utterance patterns) ───
# Used for R6 to qualify self-references
FACT_UTTERANCE_PATTERNS: list[str] = [
    "이에요",
    "예요",
    "입니다",
    "이야",
    "야",
    "아요",
    "어요",
]

PREFERENCE_UTTERANCE_PATTERNS: list[str] = [
    "좋아해요",
    "좋아해",
    "좋아합니다",
    "싫어해요",
    "싫어해",
    "싫어합니다",
    "선호해요",
    "원해요",
    "원합니다",
    "하고 싶어요",
    "하고 싶어",
]

EXPERIENCE_UTTERANCE_PATTERNS: list[str] = [
    "했어요",
    "했습니다",
    "했어",
    "해봤어요",
    "해봤어",
    "경험이 있어요",
    "경험했어요",
    "써봤어요",
    "써봤어",
    "가봤어요",
    "가봤어",
    "먹어봤어요",
]

# ─── MeCab POS tags for substantive morphemes ───
# Noun POS tags used in R7 (info density)
SUBSTANTIVE_POS_TAGS: set[str] = {
    "NNG",  # 일반명사 (common noun)
    "NNP",  # 고유명사 (proper noun)
    "NNB",  # 의존명사 (bound noun)
    "NR",  # 수사 (numeral)
    "SL",  # 외래어 (foreign word / loanword)
    "SH",  # 한자 (Chinese character)
}

# Proper noun POS tag
PROPER_NOUN_TAG: str = "NNP"

# Common/regular noun POS tags
COMMON_NOUN_TAGS: set[str] = {"NNG", "NNB", "NR", "SL", "SH"}

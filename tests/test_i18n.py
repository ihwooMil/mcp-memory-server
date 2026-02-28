"""Tests for multilingual (i18n) pattern support."""

from __future__ import annotations

import re

import pytest

from aimemory.i18n import get_patterns, available_languages


# ─── Registry ───


class TestRegistry:
    def test_available_languages(self):
        langs = available_languages()
        assert "ko" in langs
        assert "en" in langs

    def test_get_patterns_ko(self):
        lp = get_patterns("ko")
        assert lp.chars_per_token == 2.5

    def test_get_patterns_en(self):
        lp = get_patterns("en")
        assert lp.chars_per_token == 4.0

    def test_get_patterns_unknown_fallback(self):
        lp = get_patterns("xx")
        # Falls back to Korean
        assert lp.chars_per_token == 2.5


# ─── English patterns ───


class TestEnglishFeedback:
    @pytest.fixture
    def lp(self):
        return get_patterns("en")

    @pytest.mark.parametrize("text", [
        "That's right",
        "Yes, that's correct",
        "Exactly!",
        "You remembered!",
        "Perfect",
        "Spot on",
        "You're right",
    ])
    def test_positive_feedback_match(self, lp, text):
        matched = False
        for pattern_str, conf, name in lp.positive_feedback:
            if re.search(pattern_str, text):
                matched = True
                break
        assert matched, f"No positive pattern matched: {text!r}"

    @pytest.mark.parametrize("text", [
        "No, that's wrong",
        "I never said that",
        "That's incorrect",
        "You're wrong",
        "I already told you",
        "You already asked me that",
        "You misunderstood",
        "When did I say that?",
        "That's not what I said",
    ])
    def test_negative_feedback_match(self, lp, text):
        matched = False
        for pattern_str, conf, name in lp.negative_feedback:
            if re.search(pattern_str, text):
                matched = True
                break
        assert matched, f"No negative pattern matched: {text!r}"

    @pytest.mark.parametrize("text", [
        "That's helpful",
        "Thanks for remembering",
        "Good to know",
        "That helps",
        "Oh right, I forgot about that",
    ])
    def test_useful_feedback_match(self, lp, text):
        matched = False
        for pattern_str, conf, name in lp.useful_feedback:
            if re.search(pattern_str, text):
                matched = True
                break
        assert matched, f"No useful pattern matched: {text!r}"


class TestEnglishQuestionDetection:
    def test_question_mark(self):
        lp = get_patterns("en")
        pat = re.compile(lp.question_pattern)
        assert pat.search("What is your name?")
        assert pat.search("How does it work?")
        assert pat.search("Can you help me?")

    def test_question_keywords(self):
        lp = get_patterns("en")
        pat = re.compile(lp.question_pattern)
        assert pat.search("what is this")
        assert pat.search("how does it work")
        assert pat.search("do you know")

    def test_non_question(self):
        lp = get_patterns("en")
        pat = re.compile(lp.question_pattern)
        assert not pat.search("I like cats")


class TestEnglishEntityExtraction:
    def test_word_extraction(self):
        lp = get_patterns("en")
        pat = re.compile(lp.word_extraction_pattern)
        words = pat.findall("I really love Python programming")
        assert "really" in words
        assert "love" in words
        assert "Python" in words
        assert "programming" in words

    def test_short_words_excluded(self):
        lp = get_patterns("en")
        pat = re.compile(lp.word_extraction_pattern)
        words = pat.findall("I am a cat")
        # "I", "am", "a" are < 3 chars, excluded
        assert "cat" in words
        assert "I" not in words


class TestEnglishDismissive:
    @pytest.mark.parametrize("text", [
        "ok", "okay", "sure", "fine", "yeah", "yep",
        "mhm", "uh-huh", "alright", "got it", "I see",
    ])
    def test_dismissive_match(self, text):
        lp = get_patterns("en")
        pat = re.compile(lp.dismissive_pattern)
        assert pat.match(text), f"Dismissive pattern should match: {text!r}"

    def test_non_dismissive(self):
        lp = get_patterns("en")
        pat = re.compile(lp.dismissive_pattern)
        assert not pat.match("That's interesting, tell me more")


class TestEnglishImportance:
    def test_personal_info(self):
        lp = get_patterns("en")
        pats = [re.compile(p) for p in lp.personal_info_patterns]
        assert any(p.search("I'm a software engineer") for p in pats)
        assert any(p.search("My name is John") for p in pats)

    def test_preference(self):
        lp = get_patterns("en")
        pats = [re.compile(p) for p in lp.preference_patterns]
        assert any(p.search("I like Python a lot") for p in pats)
        assert any(p.search("I hate debugging") for p in pats)

    def test_tech_keywords(self):
        lp = get_patterns("en")
        pat = re.compile(lp.tech_keywords, re.IGNORECASE)
        assert pat.search("I use Python and Django")
        assert pat.search("deployed on AWS")

    def test_emotion(self):
        lp = get_patterns("en")
        pat = re.compile(lp.emotion_keywords)
        assert pat.search("I'm so happy today")
        assert pat.search("I feel tired")


# ─── Korean patterns preservation ───


class TestKoreanPreservation:
    @pytest.fixture
    def lp(self):
        return get_patterns("ko")

    def test_positive_feedback_count(self, lp):
        assert len(lp.positive_feedback) == 14

    def test_negative_feedback_count(self, lp):
        assert len(lp.negative_feedback) == 14

    def test_useful_feedback_count(self, lp):
        assert len(lp.useful_feedback) == 5

    def test_korean_positive_match(self, lp):
        matched = False
        for pattern_str, conf, name in lp.positive_feedback:
            if re.search(pattern_str, "맞아요"):
                matched = True
                break
        assert matched

    def test_korean_negative_match(self, lp):
        matched = False
        for pattern_str, conf, name in lp.negative_feedback:
            if re.search(pattern_str, "틀렸어요"):
                matched = True
                break
        assert matched

    def test_discourse_markers(self, lp):
        assert "근데" in lp.discourse_markers
        assert "그런데" in lp.discourse_markers
        assert len(lp.discourse_markers) == 19

    def test_first_person(self, lp):
        assert "저는" in lp.first_person
        assert "나는" in lp.first_person

    def test_word_extraction_korean(self, lp):
        pat = re.compile(lp.word_extraction_pattern)
        words = pat.findall("파이썬 프로그래밍을 좋아합니다")
        assert "파이썬" in words or "프로그래밍을" in words

    def test_dismissive_korean(self, lp):
        pat = re.compile(lp.dismissive_pattern)
        assert pat.match("그래")
        assert pat.match("응")
        assert pat.match("네")

    def test_correction_names(self, lp):
        assert "아닌데(요)" in lp.correction_names
        assert "착각+variants" in lp.correction_names

    def test_substantive_pos_tags(self, lp):
        assert "NNG" in lp.substantive_pos_tags
        assert "NNP" in lp.substantive_pos_tags


# ─── FeedbackDetector integration ───


class TestFeedbackDetectorI18n:
    def test_korean_detector(self):
        from aimemory.reward.feedback_detector import FeedbackDetector, FeedbackType
        from aimemory.schemas import MemoryActionType, Role, Turn

        detector = FeedbackDetector(lang="ko")
        turn = Turn(turn_id=1, role=Role.USER, content="맞아요! 잘 기억하시네요")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_english_detector(self):
        from aimemory.reward.feedback_detector import FeedbackDetector, FeedbackType
        from aimemory.schemas import MemoryActionType, Role, Turn

        detector = FeedbackDetector(lang="en")
        turn = Turn(turn_id=1, role=Role.USER, content="That's right, you remembered!")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_english_negative(self):
        from aimemory.reward.feedback_detector import FeedbackDetector, FeedbackType
        from aimemory.schemas import MemoryActionType, Role, Turn

        detector = FeedbackDetector(lang="en")
        turn = Turn(turn_id=1, role=Role.USER, content="That's wrong, I never said that")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type in (FeedbackType.MEMORY_FAILURE, FeedbackType.MEMORY_ERROR)


# ─── ImplicitRewardDetector integration ───


class TestImplicitDetectorI18n:
    def test_english_dismissive(self):
        from aimemory.reward.implicit_detector import ImplicitRewardDetector
        from aimemory.schemas import Role, Turn

        detector = ImplicitRewardDetector(lang="en")
        turns = [Turn(turn_id=1, role=Role.USER, content="ok")]
        reward = detector.detect(turns, ["some memory content here"])
        assert reward < 0

    def test_korean_dismissive(self):
        from aimemory.reward.implicit_detector import ImplicitRewardDetector
        from aimemory.schemas import Role, Turn

        detector = ImplicitRewardDetector(lang="ko")
        turns = [Turn(turn_id=1, role=Role.USER, content="그래")]
        reward = detector.detect(turns, ["기억 내용"])
        assert reward < 0


# ─── Resolution integration ───


class TestResolutionI18n:
    def test_estimate_tokens_ko(self):
        from aimemory.memory.resolution import estimate_tokens
        tokens = estimate_tokens("한국어 텍스트입니다", lang="ko")
        assert tokens == max(1, int(len("한국어 텍스트입니다") / 2.5))

    def test_estimate_tokens_en(self):
        from aimemory.memory.resolution import estimate_tokens
        tokens = estimate_tokens("This is English text", lang="en")
        assert tokens == max(1, int(len("This is English text") / 4.0))

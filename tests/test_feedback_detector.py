"""Tests for the Korean feedback detector module.

Covers all FeedbackType variants with Korean morphological variations,
context-dependent detection, edge cases, and repeated question detection.
"""

from __future__ import annotations

import pytest

from aimemory.reward.feedback_detector import (
    FeedbackDetector,
    FeedbackSignal,
    FeedbackType,
    _char_ngrams,
    _jaccard_similarity,
)
from aimemory.schemas import MemoryActionType, Role, Turn


# ─── Fixtures ───


@pytest.fixture
def detector() -> FeedbackDetector:
    return FeedbackDetector()


def _make_turn(
    content: str,
    turn_id: int = 0,
    role: Role = Role.USER,
) -> Turn:
    return Turn(turn_id=turn_id, role=role, content=content)


# ─── MEMORY_CORRECT: positive confirmation (+1.0) ───


class TestMemoryCorrect:
    """User confirms that the recalled memory is correct."""

    @pytest.mark.parametrize(
        "text",
        [
            "맞아",
            "맞아요",
            "맞습니다",
            "맞잖아",
            "맞잖아요",
            "맞네",
            "맞네요",
            "맞죠",
            "맞지",
            "맞지요",
        ],
    )
    def test_confirmation_variants(self, detector: FeedbackDetector, text: str):
        """All morphological variants of 맞아 should detect MEMORY_CORRECT."""
        turn = _make_turn(text)
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT
        assert signal.reward_value == 1.0
        assert signal.confidence > 0.0

    def test_ne_맞아요(self, detector: FeedbackDetector):
        turn = _make_turn("네 맞아요, 제가 그렇게 말했어요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_정확해요(self, detector: FeedbackDetector):
        turn = _make_turn("정확해요!")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_정확합니다(self, detector: FeedbackDetector):
        turn = _make_turn("네, 정확합니다.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_잘_기억하네요(self, detector: FeedbackDetector):
        turn = _make_turn("잘 기억하네요!")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_잘_기억하시네요(self, detector: FeedbackDetector):
        turn = _make_turn("잘 기억하시네요, 제가 Python 좋아한다고 했죠.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_기억하고_있네(self, detector: FeedbackDetector):
        turn = _make_turn("기억하고 있네, 대단하다.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_그랬지(self, detector: FeedbackDetector):
        turn = _make_turn("그랬지, 내가 그때 말했잖아.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_그랬죠(self, detector: FeedbackDetector):
        turn = _make_turn("그랬죠.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_기억나(self, detector: FeedbackDetector):
        turn = _make_turn("아 기억나! 그때 그랬지.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_기억나요(self, detector: FeedbackDetector):
        turn = _make_turn("기억나요, 맞아요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_아_그때(self, detector: FeedbackDetector):
        turn = _make_turn("아 그때 일이구나.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_덕분에_생각났다(self, detector: FeedbackDetector):
        turn = _make_turn("덕분에 생각났어요!")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_그거_말하는_거지(self, detector: FeedbackDetector):
        turn = _make_turn("아 그거 말하는 거지!")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_딱_맞아(self, detector: FeedbackDetector):
        turn = _make_turn("딱 맞아요, 제가 그렇게 얘기했어요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_완벽해요(self, detector: FeedbackDetector):
        turn = _make_turn("완벽해요!")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_기억해주셨네요(self, detector: FeedbackDetector):
        turn = _make_turn("기억해 주셨네요, 감사해요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_after_save_action(self, detector: FeedbackDetector):
        """MEMORY_CORRECT also fires after SAVE action."""
        turn = _make_turn("맞아요, 그거 기억해두면 좋겠어요.")
        signal = detector.detect(turn, [], MemoryActionType.SAVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT


# ─── MEMORY_USEFUL: helpful recall (+0.7) ───


class TestMemoryUseful:
    """User indicates the recalled info was helpful."""

    def test_도움이_됐어요(self, detector: FeedbackDetector):
        turn = _make_turn("도움이 됐어요, 감사합니다.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_USEFUL
        assert signal.reward_value == 0.7

    def test_유용하네요(self, detector: FeedbackDetector):
        turn = _make_turn("유용하네요, 그 정보가 필요했어요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_USEFUL

    def test_알려줘서_고마워(self, detector: FeedbackDetector):
        turn = _make_turn("알려줘서 고마워요!")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_USEFUL

    def test_좋은_정보(self, detector: FeedbackDetector):
        turn = _make_turn("좋은 정보네요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_USEFUL


# ─── MEMORY_FAILURE: forgetting / not recalling (-1.0) ───


class TestMemoryFailure:
    """User points out the agent forgot previously shared information."""

    def test_기억_안_나(self, detector: FeedbackDetector):
        turn = _make_turn("기억 안 나요?")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_FAILURE
        assert signal.reward_value == -1.0

    def test_기억_못_해(self, detector: FeedbackDetector):
        turn = _make_turn("기억 못 해요?")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_FAILURE

    def test_이미_말했잖아(self, detector: FeedbackDetector):
        turn = _make_turn("이미 말했잖아요, 아까 말씀드렸는데.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_FAILURE

    def test_아까도_물어봤잖아(self, detector: FeedbackDetector):
        turn = _make_turn("아까도 물어봤잖아요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_FAILURE

    def test_또_물어봐(self, detector: FeedbackDetector):
        turn = _make_turn("또 물어봐요?")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_FAILURE

    def test_방금_말했잖아(self, detector: FeedbackDetector):
        turn = _make_turn("방금 말했잖아요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_FAILURE

    def test_벌써_잊어버렸어(self, detector: FeedbackDetector):
        turn = _make_turn("벌써 잊어버렸어요?")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_FAILURE

    def test_아까도_얘기했잖아(self, detector: FeedbackDetector):
        turn = _make_turn("아까도 얘기했잖아요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_FAILURE


# ─── MEMORY_ERROR: factual correction (-1.5) ───


class TestMemoryError:
    """User corrects a factual error in the agent's memory."""

    def test_아닌데요(self, detector: FeedbackDetector):
        turn = _make_turn("아닌데요, 제가 말한 건 Python이에요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR
        assert signal.reward_value == -1.5

    def test_아닌데(self, detector: FeedbackDetector):
        turn = _make_turn("아닌데, 그건 다른 사람이야.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_그게_아니라(self, detector: FeedbackDetector):
        turn = _make_turn("그게 아니라 제가 말한 건 Rust예요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_그게_아니에요(self, detector: FeedbackDetector):
        turn = _make_turn("그게 아니에요, 전혀 다른 얘기예요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_내가_언제_그랬어(self, detector: FeedbackDetector):
        turn = _make_turn("내가 언제 그랬어?")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_내가_언제_그랬어요(self, detector: FeedbackDetector):
        turn = _make_turn("내가 언제 그랬어요?")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_잘못_기억(self, detector: FeedbackDetector):
        turn = _make_turn("잘못 기억하시는 것 같아요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_틀렸어요(self, detector: FeedbackDetector):
        turn = _make_turn("틀렸어요, 제가 말한 건 서울이 아니라 부산이에요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_틀렸습니다(self, detector: FeedbackDetector):
        turn = _make_turn("틀렸습니다.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_착각(self, detector: FeedbackDetector):
        turn = _make_turn("착각하신 것 같아요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_그건_내_말이_아니(self, detector: FeedbackDetector):
        turn = _make_turn("그건 제 말이 아니에요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR


# ─── Context-dependent detection ───


class TestContextDependence:
    """Feedback classification depends on last_action context."""

    def test_positive_after_skip_is_neutral(self, detector: FeedbackDetector):
        """맞아요 after SKIP should be NEUTRAL (not memory-related context)."""
        turn = _make_turn("맞아요, 오늘 날씨 좋죠.")
        signal = detector.detect(turn, [], MemoryActionType.SKIP)
        assert signal.signal_type == FeedbackType.NEUTRAL
        assert signal.reward_value == 0.0

    def test_negative_after_skip_is_neutral(self, detector: FeedbackDetector):
        """아닌데요 after SKIP should be NEUTRAL."""
        turn = _make_turn("아닌데요, 그건 다른 얘기예요.")
        signal = detector.detect(turn, [], MemoryActionType.SKIP)
        assert signal.signal_type == FeedbackType.NEUTRAL

    def test_positive_after_retrieve_is_correct(self, detector: FeedbackDetector):
        """맞아요 after RETRIEVE should be MEMORY_CORRECT."""
        turn = _make_turn("맞아요!")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_negative_after_retrieve_is_error(self, detector: FeedbackDetector):
        """아닌데요 after RETRIEVE should be MEMORY_ERROR."""
        turn = _make_turn("아닌데요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_positive_after_save_is_correct(self, detector: FeedbackDetector):
        """맞아요 after SAVE should be MEMORY_CORRECT."""
        turn = _make_turn("맞아요, 그거 저장해주세요.")
        signal = detector.detect(turn, [], MemoryActionType.SAVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_failure_after_skip_is_neutral(self, detector: FeedbackDetector):
        """이미 말했잖아 after SKIP should be NEUTRAL."""
        turn = _make_turn("이미 말했잖아요.")
        signal = detector.detect(turn, [], MemoryActionType.SKIP)
        assert signal.signal_type == FeedbackType.NEUTRAL


# ─── REPEATED_QUESTION detection ───


class TestRepeatedQuestion:
    """Agent asking the same question twice should be detected."""

    def test_identical_question(self, detector: FeedbackDetector):
        """Identical assistant questions should trigger REPEATED_QUESTION."""
        prev = _make_turn(
            "어떤 프로그래밍 언어를 주로 사용하시나요?",
            turn_id=1, role=Role.ASSISTANT,
        )
        current = _make_turn(
            "어떤 프로그래밍 언어를 주로 사용하시나요?",
            turn_id=3, role=Role.ASSISTANT,
        )
        signal = detector.detect(current, [prev], MemoryActionType.SKIP)
        assert signal.signal_type == FeedbackType.REPEATED_QUESTION
        assert signal.reward_value == -0.8

    def test_very_similar_question(self, detector: FeedbackDetector):
        """Very similar assistant questions should trigger REPEATED_QUESTION."""
        prev = _make_turn(
            "주로 어떤 프로그래밍 언어를 사용하세요?",
            turn_id=1, role=Role.ASSISTANT,
        )
        current = _make_turn(
            "어떤 프로그래밍 언어를 주로 사용하시나요?",
            turn_id=3, role=Role.ASSISTANT,
        )
        signal = detector.detect(current, [prev], MemoryActionType.SKIP)
        assert signal.signal_type == FeedbackType.REPEATED_QUESTION

    def test_different_questions_no_repeat(self, detector: FeedbackDetector):
        """Different questions should not trigger REPEATED_QUESTION."""
        prev = _make_turn(
            "어떤 프로그래밍 언어를 사용하세요?",
            turn_id=1, role=Role.ASSISTANT,
        )
        current = _make_turn(
            "서울에서 맛집 추천해 주세요.",
            turn_id=3, role=Role.ASSISTANT,
        )
        signal = detector.detect(current, [prev], MemoryActionType.SKIP)
        assert signal.signal_type != FeedbackType.REPEATED_QUESTION

    def test_user_turn_no_repeat_detection(self, detector: FeedbackDetector):
        """Repeated question detection only applies to assistant turns."""
        prev = _make_turn(
            "Python 설치 방법 알려주세요.",
            turn_id=0, role=Role.USER,
        )
        current = _make_turn(
            "Python 설치 방법 알려주세요.",
            turn_id=2, role=Role.USER,
        )
        signal = detector.detect(current, [prev], MemoryActionType.SKIP)
        # User repeating themselves is not detected as REPEATED_QUESTION
        assert signal.signal_type != FeedbackType.REPEATED_QUESTION

    def test_no_previous_turns(self, detector: FeedbackDetector):
        """No previous turns means no repeat possible."""
        current = _make_turn(
            "어떤 프로그래밍 언어를 사용하세요?",
            turn_id=1, role=Role.ASSISTANT,
        )
        signal = detector.detect(current, [], MemoryActionType.SKIP)
        assert signal.signal_type != FeedbackType.REPEATED_QUESTION

    def test_only_user_turns_in_history(self, detector: FeedbackDetector):
        """If history only has user turns, no repeat detected for assistant turn."""
        prev_user = _make_turn(
            "어떤 프로그래밍 언어를 사용하세요?",
            turn_id=0, role=Role.USER,
        )
        current = _make_turn(
            "어떤 프로그래밍 언어를 사용하세요?",
            turn_id=1, role=Role.ASSISTANT,
        )
        signal = detector.detect(current, [prev_user], MemoryActionType.SKIP)
        assert signal.signal_type != FeedbackType.REPEATED_QUESTION

    def test_repeated_takes_priority_over_memory_context(
        self, detector: FeedbackDetector
    ):
        """Repeated question detection runs before memory context check."""
        prev = _make_turn(
            "좋아하는 음식이 뭐예요?",
            turn_id=1, role=Role.ASSISTANT,
        )
        current = _make_turn(
            "좋아하는 음식이 뭐예요?",
            turn_id=3, role=Role.ASSISTANT,
        )
        # Even after RETRIEVE, repeated question detection should fire first
        signal = detector.detect(current, [prev], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.REPEATED_QUESTION


# ─── NEUTRAL cases ───


class TestNeutral:
    """Statements with no memory-related feedback."""

    def test_general_statement(self, detector: FeedbackDetector):
        turn = _make_turn("오늘 날씨가 좋네요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.NEUTRAL
        assert signal.reward_value == 0.0

    def test_question_no_feedback(self, detector: FeedbackDetector):
        turn = _make_turn("Python 설치하는 방법 알려주세요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.NEUTRAL

    def test_empty_content(self, detector: FeedbackDetector):
        turn = _make_turn("")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.NEUTRAL

    def test_unrelated_korean(self, detector: FeedbackDetector):
        turn = _make_turn("저는 서울에서 일하고 있어요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.NEUTRAL


# ─── Edge cases ───


class TestEdgeCases:
    """Ambiguous or mixed-signal inputs."""

    def test_negative_priority_over_positive(self, detector: FeedbackDetector):
        """When both negative and positive patterns match, negative wins."""
        # "아닌데" comes first in pattern checking
        turn = _make_turn("아닌데요, 맞아요 그건 다른 얘기예요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type in (
            FeedbackType.MEMORY_ERROR,
            FeedbackType.MEMORY_FAILURE,
        )

    def test_맞아_in_longer_sentence(self, detector: FeedbackDetector):
        """맞아 embedded in longer text should still be detected."""
        turn = _make_turn("그래요 맞아요, 제가 그때 파이썬 좋아한다고 했었어요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_합쇼체_formal_positive(self, detector: FeedbackDetector):
        """합쇼체 (formal) variants should work."""
        turn = _make_turn("맞습니다, 정확하게 기억하고 계시네요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_합쇼체_formal_negative(self, detector: FeedbackDetector):
        """합쇼체 (formal) negative variant."""
        turn = _make_turn("틀렸습니다, 다시 확인해주세요.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_반말_informal_positive(self, detector: FeedbackDetector):
        """반말 (informal) variant should be detected."""
        turn = _make_turn("맞아 맞아, 그거야.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_반말_informal_negative(self, detector: FeedbackDetector):
        """반말 (informal) negative variant."""
        turn = _make_turn("아닌데, 내가 그런 적 없어.")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_signal_has_matched_pattern(self, detector: FeedbackDetector):
        """FeedbackSignal should include the matched pattern name."""
        turn = _make_turn("맞아요")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.matched_pattern != ""
        assert len(signal.matched_pattern) > 0

    def test_neutral_has_empty_pattern(self, detector: FeedbackDetector):
        """NEUTRAL signals have empty matched_pattern."""
        turn = _make_turn("오늘 뭐 먹을까?")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert signal.matched_pattern == ""

    def test_confidence_range(self, detector: FeedbackDetector):
        """Confidence should be in [0, 1]."""
        turn = _make_turn("맞아요!")
        signal = detector.detect(turn, [], MemoryActionType.RETRIEVE)
        assert 0.0 <= signal.confidence <= 1.0

    def test_custom_threshold(self):
        """Custom repeated question threshold should be respected."""
        # Use very high threshold — should NOT detect repetition
        strict_detector = FeedbackDetector(repeated_question_threshold=0.99)
        prev = _make_turn(
            "어떤 언어를 사용하세요?",
            turn_id=1, role=Role.ASSISTANT,
        )
        current = _make_turn(
            "어떤 언어를 사용하시나요?",
            turn_id=3, role=Role.ASSISTANT,
        )
        signal = strict_detector.detect(current, [prev], MemoryActionType.SKIP)
        assert signal.signal_type != FeedbackType.REPEATED_QUESTION


# ─── Utility function tests ───


class TestCharNgrams:
    def test_basic_ngrams(self):
        ngrams = _char_ngrams("안녕하세요", n=2)
        assert "안녕" in ngrams
        assert "녕하" in ngrams
        assert "세요" in ngrams

    def test_short_text(self):
        ngrams = _char_ngrams("안", n=3)
        assert ngrams == {"안"}

    def test_empty_text(self):
        ngrams = _char_ngrams("", n=3)
        assert ngrams == set()

    def test_whitespace_removed(self):
        ngrams = _char_ngrams("안 녕 하", n=2)
        # Whitespace removed: "안녕하"
        assert "안녕" in ngrams


class TestJaccardSimilarity:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4, jaccard=0.5
        assert abs(sim - 0.5) < 1e-9

    def test_empty_sets(self):
        assert _jaccard_similarity(set(), {"a"}) == 0.0
        assert _jaccard_similarity(set(), set()) == 0.0


# ─── FeedbackSignal dataclass tests ───


class TestFeedbackSignal:
    def test_immutable(self):
        signal = FeedbackSignal(
            signal_type=FeedbackType.NEUTRAL,
            reward_value=0.0,
            confidence=1.0,
            matched_pattern="",
        )
        with pytest.raises(AttributeError):
            signal.reward_value = 1.0  # type: ignore[misc]

    def test_all_feedback_types_have_rewards(self):
        """Every FeedbackType should have a defined reward value."""
        from aimemory.reward.feedback_detector import _FEEDBACK_REWARDS
        for ft in FeedbackType:
            assert ft in _FEEDBACK_REWARDS


# ─── Integration-like tests ───


class TestDetectorIntegration:
    """Multi-turn scenarios combining different signals."""

    def test_retrieve_then_positive_feedback(self, detector: FeedbackDetector):
        """Simulate: agent retrieves memory → user confirms."""
        asst_turn = _make_turn(
            "이전에 Python을 좋아하신다고 하셨는데, 맞으시죠?",
            turn_id=3, role=Role.ASSISTANT,
        )
        user_turn = _make_turn(
            "맞아요! 잘 기억하시네요.",
            turn_id=4, role=Role.USER,
        )
        signal = detector.detect(
            user_turn, [asst_turn], MemoryActionType.RETRIEVE,
        )
        assert signal.signal_type == FeedbackType.MEMORY_CORRECT

    def test_retrieve_then_negative_feedback(self, detector: FeedbackDetector):
        """Simulate: agent retrieves wrong memory → user corrects."""
        asst_turn = _make_turn(
            "서울에 사신다고 하셨는데 맞나요?",
            turn_id=3, role=Role.ASSISTANT,
        )
        user_turn = _make_turn(
            "아닌데요, 저는 부산에 살아요.",
            turn_id=4, role=Role.USER,
        )
        signal = detector.detect(
            user_turn, [asst_turn], MemoryActionType.RETRIEVE,
        )
        assert signal.signal_type == FeedbackType.MEMORY_ERROR

    def test_skip_then_general_conversation(self, detector: FeedbackDetector):
        """After SKIP, general positive language is NEUTRAL."""
        user_turn = _make_turn(
            "맞아요, 오늘 날씨가 정말 좋죠.",
            turn_id=2, role=Role.USER,
        )
        signal = detector.detect(user_turn, [], MemoryActionType.SKIP)
        assert signal.signal_type == FeedbackType.NEUTRAL

    def test_repeated_then_user_feedback(self, detector: FeedbackDetector):
        """Repeated question is detected for assistant; user feedback is separate."""
        asst_prev = _make_turn(
            "좋아하는 프로그래밍 언어가 뭐예요?",
            turn_id=1, role=Role.ASSISTANT,
        )
        asst_repeat = _make_turn(
            "좋아하는 프로그래밍 언어가 뭐예요?",
            turn_id=5, role=Role.ASSISTANT,
        )
        # Repeated question for assistant turn
        signal = detector.detect(
            asst_repeat, [asst_prev], MemoryActionType.RETRIEVE,
        )
        assert signal.signal_type == FeedbackType.REPEATED_QUESTION

        # Separate user turn should get proper feedback classification
        user_turn = _make_turn(
            "이미 말했잖아요, Python이요.",
            turn_id=6, role=Role.USER,
        )
        signal2 = detector.detect(
            user_turn, [asst_prev, asst_repeat], MemoryActionType.RETRIEVE,
        )
        assert signal2.signal_type == FeedbackType.MEMORY_FAILURE

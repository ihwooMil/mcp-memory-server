import pytest

from aimemory.reward.implicit_detector import ImplicitRewardDetector
from aimemory.schemas import Role, Turn


@pytest.fixture
def detector():
    return ImplicitRewardDetector()


def test_continuation_reward(detector):
    """2+ user turns after memory use → +0.3 reward."""
    turns = [
        Turn(turn_id=1, role=Role.USER, content="아 맞다 그거 재미있었어요"),
        Turn(turn_id=2, role=Role.ASSISTANT, content="네 그러셨군요"),
        Turn(turn_id=3, role=Role.USER, content="그때 같이 갔던 친구도 좋아했어요"),
    ]
    reward = detector.detect(turns, memory_used=["작년에 부산 여행 다녀왔어요"])
    assert reward >= 0.3


def test_short_dismissive_penalty(detector):
    """Short response like '그래' after memory → -0.1."""
    turns = [
        Turn(turn_id=1, role=Role.USER, content="그래"),
    ]
    reward = detector.detect(turns, memory_used=["Python을 좋아해요"])
    assert reward == pytest.approx(-0.1)


def test_topic_expansion_reward(detector):
    """Memory keywords reappearing in subsequent turns → +0.2."""
    turns = [
        Turn(turn_id=1, role=Role.USER, content="맞아요 그때 재미있었죠"),
        Turn(turn_id=2, role=Role.ASSISTANT, content="네 좋은 추억이네요"),
        Turn(turn_id=3, role=Role.USER, content="부산에서 회도 먹었는데 맛있었어요"),
    ]
    reward = detector.detect(turns, memory_used=["부산 여행 다녀왔어요"])
    # continuation (+0.3) + topic expansion (+0.2) = +0.5
    assert reward >= 0.5


def test_empty_turns_returns_zero(detector):
    """No turns → 0.0 reward."""
    assert detector.detect([], memory_used=["test"]) == 0.0


def test_no_memory_used_returns_zero(detector):
    """No memory used → 0.0 reward."""
    turns = [Turn(turn_id=1, role=Role.USER, content="안녕하세요")]
    assert detector.detect(turns, memory_used=[]) == 0.0

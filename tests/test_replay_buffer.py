"""Tests for the Experience Replay Buffer."""

import numpy as np
import pytest

from aimemory.online.replay_buffer import Experience, ReplayBuffer


def make_state(value: float = 0.0, size: int = 4) -> np.ndarray:
    return np.full(size, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Basic push / sample
# ---------------------------------------------------------------------------


def test_push_and_sample_basic():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push(make_state(i), i % 3, float(i), make_state(i + 1))
    samples = buf.sample(5)
    assert len(samples) == 5
    for exp in samples:
        assert isinstance(exp, Experience)


def test_len_returns_correct_count():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    for i in range(7):
        buf.push(make_state(i), 0, 0.0, None)
    assert len(buf) == 7


# ---------------------------------------------------------------------------
# Capacity overflow — oldest items dropped
# ---------------------------------------------------------------------------


def test_capacity_overflow_drops_oldest():
    capacity = 5
    buf = ReplayBuffer(capacity=capacity)
    for i in range(10):
        buf.push(make_state(i), i, float(i), None)
    # Buffer should only hold the last `capacity` items
    assert len(buf) == capacity
    # The oldest items (action 0..4) must be gone; only 5..9 remain
    remaining_actions = {exp.action for exp in buf.sample(capacity)}
    assert remaining_actions == {5, 6, 7, 8, 9}


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


def test_sample_empty_buffer_raises():
    buf = ReplayBuffer()
    with pytest.raises(ValueError, match="empty"):
        buf.sample(1)


def test_sample_batch_larger_than_buffer_raises():
    buf = ReplayBuffer()
    buf.push(make_state(), 0, 1.0, None)
    with pytest.raises(ValueError, match="batch_size"):
        buf.sample(10)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(capacity=50)
    for i in range(20):
        ns = make_state(i + 1) if i % 2 == 0 else None
        buf.push(make_state(i), i % 3, float(i) * 0.1, ns)

    path = str(tmp_path / "replay.pkl")
    buf.save(path)

    buf2 = ReplayBuffer(capacity=50)
    buf2.load(path)

    assert len(buf2) == len(buf)
    orig = list(buf._buffer)
    loaded = list(buf2._buffer)
    for o, loaded_exp in zip(orig, loaded):
        assert o.action == loaded_exp.action
        assert o.reward == pytest.approx(loaded_exp.reward)
        np.testing.assert_array_equal(o.state, loaded_exp.state)
        if o.next_state is None:
            assert loaded_exp.next_state is None
        else:
            np.testing.assert_array_equal(o.next_state, loaded_exp.next_state)


# ---------------------------------------------------------------------------
# Experience namedtuple fields
# ---------------------------------------------------------------------------


def test_experience_namedtuple_fields():
    state = make_state(1.0)
    next_state = make_state(2.0)
    exp = Experience(state=state, action=2, reward=0.5, next_state=next_state)
    np.testing.assert_array_equal(exp.state, state)
    assert exp.action == 2
    assert exp.reward == pytest.approx(0.5)
    np.testing.assert_array_equal(exp.next_state, next_state)


def test_experience_allows_none_next_state():
    exp = Experience(state=make_state(), action=1, reward=-1.0, next_state=None)
    assert exp.next_state is None


# ---------------------------------------------------------------------------
# Multiple pushes produce valid samples
# ---------------------------------------------------------------------------


def test_multiple_pushes_sampling_valid():
    buf = ReplayBuffer(capacity=200)
    for i in range(50):
        buf.push(make_state(i), i % 4, float(i) / 10.0, make_state(i + 1))

    samples = buf.sample(32)
    assert len(samples) == 32
    for exp in samples:
        assert exp.action in range(4)
        assert 0.0 <= exp.reward < 5.0
        assert exp.state.shape == (4,)
        assert exp.next_state.shape == (4,)

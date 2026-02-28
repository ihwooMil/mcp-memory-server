"""Tests for ProgressiveAutonomy."""

from __future__ import annotations

import pytest

from aimemory.online.autonomy import ProgressiveAutonomy


def test_initial_save_threshold():
    """Initial save_threshold should be 0.7."""
    pa = ProgressiveAutonomy()
    assert pa.save_threshold == pytest.approx(0.7)


def test_initial_skip_threshold():
    """Initial skip_threshold should be 0.1."""
    pa = ProgressiveAutonomy()
    assert pa.skip_threshold == pytest.approx(0.1)


def test_initial_rl_zone_ratio():
    """Initial rl_zone_ratio should be ~0.6 (0.7 - 0.1)."""
    pa = ProgressiveAutonomy()
    assert pa.rl_zone_ratio == pytest.approx(0.6)


def test_positive_feedback_increases_confidence():
    """Positive feedback should increase confidence."""
    pa = ProgressiveAutonomy()
    pa.record_feedback(1.0)
    assert pa.confidence == pytest.approx(1.0)
    pa.record_feedback(2.0)
    assert pa.confidence == pytest.approx(3.0)


def test_save_threshold_decreases_after_enough_positive_feedback():
    """After confidence exceeds threshold (50), save_threshold should decrease."""
    pa = ProgressiveAutonomy(confidence_threshold=50)
    # Give exactly 50 positive feedback to reach threshold
    for _ in range(50):
        pa.record_feedback(1.0)
    assert pa.confidence == pytest.approx(50.0)
    # At confidence == threshold, save_threshold is still initial (progress=0)
    assert pa.save_threshold == pytest.approx(0.7)

    # Push well beyond threshold to see relaxation
    for _ in range(50):
        pa.record_feedback(1.0)
    # confidence = 100, progress = (100-50)/50 = 1.0
    # save = 0.7 - 1.0*(0.7-0.3) = 0.3
    assert pa.save_threshold == pytest.approx(0.3)


def test_skip_threshold_increases_after_enough_positive_feedback():
    """After enough positive feedback, skip_threshold should increase."""
    pa = ProgressiveAutonomy(confidence_threshold=50)
    for _ in range(100):
        pa.record_feedback(1.0)
    # confidence = 100, progress = (100-50)/50 = 1.0
    # skip = 0.1 + 1.0*(0.4-0.1) = 0.4
    assert pa.skip_threshold == pytest.approx(0.4)


def test_negative_feedback_reduces_confidence():
    """Negative feedback should reduce confidence (amplified by 1/decay_rate)."""
    pa = ProgressiveAutonomy(decay_rate=0.01)
    # First build up some confidence
    for _ in range(200):
        pa.record_feedback(1.0)
    confidence_before = pa.confidence

    # One strong negative feedback
    pa.record_feedback(-1.0)
    # Expected: confidence_before + (-1.0 / 0.01) = confidence_before - 100
    assert pa.confidence == pytest.approx(confidence_before - 100.0)


def test_negative_feedback_floors_at_zero():
    """Confidence should never go below 0 due to negative feedback."""
    pa = ProgressiveAutonomy(decay_rate=0.01)
    # Confidence starts at 0, negative feedback should floor at 0
    pa.record_feedback(-1.0)
    assert pa.confidence == pytest.approx(0.0)


def test_negative_feedback_resets_thresholds():
    """After heavy negative feedback, thresholds should reset to initial values."""
    pa = ProgressiveAutonomy(confidence_threshold=50, decay_rate=0.01)
    # Build up confidence beyond threshold
    for _ in range(100):
        pa.record_feedback(1.0)
    assert pa.save_threshold < 0.7  # Was relaxed

    # Heavy negative feedback to zero out confidence
    pa.record_feedback(-2.0)  # -2/0.01 = -200, floors to 0
    assert pa.confidence == pytest.approx(0.0)
    assert pa.save_threshold == pytest.approx(0.7)
    assert pa.skip_threshold == pytest.approx(0.1)


def test_save_load_roundtrip(tmp_path):
    """Saving and loading should preserve all state."""
    pa = ProgressiveAutonomy(confidence_threshold=50)
    for _ in range(75):
        pa.record_feedback(1.0)
    pa.record_feedback(-0.5)  # Add a negative count too

    state_path = str(tmp_path / "autonomy_state.json")
    pa.save(state_path)

    pa2 = ProgressiveAutonomy()
    pa2.load(state_path)

    assert pa2.confidence == pytest.approx(pa.confidence)
    assert pa2.save_threshold == pytest.approx(pa.save_threshold)
    assert pa2.skip_threshold == pytest.approx(pa.skip_threshold)
    assert pa2._positive_count == pa._positive_count
    assert pa2._negative_count == pa._negative_count


def test_rl_zone_ratio_changes_with_more_confidence():
    """rl_zone_ratio (save - skip) contracts as both thresholds converge toward each other.

    The formula is rl_zone_ratio = save_threshold - skip_threshold.
    As confidence grows: save decreases and skip increases, so the zone shrinks,
    eventually going negative when skip > save (full RL authority).
    """
    pa = ProgressiveAutonomy(confidence_threshold=50)
    initial_ratio = pa.rl_zone_ratio  # 0.7 - 0.1 = 0.6

    # Build well beyond threshold (confidence=100, progress=1.0)
    for _ in range(100):
        pa.record_feedback(1.0)
    full_ratio = pa.rl_zone_ratio  # 0.3 - 0.4 = -0.1

    # At full confidence, ratio should be less than the initial ratio
    assert full_ratio < initial_ratio
    # At progress=1: save=0.3, skip=0.4, so ratio=-0.1
    assert full_ratio == pytest.approx(0.3 - 0.4)

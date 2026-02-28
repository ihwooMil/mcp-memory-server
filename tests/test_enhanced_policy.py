"""Tests for EnhancedOnlinePolicy."""

from __future__ import annotations

import numpy as np
import pytest

from aimemory.online.enhanced_policy import EnhancedOnlinePolicy
from aimemory.online.policy import OnlinePolicy
from aimemory.online.replay_buffer import ReplayBuffer
from aimemory.online.autonomy import ProgressiveAutonomy


FEATURE_DIM = 394


def make_features(rng: np.random.Generator | None = None) -> np.ndarray:
    """Create a random 394d feature vector."""
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.random(FEATURE_DIM).astype(np.float32)


def make_policy(**kwargs) -> EnhancedOnlinePolicy:
    return EnhancedOnlinePolicy(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Basic action selection
# ──────────────────────────────────────────────────────────────────────────────


def test_select_action_returns_valid_index():
    policy = make_policy()
    features = make_features()
    action = policy.select_action(features)
    assert action in [0, 1, 2]


def test_select_action_with_394d_input():
    policy = make_policy()
    rng = np.random.default_rng(0)
    for _ in range(20):
        features = make_features(rng)
        action = policy.select_action(features)
        assert action in [0, 1, 2], f"Unexpected action {action}"


# ──────────────────────────────────────────────────────────────────────────────
# Update and replay buffer
# ──────────────────────────────────────────────────────────────────────────────


def test_update_returns_loss():
    policy = make_policy()
    features = make_features()
    loss = policy.update(features, action_id=0, reward=1.0)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_update_pushes_to_replay_buffer():
    policy = make_policy()
    assert len(policy.replay_buffer) == 0
    features = make_features()
    policy.update(features, action_id=1, reward=0.5)
    assert len(policy.replay_buffer) == 1
    policy.update(features, action_id=2, reward=-0.5)
    assert len(policy.replay_buffer) == 2


def test_batch_update_with_enough_data():
    policy = make_policy()
    rng = np.random.default_rng(7)
    for i in range(50):
        features = make_features(rng)
        policy.replay_buffer.push(features, i % 3, float(rng.random()), None)
    loss = policy.batch_update(batch_size=32)
    assert isinstance(loss, float)
    # Loss should be non-negative (MSE)
    assert loss >= 0.0


def test_batch_update_insufficient_data():
    policy = make_policy()
    # Push fewer than 32 experiences
    rng = np.random.default_rng(13)
    for _ in range(10):
        features = make_features(rng)
        policy.replay_buffer.push(features, 0, 1.0, None)
    result = policy.batch_update(batch_size=32)
    assert result == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Autonomy
# ──────────────────────────────────────────────────────────────────────────────


def test_autonomy_updated_on_feedback():
    policy = make_policy()
    initial_confidence = policy.autonomy.confidence
    features = make_features()
    # Positive reward should increase confidence
    policy.update(features, action_id=0, reward=1.0)
    assert policy.autonomy.confidence > initial_confidence


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint save/load
# ──────────────────────────────────────────────────────────────────────────────


def test_checkpoint_save_load_roundtrip(tmp_path):
    policy = make_policy(epsilon=0.25)
    features = make_features()
    # Do a few updates so update_count > 0
    for i in range(5):
        policy.update(features, action_id=i % 3, reward=float(i))

    ckpt_path = tmp_path / "policy.pt"
    policy.save_checkpoint(str(ckpt_path))

    # Load into a fresh policy
    policy2 = make_policy()
    policy2.load_checkpoint(str(ckpt_path))

    assert policy2.epsilon == pytest.approx(0.25)
    assert policy2._update_count == 5
    # Replay buffer should have been restored
    assert len(policy2.replay_buffer) == 5

    # Parameters should match
    params1 = policy.get_parameters()
    params2 = policy2.get_parameters()
    np.testing.assert_allclose(params1, params2, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Inheritance
# ──────────────────────────────────────────────────────────────────────────────


def test_inherits_online_policy_interface():
    policy = make_policy()
    assert isinstance(policy, OnlinePolicy)


# ──────────────────────────────────────────────────────────────────────────────
# Parameter count
# ──────────────────────────────────────────────────────────────────────────────


def test_parameter_count_larger_than_base():
    from aimemory.online.policy import _BanditMLP
    from aimemory.online.enhanced_policy import _EnhancedMLP

    base = _BanditMLP(feature_dim=10, hidden_dim=64, n_actions=3)
    enhanced = _EnhancedMLP(feature_dim=394, hidden1=256, hidden2=128, n_actions=3)

    base_params = sum(p.numel() for p in base.parameters())
    enhanced_params = sum(p.numel() for p in enhanced.parameters())

    assert enhanced_params > base_params, (
        f"Enhanced ({enhanced_params}) should have more params than base ({base_params})"
    )
    # Sanity check: enhanced should be well above 771 params
    assert enhanced_params > 771

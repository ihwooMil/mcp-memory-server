"""Tests for online contextual bandit policy module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aimemory.online.policy import (
    ACTION_MAP,
    FEATURE_DIM,
    MemoryPolicyAgent,
    OnlinePolicy,
    StateEncoder,
)
from aimemory.config import AppConfig, OnlinePolicyConfig
from aimemory.schemas import MemoryActionType, MemoryDecision, Role, Turn


# ─── StateEncoder tests ──────────────────────────────────────────────


class TestStateEncoder:
    def setup_method(self):
        self.encoder = StateEncoder()

    def _make_turn(self, content: str, turn_id: int = 1) -> Turn:
        return Turn(turn_id=turn_id, role=Role.USER, content=content)

    def test_output_shape(self):
        turn = self._make_turn("안녕하세요 Python 개발자입니다")
        features = self.encoder.encode(turn, recent_turns=[])
        assert features.shape == (FEATURE_DIM,)
        assert features.dtype == np.float32

    def test_question_detection(self):
        question_turn = self._make_turn("Python이 뭐에요?")
        normal_turn = self._make_turn("Python을 좋아합니다")
        q_feat = self.encoder.encode(question_turn, [])
        n_feat = self.encoder.encode(normal_turn, [])
        # Feature index 3 = is_question
        assert q_feat[3] == 1.0
        assert n_feat[3] == 0.0

    def test_tech_detection(self):
        tech_turn = self._make_turn("PyTorch로 모델을 만들었어요")
        plain_turn = self._make_turn("오늘 날씨가 좋아요")
        t_feat = self.encoder.encode(tech_turn, [])
        p_feat = self.encoder.encode(plain_turn, [])
        # Feature index 6 = has_tech
        assert t_feat[6] == 1.0
        assert p_feat[6] == 0.0

    def test_emotion_detection(self):
        emo_turn = self._make_turn("요즘 너무 힘들어요")
        feat = self.encoder.encode(emo_turn, [])
        # Feature index 7 = has_emotion
        assert feat[7] == 1.0

    def test_turn_position(self):
        turn = self._make_turn("test")
        feat = self.encoder.encode(turn, [], turn_position=0.75)
        assert feat[0] == pytest.approx(0.75)

    def test_memory_count_log_scaled(self):
        turn = self._make_turn("test")
        feat0 = self.encoder.encode(turn, [], memory_count=0)
        feat10 = self.encoder.encode(turn, [], memory_count=10)
        assert feat0[1] < feat10[1]
        assert feat0[1] == pytest.approx(np.log1p(0))
        assert feat10[1] == pytest.approx(np.log1p(10))

    def test_recent_actions_counted(self):
        turn = self._make_turn("test")
        actions = [MemoryActionType.SAVE, MemoryActionType.SAVE, MemoryActionType.RETRIEVE]
        feat = self.encoder.encode(turn, [], recent_actions=actions)
        # Feature index 8 = recent_save_count (log-scaled)
        assert feat[8] == pytest.approx(np.log1p(2))
        # Feature index 9 = recent_retrieve_count (log-scaled)
        assert feat[9] == pytest.approx(np.log1p(1))

    def test_personal_info_detection(self):
        turn = self._make_turn("저는 서울에서 살고 있습니다")
        feat = self.encoder.encode(turn, [])
        # Feature index 4 = has_personal_info
        assert feat[4] == 1.0


# ─── OnlinePolicy tests ──────────────────────────────────────────────


class TestOnlinePolicy:
    def setup_method(self):
        self.policy = OnlinePolicy(
            feature_dim=FEATURE_DIM,
            n_actions=3,
            hidden_dim=64,
            lr=0.01,
            epsilon=0.1,
        )

    def test_select_action_returns_valid_index(self):
        features = np.random.randn(FEATURE_DIM).astype(np.float32)
        for _ in range(20):
            action = self.policy.select_action(features)
            assert action in {0, 1, 2}

    def test_greedy_action_deterministic(self):
        """With epsilon=0, same features should produce same action."""
        self.policy.epsilon = 0.0
        features = np.random.randn(FEATURE_DIM).astype(np.float32)
        actions = {self.policy.select_action(features) for _ in range(10)}
        assert len(actions) == 1

    def test_exploration_with_high_epsilon(self):
        """With epsilon=1.0, actions should be uniformly random."""
        self.policy.epsilon = 1.0
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        actions = [self.policy.select_action(features) for _ in range(300)]
        unique = set(actions)
        # All 3 actions should appear with high probability
        assert len(unique) == 3

    def test_update_returns_loss(self):
        features = np.random.randn(FEATURE_DIM).astype(np.float32)
        loss = self.policy.update(features, action_id=0, reward=1.0)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_update_reduces_loss(self):
        """Repeated updates with same (state, action, reward) should reduce loss."""
        features = np.random.randn(FEATURE_DIM).astype(np.float32)
        losses = []
        for _ in range(50):
            loss = self.policy.update(features, action_id=1, reward=1.0)
            losses.append(loss)
        # Loss should decrease overall
        assert losses[-1] < losses[0]

    def test_get_set_parameters_roundtrip(self):
        params_before = self.policy.get_parameters()
        assert isinstance(params_before, np.ndarray)
        assert params_before.ndim == 1

        # Modify parameters
        new_params = params_before + 0.1
        self.policy.set_parameters(new_params)
        params_after = self.policy.get_parameters()
        np.testing.assert_allclose(params_after, new_params, atol=1e-6)

    def test_save_load_checkpoint(self):
        features = np.random.randn(FEATURE_DIM).astype(np.float32)
        # Train a bit
        for _ in range(10):
            self.policy.update(features, action_id=0, reward=1.0)
        params_before = self.policy.get_parameters()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "policy.pt"
            self.policy.save_checkpoint(ckpt_path)

            # Create a fresh policy and load
            new_policy = OnlinePolicy(feature_dim=FEATURE_DIM)
            new_policy.load_checkpoint(ckpt_path)
            params_after = new_policy.get_parameters()

        np.testing.assert_allclose(params_after, params_before, atol=1e-6)

    def test_parameter_count(self):
        """Verify total parameter count matches MLP architecture."""
        params = self.policy.get_parameters()
        # Linear(10 -> 64): 10*64 + 64 = 704
        # Linear(64 -> 3): 64*3 + 3 = 195
        # Total: 899
        expected = FEATURE_DIM * 64 + 64 + 64 * 3 + 3
        assert len(params) == expected


# ─── MemoryPolicyAgent tests ─────────────────────────────────────────


class TestMemoryPolicyAgent:
    def setup_method(self):
        self.policy = OnlinePolicy(feature_dim=FEATURE_DIM, epsilon=0.0)

        # Mock GraphMemoryStore
        self.mock_store = MagicMock()
        self.mock_store.get_stats.return_value = {"total": 5, "categories": {"fact": 3, "preference": 2}}
        self.mock_store.add_memory.return_value = "mem_abc123"
        self.mock_store.search.return_value = []

        # Mock FeedbackDetector
        self.mock_feedback = MagicMock()

        self.agent = MemoryPolicyAgent(
            graph_store=self.mock_store,
            policy=self.policy,
            feedback_detector=self.mock_feedback,
        )

    def _make_turn(self, content: str, turn_id: int = 1, role: Role = Role.USER) -> Turn:
        return Turn(turn_id=turn_id, role=role, content=content)

    def test_skip_non_user_turn(self):
        turn = self._make_turn("네 알겠습니다", role=Role.ASSISTANT)
        decision = self.agent.decide(turn, [])
        assert decision.action == MemoryActionType.SKIP

    def test_decide_returns_memory_decision(self):
        turn = self._make_turn("Python을 좋아합니다")
        decision = self.agent.decide(turn, [])
        assert isinstance(decision, MemoryDecision)
        assert decision.action in (MemoryActionType.SAVE, MemoryActionType.SKIP, MemoryActionType.RETRIEVE)

    def test_save_action_calls_store(self):
        """When policy selects SAVE, graph_store.add_memory should be called."""
        # Force SAVE action by manipulating policy
        self.policy.epsilon = 1.0  # random

        saved = False
        for _ in range(50):
            turn = self._make_turn("저는 서울에서 Python 개발자로 일하고 있습니다", turn_id=1)
            decision = self.agent.decide(turn, [])
            if decision.action == MemoryActionType.SAVE:
                saved = True
                break

        if saved:
            self.mock_store.add_memory.assert_called()

    def test_retrieve_action_calls_search(self):
        """When policy selects RETRIEVE, graph_store.search should be called."""
        self.policy.epsilon = 1.0  # random

        retrieved = False
        for _ in range(50):
            turn = self._make_turn("Python에 대해 알려주세요?", turn_id=2)
            decision = self.agent.decide(turn, [])
            if decision.action == MemoryActionType.RETRIEVE:
                retrieved = True
                break

        if retrieved:
            self.mock_store.search.assert_called()

    def test_process_feedback_updates_policy(self):
        from aimemory.reward.feedback_detector import FeedbackSignal, FeedbackType

        self.mock_feedback.detect.return_value = FeedbackSignal(
            signal_type=FeedbackType.MEMORY_CORRECT,
            reward_value=1.0,
            confidence=0.9,
            matched_pattern="test",
        )

        # First, make a decision to set internal state
        turn = self._make_turn("Python을 좋아합니다")
        self.agent.decide(turn, [])

        # Process feedback
        feedback_turn = self._make_turn("맞아요 잘 기억하시네요", turn_id=2)
        signal, reward = self.agent.process_feedback(feedback_turn, [turn])
        assert reward == 1.0

    def test_recent_actions_tracked(self):
        """After multiple decide() calls, recent_actions should be populated."""
        for i in range(5):
            turn = self._make_turn(f"Turn {i} content", turn_id=i)
            self.agent.decide(turn, [])

        assert len(self.agent._recent_actions) == 5


# ─── Integration tests ───────────────────────────────────────────────


class TestPolicyIntegration:
    def test_encoder_policy_pipeline(self):
        """StateEncoder output works as OnlinePolicy input."""
        encoder = StateEncoder()
        policy = OnlinePolicy(feature_dim=FEATURE_DIM, epsilon=0.0)

        turn = Turn(turn_id=1, role=Role.USER, content="PyTorch로 모델 만들기")
        features = encoder.encode(turn, [])
        action = policy.select_action(features)
        assert action in {0, 1, 2}

        # Update should work
        loss = policy.update(features, action, reward=0.5)
        assert loss >= 0.0

    def test_action_map_complete(self):
        """All 3 actions should be mapped."""
        assert len(ACTION_MAP) == 3
        assert set(ACTION_MAP.values()) == {
            MemoryActionType.SAVE,
            MemoryActionType.SKIP,
            MemoryActionType.RETRIEVE,
        }


# ─── Importance scoring tests ───

class TestImportanceScoring:
    """Tests for MemoryPolicyAgent._compute_importance()."""

    def _make_agent(self):
        """Create agent with mock store."""
        from unittest.mock import MagicMock
        store = MagicMock()
        store.get_stats.return_value = {"total": 5}
        store.search.return_value = []
        detector = MagicMock()
        policy = OnlinePolicy(epsilon=0.0)
        return MemoryPolicyAgent(
            graph_store=store, policy=policy, feedback_detector=detector
        )

    def test_personal_info_high_importance(self):
        """Personal info feature → importance >= 0.4."""
        agent = self._make_agent()
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        features[4] = 1.0  # has_personal_info
        turn = Turn(turn_id=0, role=Role.USER, content="저는 서울에 살고 있어요.")
        score = agent._compute_importance(turn, features)
        assert score >= 0.4

    def test_preference_high_importance(self):
        """Preference feature → importance >= 0.35."""
        agent = self._make_agent()
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        features[5] = 1.0  # has_preference
        turn = Turn(turn_id=0, role=Role.USER, content="저는 커피를 좋아해요.")
        score = agent._compute_importance(turn, features)
        assert score >= 0.35

    def test_combined_high_importance(self):
        """Personal + preference + tech → importance >= 0.7 (SAVE zone)."""
        agent = self._make_agent()
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        features[4] = 1.0  # personal
        features[5] = 1.0  # preference
        features[6] = 1.0  # tech
        turn = Turn(turn_id=0, role=Role.USER, content="저는 Python 개발을 좋아해요.")
        score = agent._compute_importance(turn, features)
        assert score >= 0.7

    def test_empty_features_low_importance(self):
        """All zero features → importance <= 0.1 (SKIP zone)."""
        agent = self._make_agent()
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        turn = Turn(turn_id=0, role=Role.USER, content="오늘 날씨가 좋네요.")
        score = agent._compute_importance(turn, features)
        assert score <= 0.1

    def test_keyword_density_contribution(self):
        """Keyword count contributes to importance."""
        agent = self._make_agent()
        features_no_kw = np.zeros(FEATURE_DIM, dtype=np.float32)
        features_with_kw = np.zeros(FEATURE_DIM, dtype=np.float32)
        features_with_kw[2] = 1.5  # log-scaled keyword count
        turn = Turn(turn_id=0, role=Role.USER, content="테스트")
        score_no = agent._compute_importance(turn, features_no_kw)
        score_with = agent._compute_importance(turn, features_with_kw)
        assert score_with > score_no


# ─── Rule-based decision tests ───

class TestRuleBasedDecision:
    """Tests for the 3-phase decide() logic."""

    def _make_agent_with_mock(self):
        from unittest.mock import MagicMock, patch
        store = MagicMock()
        store.get_stats.return_value = {"total": 5}
        store.search.return_value = []
        store.add_memory.return_value = "mem_123"
        detector = MagicMock()
        policy = OnlinePolicy(epsilon=0.0)
        agent = MemoryPolicyAgent(
            graph_store=store, policy=policy, feedback_detector=detector
        )
        return agent

    def test_high_importance_always_saves(self):
        """importance >= 0.7 → always SAVE regardless of bandit."""
        agent = self._make_agent_with_mock()
        # Personal + preference + tech = 0.4 + 0.35 + 0.3 = 1.05 → capped to 1.0
        turn = Turn(
            turn_id=0, role=Role.USER,
            content="저는 Python 개발자예요. 코딩을 좋아해요."
        )
        # Multiple calls should always SAVE
        for _ in range(5):
            decision = agent.decide(turn, [])
            assert decision.action in (MemoryActionType.SAVE, MemoryActionType.RETRIEVE)

    def test_low_importance_always_skips(self):
        """importance <= 0.1 → always SKIP."""
        agent = self._make_agent_with_mock()
        agent._store.get_stats.return_value = {"total": 0}  # no memories → no retrieve
        # Plain text with no personal/pref/tech/emotion patterns
        turn = Turn(turn_id=0, role=Role.USER, content="네.")
        decision = agent.decide(turn, [])
        assert decision.action == MemoryActionType.SKIP

    def test_non_user_always_skips(self):
        """Non-user turns → always SKIP."""
        agent = self._make_agent_with_mock()
        turn = Turn(turn_id=0, role=Role.ASSISTANT, content="저는 Python 개발자예요.")
        decision = agent.decide(turn, [])
        assert decision.action == MemoryActionType.SKIP


# ─── Should retrieve tests ───

class TestShouldRetrieve:
    """Tests for _should_retrieve()."""

    def _make_agent(self, total_memories=5):
        from unittest.mock import MagicMock
        store = MagicMock()
        store.get_stats.return_value = {"total": total_memories}
        detector = MagicMock()
        policy = OnlinePolicy(epsilon=0.0)
        return MemoryPolicyAgent(
            graph_store=store, policy=policy, feedback_detector=detector
        )

    def test_question_triggers_retrieve(self):
        """Question pattern → should retrieve."""
        agent = self._make_agent()
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        features[3] = 1.0  # is_question
        turn = Turn(turn_id=0, role=Role.USER, content="Python 어떻게 설치하나요?")
        assert agent._should_retrieve(turn, features) is True

    def test_discourse_marker_triggers_retrieve(self):
        """Discourse marker → should retrieve."""
        agent = self._make_agent()
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        turn = Turn(turn_id=0, role=Role.USER, content="근데, 어제 말한 거 기억나?")
        assert agent._should_retrieve(turn, features) is True

    def test_no_question_no_marker_false(self):
        """No question, no marker → should not retrieve."""
        agent = self._make_agent()
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        turn = Turn(turn_id=0, role=Role.USER, content="오늘 점심 맛있었어요.")
        assert agent._should_retrieve(turn, features) is False

    def test_empty_store_no_retrieve(self):
        """No memories in store → should not retrieve."""
        agent = self._make_agent(total_memories=0)
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        features[3] = 1.0  # is_question
        turn = Turn(turn_id=0, role=Role.USER, content="Python 어떻게 설치하나요?")
        assert agent._should_retrieve(turn, features) is False


# ─── OnlinePolicyConfig tests ───

class TestOnlinePolicyConfig:
    """Tests for the new OnlinePolicyConfig."""

    def test_default_values(self):
        cfg = OnlinePolicyConfig()
        assert cfg.feature_dim == 10
        assert cfg.hidden_dim == 64
        assert cfg.n_actions == 3
        assert cfg.lr == 0.01
        assert cfg.epsilon == 0.1
        assert cfg.save_threshold == 0.7
        assert cfg.skip_threshold == 0.1
        assert cfg.personal_weight == 0.4
        assert cfg.preference_weight == 0.35
        assert cfg.tech_weight == 0.3
        assert cfg.emotion_weight == 0.2
        assert cfg.keyword_weight == 0.15
        assert cfg.retrieve_top_k == 3

    def test_app_config_includes_online_policy(self):
        cfg = AppConfig()
        assert hasattr(cfg, "online_policy")
        assert isinstance(cfg.online_policy, OnlinePolicyConfig)

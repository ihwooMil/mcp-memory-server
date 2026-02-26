"""Tests for Pydantic schemas serialization/deserialization."""

from __future__ import annotations

import json

from aimemory.schemas import (
    Action,
    Episode,
    MemoryActionType,
    RewardBreakdown,
    Role,
    SARTriple,
    ScenarioType,
    State,
    Turn,
)


class TestTurn:
    def test_create_turn(self):
        turn = Turn(turn_id=0, role=Role.USER, content="안녕하세요", token_count=5)
        assert turn.turn_id == 0
        assert turn.role == Role.USER
        assert turn.content == "안녕하세요"

    def test_turn_serialization(self):
        turn = Turn(turn_id=1, role=Role.ASSISTANT, content="반갑습니다", token_count=6)
        data = turn.model_dump()
        restored = Turn.model_validate(data)
        assert restored.content == turn.content
        assert restored.role == turn.role

    def test_turn_json_roundtrip(self):
        turn = Turn(turn_id=0, role=Role.USER, content="테스트", token_count=3)
        json_str = turn.model_dump_json()
        restored = Turn.model_validate_json(json_str)
        assert restored == turn


class TestEpisode:
    def test_create_episode(self, sample_episode):
        assert sample_episode.scenario == ScenarioType.TECHNICAL_QA
        assert sample_episode.num_turns == 4
        assert sample_episode.num_saves == 2

    def test_episode_json_roundtrip(self, sample_episode):
        json_str = sample_episode.model_dump_json()
        restored = Episode.model_validate_json(json_str)
        assert restored.num_turns == sample_episode.num_turns
        assert restored.num_saves == sample_episode.num_saves
        assert restored.scenario == sample_episode.scenario

    def test_episode_jsonl_compatible(self, sample_episode):
        """Episodes should be writable as single-line JSON (JSONL)."""
        json_str = sample_episode.model_dump_json()
        assert "\n" not in json_str
        parsed = json.loads(json_str)
        assert "turns" in parsed
        assert "memory_decisions" in parsed


class TestRewardBreakdown:
    def test_compute_total_default_weights(self, sample_reward):
        total = sample_reward.compute_total()
        assert total > 0
        assert sample_reward.total == total

    def test_compute_total_custom_weights(self):
        reward = RewardBreakdown(r1_keyword_reappearance=1.0)
        total = reward.compute_total({"r1_keyword_reappearance": 2.0})
        assert abs(total - 2.0) < 1e-6

    def test_reward_serialization(self, sample_reward):
        sample_reward.compute_total()
        data = sample_reward.model_dump()
        restored = RewardBreakdown.model_validate(data)
        assert abs(restored.total - sample_reward.total) < 1e-6


class TestSARTriple:
    def test_create_sar_triple(self, sample_state, sample_action, sample_reward):
        sample_reward.compute_total()
        triple = SARTriple(
            episode_id="test_ep_001",
            step_index=0,
            state=sample_state,
            action=sample_action,
            reward=sample_reward,
        )
        assert triple.episode_id == "test_ep_001"
        assert triple.action.action_type == MemoryActionType.SAVE
        assert triple.reward.total > 0
        assert triple.done is False

    def test_sar_triple_json_roundtrip(self, sample_state, sample_action, sample_reward):
        sample_reward.compute_total()
        triple = SARTriple(
            episode_id="test_ep_001",
            step_index=0,
            state=sample_state,
            action=sample_action,
            reward=sample_reward,
        )
        json_str = triple.model_dump_json()
        restored = SARTriple.model_validate_json(json_str)
        assert restored.episode_id == triple.episode_id
        assert restored.step_index == triple.step_index
        assert abs(restored.reward.total - triple.reward.total) < 1e-6

    def test_sar_triple_with_next_state(self, sample_state, sample_action, sample_reward):
        next_state = sample_state.model_copy(update={"turn_id": 3, "memory_count": 2})
        triple = SARTriple(
            episode_id="test_ep_001",
            step_index=0,
            state=sample_state,
            action=sample_action,
            reward=sample_reward,
            next_state=next_state,
        )
        assert triple.next_state is not None
        assert triple.next_state.memory_count == 2


class TestAction:
    def test_save_action(self):
        action = Action(
            action_type=MemoryActionType.SAVE,
            saved_content="테스트 메모리",
            saved_keywords=["테스트"],
        )
        assert action.action_type == MemoryActionType.SAVE
        assert action.saved_content == "테스트 메모리"

    def test_skip_action(self):
        action = Action(action_type=MemoryActionType.SKIP)
        assert action.action_type == MemoryActionType.SKIP
        assert action.saved_content is None

    def test_retrieve_action(self):
        action = Action(action_type=MemoryActionType.RETRIEVE, retrieved_count=3)
        assert action.retrieved_count == 3

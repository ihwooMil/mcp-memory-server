"""Tests for dataset/builder.py - Episode → SARTriple conversion."""

from __future__ import annotations

import json

import pytest

from aimemory.config import DatasetConfig
from aimemory.dataset.builder import EpisodeBuilder
from aimemory.schemas import (
    Action,
    Episode,
    MemoryActionType,
    MemoryDecision,
    MemoryEntry,
    RewardBreakdown,
    Role,
    SARTriple,
    ScenarioType,
    Turn,
)


# ─── Fixtures ───


@pytest.fixture
def default_config() -> DatasetConfig:
    return DatasetConfig(context_window=6, random_seed=42)


@pytest.fixture
def builder(default_config) -> EpisodeBuilder:
    return EpisodeBuilder(default_config)


@pytest.fixture
def minimal_episode() -> Episode:
    """Episode with 4 turns and 4 decisions."""
    turns = [
        Turn(turn_id=0, role=Role.USER, content="안녕하세요", token_count=5),
        Turn(turn_id=1, role=Role.ASSISTANT, content="안녕하세요!", token_count=5),
        Turn(turn_id=2, role=Role.USER, content="Python을 좋아해요", token_count=8),
        Turn(turn_id=3, role=Role.ASSISTANT, content="좋네요!", token_count=4),
    ]
    entry = MemoryEntry(
        content="사용자는 Python을 좋아함",
        source_turn_id=2,
        keywords=["Python"],
        category="preference",
    )
    decisions = [
        MemoryDecision(turn_id=0, action=MemoryActionType.SKIP, reasoning="no info"),
        MemoryDecision(turn_id=1, action=MemoryActionType.SKIP, reasoning="assistant"),
        MemoryDecision(
            turn_id=2,
            action=MemoryActionType.SAVE,
            memory_entry=entry,
            reasoning="user preference",
        ),
        MemoryDecision(turn_id=3, action=MemoryActionType.SKIP, reasoning="assistant"),
    ]
    return Episode(
        scenario=ScenarioType.TECHNICAL_QA,
        turns=turns,
        memory_decisions=decisions,
        memory_store=[entry],
    )


@pytest.fixture
def episode_with_retrieve(minimal_episode) -> Episode:
    """Episode that includes a RETRIEVE action."""
    entry = minimal_episode.memory_store[0]
    minimal_episode.memory_decisions.append(
        MemoryDecision(
            turn_id=1,  # will be replaced below
            action=MemoryActionType.RETRIEVE,
            retrieved_memories=[entry],
            reasoning="retrieve for context",
        )
    )
    # Replace decisions to add a retrieve at turn_id=3
    turns = minimal_episode.turns + [
        Turn(turn_id=4, role=Role.USER, content="Python 언제부터 했나요?", token_count=10)
    ]
    decisions = minimal_episode.memory_decisions[:-1] + [
        MemoryDecision(
            turn_id=3,
            action=MemoryActionType.RETRIEVE,
            retrieved_memories=[entry],
            reasoning="retrieve at turn 3",
        ),
        MemoryDecision(turn_id=4, action=MemoryActionType.SKIP, reasoning="end"),
    ]
    return Episode(
        scenario=ScenarioType.TECHNICAL_QA,
        turns=turns,
        memory_decisions=decisions,
        memory_store=[entry],
    )


# ─── Tests: episode_to_sar_triples ───


class TestEpisodeToSarTriples:
    def test_produces_one_triple_per_decision(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        assert len(triples) == len(minimal_episode.memory_decisions)

    def test_triple_ids_are_unique(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        ids = [t.triple_id for t in triples]
        assert len(ids) == len(set(ids))

    def test_episode_id_propagated(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        for t in triples:
            assert t.episode_id == minimal_episode.episode_id

    def test_step_indices_sequential(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        for i, t in enumerate(triples):
            assert t.step_index == i

    def test_last_triple_is_done(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        assert triples[-1].done is True

    def test_non_last_triples_not_done(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        for t in triples[:-1]:
            assert t.done is False

    def test_last_triple_has_no_next_state(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        assert triples[-1].next_state is None

    def test_non_last_triples_have_next_state(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        for t in triples[:-1]:
            assert t.next_state is not None

    def test_empty_episode_returns_empty(self, builder):
        ep = Episode(scenario=ScenarioType.CASUAL_CHAT)
        result = builder.episode_to_sar_triples(ep)
        assert result == []


class TestStateBuilding:
    def test_state_contains_episode_id(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        for t in triples:
            assert t.state.episode_id == minimal_episode.episode_id

    def test_state_recent_turns_not_empty(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        for t in triples:
            assert len(t.state.recent_turns) > 0

    def test_state_recent_turns_max_window(self, builder, minimal_episode):
        """recent_turns should not exceed context_window."""
        window = builder.config.context_window
        triples = builder.episode_to_sar_triples(minimal_episode)
        for t in triples:
            assert len(t.state.recent_turns) <= window

    def test_state_memory_count_increases_after_save(self, builder, minimal_episode):
        """Memory count in state should increase after a SAVE decision."""
        triples = builder.episode_to_sar_triples(minimal_episode)
        # Find the SAVE triple (turn_id=2)
        save_triple = next(
            t for t in triples if t.action.action_type == MemoryActionType.SAVE
        )
        save_idx = save_triple.step_index

        if save_idx + 1 < len(triples):
            next_triple = triples[save_idx + 1]
            # next_state of save_triple = state of next_triple
            assert save_triple.next_state is not None
            # Memory count should be higher in next_state
            assert save_triple.next_state.memory_count > save_triple.state.memory_count

    def test_first_turn_state_has_zero_memory(self, builder, minimal_episode):
        """State at the first turn should have no memory entries yet."""
        triples = builder.episode_to_sar_triples(minimal_episode)
        first = triples[0]
        assert first.state.memory_count == 0
        assert first.state.current_memory_summary == []

    def test_turn_position_normalized(self, builder, minimal_episode):
        """turn_position should be in [0.0, 1.0]."""
        triples = builder.episode_to_sar_triples(minimal_episode)
        for t in triples:
            assert 0.0 <= t.state.turn_position <= 1.0

    def test_small_window_respected(self, minimal_episode):
        """With a small context window, recent_turns should be capped."""
        builder = EpisodeBuilder(DatasetConfig(context_window=2))
        triples = builder.episode_to_sar_triples(minimal_episode)
        for t in triples:
            assert len(t.state.recent_turns) <= 2


class TestActionConversion:
    def test_skip_action(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        skip_triples = [t for t in triples if t.action.action_type == MemoryActionType.SKIP]
        assert len(skip_triples) > 0
        for t in skip_triples:
            assert t.action.saved_content is None
            assert t.action.saved_keywords == []

    def test_save_action_has_content(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        save_triples = [t for t in triples if t.action.action_type == MemoryActionType.SAVE]
        assert len(save_triples) == 1
        save = save_triples[0]
        assert save.action.saved_content is not None
        assert len(save.action.saved_content) > 0
        assert "Python" in save.action.saved_keywords

    def test_retrieve_action(self, builder, episode_with_retrieve):
        triples = builder.episode_to_sar_triples(episode_with_retrieve)
        retrieve_triples = [
            t for t in triples if t.action.action_type == MemoryActionType.RETRIEVE
        ]
        assert len(retrieve_triples) > 0
        for t in retrieve_triples:
            assert t.action.retrieved_count > 0


class TestRewardIntegration:
    def test_default_reward_is_zero(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        for t in triples:
            assert t.reward.total == 0.0

    def test_custom_reward_map_applied(self, builder, minimal_episode):
        rb = RewardBreakdown(
            r1_keyword_reappearance=0.8,
            r3_efficiency=0.5,
            total=1.3,
        )
        reward_map = {2: rb}  # turn_id=2 is the SAVE decision
        triples = builder.episode_to_sar_triples(minimal_episode, reward_map=reward_map)
        save_triple = next(
            t for t in triples if t.action.action_type == MemoryActionType.SAVE
        )
        assert save_triple.reward.r1_keyword_reappearance == 0.8
        assert save_triple.reward.r3_efficiency == 0.5
        assert save_triple.reward.total == 1.3

    def test_unspecified_turns_get_zero_reward(self, builder, minimal_episode):
        reward_map = {999: RewardBreakdown(total=5.0)}  # non-existent turn
        triples = builder.episode_to_sar_triples(minimal_episode, reward_map=reward_map)
        for t in triples:
            assert t.reward.total == 0.0


class TestBuildFromEpisodes:
    def test_multiple_episodes(self, builder, minimal_episode, sample_episode):
        """build_from_episodes aggregates triples from all episodes."""
        triples = builder.build_from_episodes([minimal_episode, sample_episode])
        episode_ids = {t.episode_id for t in triples}
        assert minimal_episode.episode_id in episode_ids
        assert sample_episode.episode_id in episode_ids

    def test_total_triples_is_sum(self, builder, minimal_episode):
        single = builder.episode_to_sar_triples(minimal_episode)
        batch = builder.build_from_episodes([minimal_episode, minimal_episode])
        # Two identical episodes (different episode_ids since Episode creates new ones)
        # but minimal_episode is the same object; it will appear twice
        assert len(batch) == 2 * len(single)


class TestParquetRows:
    def test_rows_have_required_columns(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        rows = builder.triples_to_parquet_rows(triples)
        required = [
            "triple_id",
            "episode_id",
            "step_index",
            "done",
            "state_turn_id",
            "state_memory_count",
            "state_turn_position",
            "state_recent_turns_json",
            "state_memory_summary_json",
            "action_type",
            "reward_total",
            "next_state_json",
        ]
        for col in required:
            assert col in rows[0], f"Missing column: {col}"

    def test_json_columns_are_valid(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        rows = builder.triples_to_parquet_rows(triples)
        for row in rows:
            json.loads(row["state_recent_turns_json"])
            json.loads(row["state_memory_summary_json"])
            json.loads(row["action_saved_keywords_json"])
            # next_state_json may be null
            if row["next_state_json"]:
                parsed = json.loads(row["next_state_json"])
                # null JSON value is fine for done=True triples
                assert parsed is None or isinstance(parsed, dict)

    def test_action_type_is_string(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        rows = builder.triples_to_parquet_rows(triples)
        for row in rows:
            assert isinstance(row["action_type"], str)

    def test_done_triples_have_null_next_state(self, builder, minimal_episode):
        triples = builder.episode_to_sar_triples(minimal_episode)
        rows = builder.triples_to_parquet_rows(triples)
        done_rows = [r for r in rows if r["done"]]
        for row in done_rows:
            assert json.loads(row["next_state_json"]) is None

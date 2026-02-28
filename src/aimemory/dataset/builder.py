"""Episode → SARTriple conversion for RL training dataset.

Converts raw episodes to State-Action-Reward triples with:
- State: last 6 turns dialogue window + current memory summary
- Next state for TD learning
- Edge case handling (first/last turns, empty memory)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from aimemory.config import DatasetConfig
from aimemory.schemas import (
    Action,
    Episode,
    MemoryActionType,
    RewardBreakdown,
    SARTriple,
    State,
    Turn,
)

logger = logging.getLogger(__name__)


class EpisodeBuilder:
    """Converts Episodes into SARTriple sequences for RL training."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def _build_state(
        self,
        episode: Episode,
        turn_id: int,
        memory_entries_up_to_turn: list,
    ) -> State:
        """Build RL State from episode context at a given turn.

        Args:
            episode: The source episode.
            turn_id: Current turn index (0-based).
            memory_entries_up_to_turn: Memory entries saved up to (but not including) this turn.

        Returns:
            State with recent dialogue window and memory summary.
        """
        window = self.config.context_window  # default 6

        # Get the last `window` turns up to and including turn_id
        start = max(0, turn_id - window + 1)
        recent_turns: list[Turn] = episode.turns[start : turn_id + 1]

        # Memory summary: list of content strings from saved entries
        memory_summary = [entry.content for entry in memory_entries_up_to_turn]

        # Normalized position in episode (0.0 ~ 1.0)
        num_turns = max(1, len(episode.turns))
        turn_position = turn_id / (num_turns - 1) if num_turns > 1 else 0.0

        return State(
            episode_id=episode.episode_id,
            turn_id=turn_id,
            recent_turns=recent_turns,
            current_memory_summary=memory_summary,
            memory_count=len(memory_entries_up_to_turn),
            turn_position=turn_position,
        )

    def _decision_to_action(self, decision) -> Action:
        """Convert a MemoryDecision to an RL Action."""
        action_type = decision.action

        saved_content = None
        saved_keywords: list[str] = []
        retrieved_count = 0

        if action_type == MemoryActionType.SAVE and decision.memory_entry is not None:
            saved_content = decision.memory_entry.content
            saved_keywords = decision.memory_entry.keywords

        if action_type == MemoryActionType.RETRIEVE:
            retrieved_count = len(decision.retrieved_memories)

        return Action(
            action_type=action_type,
            saved_content=saved_content,
            saved_keywords=saved_keywords,
            retrieved_count=retrieved_count,
        )

    def episode_to_sar_triples(
        self,
        episode: Episode,
        reward_map: dict[int, RewardBreakdown] | None = None,
    ) -> list[SARTriple]:
        """Convert an Episode into a list of SARTriples.

        Args:
            episode: Source episode to convert.
            reward_map: Optional mapping of turn_id → RewardBreakdown.
                        If None, zero rewards are used (to be filled later).

        Returns:
            List of SARTriples, one per memory decision in the episode.
        """
        if not episode.memory_decisions:
            logger.warning("Episode %s has no memory decisions", episode.episode_id)
            return []

        # Build a turn_id → decision lookup
        {d.turn_id: d for d in episode.memory_decisions}

        # Track cumulative memory entries as we step through decisions
        cumulative_memory: list = []

        triples: list[SARTriple] = []
        decisions_sorted = sorted(episode.memory_decisions, key=lambda d: d.turn_id)

        for step_index, decision in enumerate(decisions_sorted):
            turn_id = decision.turn_id

            # Validate turn_id is within episode
            if turn_id >= len(episode.turns):
                logger.warning(
                    "Decision turn_id %d out of range for episode %s (len=%d)",
                    turn_id,
                    episode.episode_id,
                    len(episode.turns),
                )
                continue

            # State: memory entries before this decision
            state = self._build_state(episode, turn_id, list(cumulative_memory))

            # Action
            action = self._decision_to_action(decision)

            # Reward: use provided map or zero default
            if reward_map and turn_id in reward_map:
                reward = reward_map[turn_id]
            else:
                reward = RewardBreakdown()

            # Update cumulative memory AFTER building current state
            if decision.action == MemoryActionType.SAVE and decision.memory_entry:
                cumulative_memory.append(decision.memory_entry)

            # Next state: state at the next decision, or None if last
            done = step_index == len(decisions_sorted) - 1
            if not done:
                next_decision = decisions_sorted[step_index + 1]
                next_turn_id = next_decision.turn_id
                if next_turn_id < len(episode.turns):
                    next_state = self._build_state(episode, next_turn_id, list(cumulative_memory))
                else:
                    next_state = None
                    done = True
            else:
                next_state = None

            triple = SARTriple(
                episode_id=episode.episode_id,
                step_index=step_index,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            triples.append(triple)

        return triples

    def build_from_episodes(
        self,
        episodes: list[Episode],
        reward_maps: dict[str, dict[int, RewardBreakdown]] | None = None,
    ) -> list[SARTriple]:
        """Build SARTriples from a list of episodes.

        Args:
            episodes: List of episodes to process.
            reward_maps: Optional mapping episode_id → {turn_id → RewardBreakdown}.

        Returns:
            Flat list of all SARTriples.
        """
        all_triples: list[SARTriple] = []
        for episode in episodes:
            reward_map = None
            if reward_maps:
                reward_map = reward_maps.get(episode.episode_id)
            triples = self.episode_to_sar_triples(episode, reward_map=reward_map)
            all_triples.extend(triples)
            logger.debug("Episode %s → %d triples", episode.episode_id, len(triples))
        return all_triples

    def iter_episodes_from_jsonl(self, path: Path) -> Iterator[Episode]:
        """Iterate over Episodes from a JSONL file."""
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield Episode.model_validate(data)
                except Exception as exc:
                    logger.error("Line %d parse error in %s: %s", line_num, path, exc)

    def triples_to_parquet_rows(self, triples: list[SARTriple]) -> list[dict]:
        """Convert SARTriples to flat dicts suitable for Parquet storage.

        Scalar fields become columns; nested structures become JSON string columns.
        """
        rows = []
        for t in triples:
            row = {
                "triple_id": t.triple_id,
                "episode_id": t.episode_id,
                "step_index": t.step_index,
                "done": t.done,
                # State scalars
                "state_turn_id": t.state.turn_id,
                "state_memory_count": t.state.memory_count,
                "state_turn_position": t.state.turn_position,
                # State nested as JSON
                "state_recent_turns_json": json.dumps(
                    [turn.model_dump(mode="json") for turn in t.state.recent_turns]
                ),
                "state_memory_summary_json": json.dumps(t.state.current_memory_summary),
                # Action scalars
                "action_type": t.action.action_type.value,
                "action_retrieved_count": t.action.retrieved_count,
                # Action nested as JSON
                "action_saved_content": t.action.saved_content or "",
                "action_saved_keywords_json": json.dumps(t.action.saved_keywords),
                # Reward scalars
                "reward_r1": t.reward.r1_keyword_reappearance,
                "reward_r2": t.reward.r2_repeated_question_penalty,
                "reward_r3": t.reward.r3_efficiency,
                "reward_r4": t.reward.r4_retrieval_relevance,
                "reward_r5": t.reward.r5_speech_act_weight,
                "reward_r6": t.reward.r6_self_reference,
                "reward_r7": t.reward.r7_info_density,
                "reward_r8": t.reward.r8_preference_constraint,
                "reward_r9": t.reward.r9_emotional_salience,
                "reward_r10": t.reward.r10_topic_boundary,
                "reward_r11": t.reward.r11_user_feedback,
                "reward_total": t.reward.total,
                # Next state (nullable)
                "next_state_json": json.dumps(
                    t.next_state.model_dump(mode="json") if t.next_state else None
                ),
            }
            rows.append(row)
        return rows

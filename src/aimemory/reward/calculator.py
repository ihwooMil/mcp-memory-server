"""Reward calculator for the AI Memory System.

Combines all 11 reward signals into a single weighted RewardBreakdown.
"""

from __future__ import annotations

from aimemory.config import RewardConfig
from aimemory.schemas import (
    Action,
    Episode,
    MemoryActionType,
    MemoryDecision,
    MemoryEntry,
    RewardBreakdown,
    State,
    Turn,
)

from .signals import (
    compute_r1_keyword_reappearance,
    compute_r2_repeated_question_penalty,
    compute_r3_efficiency,
    compute_r4_retrieval_relevance,
    compute_r5_speech_act_weight,
    compute_r6_self_reference,
    compute_r7_info_density,
    compute_r8_preference_constraint,
    compute_r9_emotional_salience,
    compute_r10_topic_boundary,
    compute_r11_user_feedback,
)


class RewardCalculator:
    """Computes weighted reward breakdown for a state-action pair.

    Usage::

        from aimemory.config import RewardConfig
        from aimemory.reward.calculator import RewardCalculator

        calc = RewardCalculator(config=RewardConfig())
        breakdown = calc.compute(
            state=state,
            action=action,
            current_turn=current_turn,
            history_turns=history_turns,
            future_turns=future_turns,
            previous_topic_summary="이전 주제 요약",
        )
        print(breakdown.total)
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        state: State,
        action: Action,
        current_turn: Turn,
        history_turns: list[Turn] | None = None,
        future_turns: list[Turn] | None = None,
        previous_topic_summary: str | None = None,
    ) -> RewardBreakdown:
        """Compute all reward signals and return a RewardBreakdown.

        Args:
            state: Current RL state (includes recent_turns and memory summary).
            action: Memory action taken at this step.
            current_turn: The actual Turn object being evaluated.
            history_turns: Turns before the current turn (for R2).
            future_turns: Turns after the current turn (for R1).
            previous_topic_summary: Summary of the previous topic (for R10).

        Returns:
            RewardBreakdown with per-signal scores and weighted total.
        """
        history_turns = history_turns or []
        future_turns = future_turns or []

        text = current_turn.content

        # R1: keyword reappearance (only meaningful for SAVE actions)
        r1 = 0.0
        if action.action_type == MemoryActionType.SAVE and future_turns:
            r1 = compute_r1_keyword_reappearance(
                state=state,
                action=action,
                future_turns=future_turns,
                proper_noun_multiplier=self.config.proper_noun_multiplier,
                common_noun_multiplier=self.config.common_noun_multiplier,
            )

        # R2: repeated question penalty
        r2 = compute_r2_repeated_question_penalty(
            current_turn=current_turn,
            history_turns=history_turns,
        )

        # R3: compression efficiency (only for SAVE actions with content)
        r3 = 0.0
        if action.action_type == MemoryActionType.SAVE and action.saved_content:
            r3 = compute_r3_efficiency(
                original_content=text,
                compressed_content=action.saved_content,
            )

        # R4: retrieval relevance (only for RETRIEVE actions)
        r4 = 0.0
        if action.action_type == MemoryActionType.RETRIEVE:
            from aimemory.reward.signals import _extract_keywords_from_text

            proxy_memories = [
                MemoryEntry(
                    content=summary,
                    source_turn_id=0,
                    keywords=_extract_keywords_from_text(summary),
                )
                for summary in state.current_memory_summary
            ]
            r4 = compute_r4_retrieval_relevance(
                retrieved_memories=proxy_memories,
                current_context=text,
            )

        # R5: speech act weight
        r5 = compute_r5_speech_act_weight(text)

        # R6: self-reference
        r6 = compute_r6_self_reference(text)

        # R7: information density
        r7 = compute_r7_info_density(text)

        # R8: preference/constraint expressions
        r8 = compute_r8_preference_constraint(text)

        # R9: emotional salience
        r9 = compute_r9_emotional_salience(text)

        # R10: topic boundary
        r10 = compute_r10_topic_boundary(
            current_turn=current_turn,
            previous_summary=previous_topic_summary,
        )

        # R11: user feedback
        r11 = compute_r11_user_feedback(text)

        breakdown = RewardBreakdown(
            r1_keyword_reappearance=r1,
            r2_repeated_question_penalty=r2,
            r3_efficiency=r3,
            r4_retrieval_relevance=r4,
            r5_speech_act_weight=r5,
            r6_self_reference=r6,
            r7_info_density=r7,
            r8_preference_constraint=r8,
            r9_emotional_salience=r9,
            r10_topic_boundary=r10,
            r11_user_feedback=r11,
        )
        breakdown.compute_total(weights=self.config.weights)
        return breakdown

    def compute_from_decision(
        self,
        state: State,
        decision: MemoryDecision,
        current_turn: Turn,
        history_turns: list[Turn] | None = None,
        future_turns: list[Turn] | None = None,
        previous_topic_summary: str | None = None,
    ) -> RewardBreakdown:
        """Compute reward from a MemoryDecision object.

        Convenience wrapper that converts MemoryDecision → Action.
        """
        action = Action(
            action_type=decision.action,
            saved_content=(decision.memory_entry.content if decision.memory_entry else None),
            saved_keywords=(decision.memory_entry.keywords if decision.memory_entry else []),
            retrieved_count=len(decision.retrieved_memories),
        )
        return self.compute(
            state=state,
            action=action,
            current_turn=current_turn,
            history_turns=history_turns,
            future_turns=future_turns,
            previous_topic_summary=previous_topic_summary,
        )

    def compute_episode_rewards(self, episode: Episode) -> dict[int, RewardBreakdown]:
        """Compute rewards for all memory decisions in an episode.

        Properly constructs history_turns, future_turns, and topic summaries
        for each decision, enabling R1, R10, and R11 to produce meaningful signals.
        """
        from aimemory.reward.korean_patterns import DISCOURSE_MARKERS

        rewards: dict[int, RewardBreakdown] = {}
        cumulative_memories: list[MemoryEntry] = []
        previous_topic_summary: str | None = None

        sorted_decisions = sorted(episode.memory_decisions, key=lambda d: d.turn_id)

        for decision in sorted_decisions:
            # Find the corresponding turn
            current_turn = None
            for t in episode.turns:
                if t.turn_id == decision.turn_id:
                    current_turn = t
                    break
            if current_turn is None:
                continue

            # Build history and future turns
            history_turns = [t for t in episode.turns if t.turn_id < decision.turn_id]
            future_turns = [t for t in episode.turns if t.turn_id > decision.turn_id]

            # Build state
            recent = [t for t in episode.turns if t.turn_id <= decision.turn_id][-6:]
            state = State(
                episode_id=episode.episode_id,
                turn_id=decision.turn_id,
                recent_turns=recent,
                current_memory_summary=[m.content for m in cumulative_memories],
                memory_count=len(cumulative_memories),
                turn_position=decision.turn_id / max(len(episode.turns), 1),
            )

            # Detect topic boundary for previous_topic_summary
            has_discourse = any(marker in current_turn.content for marker in DISCOURSE_MARKERS)
            if has_discourse and cumulative_memories:
                previous_topic_summary = "; ".join(m.content for m in cumulative_memories[-3:])

            # Compute reward
            breakdown = self.compute_from_decision(
                state=state,
                decision=decision,
                current_turn=current_turn,
                history_turns=history_turns,
                future_turns=future_turns,
                previous_topic_summary=previous_topic_summary,
            )
            rewards[decision.turn_id] = breakdown

            # Update cumulative memories
            if decision.memory_entry is not None:
                cumulative_memories.append(decision.memory_entry)

        return rewards

"""Self-play engine: runs two-LLM conversation episodes with memory decisions.

User LLM and Assistant LLM take turns in a Korean conversation.
The MemoryAgent makes save/retrieve/skip decisions at each turn.
Episodes are checkpointed every N episodes for resume support.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from aimemory.config import AppConfig, SelfPlayConfig
from aimemory.schemas import (
    Episode,
    MemoryActionType,
    MemoryDecision,
    Role,
    ScenarioType,
    Turn,
)
from aimemory.selfplay.llm_client import LLMClient, is_korean_text

# Patterns that indicate meta/instructional text rather than real dialogue
_META_PATTERNS = re.compile(
    r"^\s*[\(（]|"       # starts with parenthesis
    r"예시\s*형식|"      # "예시 형식"
    r"다음\s*턴|"        # "다음 턴"
    r"이후로도|"         # "이후로도"
    r"이런\s*식으로|"    # "이런 식으로"
    r"추가\s*입력|"      # "추가 입력"
    r"계속해서\s*제공|"  # "계속해서 제공"
    r"참고로|"           # "참고로"
    r"위 내용을|"        # "위 내용을"
    r"다음과 같이|"      # "다음과 같이"
    r"아래는|"           # "아래는"
    r"^\s*Sure|"         # "Sure"
    r"^\s*Here|"         # "Here"
    r"^\s*Let me",       # "Let me"
    re.MULTILINE,
)
from aimemory.selfplay.memory_agent import MemoryAgent, MemoryStore
from aimemory.selfplay.scenarios import KOREAN_ONLY_RULE, ScenarioManager, Topic

logger = logging.getLogger(__name__)


class SelfPlayEngine:
    """Orchestrates self-play episodes between User LLM and Assistant LLM."""

    def __init__(
        self,
        config: AppConfig | None = None,
        user_client: LLMClient | None = None,
        assistant_client: LLMClient | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self._selfplay_cfg: SelfPlayConfig = self.config.selfplay
        self._rng = random.Random(seed)

        # Allow injecting mock clients for testing
        self.user_client = user_client or LLMClient(self.config.ollama)
        self.assistant_client = assistant_client or LLMClient(self.config.ollama)

        self.scenario_mgr = ScenarioManager(seed=seed)
        self.memory_agent = MemoryAgent(seed=seed)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run(self, num_episodes: int | None = None, output_dir: Path | None = None) -> list[Episode]:
        """Run self-play for num_episodes and return completed episodes.

        Supports checkpoint / resume: completed episode files are skipped.
        Episodes are numbered 0..num_episodes-1; existing files are skipped.
        """
        num_episodes = num_episodes or self.config.num_episodes
        output_dir = output_dir or self.config.paths.raw_episodes
        output_dir.mkdir(parents=True, exist_ok=True)

        episodes: list[Episode] = []

        for ep_num in range(num_episodes):
            checkpoint_path = output_dir / f"episode_{ep_num:06d}.json"

            if checkpoint_path.exists():
                logger.info("Skipping already-completed episode %d", ep_num)
                continue

            scenario = self.scenario_mgr.random_scenario()
            episode = self.run_episode(scenario, episode_index=ep_num)
            episodes.append(episode)

            # Save episode to disk
            checkpoint_path.write_text(
                episode.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info(
                "Episode %d saved: %s turns, %d saves, %d retrieves",
                ep_num,
                episode.num_turns,
                episode.num_saves,
                episode.num_retrieves,
            )

            # Log checkpoint progress
            if (ep_num + 1) % self._selfplay_cfg.checkpoint_interval == 0:
                logger.info(
                    "Checkpoint: %d/%d episodes completed", ep_num + 1, num_episodes
                )

        return episodes

    def run_episode(
        self,
        scenario: ScenarioType,
        episode_index: int = 0,
        topic: Topic | None = None,
    ) -> Episode:
        """Run a single self-play episode and return the completed Episode."""
        # Pick a topic if not provided
        if topic is None:
            topic = self.scenario_mgr.random_topic()
            scenario = topic.scenario_type

        episode = Episode(
            scenario=scenario,
            metadata={
                "episode_index": episode_index,
                "topic_id": topic.id,
                "topic_name": topic.name,
            },
        )
        memory_store = MemoryStore()

        # Build initial context with strong Korean enforcement
        scenario_hint = ScenarioManager.scenario_system_prompt(scenario, topic)
        user_messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    KOREAN_ONLY_RULE
                    + self._selfplay_cfg.user_system_prompt
                    + "\n\n"
                    + scenario_hint
                ),
            }
        ]
        assistant_messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    KOREAN_ONLY_RULE
                    + self._selfplay_cfg.assistant_system_prompt
                    + "\n\n"
                    + scenario_hint
                ),
            }
        ]

        # Seed the conversation with a topic-appropriate opening
        seed_prompt = self.scenario_mgr.get_seed_prompt(topic)
        target_turns = self._rng.randint(
            self._selfplay_cfg.min_turns, self._selfplay_cfg.max_turns
        )

        # Sliding window size: keep system prompt + last N messages
        # This prevents context from growing unbounded and slowing LLM inference.
        _MAX_HISTORY = 8  # 4 user-assistant pairs

        # ── Turn loop ──────────────────────────────────────────────────
        consecutive_non_korean = 0
        pending_feedback: str | None = None
        for turn_index in range(target_turns):
            turn_id = len(episode.turns)

            if turn_index == 0:
                # First user turn: use seed prompt directly
                user_content = seed_prompt
            else:
                # Possibly inject a memory test (reference previous topic)
                injected = self._maybe_inject_memory_test(episode, user_messages)
                if injected:
                    user_content = injected
                else:
                    # Generate next user message (with sliding window)
                    windowed_user = self._windowed(user_messages, _MAX_HISTORY)
                    # A6: Inject memory summaries when context is truncated
                    non_system = [m for m in user_messages if m["role"] != "system"]
                    if memory_store.entries and len(non_system) > _MAX_HISTORY:
                        memory_summary = "축적된 정보:\n" + "\n".join(
                            f"- {e.content}" for e in memory_store.entries
                        )
                        windowed_user = [windowed_user[0], {"role": "system", "content": memory_summary}] + windowed_user[1:]
                    user_content = self.user_client.chat(windowed_user)

                # Filter meta/instructional text — retry once
                if _META_PATTERNS.search(user_content):
                    logger.warning("Turn %d: meta-text detected, regenerating", turn_id)
                    user_content = self.user_client.chat(
                        self._windowed(user_messages, _MAX_HISTORY)
                    )
                    # If still meta, use a fallback seed prompt using current topic
                    if _META_PATTERNS.search(user_content):
                        user_content = self.scenario_mgr.get_seed_prompt(topic)

                # B4: Prepend pending feedback from previous RETRIEVE decision
                if pending_feedback is not None:
                    user_content = pending_feedback + " " + user_content
                    pending_feedback = None

            # ── Korean enforcement for user turn ──
            if turn_index > 0 and not is_korean_text(user_content):
                consecutive_non_korean += 1
                if consecutive_non_korean >= 2:
                    # Topic switch: inject a new seed prompt to reset conversation
                    logger.warning(
                        "Turn %d: consecutive non-Korean user turns, switching topic",
                        turn_id,
                    )
                    new_scenario = self.scenario_mgr.random_scenario()
                    new_topic = self.scenario_mgr.random_topic()
                    user_content = self.scenario_mgr.get_seed_prompt(new_topic)
                    # Reset conversation context to break the non-Korean loop
                    user_messages = [user_messages[0]]  # keep system prompt only
                    assistant_messages = [assistant_messages[0]]
                    consecutive_non_korean = 0
            else:
                consecutive_non_korean = 0

            user_turn = Turn(
                turn_id=turn_id,
                role=Role.USER,
                content=user_content,
                token_count=len(user_content.split()),
            )
            episode.turns.append(user_turn)

            # Memory decision for user turn
            decision = self.memory_agent.decide(user_turn, memory_store, episode.turns)
            episode.memory_decisions.append(decision)
            if decision.action == MemoryActionType.SAVE and decision.memory_entry:
                memory_store.add(decision.memory_entry)
                episode.memory_store.append(decision.memory_entry)

            # B4: Check for feedback injection on next turn
            pending_feedback = self._maybe_inject_feedback(episode, decision)

            # Update conversation history for both LLMs
            user_messages.append({"role": "user", "content": user_content})

            # Build assistant context with retrieved memories (sliding window)
            assistant_ctx = self._windowed(assistant_messages, _MAX_HISTORY)
            # A6: Inject memory summaries when assistant context is truncated
            asst_non_system = [m for m in assistant_messages if m["role"] != "system"]
            if memory_store.entries and len(asst_non_system) > _MAX_HISTORY:
                memory_summary = "축적된 정보:\n" + "\n".join(
                    f"- {e.content}" for e in memory_store.entries
                )
                assistant_ctx = [assistant_ctx[0], {"role": "system", "content": memory_summary}] + assistant_ctx[1:]
            if decision.action == MemoryActionType.RETRIEVE and decision.retrieved_memories:
                memory_ctx = "관련 기억:\n" + "\n".join(
                    f"- {m.content}" for m in decision.retrieved_memories
                )
                assistant_ctx.append({"role": "system", "content": memory_ctx})
            assistant_ctx.append({"role": "user", "content": user_content})

            # Generate assistant response
            assistant_content = self.assistant_client.chat(assistant_ctx)

            # ── Korean enforcement for assistant turn ──
            if not is_korean_text(assistant_content):
                logger.warning(
                    "Turn %d: non-Korean assistant response, requesting Korean redirect",
                    turn_id + 1,
                )
                redirect_ctx = assistant_ctx + [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": "한국어로 다시 설명해 주세요. 코드 없이 한국어로만 답변해 주세요.",
                    },
                ]
                retry_content = self.assistant_client.chat(redirect_ctx)
                if is_korean_text(retry_content):
                    assistant_content = retry_content

            assistant_turn = Turn(
                turn_id=turn_id + 1,
                role=Role.ASSISTANT,
                content=assistant_content,
                token_count=len(assistant_content.split()),
            )
            episode.turns.append(assistant_turn)

            # Memory decision for assistant turn (A2: may SAVE if paraphrase detected)
            asst_decision = self.memory_agent.decide(
                assistant_turn, memory_store, episode.turns
            )
            episode.memory_decisions.append(asst_decision)
            if asst_decision.action == MemoryActionType.SAVE and asst_decision.memory_entry:
                memory_store.add(asst_decision.memory_entry)
                episode.memory_store.append(asst_decision.memory_entry)

            # Update histories
            user_messages.append({"role": "assistant", "content": assistant_content})
            assistant_messages.append({"role": "user", "content": user_content})
            assistant_messages.append({"role": "assistant", "content": assistant_content})

        # A8: Add quality metrics to episode metadata
        total_decisions = len(episode.memory_decisions)
        save_count = episode.num_saves
        retrieve_count = episode.num_retrieves
        skip_count = total_decisions - save_count - retrieve_count
        episode.metadata["quality_metrics"] = {
            "save_rate": save_count / max(total_decisions, 1),
            "retrieve_rate": retrieve_count / max(total_decisions, 1),
            "skip_rate": skip_count / max(total_decisions, 1),
            "avg_turn_length": sum(len(t.content) for t in episode.turns) / max(len(episode.turns), 1),
            "num_memories": len(episode.memory_store),
        }

        return episode

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _windowed(messages: list[dict], max_history: int) -> list[dict]:
        """Return system prompt + last max_history non-system messages.

        This prevents the LLM context from growing unbounded as
        conversations get longer, keeping inference time constant.
        """
        system = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        if len(non_system) <= max_history:
            return list(messages)
        return system + non_system[-max_history:]

    def _maybe_inject_memory_test(
        self,
        episode: Episode,
        user_messages: list[dict],
    ) -> str | None:
        """25% chance to generate a user message that references a previous topic.

        Returns an injected prompt string, or None to use normal LLM generation.
        """
        if self._rng.random() > self._selfplay_cfg.memory_test_probability:
            return None
        if not episode.memory_store:
            return None

        # Pick a random prior memory
        prior_memory = self._rng.choice(episode.memory_store)
        inject_prompt = (
            f"이전에 '{prior_memory.content[:60]}' 라고 하셨는데, "
            "그것에 대해 좀 더 이야기해 주시겠어요?"
        )

        # Let user LLM rephrase the reference naturally
        probe_messages = list(user_messages) + [
            {
                "role": "system",
                "content": (
                    f"다음 내용을 자연스럽게 이전 대화를 참조하는 방식으로 한 문장으로 말하세요: {inject_prompt}"
                ),
            }
        ]
        try:
            rephrased = self.user_client.chat(probe_messages)
            return rephrased
        except Exception:
            # Fall back to the raw inject prompt
            return inject_prompt

    def _maybe_inject_feedback(
        self,
        episode: Episode,
        decision: MemoryDecision,
    ) -> str | None:
        """After RETRIEVE, 30% chance to inject user feedback about memory accuracy.

        Returns a feedback string to prepend to the next user turn, or None.
        """
        if decision.action != MemoryActionType.RETRIEVE:
            return None
        if not decision.retrieved_memories:
            return None
        if self._rng.random() > 0.3:
            return None

        memory = decision.retrieved_memories[0]
        if self._rng.random() < 0.5:
            # Positive feedback
            templates = [
                f"맞아요, 제가 그때 '{memory.content[:30]}' 관련해서 말했었죠.",
                "정확해요! 잘 기억하시네요.",
                "네 맞아요, 그 얘기 했었죠.",
            ]
        else:
            # Negative feedback
            templates = [
                "아니 그게 아니라, 제가 말한 건 좀 달라요.",
                "아닌데요, 제 말은 그게 아니었어요.",
                "기억이 잘못되셨네요, 제가 말한 건 다른 거예요.",
            ]
        return self._rng.choice(templates)

    @staticmethod
    def _load_completed_ids(output_dir: Path) -> set[str]:
        """Scan output directory for completed episode files."""
        ids: set[str] = set()
        for p in output_dir.glob("episode_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                ids.add(data.get("episode_id", p.stem))
            except Exception:
                pass
        return ids

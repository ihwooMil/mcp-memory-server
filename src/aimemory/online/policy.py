"""Online contextual bandit policy for memory action selection.

Provides:
- StateEncoder: converts conversation state to a feature vector
- OnlinePolicy: neural contextual bandit (PyTorch MLP, epsilon-greedy)
- MemoryPolicyAgent: integrates policy with GraphMemoryStore and FeedbackDetector
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from aimemory.i18n import LanguagePatterns, get_patterns
from aimemory.schemas import (
    MemoryActionType,
    MemoryDecision,
    MemoryEntry,
    Role,
    Turn,
)

logger = logging.getLogger(__name__)

# Action indices: 0=SAVE, 1=SKIP, 2=RETRIEVE
ACTION_MAP: dict[int, MemoryActionType] = {
    0: MemoryActionType.SAVE,
    1: MemoryActionType.SKIP,
    2: MemoryActionType.RETRIEVE,
}
ACTION_INDEX: dict[MemoryActionType, int] = {v: k for k, v in ACTION_MAP.items()}

FEATURE_DIM = 10  # default feature dimension


# ─── StateEncoder ─────────────────────────────────────────────────────


class StateEncoder:
    """Converts conversation state to a fixed-size feature vector.

    Features (10-dim):
        0: turn_position        - normalized position in conversation (0.0-1.0)
        1: memory_count         - number of memories stored (log-scaled)
        2: keyword_count        - keywords found in current turn (log-scaled)
        3: is_question          - 1.0 if current turn is a question
        4: has_personal_info    - 1.0 if personal info pattern detected
        5: has_preference       - 1.0 if preference pattern detected
        6: has_tech             - 1.0 if technical keywords detected
        7: has_emotion          - 1.0 if emotion keywords detected
        8: recent_save_count    - SAVE actions in recent turns (log-scaled)
        9: recent_retrieve_count - RETRIEVE actions in recent turns (log-scaled)
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, lang: str = "ko") -> None:
        self.feature_dim = feature_dim
        lp = get_patterns(lang)
        self._question_re = re.compile(lp.question_pattern)
        self._personal_res = [re.compile(p) for p in lp.personal_info_patterns]
        self._preference_res = [re.compile(p) for p in lp.preference_patterns]
        self._tech_re = re.compile(lp.tech_keywords, re.IGNORECASE)
        self._emotion_re = re.compile(lp.emotion_keywords)

    def encode(
        self,
        turn: Turn,
        recent_turns: list[Turn],
        memory_count: int = 0,
        recent_actions: list[MemoryActionType] | None = None,
        turn_position: float = 0.0,
    ) -> np.ndarray:
        """Encode conversation state into a feature vector."""
        text = turn.content
        recent_actions = recent_actions or []

        # Extract keywords count using tech pattern + quoted strings
        tech_matches = self._tech_re.findall(text)
        quoted_matches = re.findall(r"['\"]([^'\"]{2,30})['\"]", text)
        keyword_count = len(tech_matches) + len(quoted_matches)

        # Pattern detections
        is_question = 1.0 if self._question_re.search(text) else 0.0
        has_personal = 1.0 if any(p.search(text) for p in self._personal_res) else 0.0
        has_preference = 1.0 if any(p.search(text) for p in self._preference_res) else 0.0
        has_tech = 1.0 if self._tech_re.search(text) else 0.0
        has_emotion = 1.0 if self._emotion_re.search(text) else 0.0

        # Count recent actions
        recent_save = sum(1 for a in recent_actions if a == MemoryActionType.SAVE)
        recent_retrieve = sum(1 for a in recent_actions if a == MemoryActionType.RETRIEVE)

        features = np.array(
            [
                turn_position,
                np.log1p(memory_count),
                np.log1p(keyword_count),
                is_question,
                has_personal,
                has_preference,
                has_tech,
                has_emotion,
                np.log1p(recent_save),
                np.log1p(recent_retrieve),
            ],
            dtype=np.float32,
        )
        return features


# ─── OnlinePolicy (Neural Contextual Bandit) ─────────────────────────


class _BanditMLP(nn.Module):
    """Small MLP for contextual bandit action scoring."""

    def __init__(self, feature_dim: int, hidden_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OnlinePolicy:
    """Neural contextual bandit with epsilon-greedy exploration.

    Uses a small MLP to score actions given state features,
    updated via single-step SGD on observed rewards.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        n_actions: int = 3,
        hidden_dim: int = 64,
        lr: float = 0.01,
        epsilon: float = 0.1,
    ) -> None:
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.epsilon = epsilon
        self._model = _BanditMLP(feature_dim, hidden_dim, n_actions)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._rng = np.random.default_rng()

    def select_action(self, features: np.ndarray) -> int:
        """Select an action using epsilon-greedy strategy."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))

        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0)
            scores = self._model(x)
            return int(scores.argmax(dim=1).item())

    def update(self, features: np.ndarray, action_id: int, reward: float) -> float:
        """Single-step SGD update on the observed (state, action, reward)."""
        self._model.train()
        x = torch.from_numpy(features).float().unsqueeze(0)
        scores = self._model(x)
        q_value = scores[0, action_id]
        target = torch.tensor(reward, dtype=torch.float32)
        loss = (q_value - target) ** 2

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return float(loss.item())

    def get_parameters(self) -> np.ndarray:
        """Return flattened model parameters as a numpy array (for gossip)."""
        params = []
        for p in self._model.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_parameters(self, params: np.ndarray) -> None:
        """Set model parameters from a flattened numpy array (from gossip)."""
        offset = 0
        for p in self._model.parameters():
            numel = p.data.numel()
            chunk = params[offset : offset + numel]
            p.data.copy_(torch.from_numpy(chunk.reshape(p.data.shape)).float())
            offset += numel

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model and optimizer state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "epsilon": self.epsilon,
                "feature_dim": self.feature_dim,
                "n_actions": self.n_actions,
            },
            str(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model and optimizer state from file."""
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)


# ─── MemoryPolicyAgent ────────────────────────────────────────────────


class MemoryPolicyAgent:
    """Policy-driven memory agent that replaces the rule-based MemoryAgent."""

    def __init__(
        self,
        graph_store,  # GraphMemoryStore (import avoided for loose coupling)
        policy: OnlinePolicy,
        feedback_detector,  # FeedbackDetector
        encoder: StateEncoder | None = None,
        reranker=None,  # Optional[ReRanker] (loose coupling)
        lang: str = "ko",
    ) -> None:
        self._store = graph_store
        self._policy = policy
        self._feedback = feedback_detector
        self._encoder = encoder or StateEncoder(lang=lang)
        self._reranker = reranker
        self._recent_actions: list[MemoryActionType] = []
        self._last_features: Optional[np.ndarray] = None
        self._last_action_id: Optional[int] = None
        self._last_retrieved_ids: list[str] = []
        self._lp = get_patterns(lang)

    def _compute_importance(self, turn: Turn, features: np.ndarray) -> float:
        """Rule-based importance score [0.0, 1.0]."""
        score = 0.0
        if features[4]:  score += 0.4   # has_personal_info
        if features[5]:  score += 0.35  # has_preference
        if features[6]:  score += 0.3   # has_tech
        if features[7]:  score += 0.2   # has_emotion
        score += min(features[2] * 0.15, 0.3)  # keyword density (log-scaled)
        return min(score, 1.0)

    def _should_retrieve(self, turn: Turn, features: np.ndarray) -> bool:
        """Rule-based RETRIEVE decision."""
        is_question = bool(features[3])
        has_discourse = any(marker in turn.content for marker in self._lp.discourse_markers)
        has_memory = self._store.get_stats().get("total", 0) > 0
        return (is_question or has_discourse) and has_memory

    def decide(self, turn: Turn, recent_turns: list[Turn], turn_position: float = 0.0) -> MemoryDecision:
        """Decide memory action using rule-based scoring + MLP bandit fallback."""
        # Non-user turns are always skipped
        if turn.role != Role.USER:
            return MemoryDecision(turn_id=turn.turn_id, action=MemoryActionType.SKIP)

        memory_count = self._store.get_stats().get("total", 0)
        features = self._encoder.encode(
            turn=turn,
            recent_turns=recent_turns,
            memory_count=memory_count,
            recent_actions=self._recent_actions[-6:] if self._recent_actions else None,
            turn_position=turn_position,
        )

        importance = self._compute_importance(turn, features)

        # Phase 0: RETRIEVE check (question or discourse marker)
        if self._should_retrieve(turn, features):
            result = self._execute_retrieve(turn)
            if result.retrieved_memories:
                self._last_features = features
                self._last_action_id = ACTION_INDEX[MemoryActionType.RETRIEVE]
                self._recent_actions.append(MemoryActionType.RETRIEVE)
                return result

        # Phase 1: Rule-based high/low confidence
        if importance >= 0.7:
            self._last_features = features
            self._last_action_id = ACTION_INDEX[MemoryActionType.SAVE]
            self._recent_actions.append(MemoryActionType.SAVE)
            return self._execute_save(turn)

        if importance <= 0.1:
            self._last_features = features
            self._last_action_id = ACTION_INDEX[MemoryActionType.SKIP]
            self._recent_actions.append(MemoryActionType.SKIP)
            return MemoryDecision(turn_id=turn.turn_id, action=MemoryActionType.SKIP)

        # Phase 2: Mid-confidence → MLP bandit decides
        action_id = self._policy.select_action(features)
        action_type = ACTION_MAP[action_id]
        self._last_features = features
        self._last_action_id = action_id
        self._recent_actions.append(action_type)
        return self._execute_action(turn, action_type)

    def process_feedback(
        self,
        user_turn: Turn,
        previous_turns: list[Turn],
    ) -> tuple:
        """Process user feedback and update policy."""
        from aimemory.reward.feedback_detector import FeedbackType

        last_action = (
            self._recent_actions[-1] if self._recent_actions else MemoryActionType.SKIP
        )
        signal = self._feedback.detect(user_turn, previous_turns, last_action)

        # Update policy if we have stored state
        if self._last_features is not None and self._last_action_id is not None:
            self._policy.update(
                self._last_features, self._last_action_id, signal.reward_value
            )

        if (
            self._reranker is not None
            and self._last_action_id == ACTION_INDEX[MemoryActionType.RETRIEVE]
            and self._reranker.has_pending_state
            and signal.signal_type != FeedbackType.NEUTRAL
        ):
            self._reranker.update_from_feedback(signal.reward_value)

        return signal, signal.reward_value

    def _execute_action(self, turn: Turn, action_type: MemoryActionType) -> MemoryDecision:
        """Execute the selected memory action."""
        if action_type == MemoryActionType.SAVE:
            return self._execute_save(turn)
        elif action_type == MemoryActionType.RETRIEVE:
            return self._execute_retrieve(turn)
        else:
            return MemoryDecision(
                turn_id=turn.turn_id,
                action=MemoryActionType.SKIP,
                reasoning="Policy selected SKIP",
            )

    def _execute_save(self, turn: Turn) -> MemoryDecision:
        """Save turn content to graph memory store."""
        from aimemory.selfplay.memory_agent import (
            classify_category,
            extract_keywords,
        )

        keywords = extract_keywords(turn.content)
        if not keywords:
            return MemoryDecision(
                turn_id=turn.turn_id,
                action=MemoryActionType.SKIP,
                reasoning="SAVE selected but no keywords extracted; fallback to SKIP",
            )

        category = classify_category(turn.content, keywords)
        cat_map = {
            "general": "fact",
            "personal": "fact",
            "technical": "technical",
            "preference": "preference",
        }
        graph_category = cat_map.get(category, "fact")

        content = turn.content[:150].strip()
        memory_id = self._store.add_memory(
            content=content,
            keywords=keywords,
            category=graph_category,
            source_turn_id=turn.turn_id,
        )

        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            source_turn_id=turn.turn_id,
            keywords=keywords,
            category=category,
        )
        return MemoryDecision(
            turn_id=turn.turn_id,
            action=MemoryActionType.SAVE,
            memory_entry=entry,
            reasoning="Policy selected SAVE",
        )

    def _execute_retrieve(self, turn: Turn) -> MemoryDecision:
        """Retrieve relevant memories from graph store, optionally re-ranked."""
        if self._reranker is not None and self._reranker.enabled:
            candidates = self._store.search(turn.content, top_k=10)
            if candidates:
                results = self._reranker.rerank(
                    query=turn.content,
                    candidates=candidates,
                )
                self._last_retrieved_ids = [n.memory_id for n in results]
            else:
                results = []
                self._last_retrieved_ids = []
        else:
            results = self._store.search(turn.content, top_k=3)
            self._last_retrieved_ids = [n.memory_id for n in results]

        if not results:
            return MemoryDecision(
                turn_id=turn.turn_id,
                action=MemoryActionType.RETRIEVE,
                retrieved_memories=[],
                reasoning="Policy selected RETRIEVE but no results found",
            )

        retrieved = [
            MemoryEntry(
                memory_id=node.memory_id,
                content=node.content,
                source_turn_id=0,
                keywords=node.keywords,
                category=node.category,
            )
            for node in results
        ]
        return MemoryDecision(
            turn_id=turn.turn_id,
            action=MemoryActionType.RETRIEVE,
            retrieved_memories=retrieved,
            reasoning=f"Policy selected RETRIEVE, {len(retrieved)} results",
        )

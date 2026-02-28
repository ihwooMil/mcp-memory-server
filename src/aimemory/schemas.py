"""Data schemas for the AI Memory System.

Defines Pydantic models for:
- Conversation-level: Turn, MemoryEntry, MemoryDecision, Episode
- RL training: State, Action, RewardBreakdown, SARTriple
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ─── Conversation-level schemas ───


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Turn(BaseModel):
    """A single dialogue turn."""

    turn_id: int
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    token_count: int = 0


class MemoryEntry(BaseModel):
    """A piece of information stored in memory."""

    memory_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str
    source_turn_id: int
    keywords: list[str] = Field(default_factory=list)
    category: str = "general"
    created_at: datetime = Field(default_factory=datetime.now)


class MemoryActionType(str, Enum):
    SAVE = "save"
    SKIP = "skip"
    RETRIEVE = "retrieve"


class MemoryDecision(BaseModel):
    """A decision made by the memory agent at a specific turn."""

    turn_id: int
    action: MemoryActionType
    memory_entry: Optional[MemoryEntry] = None
    retrieved_memories: list[MemoryEntry] = Field(default_factory=list)
    reasoning: str = ""


class ScenarioType(str, Enum):
    CASUAL_CHAT = "casual_chat"
    TECHNICAL_QA = "technical_qa"
    PROJECT_DISCUSSION = "project_discussion"
    PERSONAL_PREFERENCES = "personal_preferences"
    LEARNING_TUTORING = "learning_tutoring"
    TROUBLESHOOTING = "troubleshooting"


class Episode(BaseModel):
    """A complete self-play conversation episode."""

    episode_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    scenario: ScenarioType
    turns: list[Turn] = Field(default_factory=list)
    memory_decisions: list[MemoryDecision] = Field(default_factory=list)
    memory_store: list[MemoryEntry] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def num_saves(self) -> int:
        return sum(1 for d in self.memory_decisions if d.action == MemoryActionType.SAVE)

    @property
    def num_retrieves(self) -> int:
        return sum(1 for d in self.memory_decisions if d.action == MemoryActionType.RETRIEVE)


# ─── RL Training schemas ───


class State(BaseModel):
    """RL state: recent conversation window + current memory summary."""

    episode_id: str
    turn_id: int
    recent_turns: list[Turn]  # last 6 turns
    current_memory_summary: list[str] = Field(default_factory=list)
    memory_count: int = 0
    turn_position: float = 0.0  # normalized position in episode (0.0 ~ 1.0)


class Action(BaseModel):
    """RL action: the memory decision made."""

    action_type: MemoryActionType
    saved_content: Optional[str] = None
    saved_keywords: list[str] = Field(default_factory=list)
    retrieved_count: int = 0


class RewardBreakdown(BaseModel):
    """Detailed breakdown of reward signals."""

    r1_keyword_reappearance: float = 0.0
    r2_repeated_question_penalty: float = 0.0
    r3_efficiency: float = 0.0
    r4_retrieval_relevance: float = 0.0
    r5_speech_act_weight: float = 0.0
    r6_self_reference: float = 0.0
    r7_info_density: float = 0.0
    r8_preference_constraint: float = 0.0
    r9_emotional_salience: float = 0.0
    r10_topic_boundary: float = 0.0
    r11_user_feedback: float = 0.0
    total: float = 0.0

    def compute_total(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted sum of all reward signals."""
        default_weights = {
            "r1_keyword_reappearance": 1.0,
            "r2_repeated_question_penalty": 1.0,
            "r3_efficiency": 0.8,
            "r4_retrieval_relevance": 1.2,
            "r5_speech_act_weight": 0.7,
            "r6_self_reference": 0.7,
            "r7_info_density": 0.5,
            "r8_preference_constraint": 0.9,
            "r9_emotional_salience": 0.4,
            "r10_topic_boundary": 0.6,
            "r11_user_feedback": 1.0,
        }
        w = weights or default_weights
        self.total = sum(w.get(field, 1.0) * getattr(self, field) for field in default_weights)
        return self.total


class SARTriple(BaseModel):
    """State-Action-Reward triple for RL training."""

    triple_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    episode_id: str
    step_index: int
    state: State
    action: Action
    reward: RewardBreakdown
    next_state: Optional[State] = None
    done: bool = False

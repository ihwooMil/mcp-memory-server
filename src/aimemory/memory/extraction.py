"""Memory extraction from conversation logs.

Provides heuristic and RL-based extractors that decide which conversation
turns contain memories worth saving, plus a progressive transition manager.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from aimemory.memory.graph_store import GraphMemoryStore

logger = logging.getLogger(__name__)


# ── Pattern sets (reused from selfplay.memory_agent) ─────────────────

_TECH_KEYWORDS = re.compile(
    r"(?<![가-힣a-zA-Z_])(?:"
    r"Python|Java(?:Script)?|TypeScript|Rust|Go|C\+\+|Ruby|Swift|Kotlin|Dart|"
    r"React|Vue|Angular|Next\.js|Django|Flask|FastAPI|Spring|Rails|"
    r"Docker|Kubernetes|k8s|AWS|GCP|Azure|Linux|Ubuntu|macOS|Windows|"
    r"MySQL|PostgreSQL|SQLite|Redis|MongoDB|Elasticsearch|"
    r"Git|GitHub|GitLab|CI/CD|DevOps|MLOps|"
    r"pandas|numpy|scipy|sklearn|scikit-learn|TensorFlow|PyTorch|Keras|"
    r"LLM|GPT|Claude|Gemini|머신러닝|딥러닝|인공지능|AI|"
    r"API|REST|GraphQL|WebSocket|gRPC|"
    r"알고리즘|자료구조|데이터베이스|클라우드|마이크로서비스"
    r")(?![a-zA-Z_])",
    re.IGNORECASE,
)

_PREFERENCE_PATTERNS = [
    re.compile(r"좋아(?:해요|합니다|하는|함)"),
    re.compile(r"싫어(?:해요|합니다|하는|함)"),
    re.compile(r"선호(?:해요|합니다|하는|함|하다)"),
    re.compile(r"취미(?:가|는|로)?"),
    re.compile(r"주로\s+\S+"),
    re.compile(r"즐겨\s*\S+"),
]

_PERSONAL_INFO_PATTERNS = [
    re.compile(r"저는?\s+(.{2,20})(?:이에요|입니다|예요|인데|거든요|이라서)"),
    re.compile(r"제\s+(?:이름|나이|직업|전공|회사|팀|프로젝트)"),
    re.compile(r"살고\s*있(?:어요|습니다)"),
    re.compile(r"다니고\s*있(?:어요|습니다)"),
    re.compile(r"일하고\s*있(?:어요|습니다)"),
]

_EMOTION_KEYWORDS = re.compile(
    r"기쁘|슬프|화나|무서|불안|설레|걱정|힘들|어렵|좋아|싫어|즐거|행복|우울|피곤|신나"
)


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class ExtractionCandidate:
    """A conversation turn evaluated for memory extraction."""

    turn_id: int
    conversation_id: str
    role: str
    content: str
    should_extract: bool = False
    category: str = "fact"
    keywords: list[str] = field(default_factory=list)
    info_density: float = 0.0
    extraction_source: str = "heuristic"


@dataclass
class ExtractionResult:
    """Result summary from a batch extraction run."""

    turns_processed: int = 0
    memories_extracted: int = 0
    memories_deduplicated: int = 0
    extraction_mode: str = ""
    errors: list[str] = field(default_factory=list)


# ── Heuristic Extractor ─────────────────────────────────────────────


class HeuristicMemoryExtractor:
    """Pattern-matching based memory extractor.

    Reuses classification logic from selfplay.memory_agent:
    - extract_keywords() for keyword extraction
    - classify_category() for category assignment
    - Info density (keyword / token ratio) filtering
    """

    def __init__(self, min_info_density: float = 0.1) -> None:
        self._min_info_density = min_info_density

    def evaluate(self, content: str, role: str = "user") -> ExtractionCandidate:
        """Evaluate whether a turn should be extracted as a memory."""
        from aimemory.selfplay.memory_agent import classify_category, extract_keywords

        candidate = ExtractionCandidate(
            turn_id=0,
            conversation_id="",
            role=role,
            content=content,
            extraction_source="heuristic",
        )

        # Skip very short content
        if len(content.strip()) <= 20:
            return candidate

        keywords = extract_keywords(content)
        candidate.keywords = keywords

        # Compute info density: keywords per token
        tokens = content.split()
        token_count = max(len(tokens), 1)
        candidate.info_density = len(keywords) / token_count

        # Skip low-density turns (greetings, fillers)
        if candidate.info_density < self._min_info_density and not self._has_pattern_match(content):
            return candidate

        # Pattern matching for explicit signals
        has_personal = any(p.search(content) for p in _PERSONAL_INFO_PATTERNS)
        has_preference = any(p.search(content) for p in _PREFERENCE_PATTERNS)
        has_tech = bool(_TECH_KEYWORDS.search(content))
        has_emotion = bool(_EMOTION_KEYWORDS.search(content))

        if has_personal or has_preference or has_tech or has_emotion or len(keywords) >= 2:
            candidate.should_extract = True

            # Category classification
            raw_category = classify_category(content, keywords)
            cat_map = {
                "general": "fact",
                "personal": "fact",
                "technical": "technical",
                "preference": "preference",
            }
            candidate.category = cat_map.get(raw_category, "fact")

            # Add emotion category if emotion is the dominant signal
            if has_emotion and not (has_personal or has_preference or has_tech):
                candidate.category = "emotion"

        return candidate

    def _has_pattern_match(self, content: str) -> bool:
        """Check if content matches any extraction-worthy pattern."""
        if any(p.search(content) for p in _PERSONAL_INFO_PATTERNS):
            return True
        if any(p.search(content) for p in _PREFERENCE_PATTERNS):
            return True
        if _TECH_KEYWORDS.search(content):
            return True
        if _EMOTION_KEYWORDS.search(content):
            return True
        return False

    def get_features(self, content: str) -> np.ndarray:
        """Extract numeric features for RL training.

        Returns 6-dim vector: [keyword_count, info_density,
            has_personal, has_preference, has_tech, has_emotion].
        """
        from aimemory.selfplay.memory_agent import extract_keywords

        keywords = extract_keywords(content)
        tokens = content.split()
        token_count = max(len(tokens), 1)

        return np.array([
            np.log1p(len(keywords)),
            len(keywords) / token_count,
            1.0 if any(p.search(content) for p in _PERSONAL_INFO_PATTERNS) else 0.0,
            1.0 if any(p.search(content) for p in _PREFERENCE_PATTERNS) else 0.0,
            1.0 if _TECH_KEYWORDS.search(content) else 0.0,
            1.0 if _EMOTION_KEYWORDS.search(content) else 0.0,
        ], dtype=np.float32)


# ── RL Extractor ─────────────────────────────────────────────────────


class _ExtractionMLP(nn.Module):
    """Binary classifier MLP for EXTRACT / SKIP decision."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 0=SKIP, 1=EXTRACT
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RLMemoryExtractor:
    """RL-based memory extractor using MLP bandit.

    Input: heuristic features (6d) from HeuristicMemoryExtractor.
    Output: EXTRACT (1) or SKIP (0) binary decision.

    Initially learns via imitation of heuristic decisions,
    then transitions to independent RL-based decisions.
    """

    def __init__(
        self,
        feature_dim: int = 6,
        hidden_dim: int = 32,
        lr: float = 0.005,
        epsilon: float = 0.1,
    ) -> None:
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self._model = _ExtractionMLP(feature_dim, hidden_dim)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._rng = np.random.default_rng()
        self._total_decisions: int = 0
        self._correct_imitations: int = 0

    def predict(self, features: np.ndarray) -> int:
        """Predict EXTRACT (1) or SKIP (0) using epsilon-greedy."""
        self._total_decisions += 1
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, 2))

        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0)
            scores = self._model(x)
            return int(scores.argmax(dim=1).item())

    def update(self, features: np.ndarray, action: int, reward: float) -> float:
        """Single-step SGD update. Returns loss value."""
        self._model.train()
        x = torch.from_numpy(features).float().unsqueeze(0)
        scores = self._model(x)
        q_value = scores[0, action]
        target = torch.tensor(reward, dtype=torch.float32)
        loss = (q_value - target) ** 2

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return float(loss.item())

    def imitation_update(self, features: np.ndarray, heuristic_decision: bool) -> float:
        """Learn from heuristic decision (imitation learning)."""
        action = 1 if heuristic_decision else 0

        # Check if RL agrees with heuristic (before update)
        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0)
            rl_action = int(self._model(x).argmax(dim=1).item())
        if rl_action == action:
            self._correct_imitations += 1

        # Imitation reward: +1 for matching heuristic
        return self.update(features, action, reward=1.0)

    @property
    def imitation_accuracy(self) -> float:
        """Fraction of RL decisions that agree with heuristic."""
        if self._total_decisions == 0:
            return 0.0
        return self._correct_imitations / self._total_decisions

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_decisions": self._total_decisions,
                "correct_imitations": self._correct_imitations,
            },
            str(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state."""
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self._total_decisions = checkpoint.get("total_decisions", 0)
        self._correct_imitations = checkpoint.get("correct_imitations", 0)


# ── Progressive Extraction ───────────────────────────────────────────


class ProgressiveExtraction:
    """Manages the transition from heuristic to RL-based extraction.

    Stages:
    - heuristic_only: RL observes and learns via imitation
    - rl_assisted: RL + heuristic must agree (consensus)
    - rl_primary: RL decides independently, heuristic is safety net

    Follows the same pattern as online.autonomy.ProgressiveAutonomy.
    """

    def __init__(
        self,
        heuristic: HeuristicMemoryExtractor,
        rl_extractor: RLMemoryExtractor,
        confidence_threshold: int = 50,
    ) -> None:
        self._heuristic = heuristic
        self._rl = rl_extractor
        self._confidence_threshold = confidence_threshold
        self._confidence: float = 0.0
        self._decision_count: int = 0

    @property
    def mode(self) -> str:
        """Current extraction mode."""
        if self._decision_count < self._confidence_threshold:
            return "heuristic_only"
        if self._confidence < self._confidence_threshold:
            return "rl_assisted"
        return "rl_primary"

    @property
    def confidence(self) -> float:
        return self._confidence

    def evaluate(self, content: str, role: str = "user") -> ExtractionCandidate:
        """Evaluate a turn using the current mode's strategy."""
        self._decision_count += 1

        # Always run heuristic
        heuristic_result = self._heuristic.evaluate(content, role)
        features = self._heuristic.get_features(content)

        current_mode = self.mode

        if current_mode == "heuristic_only":
            # RL observes and learns via imitation
            self._rl.imitation_update(features, heuristic_result.should_extract)
            heuristic_result.extraction_source = "heuristic"
            return heuristic_result

        rl_decision = self._rl.predict(features) == 1

        if current_mode == "rl_assisted":
            # Both must agree to extract
            if heuristic_result.should_extract and rl_decision:
                heuristic_result.extraction_source = "rl_assisted"
                return heuristic_result
            elif heuristic_result.should_extract and not rl_decision:
                # Heuristic says yes, RL says no - use heuristic but train RL
                self._rl.imitation_update(features, True)
                heuristic_result.extraction_source = "heuristic"
                return heuristic_result
            else:
                heuristic_result.should_extract = False
                return heuristic_result

        # rl_primary mode
        if rl_decision:
            # RL says extract - use it
            candidate = heuristic_result  # reuse keywords/category from heuristic
            candidate.should_extract = True
            candidate.extraction_source = "rl"
            return candidate
        elif heuristic_result.should_extract:
            # Safety net: heuristic catches what RL misses
            heuristic_result.extraction_source = "heuristic"
            return heuristic_result
        else:
            heuristic_result.should_extract = False
            return heuristic_result

    def record_feedback(self, features: np.ndarray, action: int, reward: float) -> None:
        """Record feedback from memory usage to update RL and confidence."""
        self._rl.update(features, action, reward)
        if reward > 0:
            self._confidence += reward
        elif reward < 0:
            self._confidence = max(0.0, self._confidence + reward * 10)  # amplified penalty

    def save(self, path: str | Path) -> None:
        """Save extraction state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "confidence": self._confidence,
            "decision_count": self._decision_count,
            "confidence_threshold": self._confidence_threshold,
        }
        path.write_text(json.dumps(data))

    def load(self, path: str | Path) -> None:
        """Load extraction state."""
        data = json.loads(Path(path).read_text())
        self._confidence = data.get("confidence", 0.0)
        self._decision_count = data.get("decision_count", 0)
        self._confidence_threshold = data.get("confidence_threshold", self._confidence_threshold)


# ── Batch Extraction (for sleep cycle) ───────────────────────────────


def extract_from_turns(
    turns: list[dict],
    store: GraphMemoryStore,
    extractor: ProgressiveExtraction,
    dedup_threshold: float = 0.90,
) -> ExtractionResult:
    """Process a batch of conversation turns and extract memories.

    Args:
        turns: List of turn dicts from ConversationLog.get_unprocessed_turns().
        store: GraphMemoryStore to save extracted memories and check duplicates.
        extractor: ProgressiveExtraction instance for evaluation.
        dedup_threshold: Similarity threshold for deduplication (>=threshold → skip).

    Returns:
        ExtractionResult with counts of processed/extracted/deduplicated turns.
    """
    result = ExtractionResult(extraction_mode=extractor.mode)

    # Group by conversation_id for context
    conversations: dict[str, list[dict]] = {}
    for turn in turns:
        conv_id = turn["conversation_id"]
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(turn)

    for conv_id, conv_turns in conversations.items():
        for turn in conv_turns:
            result.turns_processed += 1
            try:
                candidate = extractor.evaluate(turn["content"], turn["role"])

                if not candidate.should_extract:
                    continue

                # Dedup check: search for similar existing memories
                existing = store.search(
                    turn["content"],
                    top_k=1,
                    track_access=False,
                )
                if existing and existing[0].similarity_score is not None:
                    if existing[0].similarity_score >= dedup_threshold:
                        result.memories_deduplicated += 1
                        continue

                # Save to store
                content = turn["content"][:300].strip()
                store.add_memory(
                    content=content,
                    keywords=candidate.keywords,
                    category=candidate.category,
                    conversation_id=conv_id,
                    extraction_source=candidate.extraction_source,
                )
                result.memories_extracted += 1

            except Exception as exc:
                msg = f"Extraction error (conv={conv_id}, turn={turn.get('turn_index')}): {exc}"
                result.errors.append(msg)
                logger.warning(msg)

    return result

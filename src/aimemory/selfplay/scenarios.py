"""RAG-based scenario management with Korean-only enforcement.

Loads diverse conversation topics from data/prompts/topics.json.
Each episode is assigned a single topic for focused, natural Korean dialogue.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from aimemory.schemas import ScenarioType

logger = logging.getLogger(__name__)

# ─── Korean enforcement prefix (injected into EVERY system prompt) ───
KOREAN_ONLY_RULE = (
    "## 절대 규칙 — 한국어만 사용\n"
    "1. 반드시 한국어로만 말하세요.\n"
    "2. 영어 문장을 쓰지 마세요. 중국어 문장을 쓰지 마세요. 일본어 문장을 쓰지 마세요.\n"
    "3. 코드 블록(```)을 절대 작성하지 마세요.\n"
    "4. 프로그래밍 코드를 직접 쓰지 마세요.\n"
    "5. 기술 용어는 짧게만 쓰세요 (예: 'Python', 'API' 정도). 나머지는 한국어로 풀어서 설명하세요.\n"  # noqa: E501
    "6. 이 규칙을 어기면 대화가 실패합니다. 반드시 지켜주세요.\n\n"
)

# ─── Default prompts directory ───
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "prompts"


@dataclass
class Topic:
    """A single conversation topic loaded from RAG."""

    id: str
    category: str
    name: str
    system_hint: str
    seed_prompts: list[str] = field(default_factory=list)

    @property
    def scenario_type(self) -> ScenarioType:
        """Map category string to ScenarioType enum."""
        mapping = {
            "casual_chat": ScenarioType.CASUAL_CHAT,
            "technical_qa": ScenarioType.TECHNICAL_QA,
            "project_discussion": ScenarioType.PROJECT_DISCUSSION,
            "personal_preferences": ScenarioType.PERSONAL_PREFERENCES,
            "learning_tutoring": ScenarioType.LEARNING_TUTORING,
            "troubleshooting": ScenarioType.TROUBLESHOOTING,
        }
        return mapping.get(self.category, ScenarioType.CASUAL_CHAT)


class ScenarioManager:
    """RAG-based scenario manager that loads topics from JSON files."""

    def __init__(
        self,
        seed: int | None = None,
        prompts_dir: Path | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._prompts_dir = prompts_dir or _DEFAULT_PROMPTS_DIR
        self._topics: list[Topic] = []
        self._topics_by_category: dict[str, list[Topic]] = {}
        self._load_topics()

    def _load_topics(self) -> None:
        """Load all topic files from the prompts directory."""
        topics_file = self._prompts_dir / "topics.json"
        if topics_file.exists():
            try:
                data = json.loads(topics_file.read_text(encoding="utf-8"))
                for t in data.get("topics", []):
                    topic = Topic(
                        id=t["id"],
                        category=t["category"],
                        name=t["name"],
                        system_hint=t["system_hint"],
                        seed_prompts=t.get("seed_prompts", []),
                    )
                    self._topics.append(topic)
                    self._topics_by_category.setdefault(topic.category, []).append(topic)
                logger.info("Loaded %d topics from %s", len(self._topics), topics_file)
            except Exception as e:
                logger.warning("Failed to load topics from %s: %s", topics_file, e)

        if not self._topics:
            logger.warning("No RAG topics loaded, using fallback prompts")
            self._load_fallback()

    def _load_fallback(self) -> None:
        """Fallback: create minimal topics if RAG files are missing."""
        fallback = [
            Topic(
                id="fallback_casual",
                category="casual_chat",
                name="일상 대화",
                system_hint="가벼운 일상 대화를 합니다.",
                seed_prompts=[
                    "오늘 날씨가 좋네요. 요즘 어떻게 지내세요?",
                    "주말에 뭐 하셨어요?",
                    "좋아하는 음식이 뭐예요?",
                ],
            ),
            Topic(
                id="fallback_preference",
                category="personal_preferences",
                name="취향 공유",
                system_hint="자신의 취향과 선호를 이야기합니다.",
                seed_prompts=[
                    "커피랑 차 중에 뭘 더 좋아하세요?",
                    "좋아하는 영화 장르가 뭐예요?",
                ],
            ),
        ]
        for t in fallback:
            self._topics.append(t)
            self._topics_by_category.setdefault(t.category, []).append(t)

    @property
    def topic_count(self) -> int:
        return len(self._topics)

    @property
    def all_topics(self) -> list[Topic]:
        return list(self._topics)

    def random_topic(self) -> Topic:
        """Select a random topic with uniform probability."""
        return self._rng.choice(self._topics)

    def random_topic_by_category(self, category: str) -> Topic | None:
        """Select a random topic within a specific category."""
        topics = self._topics_by_category.get(category, [])
        if not topics:
            return None
        return self._rng.choice(topics)

    def get_seed_prompt(self, scenario_or_topic: ScenarioType | Topic | None = None) -> str:
        """Return a random seed prompt.

        Accepts a ScenarioType (for backward compat), a Topic, or None (random).
        """
        if isinstance(scenario_or_topic, Topic):
            return self._rng.choice(scenario_or_topic.seed_prompts)
        if isinstance(scenario_or_topic, ScenarioType):
            # Find topics matching this scenario type
            matching = [t for t in self._topics if t.scenario_type == scenario_or_topic]
            if matching:
                topic = self._rng.choice(matching)
                return self._rng.choice(topic.seed_prompts)
        # Fallback: random from all topics
        topic = self._rng.choice(self._topics)
        return self._rng.choice(topic.seed_prompts)

    def random_scenario(self) -> ScenarioType:
        """Select a random scenario type based on available topics."""
        topic = self._rng.choice(self._topics)
        return topic.scenario_type

    def round_robin_topics(self, count: int) -> list[Topic]:
        """Return `count` topics in round-robin order for even distribution."""
        if not self._topics:
            return []
        result: list[Topic] = []
        topics = list(self._topics)
        self._rng.shuffle(topics)
        for i in range(count):
            result.append(topics[i % len(topics)])
        return result

    def all_seed_prompts(self, scenario: ScenarioType) -> list[str]:
        """Return all seed prompts for a given scenario type."""
        prompts = []
        for t in self._topics:
            if t.scenario_type == scenario:
                prompts.extend(t.seed_prompts)
        return prompts

    @staticmethod
    def scenario_system_prompt(scenario: ScenarioType, topic: Topic | None = None) -> str:
        """Return a system prompt with Korean enforcement + scenario/topic hint."""
        base_hints: dict[ScenarioType, str] = {
            ScenarioType.CASUAL_CHAT: "가벼운 일상 대화를 합니다.",
            ScenarioType.TECHNICAL_QA: "기술 관련 경험을 한국어로만 이야기합니다. 코드를 쓰지 마세요.",  # noqa: E501
            ScenarioType.PROJECT_DISCUSSION: "프로젝트 관련 이야기를 한국어로만 합니다. 코드를 쓰지 마세요.",  # noqa: E501
            ScenarioType.PERSONAL_PREFERENCES: "자신의 취향과 선호를 이야기합니다.",
            ScenarioType.LEARNING_TUTORING: "배움과 학습 경험에 대해 한국어로만 이야기합니다.",  # noqa: E501
            ScenarioType.TROUBLESHOOTING: "문제 해결 경험을 한국어로만 이야기합니다. 코드를 쓰지 마세요.",  # noqa: E501
        }

        hint = base_hints.get(scenario, "자연스럽게 대화하세요.")
        if topic:
            hint = topic.system_hint

        return hint

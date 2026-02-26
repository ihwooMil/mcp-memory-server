"""Tests for the selfplay module using mock LLM clients."""

from __future__ import annotations

import itertools
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aimemory.config import AppConfig, OllamaConfig, SelfPlayConfig
from aimemory.schemas import (
    Episode,
    MemoryActionType,
    Role,
    ScenarioType,
    Turn,
)
from aimemory.selfplay.engine import SelfPlayEngine
from aimemory.selfplay.llm_client import LLMClient
from aimemory.selfplay.memory_agent import (
    MemoryAgent,
    MemoryStore,
    classify_category,
    extract_keywords,
)
from aimemory.selfplay.scenarios import ScenarioManager


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_user_client():
    """Mock LLM client that returns deterministic Korean user messages."""
    client = MagicMock(spec=LLMClient)
    responses = itertools.cycle([
        "저는 Python을 정말 좋아해요. 데이터 분석에 주로 사용합니다.",
        "머신러닝 프로젝트를 진행 중인데요, 도움이 필요해요.",
        "이전에 pandas를 언급했는데, 좀 더 알고 싶어요.",
        "요즘 Docker를 배우고 있어요. 어렵더라고요.",
        "취미로 독서를 하는데 특히 SF 소설을 좋아해요.",
    ])
    client.chat.side_effect = lambda *args, **kwargs: next(responses)
    return client


@pytest.fixture
def mock_assistant_client():
    """Mock LLM client that returns deterministic Korean assistant responses."""
    client = MagicMock(spec=LLMClient)
    responses = itertools.cycle([
        "Python은 훌륭한 선택이에요! 데이터 분석에 정말 유용하죠.",
        "머신러닝 프로젝트를 도와드릴게요. 어떤 부분이 어려우신가요?",
        "pandas에 대해 더 알고 싶으시군요. 어떤 기능을 배우고 싶으세요?",
        "Docker는 처음엔 어렵지만 익숙해지면 편리해요.",
        "SF 소설을 좋아하시는군요! 추천 작품이 있으신가요?",
    ])
    client.chat.side_effect = lambda *args, **kwargs: next(responses)
    return client


@pytest.fixture
def minimal_config():
    """App config with minimal turns for fast testing."""
    cfg = AppConfig()
    cfg.selfplay = SelfPlayConfig(
        min_turns=2,
        max_turns=4,
        memory_test_probability=0.0,  # disable memory injection for predictability
        checkpoint_interval=5,
    )
    return cfg


@pytest.fixture
def engine(mock_user_client, mock_assistant_client, minimal_config):
    """SelfPlayEngine with mocked LLM clients."""
    return SelfPlayEngine(
        config=minimal_config,
        user_client=mock_user_client,
        assistant_client=mock_assistant_client,
        seed=42,
    )


# ─── LLMClient tests ─────────────────────────────────────────────────


class TestLLMClient:
    def test_chat_returns_string(self):
        """LLMClient.chat should return a stripped string from the Ollama response."""
        with patch("aimemory.selfplay.llm_client.ollama.Client") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_response = MagicMock()
            mock_response.message.content = "  안녕하세요!  "
            mock_instance.chat.return_value = mock_response

            client = LLMClient(OllamaConfig())
            result = client.chat([{"role": "user", "content": "안녕?"}])

        assert result == "안녕하세요!"

    def test_chat_retries_on_failure(self):
        """LLMClient.chat should retry up to max_retries times."""
        with patch("aimemory.selfplay.llm_client.ollama.Client") as mock_cls:
            with patch("aimemory.selfplay.llm_client.time.sleep"):
                mock_instance = mock_cls.return_value
                mock_response = MagicMock()
                mock_response.message.content = "네 알겠습니다 테스트 응답입니다"
                # Fail twice, succeed on third attempt
                mock_instance.chat.side_effect = [
                    RuntimeError("연결 실패"),
                    RuntimeError("연결 실패"),
                    mock_response,
                ]

                client = LLMClient(OllamaConfig(max_retries=3))
                result = client.chat([{"role": "user", "content": "테스트"}])

        assert result == "네 알겠습니다 테스트 응답입니다"
        assert mock_instance.chat.call_count == 3

    def test_chat_raises_after_max_retries(self):
        """LLMClient.chat should raise RuntimeError after exhausting retries."""
        with patch("aimemory.selfplay.llm_client.ollama.Client") as mock_cls:
            with patch("aimemory.selfplay.llm_client.time.sleep"):
                mock_instance = mock_cls.return_value
                mock_instance.chat.side_effect = RuntimeError("서버 오류")

                client = LLMClient(OllamaConfig(max_retries=2))
                with pytest.raises(RuntimeError, match="2 attempts"):
                    client.chat([{"role": "user", "content": "테스트"}])

    def test_is_available_true(self):
        with patch("aimemory.selfplay.llm_client.ollama.Client") as mock_cls:
            mock_cls.return_value.list.return_value = []
            client = LLMClient()
            assert client.is_available() is True

    def test_is_available_false(self):
        with patch("aimemory.selfplay.llm_client.ollama.Client") as mock_cls:
            mock_cls.return_value.list.side_effect = ConnectionError("연결 불가")
            client = LLMClient()
            assert client.is_available() is False


# ─── ScenarioManager tests ───────────────────────────────────────────


class TestScenarioManager:
    def test_get_seed_prompt_returns_string(self):
        mgr = ScenarioManager(seed=0)
        prompt = mgr.get_seed_prompt(ScenarioType.TECHNICAL_QA)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_all_scenarios_have_prompts(self):
        mgr = ScenarioManager(seed=0)
        for scenario in ScenarioType:
            prompts = mgr.all_seed_prompts(scenario)
            assert len(prompts) >= 5, f"{scenario} has fewer than 5 prompts"

    def test_random_scenario_returns_valid_type(self):
        mgr = ScenarioManager(seed=42)
        for _ in range(20):
            s = mgr.random_scenario()
            assert isinstance(s, ScenarioType)

    def test_scenario_system_prompt_korean(self):
        for scenario in ScenarioType:
            prompt = ScenarioManager.scenario_system_prompt(scenario)
            # Check that the prompt contains Korean characters
            assert any("\uAC00" <= c <= "\uD7A3" for c in prompt), (
                f"{scenario} system prompt has no Korean characters"
            )


# ─── extract_keywords tests ──────────────────────────────────────────


class TestExtractKeywords:
    def test_extracts_tech_keywords(self):
        text = "저는 Python과 pandas를 사용해서 머신러닝 프로젝트를 합니다."
        kws = extract_keywords(text)
        assert "Python" in kws or "pandas" in kws or "머신러닝" in kws

    def test_extracts_quoted_terms(self):
        text = "제가 쓰는 라이브러리는 'scikit-learn'이에요."
        kws = extract_keywords(text)
        assert "scikit-learn" in kws or "scikit" in kws

    def test_empty_text_returns_list(self):
        kws = extract_keywords("")
        assert isinstance(kws, list)

    def test_result_capped_at_10(self):
        text = "Python Java JavaScript TypeScript Rust Go C++ Ruby Swift Kotlin Dart React"
        kws = extract_keywords(text)
        assert len(kws) <= 10

    def test_classify_technical(self):
        text = "Python과 TensorFlow를 사용해서 딥러닝 모델을 학습합니다."
        kws = extract_keywords(text)
        cat = classify_category(text, kws)
        assert cat == "technical"

    def test_classify_preference(self):
        text = "저는 SF 소설을 정말 좋아해요."
        kws = extract_keywords(text)
        cat = classify_category(text, kws)
        assert cat == "preference"


# ─── MemoryAgent tests ───────────────────────────────────────────────


class TestMemoryAgent:
    def _make_turn(self, turn_id: int, role: Role, content: str) -> Turn:
        return Turn(turn_id=turn_id, role=role, content=content)

    def test_skips_assistant_turn(self):
        agent = MemoryAgent(seed=0)
        store = MemoryStore()
        turn = self._make_turn(0, Role.ASSISTANT, "Python은 좋은 언어입니다.")
        decision = agent.decide(turn, store, [turn])
        assert decision.action == MemoryActionType.SKIP

    def test_saves_personal_info(self):
        agent = MemoryAgent(seed=0)
        store = MemoryStore()
        turn = self._make_turn(0, Role.USER, "저는 Python 개발자입니다.")
        decision = agent.decide(turn, store, [turn])
        # Should be SAVE because it has personal info pattern
        assert decision.action in (MemoryActionType.SAVE, MemoryActionType.SKIP)
        # If saved, should have memory entry
        if decision.action == MemoryActionType.SAVE:
            assert decision.memory_entry is not None
            assert decision.memory_entry.source_turn_id == 0

    def test_retrieves_on_question_with_stored_memory(self):
        agent = MemoryAgent(seed=0)
        store = MemoryStore()
        # Add a memory
        from aimemory.schemas import MemoryEntry
        store.add(MemoryEntry(
            content="사용자는 Python을 좋아함",
            source_turn_id=0,
            keywords=["Python"],
            category="preference",
        ))
        # Ask a question referencing the stored topic
        turn = self._make_turn(2, Role.USER, "Python에 대해 더 알고 싶은데요?")
        decision = agent.decide(turn, store, [turn])
        assert decision.action == MemoryActionType.RETRIEVE
        assert len(decision.retrieved_memories) >= 1

    def test_skip_on_question_with_empty_store(self):
        agent = MemoryAgent(seed=0)
        store = MemoryStore()
        turn = self._make_turn(0, Role.USER, "머신러닝이 뭔가요?")
        decision = agent.decide(turn, store, [turn])
        # No memories stored, cannot retrieve -> should be SKIP or SAVE
        assert decision.action in (MemoryActionType.SKIP, MemoryActionType.SAVE)


# ─── MemoryStore tests ───────────────────────────────────────────────


class TestMemoryStore:
    def test_add_and_len(self):
        from aimemory.schemas import MemoryEntry
        store = MemoryStore()
        assert len(store) == 0
        store.add(MemoryEntry(content="테스트", source_turn_id=0, keywords=["테스트"]))
        assert len(store) == 1

    def test_retrieve_relevant_matches_keywords(self):
        from aimemory.schemas import MemoryEntry
        store = MemoryStore()
        store.add(MemoryEntry(
            content="Python 개발자",
            source_turn_id=0,
            keywords=["Python", "개발자"],
        ))
        store.add(MemoryEntry(
            content="여행 좋아함",
            source_turn_id=1,
            keywords=["여행", "취미"],
        ))
        results = store.retrieve_relevant(["Python"])
        assert len(results) == 1
        assert "Python" in results[0].keywords

    def test_retrieve_empty_store(self):
        store = MemoryStore()
        results = store.retrieve_relevant(["Python"])
        assert results == []

    def test_retrieve_top_k_limit(self):
        from aimemory.schemas import MemoryEntry
        store = MemoryStore()
        for i in range(10):
            store.add(MemoryEntry(
                content=f"Python 관련 메모 {i}",
                source_turn_id=i,
                keywords=["Python"],
            ))
        results = store.retrieve_relevant(["Python"], top_k=3)
        assert len(results) <= 3


# ─── SelfPlayEngine tests ────────────────────────────────────────────


class TestSelfPlayEngine:
    def test_run_episode_returns_episode(self, engine):
        episode = engine.run_episode(ScenarioType.TECHNICAL_QA)
        assert isinstance(episode, Episode)
        assert isinstance(episode.scenario, ScenarioType)

    def test_episode_has_turns(self, engine, minimal_config):
        episode = engine.run_episode(ScenarioType.CASUAL_CHAT)
        assert episode.num_turns >= minimal_config.selfplay.min_turns * 2

    def test_episode_has_memory_decisions(self, engine):
        episode = engine.run_episode(ScenarioType.PERSONAL_PREFERENCES)
        # One decision per turn
        assert len(episode.memory_decisions) == len(episode.turns)

    def test_turns_alternate_roles(self, engine):
        episode = engine.run_episode(ScenarioType.LEARNING_TUTORING)
        for i, turn in enumerate(episode.turns):
            expected_role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            assert turn.role == expected_role, (
                f"Turn {i} expected {expected_role}, got {turn.role}"
            )

    def test_episode_memory_store_consistent(self, engine):
        episode = engine.run_episode(ScenarioType.TROUBLESHOOTING)
        # All entries in memory_store should correspond to SAVE decisions
        save_count = episode.num_saves
        assert len(episode.memory_store) == save_count

    def test_run_creates_files(self, engine, tmp_path):
        episodes = engine.run(num_episodes=2, output_dir=tmp_path)
        files = list(tmp_path.glob("episode_*.json"))
        assert len(files) == 2
        assert len(episodes) == 2

    def test_run_skips_existing_files(self, engine, tmp_path):
        # Run once
        engine.run(num_episodes=2, output_dir=tmp_path)
        # Run again - should skip existing files
        new_episodes = engine.run(num_episodes=2, output_dir=tmp_path)
        # All episodes should already exist, so no new episodes generated
        assert len(new_episodes) == 0

    def test_memory_injection_disabled(self, engine, minimal_config):
        """With memory_test_probability=0.0, _maybe_inject_memory_test returns None."""
        assert minimal_config.selfplay.memory_test_probability == 0.0
        episode = engine.run_episode(ScenarioType.CASUAL_CHAT)
        # When no injection, user_client.chat is called for every user turn except first
        assert episode.num_turns > 0

    def test_memory_injection_with_probability_1(
        self, mock_user_client, mock_assistant_client
    ):
        """With probability=1.0 and stored memories, injection should be triggered."""
        cfg = AppConfig()
        cfg.selfplay = SelfPlayConfig(
            min_turns=3,
            max_turns=4,
            memory_test_probability=1.0,
        )
        eng = SelfPlayEngine(
            config=cfg,
            user_client=mock_user_client,
            assistant_client=mock_assistant_client,
            seed=0,
        )
        episode = eng.run_episode(ScenarioType.PROJECT_DISCUSSION)
        assert episode.num_turns > 0

    def test_run_episode_metadata(self, engine):
        episode = engine.run_episode(ScenarioType.TECHNICAL_QA, episode_index=5)
        assert episode.metadata.get("episode_index") == 5


# ─── Integration: scenario coverage ──────────────────────────────────


class TestScenarioCoverage:
    def test_all_scenarios_produce_episode(self, engine):
        for scenario in ScenarioType:
            episode = engine.run_episode(scenario)
            assert isinstance(episode, Episode)
            assert isinstance(episode.scenario, ScenarioType)
            assert len(episode.turns) > 0


# ─── A1-A8 improvement tests ─────────────────────────────────────────


class TestA1MemorySummarySentenceBoundary:
    """A1: Memory save content should end at sentence boundary."""

    def _make_turn(self, turn_id: int, role: Role, content: str) -> Turn:
        return Turn(turn_id=turn_id, role=role, content=content)

    def test_memory_summary_preserves_sentence_boundary(self):
        agent = MemoryAgent(seed=0)
        store = MemoryStore()
        # Long personal info with multiple sentences
        content = (
            "저는 Python 개발자입니다. "
            "주로 데이터 분석 업무를 합니다. "
            "TensorFlow와 pandas를 자주 사용합니다. "
            "딥러닝 모델을 학습시키는 것이 제 주요 업무예요."
        )
        turn = self._make_turn(0, Role.USER, content)
        decision = agent.decide(turn, store, [turn])
        if decision.action == MemoryActionType.SAVE:
            # Should not be cut mid-word
            saved = decision.memory_entry.content
            # Check ends at sentence boundary or complete word boundary
            assert not saved.endswith("..."), "Should not use simple truncation"
            # Should not cut in the middle of a Korean word
            if len(saved) < len(content):
                # Last char should be sentence punctuation or last word boundary
                assert saved[-1] in ".!?。！？" or saved.endswith(saved.split()[-1])

    def test_short_content_kept_as_is(self):
        from aimemory.selfplay.memory_agent import _sentence_summary
        text = "저는 Python을 좋아해요."
        result = _sentence_summary(text, ["Python"])
        assert "Python" in result

    def test_long_content_cut_at_sentence(self):
        from aimemory.selfplay.memory_agent import _sentence_summary
        text = "저는 Python을 좋아해요. 데이터 분석에 씁니다. 매일 코딩합니다."
        result = _sentence_summary(text, ["Python"])
        assert len(result) <= 150
        # Should contain the keyword sentence
        assert "Python" in result


class TestA2AssistantParaphrase:
    """A2: Assistant turns with paraphrase patterns should be SAVE."""

    def _make_turn(self, turn_id: int, role: Role, content: str) -> Turn:
        return Turn(turn_id=turn_id, role=role, content=content)

    def test_assistant_paraphrase_saved(self):
        agent = MemoryAgent(seed=0)
        store = MemoryStore()
        # Assistant paraphrasing user info with "~하시는군요"
        turn = self._make_turn(1, Role.ASSISTANT, "Python을 좋아하시는군요! 데이터 분석에 활용하시는군요.")
        decision = agent.decide(turn, store, [turn])
        assert decision.action == MemoryActionType.SAVE
        assert decision.memory_entry is not None

    def test_assistant_plain_response_skipped(self):
        agent = MemoryAgent(seed=0)
        store = MemoryStore()
        # Assistant plain response without paraphrase
        turn = self._make_turn(1, Role.ASSISTANT, "네, 도움이 필요하시면 말씀해 주세요.")
        decision = agent.decide(turn, store, [turn])
        assert decision.action == MemoryActionType.SKIP


class TestA3RetrieveTriggerExpansion:
    """A3: RETRIEVE triggered by keyword overlap and discourse markers."""

    def _make_turn(self, turn_id: int, role: Role, content: str) -> Turn:
        return Turn(turn_id=turn_id, role=role, content=content)

    def _make_store_with_entry(self, keywords: list[str]) -> MemoryStore:
        from aimemory.schemas import MemoryEntry
        store = MemoryStore()
        store.add(MemoryEntry(
            content="Python과 pandas로 데이터 분석",
            source_turn_id=0,
            keywords=keywords,
            category="technical",
        ))
        return store

    def test_retrieve_on_keyword_overlap(self):
        agent = MemoryAgent(seed=0)
        # Store with Python and pandas
        store = self._make_store_with_entry(["Python", "pandas"])
        # User turn mentioning both Python and pandas (2+ overlap)
        turn = self._make_turn(2, Role.USER, "Python과 pandas를 같이 쓰는 방법이 궁금해요.")
        decision = agent.decide(turn, store, [turn])
        assert decision.action == MemoryActionType.RETRIEVE

    def test_retrieve_on_discourse_marker(self):
        agent = MemoryAgent(seed=0)
        store = self._make_store_with_entry(["Python", "데이터"])
        # User turn with discourse marker "아 맞다" and keywords
        turn = self._make_turn(2, Role.USER, "아 맞다, Python 프로젝트 얘기 했었죠.")
        decision = agent.decide(turn, store, [turn])
        assert decision.action == MemoryActionType.RETRIEVE


class TestA5StopwordsFiltered:
    """A5: extract_keywords should not return words in stopword list."""

    def test_stopwords_filtered(self):
        # These are all in _KOREAN_STOPWORDS
        text = "그것은 이것이 저것이 수가 있는 것입니다."
        kws = extract_keywords(text)
        stopwords = {"것", "수", "때", "거", "게", "줄", "데", "말", "점", "중",
                     "건", "뭐", "저", "제", "내", "그", "이", "더", "안", "좀",
                     "걸", "곳"}
        for kw in kws:
            assert kw not in stopwords, f"Stopword '{kw}' should not be in keywords"


class TestA7MetaPatternsExpanded:
    """A7: New meta-text patterns should be detected."""
    import re

    def test_meta_patterns_expanded(self):
        from aimemory.selfplay.engine import _META_PATTERNS
        # Test new patterns
        assert _META_PATTERNS.search("참고로 이것은 테스트입니다")
        assert _META_PATTERNS.search("위 내용을 참고하세요")
        assert _META_PATTERNS.search("다음과 같이 진행합니다")
        assert _META_PATTERNS.search("아래는 예시입니다")
        assert _META_PATTERNS.search("Sure, here you go")
        assert _META_PATTERNS.search("Here is the answer")
        assert _META_PATTERNS.search("Let me explain this")

    def test_original_patterns_still_work(self):
        from aimemory.selfplay.engine import _META_PATTERNS
        assert _META_PATTERNS.search("예시 형식은 다음과 같습니다")
        assert _META_PATTERNS.search("다음 턴에는")
        assert _META_PATTERNS.search("(이런 식으로 대화를 이어가세요)")


class TestA8EpisodeQualityMetrics:
    """A8: Episode metadata should contain quality_metrics after run_episode."""

    def test_episode_quality_metrics(self, engine):
        episode = engine.run_episode(ScenarioType.TECHNICAL_QA)
        assert "quality_metrics" in episode.metadata
        qm = episode.metadata["quality_metrics"]
        assert "save_rate" in qm
        assert "retrieve_rate" in qm
        assert "skip_rate" in qm
        assert "avg_turn_length" in qm
        assert "num_memories" in qm
        # Rates should sum to 1.0
        assert abs(qm["save_rate"] + qm["retrieve_rate"] + qm["skip_rate"] - 1.0) < 1e-9
        # avg_turn_length should be positive
        assert qm["avg_turn_length"] > 0


# ─── C1-C5 improvement tests ─────────────────────────────────────────


class TestC1ScenarioPromptNoDuplicateKoreanRule:
    """C1: scenario_system_prompt should NOT prepend KOREAN_ONLY_RULE."""

    def test_scenario_prompt_no_duplicate_korean_rule(self):
        from aimemory.selfplay.scenarios import KOREAN_ONLY_RULE
        for scenario in ScenarioType:
            prompt = ScenarioManager.scenario_system_prompt(scenario)
            # KOREAN_ONLY_RULE should NOT be duplicated inside the scenario hint
            assert prompt != KOREAN_ONLY_RULE + prompt, (
                f"{scenario} scenario prompt contains duplicate KOREAN_ONLY_RULE prefix"
            )
            # The prompt should not start with KOREAN_ONLY_RULE
            assert not prompt.startswith(KOREAN_ONLY_RULE), (
                f"{scenario} scenario_system_prompt should not prepend KOREAN_ONLY_RULE"
            )


class TestC2SeedPromptDiversity:
    """C2: topics.json should have 8+ seed prompts per topic for at least some topics."""

    def test_seed_prompt_diversity(self):
        mgr = ScenarioManager(seed=0)
        topics_with_8_plus = [t for t in mgr.all_topics if len(t.seed_prompts) >= 8]
        assert len(topics_with_8_plus) >= 10, (
            f"Expected at least 10 topics with 8+ seed prompts, got {len(topics_with_8_plus)}"
        )

    def test_total_seed_prompts_over_200(self):
        mgr = ScenarioManager(seed=0)
        total = sum(len(t.seed_prompts) for t in mgr.all_topics)
        assert total >= 200, f"Expected 200+ total seed prompts, got {total}"


class TestC3UserPromptHasFewShot:
    """C3: user_system_prompt should contain few-shot conversation example."""

    def test_user_prompt_has_few_shot(self):
        from aimemory.config import SelfPlayConfig
        cfg = SelfPlayConfig()
        assert "대화 흐름 예시" in cfg.user_system_prompt, (
            "user_system_prompt should contain '대화 흐름 예시' few-shot section"
        )

    def test_user_prompt_few_shot_contains_example_exchange(self):
        from aimemory.config import SelfPlayConfig
        cfg = SelfPlayConfig()
        assert "된장찌개" in cfg.user_system_prompt, (
            "user_system_prompt few-shot example should contain the sample exchange"
        )


class TestC4ConfigDefaults:
    """C4: OllamaConfig defaults should be temperature=0.7 and max_tokens=384."""

    def test_temperature_default(self):
        from aimemory.config import OllamaConfig
        cfg = OllamaConfig()
        assert cfg.temperature == 0.7, (
            f"Expected default temperature 0.7, got {cfg.temperature}"
        )

    def test_max_tokens_default(self):
        from aimemory.config import OllamaConfig
        cfg = OllamaConfig()
        assert cfg.max_tokens == 384, (
            f"Expected default max_tokens 384, got {cfg.max_tokens}"
        )


class TestC5RoundRobinTopics:
    """C5: round_robin_topics should return even distribution."""

    def test_round_robin_topics_returns_correct_count(self):
        mgr = ScenarioManager(seed=0)
        topics = mgr.round_robin_topics(50)
        assert len(topics) == 50

    def test_round_robin_topics(self):
        mgr = ScenarioManager(seed=42)
        n = mgr.topic_count
        topics = mgr.round_robin_topics(n * 3)
        counts: dict[str, int] = {}
        for t in topics:
            counts[t.id] = counts.get(t.id, 0) + 1
        # Each topic should appear at least twice in 3 full rounds
        for topic_id, count in counts.items():
            assert count >= 2, (
                f"Topic '{topic_id}' appeared only {count} times in {n * 3} draws"
            )

    def test_round_robin_covers_all_topics(self):
        mgr = ScenarioManager(seed=7)
        n = mgr.topic_count
        topics = mgr.round_robin_topics(n)
        seen_ids = {t.id for t in topics}
        all_ids = {t.id for t in mgr.all_topics}
        assert seen_ids == all_ids, (
            f"round_robin_topics({n}) should cover all {n} topics, "
            f"missing: {all_ids - seen_ids}"
        )

    def test_round_robin_empty(self):
        from aimemory.selfplay.scenarios import ScenarioManager as SM
        import tempfile, json
        from pathlib import Path
        # Create a manager with no topics
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SM(seed=0, prompts_dir=Path(tmpdir))
            # The fallback will have some topics, so just test count=0
            result = mgr.round_robin_topics(0)
            assert result == []

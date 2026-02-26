"""Shared test fixtures for the AI Memory System."""

from __future__ import annotations

from datetime import datetime

import pytest

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
    State,
    Turn,
)


@pytest.fixture
def sample_turns() -> list[Turn]:
    return [
        Turn(turn_id=0, role=Role.USER, content="안녕하세요! 저는 Python을 좋아해요.", token_count=15),
        Turn(turn_id=1, role=Role.ASSISTANT, content="안녕하세요! Python을 좋아하시는군요. 어떤 분야에서 사용하시나요?", token_count=20),
        Turn(turn_id=2, role=Role.USER, content="주로 데이터 분석이랑 머신러닝에 사용해요. 특히 pandas를 많이 써요.", token_count=22),
        Turn(turn_id=3, role=Role.ASSISTANT, content="pandas는 정말 유용한 라이브러리죠! 어떤 데이터를 다루시나요?", token_count=18),
    ]


@pytest.fixture
def sample_memory_entry() -> MemoryEntry:
    return MemoryEntry(
        content="사용자는 Python을 좋아하며, 데이터 분석과 머신러닝에 사용함",
        source_turn_id=0,
        keywords=["Python", "데이터 분석", "머신러닝", "pandas"],
        category="preference",
    )


@pytest.fixture
def sample_episode(sample_turns, sample_memory_entry) -> Episode:
    decisions = [
        MemoryDecision(
            turn_id=0,
            action=MemoryActionType.SAVE,
            memory_entry=sample_memory_entry,
            reasoning="사용자의 기술 선호도 정보",
        ),
        MemoryDecision(
            turn_id=1,
            action=MemoryActionType.SKIP,
            reasoning="어시스턴트 응답, 저장할 새 정보 없음",
        ),
        MemoryDecision(
            turn_id=2,
            action=MemoryActionType.SAVE,
            memory_entry=MemoryEntry(
                content="사용자는 pandas를 많이 사용하며, 데이터 분석과 머신러닝이 주 분야",
                source_turn_id=2,
                keywords=["pandas", "데이터 분석", "머신러닝"],
                category="technical",
            ),
            reasoning="구체적 기술 스택 정보",
        ),
        MemoryDecision(
            turn_id=3,
            action=MemoryActionType.SKIP,
            reasoning="어시스턴트 질문, 저장 불필요",
        ),
    ]
    return Episode(
        scenario=ScenarioType.TECHNICAL_QA,
        turns=sample_turns,
        memory_decisions=decisions,
        memory_store=[sample_memory_entry],
    )


@pytest.fixture
def sample_state(sample_turns) -> State:
    return State(
        episode_id="test_ep_001",
        turn_id=2,
        recent_turns=sample_turns[:3],
        current_memory_summary=["사용자는 Python을 좋아함"],
        memory_count=1,
        turn_position=0.5,
    )


@pytest.fixture
def sample_action() -> Action:
    return Action(
        action_type=MemoryActionType.SAVE,
        saved_content="사용자는 pandas를 많이 사용함",
        saved_keywords=["pandas", "데이터 분석"],
    )


@pytest.fixture
def sample_reward() -> RewardBreakdown:
    return RewardBreakdown(
        r1_keyword_reappearance=0.5,
        r3_efficiency=0.3,
        r6_self_reference=0.4,
        r7_info_density=0.6,
    )

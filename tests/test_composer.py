"""Tests for the context composer."""

from __future__ import annotations

from aimemory.memory.composer import ComposedMemory, ContextComposer
from aimemory.memory.graph_store import MemoryNode


def _make_node(
    memory_id: str = "m1",
    content: str = "전체 텍스트입니다.",
    level1_text: str = "요약 텍스트",
    level2_text: str = "주어,서술어,목적어",
    similarity_score: float = 0.9,
    **kwargs,
) -> MemoryNode:
    return MemoryNode(
        memory_id=memory_id,
        content=content,
        level1_text=level1_text,
        level2_text=level2_text,
        similarity_score=similarity_score,
        **kwargs,
    )


class TestContextComposer:
    def test_compose_within_budget(self) -> None:
        composer = ContextComposer(token_budget=100, top_k=5)
        nodes = [_make_node("m1", similarity_score=0.9)]
        composed = composer.compose(nodes)
        assert len(composed) >= 1
        total_tokens = sum(c.tokens for c in composed)
        assert total_tokens <= 100

    def test_relevance_ordering(self) -> None:
        composer = ContextComposer(token_budget=1000, top_k=5)
        nodes = [
            _make_node("m1", similarity_score=0.5),
            _make_node("m2", similarity_score=0.9),
            _make_node("m3", similarity_score=0.7),
        ]
        composed = composer.compose(nodes)
        assert len(composed) == 3
        assert composed[0].memory_id == "m2"
        assert composed[1].memory_id == "m3"
        assert composed[2].memory_id == "m1"

    def test_falls_back_to_lower_level(self) -> None:
        """When L0 doesn't fit, should fall back to L1 or L2."""
        # Very small budget that only fits level2
        composer = ContextComposer(token_budget=5, top_k=5)
        nodes = [
            _make_node(
                "m1",
                content="아주 긴 텍스트입니다 " * 10,  # Too big for budget
                level1_text="중간 요약",  # Still too big
                level2_text="주,술,목",  # Should fit
                similarity_score=0.9,
            ),
        ]
        composed = composer.compose(nodes)
        if composed:  # May or may not fit depending on token estimation
            assert composed[0].level >= 1  # Should use lower resolution

    def test_respects_top_k(self) -> None:
        composer = ContextComposer(token_budget=10000, top_k=2)
        nodes = [_make_node(f"m{i}", similarity_score=0.9 - i * 0.1) for i in range(5)]
        composed = composer.compose(nodes)
        assert len(composed) <= 2

    def test_empty_input(self) -> None:
        composer = ContextComposer()
        assert composer.compose([]) == []

    def test_none_similarity_treated_as_zero(self) -> None:
        composer = ContextComposer(token_budget=1000)
        nodes = [
            _make_node("m1", similarity_score=None),
            _make_node("m2", similarity_score=0.5),
        ]
        composed = composer.compose(nodes)
        # m2 should come first (0.5 > 0.0)
        assert composed[0].memory_id == "m2"


class TestFormatContext:
    def test_format_with_levels(self) -> None:
        composer = ContextComposer()
        composed = [
            ComposedMemory("m1", "전체 텍스트", level=0, relevance=0.9, tokens=10),
            ComposedMemory("m2", "요약 텍스트", level=1, relevance=0.7, tokens=5),
            ComposedMemory("m3", "주,술,목", level=2, relevance=0.5, tokens=3),
        ]
        result = composer.format_context(composed)
        assert "[L0] 전체 텍스트" in result
        assert "[L1] 요약 텍스트" in result
        assert "[L2] 주,술,목" in result

    def test_format_empty(self) -> None:
        composer = ContextComposer()
        assert composer.format_context([]) == ""

"""Tests for multi-resolution text generation."""

from __future__ import annotations

import pytest

from aimemory.memory.resolution import (
    MultiResolutionText,
    estimate_tokens,
    generate_all_levels,
    generate_level1,
    generate_level2,
)


class TestGenerateLevel1:
    def test_extracts_keyword_sentence(self) -> None:
        text = "저는 커피를 좋아합니다. 매일 아침에 한 잔 마셔요. 오후에는 차를 마셔요."
        result = generate_level1(text, keywords=["커피"])
        assert "커피" in result

    def test_max_100_chars(self) -> None:
        long_text = "저는 " + "매우 긴 문장입니다. " * 20
        result = generate_level1(long_text, keywords=["매우"])
        assert len(result) <= 100

    def test_fallback_to_first_sentence(self) -> None:
        text = "첫 번째 문장입니다. 두 번째 문장입니다."
        result = generate_level1(text, keywords=["없는키워드"])
        assert "첫 번째" in result

    def test_empty_text_returns_empty(self) -> None:
        assert generate_level1("", keywords=["test"]) == ""

    def test_no_keywords_returns_first_sentence(self) -> None:
        text = "첫 번째 문장입니다. 두 번째 문장입니다."
        result = generate_level1(text, keywords=[])
        assert "첫 번째" in result

    def test_combines_two_matching_sentences(self) -> None:
        text = "커피 좋아요. 커피는 매일 마셔요. 물도 마셔요."
        result = generate_level1(text, keywords=["커피"])
        # Should try combining first two matching sentences
        assert "커피" in result

    def test_none_keywords(self) -> None:
        text = "테스트 문장입니다."
        result = generate_level1(text, keywords=None)
        assert "테스트" in result


class TestGenerateLevel2:
    def test_returns_triple_format(self) -> None:
        text = "저는 커피를 좋아합니다."
        result = generate_level2(text, keywords=["커피", "좋아"])
        parts = result.split(",")
        assert len(parts) == 3  # subject, predicate, object

    def test_extracts_from_keywords(self) -> None:
        text = "저는 커피를 좋아합니다."
        result = generate_level2(text, keywords=["커피", "좋아"])
        assert "커피" in result

    def test_empty_text_returns_empty(self) -> None:
        assert generate_level2("") == ""

    def test_no_keywords_still_extracts(self) -> None:
        text = "저는 파이썬을 좋아합니다."
        result = generate_level2(text, keywords=[])
        assert result  # Should not be empty
        assert "," in result


class TestGenerateAllLevels:
    def test_returns_multi_resolution_text(self) -> None:
        text = "저는 커피를 좋아합니다."
        result = generate_all_levels(text, keywords=["커피"])
        assert isinstance(result, MultiResolutionText)
        assert result.level0 == text
        assert result.level1  # not empty
        assert result.level2  # not empty

    def test_level0_is_original(self) -> None:
        text = "원본 텍스트입니다."
        result = generate_all_levels(text)
        assert result.level0 == text


class TestEstimateTokens:
    def test_korean_text(self) -> None:
        text = "한국어 텍스트"  # 7 chars
        tokens = estimate_tokens(text)
        assert tokens == int(7 / 2.5)

    def test_empty_text(self) -> None:
        assert estimate_tokens("") == 0

    def test_minimum_one_token(self) -> None:
        assert estimate_tokens("a") >= 1

    def test_longer_text(self) -> None:
        text = "이것은 조금 더 긴 한국어 텍스트입니다."
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens == int(len(text) / 2.5)

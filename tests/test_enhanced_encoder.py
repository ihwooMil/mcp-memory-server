"""Tests for EnhancedStateEncoder."""
from __future__ import annotations

import numpy as np
import pytest

from aimemory.online.enhanced_encoder import EnhancedStateEncoder
from aimemory.online.policy import StateEncoder
from aimemory.schemas import MemoryActionType, Role, Turn

EMB_DIM = 384
TOTAL_DIM = EMB_DIM + 10  # 394


def make_turn(content: str = "안녕하세요", turn_id: int = 1) -> Turn:
    return Turn(turn_id=turn_id, role=Role.USER, content=content)


def mock_embedding_fn(texts: list[str]) -> list[list[float]]:
    """Mock embedding function returning known vectors."""
    return [[float(i % 10) * 0.1] * EMB_DIM for i, _ in enumerate(texts)]


class TestEnhancedStateEncoderOutputDimension:
    def test_output_is_394d(self):
        encoder = EnhancedStateEncoder()
        turn = make_turn()
        result = encoder.encode(turn=turn, recent_turns=[], memory_count=0)
        assert result.shape == (TOTAL_DIM,)

    def test_output_is_394d_with_embedding_fn(self):
        encoder = EnhancedStateEncoder()
        encoder.set_embedding_fn(mock_embedding_fn)
        turn = make_turn()
        result = encoder.encode(turn=turn, recent_turns=[], memory_count=0)
        assert result.shape == (TOTAL_DIM,)


class TestEnhancedStateEncoderBaseFeatures:
    def test_last_10_match_base_encoder(self):
        encoder = EnhancedStateEncoder()
        base_encoder = StateEncoder()
        turn = make_turn("저는 Python을 좋아해요")
        recent = [make_turn("이전 턴", turn_id=0)]
        actions = [MemoryActionType.SAVE]

        enhanced = encoder.encode(
            turn=turn,
            recent_turns=recent,
            memory_count=3,
            recent_actions=actions,
            turn_position=0.5,
        )
        base = base_encoder.encode(
            turn=turn,
            recent_turns=recent,
            memory_count=3,
            recent_actions=actions,
            turn_position=0.5,
        )

        np.testing.assert_array_equal(enhanced[-10:], base)

    def test_base_features_preserved_with_embedding_fn(self):
        encoder = EnhancedStateEncoder()
        encoder.set_embedding_fn(mock_embedding_fn)
        base_encoder = StateEncoder()
        turn = make_turn("제 이름은 김철수입니다")

        enhanced = encoder.encode(turn=turn, recent_turns=[], memory_count=5)
        base = base_encoder.encode(turn=turn, recent_turns=[], memory_count=5)

        np.testing.assert_array_equal(enhanced[-10:], base)


class TestEnhancedStateEncoderEmbeddingInjection:
    def test_embedding_fn_injection_works(self):
        encoder = EnhancedStateEncoder()

        def constant_fn(texts):
            return [[0.5] * EMB_DIM for _ in texts]

        encoder.set_embedding_fn(constant_fn)
        turn = make_turn("test content")
        result = encoder.encode(turn=turn, recent_turns=[])

        np.testing.assert_allclose(result[:EMB_DIM], np.full(EMB_DIM, 0.5, dtype=np.float32))

    def test_embedding_fn_result_in_first_dims(self):
        encoder = EnhancedStateEncoder()
        encoder.set_embedding_fn(mock_embedding_fn)
        turn = make_turn("hello world")
        result = encoder.encode(turn=turn, recent_turns=[])

        expected_embedding = np.full(EMB_DIM, 0.0, dtype=np.float32)
        np.testing.assert_allclose(result[:EMB_DIM], expected_embedding)


class TestEnhancedStateEncoderWithoutEmbeddingFn:
    def test_first_dims_are_zeros_without_fn(self):
        encoder = EnhancedStateEncoder()
        turn = make_turn("안녕하세요")
        result = encoder.encode(turn=turn, recent_turns=[])

        np.testing.assert_array_equal(result[:EMB_DIM], np.zeros(EMB_DIM, dtype=np.float32))

    def test_zeros_even_with_nonempty_content(self):
        encoder = EnhancedStateEncoder()
        turn = make_turn("저는 Python과 React를 좋아합니다. 취미가 있어요.")
        result = encoder.encode(turn=turn, recent_turns=[], memory_count=10, turn_position=0.8)

        np.testing.assert_array_equal(result[:EMB_DIM], np.zeros(EMB_DIM, dtype=np.float32))


class TestEnhancedStateEncoderEdgeCases:
    def test_empty_content(self):
        encoder = EnhancedStateEncoder()
        turn = make_turn(content="")
        result = encoder.encode(turn=turn, recent_turns=[])

        assert result.shape == (TOTAL_DIM,)
        assert result.dtype == np.float32

    def test_minimal_input(self):
        encoder = EnhancedStateEncoder()
        turn = make_turn()
        result = encoder.encode(turn=turn, recent_turns=[])

        assert result.shape == (TOTAL_DIM,)

    def test_dtype_is_float32(self):
        encoder = EnhancedStateEncoder()
        encoder.set_embedding_fn(mock_embedding_fn)
        turn = make_turn("test")
        result = encoder.encode(turn=turn, recent_turns=[])

        assert result.dtype == np.float32

    def test_dtype_is_float32_without_fn(self):
        encoder = EnhancedStateEncoder()
        turn = make_turn("test")
        result = encoder.encode(turn=turn, recent_turns=[])

        assert result.dtype == np.float32

    def test_with_multiple_recent_turns_and_actions(self):
        encoder = EnhancedStateEncoder()
        turn = make_turn("현재 턴", turn_id=5)
        recent = [make_turn(f"턴 {i}", turn_id=i) for i in range(5)]
        actions = [MemoryActionType.SAVE, MemoryActionType.SKIP, MemoryActionType.RETRIEVE]

        result = encoder.encode(
            turn=turn,
            recent_turns=recent,
            memory_count=7,
            recent_actions=actions,
            turn_position=0.9,
        )

        assert result.shape == (TOTAL_DIM,)
        assert result.dtype == np.float32

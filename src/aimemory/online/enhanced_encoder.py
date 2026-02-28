from __future__ import annotations
import numpy as np
from typing import Callable
from aimemory.online.policy import StateEncoder
from aimemory.schemas import MemoryActionType, Turn

class EnhancedStateEncoder:
    """384d embedding + 10d hand-crafted → 394d state."""

    def __init__(self, st_model: str = "intfloat/multilingual-e5-small", emb_dim: int = 384, lang: str = "ko"):
        self._base_encoder = StateEncoder(lang=lang)
        self._embedding_fn: Callable | None = None
        self._st_model = st_model
        self._emb_dim = emb_dim

    def set_embedding_fn(self, fn: Callable) -> None:
        """Inject external embedding function to avoid duplicate model loading.
        The fn should accept a list of strings and return a list of embeddings."""
        self._embedding_fn = fn

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text. Uses injected fn or falls back to zeros."""
        if self._embedding_fn is not None:
            result = self._embedding_fn([text])
            # Handle various return types (list of lists, numpy arrays, etc.)
            if hasattr(result, '__iter__'):
                emb = result[0] if len(result) > 0 else np.zeros(self._emb_dim)
                return np.array(emb, dtype=np.float32)
        return np.zeros(self._emb_dim, dtype=np.float32)

    def encode(
        self,
        turn: Turn,
        recent_turns: list[Turn],
        memory_count: int = 0,
        recent_actions: list[MemoryActionType] | None = None,
        turn_position: float = 0.0,
    ) -> np.ndarray:
        """Encode to 394d: 384d embedding + 10d hand-crafted features."""
        # Get base 10d features
        base_features = self._base_encoder.encode(
            turn=turn,
            recent_turns=recent_turns,
            memory_count=memory_count,
            recent_actions=recent_actions,
            turn_position=turn_position,
        )

        # Get embedding
        embedding = self._get_embedding(turn.content)

        # Concatenate: [384d embedding, 10d features] = 394d
        return np.concatenate([embedding, base_features])

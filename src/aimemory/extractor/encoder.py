"""Enhanced state encoder using sentence-transformer embeddings.

Combines ko-sroberta embeddings with hand-crafted features from the
existing StateEncoder for richer state representation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from aimemory.online.policy import StateEncoder
from aimemory.schemas import MemoryActionType, Turn

logger = logging.getLogger(__name__)

# Default sentence-transformer model
DEFAULT_ST_MODEL = "jhgan/ko-sroberta-multitask"


class EnhancedStateEncoder:
    """Encodes conversation state into (turn_emb, mem_emb, hand_features).

    Uses ko-sroberta-multitask for dense text embeddings and
    the existing StateEncoder for hand-crafted features.

    Args:
        st_model_name: Name of the sentence-transformer model.
        hand_feature_dim: Dimension of hand-crafted features (default: 10).
        device: Torch device for the embedding model.
    """

    def __init__(
        self,
        st_model_name: str = DEFAULT_ST_MODEL,
        hand_feature_dim: int = 10,
        device: str | None = None,
    ) -> None:
        self._st_model_name = st_model_name
        self._st_model: SentenceTransformer | None = None
        self._hand_encoder = StateEncoder(feature_dim=hand_feature_dim)
        self._device = device

    def _get_st_model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                self._st_model_name, device=self._device
            )
            logger.info(
                "Loaded sentence-transformer: %s on %s",
                self._st_model_name,
                self._device or "default",
            )
        return self._st_model

    def encode(
        self,
        turn: Turn,
        recent_turns: list[Turn],
        memory_summaries: list[str] | None = None,
        memory_count: int = 0,
        recent_actions: list[MemoryActionType] | None = None,
        turn_position: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode conversation state into three components.

        Args:
            turn: Current dialogue turn.
            recent_turns: Recent conversation history.
            memory_summaries: List of memory summary strings.
            memory_count: Number of stored memories.
            recent_actions: Recent memory actions taken.
            turn_position: Normalized position in conversation.

        Returns:
            Tuple of (turn_emb[768], mem_emb[768], hand_features[10]).
        """
        model = self._get_st_model()

        # Turn embedding: encode current turn text
        turn_emb = model.encode(
            turn.content, convert_to_numpy=True, show_progress_bar=False
        )

        # Memory context embedding: mean of memory summaries or zero vector
        if memory_summaries:
            mem_embs = model.encode(
                memory_summaries, convert_to_numpy=True, show_progress_bar=False
            )
            mem_emb = np.mean(mem_embs, axis=0)
        else:
            mem_emb = np.zeros(turn_emb.shape[-1], dtype=np.float32)

        # Hand-crafted features from existing StateEncoder
        hand = self._hand_encoder.encode(
            turn=turn,
            recent_turns=recent_turns,
            memory_count=memory_count,
            recent_actions=recent_actions,
            turn_position=turn_position,
        )

        return (
            turn_emb.astype(np.float32),
            mem_emb.astype(np.float32),
            hand.astype(np.float32),
        )

    def encode_text_batch(
        self,
        texts: list[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        """Batch-encode texts into embeddings.

        Args:
            texts: List of text strings.
            batch_size: Encoding batch size.

        Returns:
            ndarray of shape [N, 768].
        """
        model = self._get_st_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def build_turn_text(recent_turns_json: str) -> str:
        """Extract current turn text from the JSON-encoded recent_turns field.

        Uses the last turn's content as the turn text for embedding.

        Args:
            recent_turns_json: JSON string of recent turns list.

        Returns:
            Text content of the last (current) turn.
        """
        turns = json.loads(recent_turns_json)
        if turns:
            return turns[-1].get("content", "")
        return ""

    @staticmethod
    def build_memory_text(memory_summary_json: str) -> str:
        """Combine memory summaries into a single text for embedding.

        Args:
            memory_summary_json: JSON string of memory summary list.

        Returns:
            Concatenated memory text or empty string.
        """
        summaries = json.loads(memory_summary_json)
        if summaries:
            return " ".join(summaries)
        return ""

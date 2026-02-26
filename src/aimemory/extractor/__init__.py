"""RL Feature Extractor package.

Provides DualHeadDQN model, EnhancedStateEncoder, DQNPolicy,
EmbeddingDataset, and OfflineDQNTrainer for offline Double DQN training.
"""

from aimemory.extractor.model import DualHeadDQN
from aimemory.extractor.encoder import EnhancedStateEncoder
from aimemory.extractor.policy import DQNPolicy
from aimemory.extractor.dataset import EmbeddingDataset
from aimemory.extractor.trainer import OfflineDQNTrainer

__all__ = [
    "DualHeadDQN",
    "EnhancedStateEncoder",
    "DQNPolicy",
    "EmbeddingDataset",
    "OfflineDQNTrainer",
]

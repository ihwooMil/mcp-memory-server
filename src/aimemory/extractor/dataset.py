"""Embedding dataset for offline DQN training.

Loads pre-computed embeddings from memory-mapped numpy arrays
for efficient batch training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """PyTorch Dataset backed by memory-mapped numpy arrays.

    Expected directory structure:
        data_dir/
            turn_emb.npy      # [N, 768] turn embeddings
            mem_emb.npy       # [N, 768] memory context embeddings
            hand_features.npy # [N, 10]  hand-crafted features
            actions.npy       # [N]      action indices (0/1/2)
            rewards.npy       # [N]      reward values
            dones.npy         # [N]      episode termination flags
            next_turn_emb.npy # [N, 768] next-state turn embeddings
            next_mem_emb.npy  # [N, 768] next-state memory embeddings
            next_hand_features.npy  # [N, 10] next-state hand features

    Args:
        data_dir: Path to the directory containing numpy arrays.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._validate_dir()

        # Memory-mapped loading for large datasets
        self.turn_emb = np.load(self.data_dir / "turn_emb.npy", mmap_mode="r")
        self.mem_emb = np.load(self.data_dir / "mem_emb.npy", mmap_mode="r")
        self.hand_features = np.load(
            self.data_dir / "hand_features.npy", mmap_mode="r"
        )
        self.actions = np.load(self.data_dir / "actions.npy", mmap_mode="r")
        self.rewards = np.load(self.data_dir / "rewards.npy", mmap_mode="r")
        self.dones = np.load(self.data_dir / "dones.npy", mmap_mode="r")
        self.next_turn_emb = np.load(
            self.data_dir / "next_turn_emb.npy", mmap_mode="r"
        )
        self.next_mem_emb = np.load(
            self.data_dir / "next_mem_emb.npy", mmap_mode="r"
        )
        self.next_hand_features = np.load(
            self.data_dir / "next_hand_features.npy", mmap_mode="r"
        )

        self._len = len(self.turn_emb)

    def _validate_dir(self) -> None:
        """Check that all required files exist."""
        required = [
            "turn_emb.npy",
            "mem_emb.npy",
            "hand_features.npy",
            "actions.npy",
            "rewards.npy",
            "dones.npy",
            "next_turn_emb.npy",
            "next_mem_emb.npy",
            "next_hand_features.npy",
        ]
        missing = [f for f in required if not (self.data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing files in {self.data_dir}: {missing}"
            )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "turn_emb": torch.from_numpy(np.array(self.turn_emb[idx])).float(),
            "mem_emb": torch.from_numpy(np.array(self.mem_emb[idx])).float(),
            "hand_features": torch.from_numpy(
                np.array(self.hand_features[idx])
            ).float(),
            "action": torch.tensor(int(self.actions[idx]), dtype=torch.long),
            "reward": torch.tensor(float(self.rewards[idx]), dtype=torch.float32),
            "done": torch.tensor(bool(self.dones[idx]), dtype=torch.bool),
            "next_turn_emb": torch.from_numpy(
                np.array(self.next_turn_emb[idx])
            ).float(),
            "next_mem_emb": torch.from_numpy(
                np.array(self.next_mem_emb[idx])
            ).float(),
            "next_hand_features": torch.from_numpy(
                np.array(self.next_hand_features[idx])
            ).float(),
        }

"""DQN-based policy for memory action selection.

Wraps DualHeadDQN with epsilon-greedy action selection,
feature extraction, and checkpoint/gossip support.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from aimemory.extractor.model import DualHeadDQN


class DQNPolicy:
    """Policy wrapper around DualHeadDQN for inference and gossip.

    Args:
        emb_dim: Sentence-transformer embedding dimension.
        proj_dim: Projection dimension per stream.
        hand_dim: Hand-crafted feature dimension.
        trunk_dim: Trunk hidden dimension.
        n_actions: Number of actions (3: SAVE, SKIP, RETRIEVE).
        feature_dim: Feature head output dimension.
        dropout: Dropout rate.
        epsilon: Exploration rate for epsilon-greedy.
        device: Torch device string.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        proj_dim: int = 128,
        hand_dim: int = 10,
        trunk_dim: int = 128,
        n_actions: int = 3,
        feature_dim: int = 64,
        dropout: float = 0.1,
        epsilon: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.device = device

        self._model = DualHeadDQN(
            emb_dim=emb_dim,
            proj_dim=proj_dim,
            hand_dim=hand_dim,
            trunk_dim=trunk_dim,
            n_actions=n_actions,
            feature_dim=feature_dim,
            dropout=dropout,
        ).to(device)
        self._rng = np.random.default_rng()

    @property
    def model(self) -> DualHeadDQN:
        """Access the underlying DualHeadDQN model."""
        return self._model

    def select_action(
        self,
        turn_emb: np.ndarray,
        mem_emb: np.ndarray,
        hand: np.ndarray,
    ) -> int:
        """Select an action using epsilon-greedy strategy.

        Args:
            turn_emb: Turn embedding [768].
            mem_emb: Memory embedding [768].
            hand: Hand-crafted features [10].

        Returns:
            Action index (0=SAVE, 1=SKIP, 2=RETRIEVE).
        """
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))

        self._model.eval()
        with torch.no_grad():
            t = torch.from_numpy(turn_emb).float().unsqueeze(0).to(self.device)
            m = torch.from_numpy(mem_emb).float().unsqueeze(0).to(self.device)
            h = torch.from_numpy(hand).float().unsqueeze(0).to(self.device)
            q_values, _ = self._model(t, m, h)
            return int(q_values.argmax(dim=1).item())

    def extract_features(
        self,
        turn_emb: np.ndarray,
        mem_emb: np.ndarray,
        hand: np.ndarray,
    ) -> np.ndarray:
        """Extract feature vector from the feature head.

        Args:
            turn_emb: Turn embedding [768].
            mem_emb: Memory embedding [768].
            hand: Hand-crafted features [10].

        Returns:
            Feature vector of shape [64].
        """
        self._model.eval()
        with torch.no_grad():
            t = torch.from_numpy(turn_emb).float().unsqueeze(0).to(self.device)
            m = torch.from_numpy(mem_emb).float().unsqueeze(0).to(self.device)
            h = torch.from_numpy(hand).float().unsqueeze(0).to(self.device)
            _, features = self._model(t, m, h)
            return features.squeeze(0).cpu().numpy()

    def get_parameters(self) -> np.ndarray:
        """Return flattened model parameters as numpy array (for gossip)."""
        params = []
        for p in self._model.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_parameters(self, params: np.ndarray) -> None:
        """Set model parameters from flattened numpy array (from gossip)."""
        offset = 0
        for p in self._model.parameters():
            numel = p.data.numel()
            chunk = params[offset : offset + numel]
            p.data.copy_(
                torch.from_numpy(chunk.reshape(p.data.shape)).float().to(self.device)
            )
            offset += numel

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "epsilon": self.epsilon,
                "config": {
                    "emb_dim": self._model.emb_dim,
                    "proj_dim": self._model.proj_dim,
                    "hand_dim": self._model.hand_dim,
                    "trunk_dim": self._model.trunk_dim,
                    "n_actions": self._model.n_actions,
                    "feature_dim": self._model.feature_dim,
                },
            },
            str(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state from file."""
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)

    def create_target_network(self) -> DualHeadDQN:
        """Create a copy of the model for use as a target network."""
        target = copy.deepcopy(self._model)
        target.eval()
        for p in target.parameters():
            p.requires_grad = False
        return target

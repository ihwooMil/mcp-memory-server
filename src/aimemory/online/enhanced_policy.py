"""Enhanced online policy with 394d input, replay buffer, and progressive autonomy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from aimemory.online.policy import OnlinePolicy
from aimemory.online.replay_buffer import ReplayBuffer
from aimemory.online.autonomy import ProgressiveAutonomy

logger = logging.getLogger(__name__)


class _EnhancedMLP(nn.Module):
    """Larger MLP: 394d → 256 → 128 → 3."""

    def __init__(self, feature_dim: int = 394, hidden1: int = 256, hidden2: int = 128, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnhancedOnlinePolicy(OnlinePolicy):
    """Drop-in replacement with 394d input, replay buffer, progressive autonomy.

    Inherits OnlinePolicy interface so MemoryPolicyAgent works without modification.
    """

    def __init__(
        self,
        enhanced_encoder=None,
        replay_buffer: ReplayBuffer | None = None,
        autonomy: ProgressiveAutonomy | None = None,
        feature_dim: int = 394,
        n_actions: int = 3,
        hidden1: int = 256,
        hidden2: int = 128,
        lr: float = 0.001,
        epsilon: float = 0.1,
        batch_update_interval: int = 10,
    ):
        # Call parent init (sets up self.feature_dim, self.n_actions, self.epsilon, etc.)
        super().__init__(feature_dim=feature_dim, n_actions=n_actions, lr=lr, epsilon=epsilon)

        # Replace the parent's _model with enhanced MLP
        self._model = _EnhancedMLP(feature_dim, hidden1, hidden2, n_actions)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        self._encoder = enhanced_encoder
        self._replay_buffer = replay_buffer or ReplayBuffer()
        self._autonomy = autonomy or ProgressiveAutonomy()
        self._batch_update_interval = batch_update_interval
        self._update_count = 0

    @property
    def autonomy(self) -> ProgressiveAutonomy:
        return self._autonomy

    @property
    def replay_buffer(self) -> ReplayBuffer:
        return self._replay_buffer

    def select_action(self, features: np.ndarray) -> int:
        """Select action using epsilon-greedy, same interface as parent."""
        return super().select_action(features)

    def update(self, features: np.ndarray, action_id: int, reward: float) -> float:
        """Push to replay buffer + periodic batch update. Updates autonomy too."""
        # Push experience to replay buffer
        self._replay_buffer.push(features, action_id, reward, None)

        # Update autonomy with feedback
        self._autonomy.record_feedback(reward)

        # Single-step SGD (like parent)
        loss = super().update(features, action_id, reward)

        # Periodic batch update from replay buffer
        self._update_count += 1
        if (self._update_count % self._batch_update_interval == 0
                and len(self._replay_buffer) >= 32):
            self.batch_update()

        return loss

    def batch_update(self, batch_size: int = 32) -> float:
        """Batch SGD from replay buffer. Returns average loss."""
        if len(self._replay_buffer) < batch_size:
            return 0.0

        batch = self._replay_buffer.sample(batch_size)

        self._model.train()

        # Prepare batch tensors
        states = torch.stack([torch.from_numpy(exp.state).float() for exp in batch])
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)

        # Forward pass
        q_values = self._model(states)  # (batch, n_actions)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # MSE loss
        loss = torch.nn.functional.mse_loss(q_selected, rewards)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return float(loss.item())

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, optimizer, replay buffer, and autonomy state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "epsilon": self.epsilon,
                "feature_dim": self.feature_dim,
                "n_actions": self.n_actions,
                "update_count": self._update_count,
            },
            str(path),
        )
        # Save replay buffer and autonomy alongside
        buffer_path = path.parent / (path.stem + "_buffer.pkl")
        autonomy_path = path.parent / (path.stem + "_autonomy.json")
        self._replay_buffer.save(str(buffer_path))
        self._autonomy.save(str(autonomy_path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model, optimizer, replay buffer, and autonomy state."""
        path = Path(path)
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self._update_count = checkpoint.get("update_count", 0)

        # Load replay buffer and autonomy if they exist
        buffer_path = path.parent / (path.stem + "_buffer.pkl")
        autonomy_path = path.parent / (path.stem + "_autonomy.json")
        if buffer_path.exists():
            self._replay_buffer.load(str(buffer_path))
        if autonomy_path.exists():
            self._autonomy.load(str(autonomy_path))

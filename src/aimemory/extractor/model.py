"""Dual-head DQN model for memory action selection.

Architecture:
    Turn text  --[ko-sroberta 768d]--> turn_proj(768->128) -> ReLU --+
    Memory ctx --[ko-sroberta 768d]--> mem_proj(768->128)  -> ReLU --+-- concat(266)
    Hand-crafted features (10d) ------------------------------------- +      |
                                                                            v
                                            trunk: Linear(266->128) -> BN -> ReLU -> Dropout(0.1)
                                                        |                    |
                                                action_head(128->3)    feature_head(128->64)
                                                  Q-values              Feature vector
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DualHeadDQN(nn.Module):
    """Dual-head Deep Q-Network with action and feature outputs.

    Args:
        emb_dim: Dimension of sentence-transformer embeddings (default: 768).
        proj_dim: Projection dimension for each embedding stream (default: 128).
        hand_dim: Dimension of hand-crafted features (default: 10).
        trunk_dim: Hidden dimension of the shared trunk (default: 128).
        n_actions: Number of actions (SAVE=0, SKIP=1, RETRIEVE=2).
        feature_dim: Output dimension of the feature head (default: 64).
        dropout: Dropout rate in the trunk (default: 0.1).
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
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.hand_dim = hand_dim
        self.trunk_dim = trunk_dim
        self.n_actions = n_actions
        self.feature_dim = feature_dim

        # Embedding projections
        self.turn_proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
        )
        self.mem_proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
        )

        # Shared trunk
        concat_dim = proj_dim * 2 + hand_dim  # 128 + 128 + 10 = 266
        self.trunk = nn.Sequential(
            nn.Linear(concat_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.action_head = nn.Linear(trunk_dim, n_actions)
        self.feature_head = nn.Linear(trunk_dim, feature_dim)

    def forward(
        self,
        turn_emb: torch.Tensor,
        mem_emb: torch.Tensor,
        hand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            turn_emb: Turn text embedding [B, 768].
            mem_emb: Memory context embedding [B, 768].
            hand: Hand-crafted features [B, 10].

        Returns:
            Tuple of (q_values [B, 3], features [B, 64]).
        """
        t = self.turn_proj(turn_emb)
        m = self.mem_proj(mem_emb)
        x = torch.cat([t, m, hand], dim=-1)
        shared = self.trunk(x)
        q_values = self.action_head(shared)
        features = self.feature_head(shared)
        return q_values, features

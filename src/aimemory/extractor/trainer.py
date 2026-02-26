"""Offline Double DQN trainer with class-weighted Huber loss.

Trains DualHeadDQN on pre-computed embeddings using offline RL.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aimemory.extractor.model import DualHeadDQN

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    """Metrics collected during training."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    val_class_accuracy: dict[str, float] = field(default_factory=dict)
    lr: float = 0.0


class OfflineDQNTrainer:
    """Offline Double DQN trainer.

    Features:
        - Double DQN with target network (synced every target_sync steps)
        - Class-weighted Huber loss to handle action imbalance
        - AdamW optimizer with cosine annealing LR schedule
        - Gradient clipping (max_norm=1.0)
        - Early stopping (patience=3)

    Args:
        model: DualHeadDQN online network.
        lr: Learning rate for AdamW.
        gamma: Discount factor.
        target_sync: Steps between target network syncs.
        class_weights: Per-action loss weights {0: SAVE, 1: SKIP, 2: RETRIEVE}.
        max_grad_norm: Gradient clipping threshold.
        device: Torch device string.
    """

    def __init__(
        self,
        model: DualHeadDQN,
        lr: float = 3e-4,
        gamma: float = 0.99,
        target_sync: int = 1000,
        class_weights: dict[int, float] | None = None,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.target_sync = target_sync
        self.max_grad_norm = max_grad_norm

        # Online and target networks
        self.online = model.to(device)
        self.target = copy.deepcopy(model).to(device)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        # Class weights: SAVE=1.0, SKIP=0.7, RETRIEVE=3.0
        cw = class_weights or {0: 1.0, 1: 0.7, 2: 3.0}
        self._class_weights = torch.tensor(
            [cw[i] for i in range(len(cw))], dtype=torch.float32, device=device
        )

        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=lr)
        self.huber = nn.SmoothL1Loss(reduction="none")

        # Step counter for target sync
        self._global_step = 0

    def _sync_target(self) -> None:
        """Copy online network weights to target network."""
        self.target.load_state_dict(self.online.state_dict())

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute class-weighted Double DQN Huber loss."""
        turn_emb = batch["turn_emb"].to(self.device)
        mem_emb = batch["mem_emb"].to(self.device)
        hand = batch["hand_features"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch["done"].to(self.device)
        next_turn = batch["next_turn_emb"].to(self.device)
        next_mem = batch["next_mem_emb"].to(self.device)
        next_hand = batch["next_hand_features"].to(self.device)

        # Current Q-values for selected actions
        q_values, _ = self.online(turn_emb, mem_emb, hand)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: action selection from online, evaluation from target
        with torch.no_grad():
            next_q_online, _ = self.online(next_turn, next_mem, next_hand)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target, _ = self.target(next_turn, next_mem, next_hand)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (~dones).float()

        # Per-sample Huber loss with class weights
        per_sample_loss = self.huber(q_selected, target_q)
        weights = self._class_weights[actions]
        loss = (per_sample_loss * weights).mean()
        return loss

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Average loss for the epoch.
        """
        self.online.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            loss = self._compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self._global_step += 1

            # Sync target network
            if self._global_step % self.target_sync == 0:
                self._sync_target()
                logger.debug(
                    "Target network synced at step %d", self._global_step
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> TrainMetrics:
        """Evaluate on validation data.

        Args:
            dataloader: Validation data loader.

        Returns:
            TrainMetrics with loss, accuracy, and per-class accuracy.
        """
        self.online.eval()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}

        action_names = {0: "SAVE", 1: "SKIP", 2: "RETRIEVE"}

        for batch in dataloader:
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            n_batches += 1

            # Accuracy
            turn_emb = batch["turn_emb"].to(self.device)
            mem_emb = batch["mem_emb"].to(self.device)
            hand = batch["hand_features"].to(self.device)
            actions = batch["action"].to(self.device)

            q_values, _ = self.online(turn_emb, mem_emb, hand)
            predicted = q_values.argmax(dim=1)
            correct += (predicted == actions).sum().item()
            total += actions.size(0)

            for cls in class_correct:
                mask = actions == cls
                class_correct[cls] += (predicted[mask] == actions[mask]).sum().item()
                class_total[cls] += mask.sum().item()

        metrics = TrainMetrics(
            val_loss=total_loss / max(n_batches, 1),
            val_accuracy=correct / max(total, 1),
            val_class_accuracy={
                action_names[cls]: class_correct[cls] / max(class_total[cls], 1)
                for cls in class_correct
            },
        )
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 20,
        patience: int = 3,
        output_dir: str | Path = "checkpoints",
    ) -> list[TrainMetrics]:
        """Full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            max_epochs: Maximum number of epochs.
            patience: Early stopping patience (epochs without improvement).
            output_dir: Directory to save checkpoints.

        Returns:
            List of TrainMetrics for each epoch.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )

        history: list[TrainMetrics] = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            metrics = self.evaluate(val_loader)
            metrics.epoch = epoch
            metrics.train_loss = train_loss
            metrics.lr = scheduler.get_last_lr()[0]
            history.append(metrics)

            scheduler.step()

            logger.info(
                "Epoch %d/%d - train_loss: %.4f, val_loss: %.4f, "
                "val_acc: %.4f, class_acc: %s",
                epoch,
                max_epochs,
                train_loss,
                metrics.val_loss,
                metrics.val_accuracy,
                metrics.val_class_accuracy,
            )

            # Checkpointing
            if metrics.val_loss < best_val_loss:
                best_val_loss = metrics.val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.online.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": metrics.val_loss,
                        "val_accuracy": metrics.val_accuracy,
                    },
                    str(output_dir / "best_model.pt"),
                )
                logger.info("Saved best model at epoch %d", epoch)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)", epoch, patience
                    )
                    break

        return history

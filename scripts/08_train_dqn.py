#!/usr/bin/env python3
"""Train offline Double DQN on pre-computed embeddings.

Usage:
    uv run python scripts/08_train_dqn.py \
        --data-dir data/embeddings \
        --output-dir checkpoints/extractor \
        --epochs 20 \
        --batch-size 512 \
        --lr 3e-4 \
        --device cpu
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from aimemory.extractor.dataset import EmbeddingDataset
from aimemory.extractor.model import DualHeadDQN
from aimemory.extractor.trainer import OfflineDQNTrainer
from torch.utils.data import DataLoader

from aimemory.config import ExtractorConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train offline Double DQN")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Directory with {train,val}/  embedding arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/extractor"),
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-sync", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = ExtractorConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        target_sync=args.target_sync,
        max_epochs=args.epochs,
        patience=args.patience,
    )

    # Load datasets
    logger.info("Loading training data from %s/train", args.data_dir)
    train_ds = EmbeddingDataset(args.data_dir / "train")
    val_ds = EmbeddingDataset(args.data_dir / "val")
    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )

    # Create model
    model = DualHeadDQN(
        emb_dim=cfg.emb_dim,
        proj_dim=cfg.proj_dim,
        hand_dim=cfg.hand_dim,
        trunk_dim=cfg.trunk_dim,
        n_actions=cfg.n_actions,
        feature_dim=cfg.feature_dim,
        dropout=cfg.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (~%.1fK)", total_params, total_params / 1000)

    # Create trainer
    trainer = OfflineDQNTrainer(
        model=model,
        lr=cfg.lr,
        gamma=cfg.gamma,
        target_sync=cfg.target_sync,
        class_weights=cfg.class_weights,
        max_grad_norm=cfg.max_grad_norm,
        device=args.device,
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        output_dir=args.output_dir,
    )

    # Summary
    if history:
        best = min(history, key=lambda m: m.val_loss)
        logger.info(
            "Best epoch %d: val_loss=%.4f, val_acc=%.4f, class_acc=%s",
            best.epoch,
            best.val_loss,
            best.val_accuracy,
            best.val_class_accuracy,
        )

    # Save final model
    final_path = args.output_dir / "final_model.pt"
    torch.save(
        {"model_state_dict": trainer.online.state_dict()},
        str(final_path),
    )
    logger.info("Saved final model to %s", final_path)


if __name__ == "__main__":
    main()

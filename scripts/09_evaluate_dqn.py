#!/usr/bin/env python3
"""Evaluate trained DQN model on test data.

Reports:
- Overall and per-class accuracy
- Feature vector quality (same-episode vs cross-episode cosine similarity)
- Confusion matrix

Usage:
    uv run python scripts/09_evaluate_dqn.py \
        --data-dir data/embeddings/test \
        --checkpoint checkpoints/extractor/best_model.pt \
        --device cpu
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from aimemory.extractor.dataset import EmbeddingDataset
from aimemory.extractor.model import DualHeadDQN
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ACTION_NAMES = {0: "SAVE", 1: "SKIP", 2: "RETRIEVE"}


def compute_accuracy(
    model: DualHeadDQN,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Compute overall and per-class accuracy."""
    model.eval()
    correct = 0
    total = 0
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    confusion = np.zeros((3, 3), dtype=int)

    with torch.no_grad():
        for batch in dataloader:
            turn_emb = batch["turn_emb"].to(device)
            mem_emb = batch["mem_emb"].to(device)
            hand = batch["hand_features"].to(device)
            actions = batch["action"].to(device)

            q_values, _ = model(turn_emb, mem_emb, hand)
            predicted = q_values.argmax(dim=1)

            correct += (predicted == actions).sum().item()
            total += actions.size(0)

            for cls in range(3):
                mask = actions == cls
                class_correct[cls] += (predicted[mask] == actions[mask]).sum().item()
                class_total[cls] += mask.sum().item()

            for true, pred in zip(actions.cpu().numpy(), predicted.cpu().numpy()):
                confusion[true][pred] += 1

    overall_acc = correct / max(total, 1)
    class_acc = {
        ACTION_NAMES[cls]: class_correct[cls] / max(class_total[cls], 1) for cls in range(3)
    }
    class_recall = class_acc  # same as accuracy when measured per-class

    return {
        "overall_accuracy": overall_acc,
        "class_accuracy": class_acc,
        "class_recall": class_recall,
        "class_total": {ACTION_NAMES[k]: v for k, v in class_total.items()},
        "confusion_matrix": confusion,
    }


def compute_feature_quality(
    model: DualHeadDQN,
    dataloader: DataLoader,
    device: str,
    max_samples: int = 5000,
) -> dict:
    """Evaluate feature vector quality via cosine similarity analysis."""
    model.eval()
    all_features = []
    all_actions = []

    with torch.no_grad():
        collected = 0
        for batch in dataloader:
            if collected >= max_samples:
                break

            turn_emb = batch["turn_emb"].to(device)
            mem_emb = batch["mem_emb"].to(device)
            hand = batch["hand_features"].to(device)
            actions = batch["action"]

            _, features = model(turn_emb, mem_emb, hand)
            all_features.append(features.cpu().numpy())
            all_actions.append(actions.numpy())
            collected += len(actions)

    features = np.concatenate(all_features, axis=0)[:max_samples]
    actions = np.concatenate(all_actions, axis=0)[:max_samples]

    # Normalize for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features_norm = features / norms

    # Same-action vs cross-action similarity
    results = {}
    for cls in range(3):
        mask = actions == cls
        if mask.sum() < 2:
            continue

        cls_features = features_norm[mask]
        # Intra-class: sample pairs within the same class
        n_cls = min(len(cls_features), 500)
        indices = np.random.choice(len(cls_features), n_cls, replace=False)
        sub = cls_features[indices]
        sim_matrix = sub @ sub.T
        # Upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(n_cls, k=1)
        intra_sim = sim_matrix[triu_idx].mean()

        # Inter-class: sample pairs from different classes
        other_mask = actions != cls
        if other_mask.sum() > 0:
            n_other = min(other_mask.sum(), 500)
            other_indices = np.random.choice(np.where(other_mask)[0], n_other, replace=False)
            other_features = features_norm[other_indices]
            inter_sim = (sub @ other_features.T).mean()
        else:
            inter_sim = 0.0

        results[ACTION_NAMES[cls]] = {
            "intra_similarity": float(intra_sim),
            "inter_similarity": float(inter_sim),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/embeddings/test"),
        help="Test embedding directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/extractor/best_model.pt"),
        help="Model checkpoint path",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(str(args.checkpoint), map_location=args.device, weights_only=True)
    model = DualHeadDQN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    # Load test data
    test_ds = EmbeddingDataset(args.data_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Test samples: %d", len(test_ds))

    # Accuracy evaluation
    acc_results = compute_accuracy(model, test_loader, args.device)
    logger.info("=" * 60)
    logger.info("Overall accuracy: %.4f", acc_results["overall_accuracy"])
    logger.info("Per-class accuracy:")
    for cls, acc in acc_results["class_accuracy"].items():
        count = acc_results["class_total"][cls]
        logger.info("  %s: %.4f (%d samples)", cls, acc, count)

    logger.info("Confusion matrix (rows=true, cols=pred):")
    cm = acc_results["confusion_matrix"]
    logger.info("          SAVE   SKIP   RETR")
    for i, name in enumerate(["SAVE", "SKIP", "RETR"]):
        logger.info("  %s  %6d %6d %6d", name, cm[i][0], cm[i][1], cm[i][2])

    # Feature quality
    logger.info("=" * 60)
    logger.info("Feature vector quality (cosine similarity):")
    feat_results = compute_feature_quality(model, test_loader, args.device)
    for cls, sims in feat_results.items():
        logger.info(
            "  %s: intra=%.4f, inter=%.4f",
            cls,
            sims["intra_similarity"],
            sims["inter_similarity"],
        )


if __name__ == "__main__":
    main()

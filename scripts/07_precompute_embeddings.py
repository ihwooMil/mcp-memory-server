#!/usr/bin/env python3
"""Pre-compute ko-sroberta embeddings for offline DQN training.

Reads SARTriple parquet files, encodes turn and memory texts into
768-dim embeddings, and saves as numpy arrays for memory-mapped loading.

Usage:
    uv run python scripts/07_precompute_embeddings.py \
        --splits-dir data/splits \
        --output-dir data/embeddings \
        --batch-size 256 \
        --chunk-size 10000

Output structure:
    data/embeddings/{train,val,test}/
        turn_emb.npy        [N, 768]
        mem_emb.npy          [N, 768]
        hand_features.npy    [N, 10]
        actions.npy          [N]
        rewards.npy          [N]
        dones.npy            [N]
        next_turn_emb.npy    [N, 768]
        next_mem_emb.npy     [N, 768]
        next_hand_features.npy [N, 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Action type to index mapping
ACTION_INDEX = {"save": 0, "skip": 1, "retrieve": 2}


def extract_turn_text(recent_turns_json: str) -> str:
    """Extract current turn text from recent_turns JSON."""
    try:
        turns = json.loads(recent_turns_json)
        if turns:
            return turns[-1].get("content", "")
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def extract_memory_text(memory_summary_json: str) -> str:
    """Combine memory summaries into a single text."""
    try:
        summaries = json.loads(memory_summary_json)
        if summaries:
            return " ".join(summaries)
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def extract_next_state_texts(next_state_json: str) -> tuple[str, str]:
    """Extract turn text and memory text from next_state JSON."""
    try:
        ns = json.loads(next_state_json) if isinstance(next_state_json, str) else None
        if ns:
            turns = ns.get("recent_turns", [])
            turn_text = turns[-1]["content"] if turns else ""
            summaries = ns.get("current_memory_summary", [])
            mem_text = " ".join(summaries) if summaries else ""
            return turn_text, mem_text
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        pass
    return "", ""


# ─── Hand-crafted feature extraction (mirror of StateEncoder) ──────

_QUESTION_PATTERN = re.compile(
    r"[?\uff1f]|(?:\uc778\uac00\uc694|\ub098\uc694|\uc744\uae4c\uc694|\u3139\uae4c\uc694|\uc5b4\uc694\?|\uc2b5\ub2c8\uae4c)"
)
_PERSONAL_INFO_PATTERNS = [
    re.compile(
        r"\uc800\ub294?\s+(.{2,20})(?:\uc774\uc5d0\uc694|\uc785\ub2c8\ub2e4|\uc608\uc694|\uc778\ub370|\uac70\ub4e0\uc694|\uc774\ub77c\uc11c)"
    ),
    re.compile(
        r"\uc81c\s+(?:\uc774\ub984|\ub098\uc774|\uc9c1\uc5c5|\uc804\uacf5|\ud68c\uc0ac|\ud300|\ud504\ub85c\uc81d\ud2b8)"
    ),
    re.compile(r"\uc0b4\uace0\s*\uc788(?:\uc5b4\uc694|\uc2b5\ub2c8\ub2e4)"),
    re.compile(r"\ub2e4\ub2c8\uace0\s*\uc788(?:\uc5b4\uc694|\uc2b5\ub2c8\ub2e4)"),
    re.compile(r"\uc77c\ud558\uace0\s*\uc788(?:\uc5b4\uc694|\uc2b5\ub2c8\ub2e4)"),
]
_PREFERENCE_PATTERNS = [
    re.compile(r"\uc88b\uc544(?:\ud574\uc694|\ud569\ub2c8\ub2e4|\ud558\ub294|\ud568)"),
    re.compile(r"\uc2eb\uc5b4(?:\ud574\uc694|\ud569\ub2c8\ub2e4|\ud558\ub294|\ud568)"),
    re.compile(r"\uc120\ud638(?:\ud574\uc694|\ud569\ub2c8\ub2e4|\ud558\ub294|\ud568|\ud558\ub2e4)"),
    re.compile(r"\ucde8\ubbf8(?:\uac00|\ub294|\ub85c)?"),
    re.compile(r"\uc8fc\ub85c\s+\S+"),
    re.compile(r"\uc990\uaca8\s*\S+"),
]
_TECH_KEYWORDS = re.compile(
    r"(?<![가-힣a-zA-Z_])(?:"
    r"Python|Java(?:Script)?|TypeScript|Rust|Go|C\+\+|Ruby|Swift|Kotlin|Dart|"
    r"React|Vue|Angular|Next\.js|Django|Flask|FastAPI|Spring|Rails|"
    r"Docker|Kubernetes|k8s|AWS|GCP|Azure|Linux|Ubuntu|macOS|Windows|"
    r"MySQL|PostgreSQL|SQLite|Redis|MongoDB|Elasticsearch|"
    r"Git|GitHub|GitLab|CI/CD|DevOps|MLOps|"
    r"pandas|numpy|scipy|sklearn|scikit-learn|TensorFlow|PyTorch|Keras|"
    r"LLM|GPT|Claude|Gemini|\uba38\uc2e0\ub7ec\ub2dd|\ub525\ub7ec\ub2dd|\uc778\uacf5\uc9c0\ub2a5|AI|"
    r"API|REST|GraphQL|WebSocket|gRPC|"
    r"\uc54c\uace0\ub9ac\uc998|\uc790\ub8cc\uad6c\uc870|\ub370\uc774\ud130\ubca0\uc774\uc2a4|\ud074\ub77c\uc6b0\ub4dc|\ub9c8\uc774\ud06c\ub85c\uc11c\ube44\uc2a4"
    r")(?![a-zA-Z_])",
    re.IGNORECASE,
)
_EMOTION_KEYWORDS = re.compile(
    r"\uae30\uc058|\uc2ac\ud504|\ud654\ub098|\ubb34\uc11c|\ubd88\uc548|\uc124\ub808|\uac71\uc815|\ud798\ub4e4|\uc5b4\ub835|\uc88b\uc544|\uc2eb\uc5b4|\uc990\uac70|\ud589\ubcf5|\uc6b0\uc6b8|\ud53c\uace4|\uc2e0\ub098"
)


def compute_hand_features(row: pd.Series) -> np.ndarray:
    """Compute 10-dim hand-crafted features for a single row."""
    text = extract_turn_text(row["state_recent_turns_json"])
    turn_position = float(row["state_turn_position"])
    memory_count = int(row["state_memory_count"])

    tech_matches = _TECH_KEYWORDS.findall(text)
    quoted_matches = re.findall(r"['\"]([^'\"]{2,30})['\"]", text)
    keyword_count = len(tech_matches) + len(quoted_matches)

    is_question = 1.0 if _QUESTION_PATTERN.search(text) else 0.0
    has_personal = 1.0 if any(p.search(text) for p in _PERSONAL_INFO_PATTERNS) else 0.0
    has_preference = 1.0 if any(p.search(text) for p in _PREFERENCE_PATTERNS) else 0.0
    has_tech = 1.0 if _TECH_KEYWORDS.search(text) else 0.0
    has_emotion = 1.0 if _EMOTION_KEYWORDS.search(text) else 0.0

    # For recent actions, we approximate from step_index (not available per-row)
    recent_save = 0.0
    recent_retrieve = 0.0

    return np.array(
        [
            turn_position,
            np.log1p(memory_count),
            np.log1p(keyword_count),
            is_question,
            has_personal,
            has_preference,
            has_tech,
            has_emotion,
            np.log1p(recent_save),
            np.log1p(recent_retrieve),
        ],
        dtype=np.float32,
    )


def compute_next_hand_features(row: pd.Series) -> np.ndarray:
    """Compute hand features for next state."""
    try:
        ns = json.loads(row["next_state_json"]) if isinstance(row["next_state_json"], str) else None
        if ns:
            turns = ns.get("recent_turns", [])
            text = turns[-1]["content"] if turns else ""
            turn_position = float(ns.get("turn_position", 0.0))
            memory_count = int(ns.get("memory_count", 0))
        else:
            return np.zeros(10, dtype=np.float32)
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        return np.zeros(10, dtype=np.float32)

    tech_matches = _TECH_KEYWORDS.findall(text)
    quoted_matches = re.findall(r"['\"]([^'\"]{2,30})['\"]", text)
    keyword_count = len(tech_matches) + len(quoted_matches)

    is_question = 1.0 if _QUESTION_PATTERN.search(text) else 0.0
    has_personal = 1.0 if any(p.search(text) for p in _PERSONAL_INFO_PATTERNS) else 0.0
    has_preference = 1.0 if any(p.search(text) for p in _PREFERENCE_PATTERNS) else 0.0
    has_tech = 1.0 if _TECH_KEYWORDS.search(text) else 0.0
    has_emotion = 1.0 if _EMOTION_KEYWORDS.search(text) else 0.0

    return np.array(
        [
            turn_position,
            np.log1p(memory_count),
            np.log1p(keyword_count),
            is_question,
            has_personal,
            has_preference,
            has_tech,
            has_emotion,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def process_split(
    split_name: str,
    parquet_path: Path,
    output_dir: Path,
    st_model,
    batch_size: int = 256,
    chunk_size: int = 10000,
) -> None:
    """Process a single data split (train/val/test)."""
    out = output_dir / split_name
    out.mkdir(parents=True, exist_ok=True)

    # Check for checkpoint
    checkpoint_path = out / "_checkpoint.json"
    start_chunk = 0
    if checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text())
        start_chunk = ckpt.get("completed_chunks", 0)
        logger.info("Resuming %s from chunk %d", split_name, start_chunk)

    df = pd.read_parquet(parquet_path)
    n = len(df)
    n_chunks = (n + chunk_size - 1) // chunk_size
    logger.info("Processing %s: %d rows, %d chunks", split_name, n, n_chunks)

    # Pre-allocate or load existing arrays
    emb_dim = 768
    hand_dim = 10

    def init_or_load(name: str, shape: tuple) -> np.ndarray:
        path = out / name
        if path.exists() and start_chunk > 0:
            return np.load(str(path), mmap_mode="r+")
        arr = np.zeros(shape, dtype=np.float32)
        np.save(str(path), arr)
        return np.load(str(path), mmap_mode="r+")

    turn_emb = init_or_load("turn_emb.npy", (n, emb_dim))
    mem_emb = init_or_load("mem_emb.npy", (n, emb_dim))
    hand_features = init_or_load("hand_features.npy", (n, hand_dim))
    actions = init_or_load("actions.npy", (n,))
    rewards = init_or_load("rewards.npy", (n,))
    dones = init_or_load("dones.npy", (n,))
    next_turn_emb = init_or_load("next_turn_emb.npy", (n, emb_dim))
    next_mem_emb = init_or_load("next_mem_emb.npy", (n, emb_dim))
    next_hand_features = init_or_load("next_hand_features.npy", (n, hand_dim))

    for chunk_idx in tqdm(range(n_chunks), desc=f"{split_name}"):
        if chunk_idx < start_chunk:
            continue

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n)
        chunk_df = df.iloc[start:end]

        # Extract texts
        turn_texts = [extract_turn_text(r) for r in chunk_df["state_recent_turns_json"]]
        mem_texts = [extract_memory_text(r) for r in chunk_df["state_memory_summary_json"]]
        next_pairs = [extract_next_state_texts(r) for r in chunk_df["next_state_json"]]
        next_turn_texts = [p[0] for p in next_pairs]
        next_mem_texts = [p[1] for p in next_pairs]

        # Replace empty strings for encoding
        turn_texts = [t if t else " " for t in turn_texts]
        mem_texts = [t if t else " " for t in mem_texts]
        next_turn_texts = [t if t else " " for t in next_turn_texts]
        next_mem_texts = [t if t else " " for t in next_mem_texts]

        # Batch encode with sentence-transformer
        turn_emb[start:end] = st_model.encode(
            turn_texts, batch_size=batch_size, show_progress_bar=False
        )
        mem_emb[start:end] = st_model.encode(
            mem_texts, batch_size=batch_size, show_progress_bar=False
        )
        next_turn_emb[start:end] = st_model.encode(
            next_turn_texts, batch_size=batch_size, show_progress_bar=False
        )
        next_mem_emb[start:end] = st_model.encode(
            next_mem_texts, batch_size=batch_size, show_progress_bar=False
        )

        # Hand-crafted features
        for i, (_, row) in enumerate(chunk_df.iterrows()):
            hand_features[start + i] = compute_hand_features(row)
            next_hand_features[start + i] = compute_next_hand_features(row)

        # Scalar fields
        for i, (_, row) in enumerate(chunk_df.iterrows()):
            actions[start + i] = ACTION_INDEX.get(row["action_type"], 1)
            rewards[start + i] = float(row["reward_total"])
            dones[start + i] = float(row["done"])

        # Flush memory-mapped arrays
        turn_emb.flush()
        mem_emb.flush()
        hand_features.flush()
        actions.flush()
        rewards.flush()
        dones.flush()
        next_turn_emb.flush()
        next_mem_emb.flush()
        next_hand_features.flush()

        # Save checkpoint
        checkpoint_path.write_text(json.dumps({"completed_chunks": chunk_idx + 1, "total_rows": n}))

    # Remove checkpoint on completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    logger.info("Completed %s: %d rows", split_name, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for DQN training")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing {train,val,test}.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Output directory for numpy arrays",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer model...")
    st_model = SentenceTransformer("jhgan/ko-sroberta-multitask", device=args.device)

    for split in ["train", "val", "test"]:
        parquet_path = args.splits_dir / f"{split}.parquet"
        if parquet_path.exists():
            process_split(
                split_name=split,
                parquet_path=parquet_path,
                output_dir=args.output_dir,
                st_model=st_model,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
            )
        else:
            logger.warning("Skipping %s: %s not found", split, parquet_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Script 02b: Convert public Korean dialogue data → Episode JSONL.

Converts two public datasets into the Episode JSONL format expected by
the existing pipeline (03_compute_rewards → 04_build_dataset).

Supported datasets:
  1. multi_session_dialogue (76K rows → up to 152K episodes)
  2. korean_role_playing (35K rows → 35K episodes)

Usage:
    uv run scripts/02b_convert_public_data.py [OPTIONS]

Options:
    --output-dir PATH   Output directory (default: data/raw/episodes)
    --max-per-source INT  Max episodes per source file (default: unlimited)
    --seed INT          Random seed (default: 42)
    --dry-run           Count & validate without writing
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aimemory.config import DataPaths
from aimemory.schemas import (
    Episode,
    Role,
    ScenarioType,
    Turn,
)
from aimemory.selfplay.memory_agent import MemoryAgent, MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("02b_convert_public_data")

# ─── Korean text detection (inline to avoid importing ollama) ───

_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")
_MIN_KOREAN_CHARS = 5


def is_korean_text(text: str) -> bool:
    """Check if text contains enough Korean characters."""
    if not text or not text.strip():
        return False
    return len(_HANGUL_RE.findall(text)) >= _MIN_KOREAN_CHARS


# ─── topicType → ScenarioType mapping ───

_TOPIC_TYPE_MAP: dict[str, ScenarioType] = {
    "일과 직업": ScenarioType.CASUAL_CHAT,
    "개인 및 관계": ScenarioType.PERSONAL_PREFERENCES,
    "교육": ScenarioType.LEARNING_TUTORING,
    "식음료": ScenarioType.CASUAL_CHAT,
    "여가와 오락": ScenarioType.CASUAL_CHAT,
    "예술문화 생활": ScenarioType.CASUAL_CHAT,
    "미용과 건강": ScenarioType.CASUAL_CHAT,
    "미용과건강": ScenarioType.CASUAL_CHAT,
    "시사/사회": ScenarioType.CASUAL_CHAT,
    "기후": ScenarioType.CASUAL_CHAT,
    "교통": ScenarioType.CASUAL_CHAT,
    "주거와 생활": ScenarioType.CASUAL_CHAT,
    "상거래 전반": ScenarioType.CASUAL_CHAT,
    "기술": ScenarioType.TECHNICAL_QA,
}


def map_topic_type(topic_type: str) -> ScenarioType:
    """Map a topicType string (e.g. '일과 직업>군대') to ScenarioType."""
    # Try the part before '>'
    major = topic_type.split(">")[0].strip()
    return _TOPIC_TYPE_MAP.get(major, ScenarioType.CASUAL_CHAT)


# ─── Turn construction helpers ───


def make_turn(turn_id: int, role: Role, content: str) -> Turn:
    return Turn(
        turn_id=turn_id,
        role=role,
        content=content.strip(),
        token_count=len(content.split()),
    )


# ─── Dataset-specific converters ───


def parse_multi_session(
    row: dict,
    agent: MemoryAgent,
    max_episodes: int | None = None,
) -> list[Episode]:
    """Convert one multi_session_dialogue row into 1–2 Episodes."""
    episodes: list[Episode] = []
    scenario = map_topic_type(row.get("topicType", ""))
    persona_cl = row.get("personaInfo_cl", [])
    persona_cp = row.get("personaInfo_cp", [])

    for session_key in ("session1", "session2"):
        raw = row.get(session_key, "")
        if not raw:
            continue

        # Parse the session string into a list of utterances
        if isinstance(raw, str):
            try:
                utterances = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
        elif isinstance(raw, list):
            utterances = raw
        else:
            continue

        if not isinstance(utterances, list) or len(utterances) < 4:
            continue

        # Build turns: alternate USER / ASSISTANT
        turns: list[Turn] = []
        has_non_korean = False
        for i, utt in enumerate(utterances):
            if not isinstance(utt, str) or not utt.strip():
                has_non_korean = True
                break
            if not is_korean_text(utt):
                has_non_korean = True
                break
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            turns.append(make_turn(turn_id=i, role=role, content=utt))

        if has_non_korean or len(turns) < 4:
            continue

        # Run MemoryAgent on each turn
        memory_store = MemoryStore()
        decisions = []
        for turn in turns:
            recent = turns[max(0, turn.turn_id - 6) : turn.turn_id]
            decision = agent.decide(turn, memory_store, recent)
            decisions.append(decision)
            if decision.memory_entry is not None:
                memory_store.add(decision.memory_entry)

        metadata: dict = {"source": "multi_session_dialogue", "session": session_key}
        if persona_cl:
            metadata["persona_cl"] = persona_cl if isinstance(persona_cl, list) else str(persona_cl)
        if persona_cp:
            metadata["persona_cp"] = persona_cp if isinstance(persona_cp, list) else str(persona_cp)
        if row.get("topicType"):
            metadata["topic_type"] = row["topicType"]

        ep = Episode(
            scenario=scenario,
            turns=turns,
            memory_decisions=decisions,
            memory_store=memory_store.entries,
            metadata=metadata,
        )
        episodes.append(ep)

    return episodes


def parse_role_playing(
    row: dict,
    agent: MemoryAgent,
    source_name: str,
) -> Episode | None:
    """Convert one korean_role_playing row into an Episode.

    The data has a `text` field containing a list of {role, content} dicts.
    """
    text_field = row.get("text", [])

    # Parse if it's a JSON string
    if isinstance(text_field, str):
        try:
            text_field = json.loads(text_field)
        except json.JSONDecodeError:
            return None

    if not isinstance(text_field, list) or len(text_field) < 4:
        return None

    turns: list[Turn] = []
    for i, msg in enumerate(text_field):
        if not isinstance(msg, dict):
            return None
        content = msg.get("content", "").strip()
        role_str = msg.get("role", "").lower()

        if not content:
            return None
        if not is_korean_text(content):
            return None

        if role_str == "user":
            role = Role.USER
        elif role_str == "assistant":
            role = Role.ASSISTANT
        elif role_str == "system":
            # Skip system messages
            continue
        else:
            return None

        turns.append(make_turn(turn_id=len(turns), role=role, content=content))

    if len(turns) < 4:
        return None

    # Run MemoryAgent
    memory_store = MemoryStore()
    decisions = []
    for turn in turns:
        recent = turns[max(0, turn.turn_id - 6) : turn.turn_id]
        decision = agent.decide(turn, memory_store, recent)
        decisions.append(decision)
        if decision.memory_entry is not None:
            memory_store.add(decision.memory_entry)

    # Determine scenario
    topic = row.get("topic", "")
    scenario = ScenarioType.CASUAL_CHAT
    if topic and ("기술" in topic or "코딩" in topic or "프로그래밍" in topic):
        scenario = ScenarioType.TECHNICAL_QA

    metadata: dict = {"source": source_name}
    if topic:
        metadata["topic"] = topic

    return Episode(
        scenario=scenario,
        turns=turns,
        memory_decisions=decisions,
        memory_store=memory_store.entries,
        metadata=metadata,
    )


# ─── Main ───


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert public Korean dialogue data to Episode JSONL"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        help="Max episodes per source file (0 = unlimited)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def process_multi_session(
    input_path: Path,
    output_path: Path,
    agent: MemoryAgent,
    max_eps: int,
    dry_run: bool,
) -> dict:
    """Process multi_session_dialogue dataset."""
    stats = Counter({"rows": 0, "episodes": 0, "skipped": 0, "turns": 0})
    action_stats = Counter()

    out_f = None if dry_run else open(output_path, "w", encoding="utf-8")

    try:
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["rows"] += 1

                row = json.loads(line)
                episodes = parse_multi_session(row, agent)

                for ep in episodes:
                    if max_eps and stats["episodes"] >= max_eps:
                        break
                    stats["episodes"] += 1
                    stats["turns"] += len(ep.turns)
                    for d in ep.memory_decisions:
                        action_stats[d.action.value] += 1
                    if out_f:
                        out_f.write(ep.model_dump_json() + "\n")

                if max_eps and stats["episodes"] >= max_eps:
                    break

                if stats["rows"] % 10000 == 0:
                    logger.info(
                        "  multi_session: %d rows → %d episodes",
                        stats["rows"],
                        stats["episodes"],
                    )
    finally:
        if out_f:
            out_f.close()

    return {"stats": dict(stats), "actions": dict(action_stats)}


def process_role_playing(
    input_dir: Path,
    output_path: Path,
    agent: MemoryAgent,
    max_eps: int,
    dry_run: bool,
) -> dict:
    """Process korean_role_playing dataset (all JSONL files in directory)."""
    stats = Counter({"rows": 0, "episodes": 0, "skipped": 0, "turns": 0})
    action_stats = Counter()

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL files found in %s", input_dir)
        return {"stats": dict(stats), "actions": dict(action_stats)}

    out_f = None if dry_run else open(output_path, "w", encoding="utf-8")

    try:
        for jsonl_file in jsonl_files:
            source_name = jsonl_file.stem
            logger.info("  Processing %s ...", jsonl_file.name)

            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    stats["rows"] += 1

                    row = json.loads(line)
                    ep = parse_role_playing(row, agent, source_name)

                    if ep is None:
                        stats["skipped"] += 1
                        continue

                    if max_eps and stats["episodes"] >= max_eps:
                        break

                    stats["episodes"] += 1
                    stats["turns"] += len(ep.turns)
                    for d in ep.memory_decisions:
                        action_stats[d.action.value] += 1
                    if out_f:
                        out_f.write(ep.model_dump_json() + "\n")

            if max_eps and stats["episodes"] >= max_eps:
                break
    finally:
        if out_f:
            out_f.close()

    return {"stats": dict(stats), "actions": dict(action_stats)}


def print_summary(label: str, result: dict) -> None:
    stats = result["stats"]
    actions = result["actions"]
    total_decisions = sum(actions.values()) or 1

    logger.info("─── %s ───", label)
    logger.info("  Rows read:    %d", stats.get("rows", 0))
    logger.info("  Episodes:     %d", stats.get("episodes", 0))
    logger.info("  Skipped:      %d", stats.get("skipped", 0))
    logger.info("  Total turns:  %d", stats.get("turns", 0))
    logger.info("  Action distribution:")
    for action_name in ("save", "retrieve", "skip"):
        count = actions.get(action_name, 0)
        pct = count / total_decisions * 100
        logger.info("    %-10s %6d  (%.1f%%)", action_name, count, pct)


def main() -> None:
    args = parse_args()
    paths = DataPaths()
    output_dir: Path = args.output_dir or paths.raw_episodes
    output_dir.mkdir(parents=True, exist_ok=True)

    public_dir = paths.root / "raw" / "public"
    multi_session_path = public_dir / "multi_session_dialogue" / "train.jsonl"
    role_playing_dir = public_dir / "korean_role_playing"

    agent = MemoryAgent(seed=args.seed)
    max_eps = args.max_per_source or 0

    start = time.time()

    # 1. multi_session_dialogue
    if multi_session_path.exists():
        logger.info("Converting multi_session_dialogue ...")
        ms_out = output_dir / "episodes_public_multi_session.jsonl"
        ms_result = process_multi_session(multi_session_path, ms_out, agent, max_eps, args.dry_run)
        print_summary("multi_session_dialogue", ms_result)
    else:
        logger.warning("Not found: %s", multi_session_path)
        ms_result = None

    # 2. korean_role_playing
    if role_playing_dir.exists():
        logger.info("Converting korean_role_playing ...")
        rp_out = output_dir / "episodes_public_role_playing.jsonl"
        rp_result = process_role_playing(role_playing_dir, rp_out, agent, max_eps, args.dry_run)
        print_summary("korean_role_playing", rp_result)
    else:
        logger.warning("Not found: %s", role_playing_dir)
        rp_result = None

    elapsed = time.time() - start

    # Overall summary
    logger.info("=" * 60)
    logger.info("Conversion complete in %.1f min", elapsed / 60)

    total_eps = 0
    if ms_result:
        total_eps += ms_result["stats"].get("episodes", 0)
    if rp_result:
        total_eps += rp_result["stats"].get("episodes", 0)

    logger.info("Total episodes: %d", total_eps)
    if not args.dry_run:
        logger.info("Output files in: %s", output_dir)
        logger.info("  → Run scripts/03_compute_rewards.py next")


if __name__ == "__main__":
    main()

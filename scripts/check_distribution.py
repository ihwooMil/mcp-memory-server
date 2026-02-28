"""Generate episodes and check action distribution."""
from datetime import datetime
from pathlib import Path

from aimemory.config import AppConfig
from aimemory.selfplay.engine import SelfPlayEngine
from aimemory.schemas import MemoryActionType
from collections import Counter

config = AppConfig()
engine = SelfPlayEngine(config=config, seed=42)

# Output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("data/raw/episodes") / f"batch_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

episodes = []
all_actions = Counter()

for i in range(10):
    scenario = engine.scenario_mgr.random_scenario()
    ep = engine.run_episode(scenario, episode_index=i)
    episodes.append(ep)

    saves = sum(1 for d in ep.memory_decisions if d.action == MemoryActionType.SAVE)
    retrieves = sum(1 for d in ep.memory_decisions if d.action == MemoryActionType.RETRIEVE)
    skips = sum(1 for d in ep.memory_decisions if d.action == MemoryActionType.SKIP)
    # Save episode to file
    ep_path = output_dir / f"episode_{i:04d}.json"
    ep_path.write_text(ep.model_dump_json(indent=2), encoding="utf-8")

    print(f"Ep {i:2d}: {ep.num_turns:2d} turns | SAVE={saves:2d} RETRIEVE={retrieves:2d} SKIP={skips:2d} | {ep.scenario}")

    for d in ep.memory_decisions:
        all_actions[d.action] += 1

total = sum(all_actions.values())
print(f"\n=== Overall Distribution ({total} decisions) ===")
for action in [MemoryActionType.SAVE, MemoryActionType.RETRIEVE, MemoryActionType.SKIP]:
    count = all_actions.get(action, 0)
    pct = count / total * 100 if total else 0
    bar = "█" * int(pct / 2)
    print(f"{action.value:>8s}: {count:3d} ({pct:5.1f}%) {bar}")

print(f"\nSaved to: {output_dir}")

# User-only stats (excluding assistant SKIP)
user_actions = Counter()
for ep in episodes:
    for d in ep.memory_decisions:
        reasoning = d.reasoning or ""
        if d.action != MemoryActionType.SKIP or "어시스턴트" not in reasoning:
            user_actions[d.action] += 1

user_total = sum(user_actions.values())
print(f"\n=== User-Turn Only ({user_total} decisions) ===")
for action in [MemoryActionType.SAVE, MemoryActionType.RETRIEVE, MemoryActionType.SKIP]:
    count = user_actions.get(action, 0)
    pct = count / user_total * 100 if user_total else 0
    bar = "█" * int(pct / 2)
    print(f"{action.value:>8s}: {count:3d} ({pct:5.1f}%) {bar}")

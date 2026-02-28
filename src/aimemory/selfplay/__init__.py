"""Self-play module for AI Memory System.

Note: SelfPlayEngine, LLMClient, and ScenarioManager require the 'train'
optional dependency group (ollama). Import them directly from their modules
or use the lazy imports below.
"""

from aimemory.selfplay.memory_agent import MemoryAgent, MemoryStore, extract_keywords


def __getattr__(name: str):
    if name == "SelfPlayEngine":
        from aimemory.selfplay.engine import SelfPlayEngine
        return SelfPlayEngine
    if name == "LLMClient":
        from aimemory.selfplay.llm_client import LLMClient
        return LLMClient
    if name == "ScenarioManager":
        from aimemory.selfplay.scenarios import ScenarioManager
        return ScenarioManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SelfPlayEngine",
    "LLMClient",
    "MemoryAgent",
    "MemoryStore",
    "extract_keywords",
    "ScenarioManager",
]

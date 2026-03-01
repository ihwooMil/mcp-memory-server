"""Memory graph visualization using pyvis.

Generates an interactive HTML visualization of the memory graph,
including memory nodes, relationships, and knowledge graph entity triples.

Usage:
    python -m aimemory.visualize [--db-path PATH] [--output PATH]
                                [--no-browser] [--include-inactive]
    uv run python -m aimemory.visualize --no-browser
"""

from __future__ import annotations

import argparse
import logging
import os
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aimemory.config import PROJECT_ROOT

if TYPE_CHECKING:
    from aimemory.mcp.bridge import MemoryBridge
    from aimemory.memory.graph_store import GraphMemoryStore, MemoryNode
    from aimemory.memory.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Visual encoding constants
CATEGORY_COLORS: dict[str, str] = {
    "fact": "#4C72B0",
    "preference": "#DD8452",
    "experience": "#55A868",
    "emotion": "#C44E52",
    "technical": "#8172B2",
    "core_principle": "#937860",
}
KG_ENTITY_COLOR = "#CCB974"
MEMORY_EDGE_COLOR = "#AAAAAA"
KG_EDGE_COLOR = "#E8A838"
PINNED_BORDER_COLOR = "#FFD700"


def _check_pyvis():
    """Lazy import check for pyvis. Raises ImportError with install hint."""
    try:
        import pyvis  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyvis is required for visualization. "
            "Install it with: pip install 'mcp-memory-server[viz]' "
            "or: uv sync --extra viz"
        )


def _compute_node_size(access_count: int, max_access: int) -> int:
    """Scale node size (15–50px) by access_count relative to max."""
    if max_access <= 0:
        return 15
    ratio = access_count / max_access
    return int(15 + 35 * ratio)


def _truncate_content(content: str, max_chars: int = 30) -> str:
    """Truncate content for use as a node label."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "…"


def _build_node_title(node: MemoryNode) -> str:
    """Build an HTML popup string showing full memory details."""
    lines = [
        f"<b>ID:</b> {node.memory_id}",
        f"<b>Category:</b> {node.category}",
        f"<b>Content:</b> {node.content}",
        f"<b>Keywords:</b> {', '.join(node.keywords)}",
        f"<b>Access Count:</b> {node.access_count}",
        f"<b>Created:</b> {node.created_at}",
        f"<b>Active:</b> {node.active}",
        f"<b>Pinned:</b> {node.pinned}",
        f"<b>Immutable:</b> {node.immutable}",
    ]
    if node.related_ids:
        lines.append(f"<b>Related:</b> {', '.join(node.related_ids)}")
    if node.level1_text:
        lines.append(f"<b>Summary:</b> {node.level1_text}")
    if node.level2_text:
        lines.append(f"<b>Triple:</b> {node.level2_text}")
    return "<br>".join(lines)


def _resolve_output_path(output_path: str | Path | None) -> Path:
    """Resolve the output path, creating directories as needed."""
    if output_path:
        path = Path(output_path)
    else:
        path = PROJECT_ROOT / "data" / "visualizations" / "memory_graph.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_graph(
    store: GraphMemoryStore,
    kg: KnowledgeGraph | None = None,
    include_inactive: bool = False,
    output_path: str | Path | None = None,
) -> str:
    """Build a pyvis Network from the memory store and optional knowledge graph.

    Returns the path to the generated HTML file.
    """
    _check_pyvis()
    from pyvis.network import Network

    net = Network(
        height="100%",
        width="100%",
        directed=True,
        bgcolor="#222222",
        font_color="white",
        select_menu=False,
        filter_menu=False,
    )

    # ForceAtlas2-based physics
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=150,
        spring_strength=0.08,
        damping=0.4,
    )
    net.set_options("""
    {
        "physics": {
            "stabilization": {
                "enabled": true,
                "iterations": 150
            }
        }
    }
    """)

    # Get all memories
    memories = store.get_all_memories(include_inactive=include_inactive)

    # Handle empty graph
    if not memories and (kg is None or kg._graph.number_of_nodes() == 0):
        net.add_node(
            "empty",
            label="No memories found",
            shape="text",
            font={"size": 20, "color": "white"},
        )
        out = _resolve_output_path(output_path)
        net.save_graph(str(out))
        _inject_utf8_meta(out)
        return str(out)

    # Compute max access count for sizing
    max_access = max((m.access_count for m in memories), default=0)

    # Track which memory IDs exist for edge validation
    memory_ids = {m.memory_id for m in memories}

    # Add memory nodes
    for node in memories:
        color = CATEGORY_COLORS.get(node.category, "#999999")
        size = _compute_node_size(node.access_count, max_access)
        label = _truncate_content(node.content)
        title = _build_node_title(node)

        border_color = PINNED_BORDER_COLOR if node.pinned else color
        border_width = 3 if node.pinned else 1

        net.add_node(
            node.memory_id,
            label=label,
            title=title,
            shape="circle" if node.active else "dot",
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": color, "border": "#FFFFFF"},
            },
            size=size,
            borderWidth=border_width,
            font={"size": 10, "color": "white"},
        )

    # Add memory-to-memory edges
    for node in memories:
        for related_id in node.related_ids:
            if related_id in memory_ids:
                net.add_edge(
                    node.memory_id,
                    related_id,
                    color=MEMORY_EDGE_COLOR,
                    width=1,
                    arrows="to",
                )

    # Add KG entities and edges
    if kg is not None:
        kg_entity_ids: set[str] = set()
        for u, v, data in kg._graph.edges(data=True):
            predicate = data.get("predicate", "")
            # Add entity nodes if not already present
            for entity in (u, v):
                entity_id = f"kg:{entity}"
                if entity_id not in kg_entity_ids:
                    kg_entity_ids.add(entity_id)
                    net.add_node(
                        entity_id,
                        label=entity,
                        title=f"<b>KG Entity:</b> {entity}",
                        shape="diamond",
                        color=KG_ENTITY_COLOR,
                        size=10,
                        font={"size": 9, "color": "white"},
                    )
            # KG triple edge (entity → entity)
            net.add_edge(
                f"kg:{u}",
                f"kg:{v}",
                label=predicate,
                color=KG_EDGE_COLOR,
                width=2,
                arrows="to",
                font={"size": 8, "color": KG_EDGE_COLOR},
            )
            # Link KG edge to source memory if it exists
            memory_id = data.get("memory_id", "")
            if memory_id and memory_id in memory_ids:
                net.add_edge(
                    memory_id,
                    f"kg:{u}",
                    color=KG_EDGE_COLOR,
                    width=1,
                    arrows="to",
                    dashes=True,
                )

    out = _resolve_output_path(output_path)
    net.save_graph(str(out))
    _inject_utf8_meta(out)
    return str(out)


def _inject_utf8_meta(path: Path) -> None:
    """Inject UTF-8 meta charset tag into the HTML file for Korean text support."""
    content = path.read_text(encoding="utf-8")
    if '<meta charset="utf-8">' not in content.lower():
        content = content.replace(
            "<head>",
            '<head>\n<meta charset="utf-8">',
            1,
        )
        path.write_text(content, encoding="utf-8")


def visualize_from_bridge(
    bridge: MemoryBridge,
    output_path: str | Path | None = None,
    include_inactive: bool = False,
) -> dict[str, Any]:
    """Bridge-level wrapper that builds the graph and returns a result dict.

    Returns:
        dict with keys: success, file_path, node_count, edge_count, kg_stats
    """
    try:
        _check_pyvis()
    except ImportError as exc:
        return {"success": False, "error": str(exc)}

    store = bridge.store
    kg = bridge._kg

    file_path = build_graph(
        store=store,
        kg=kg,
        include_inactive=include_inactive,
        output_path=output_path,
    )

    memories = store.get_all_memories(include_inactive=include_inactive)
    edge_count = sum(
        len([rid for rid in m.related_ids if rid in {n.memory_id for n in memories}])
        for m in memories
    )

    result: dict[str, Any] = {
        "success": True,
        "file_path": file_path,
        "node_count": len(memories),
        "edge_count": edge_count,
    }

    if kg is not None:
        result["kg_stats"] = kg.stats()

    return result


def main() -> None:
    """CLI entry point for memory graph visualization."""
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML visualization of the memory graph.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to the ChromaDB database directory (default: ./memory_db)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output HTML file path (default: data/visualizations/memory_graph.html)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the generated HTML in the browser",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive (forgotten) memories in the graph",
    )
    args = parser.parse_args()

    _check_pyvis()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Build bridge with optional custom db path
    from aimemory.mcp.bridge import MemoryBridge

    kwargs: dict[str, Any] = {}
    if args.db_path:
        kwargs["persist_directory"] = args.db_path
    # Enable graph RAG if env var is set
    if os.environ.get("AIMEMORY_GRAPH_RAG") == "1":
        kwargs["use_graph_rag"] = True

    bridge = MemoryBridge(**kwargs)

    result = visualize_from_bridge(
        bridge=bridge,
        output_path=args.output,
        include_inactive=args.include_inactive,
    )

    if result["success"]:
        logger.info("Graph saved to: %s", result["file_path"])
        logger.info(
            "Nodes: %d, Edges: %d",
            result["node_count"],
            result["edge_count"],
        )
        if result.get("kg_stats"):
            logger.info("KG: %s", result["kg_stats"])
        if not args.no_browser:
            webbrowser.open(f"file://{result['file_path']}")
    else:
        logger.error("Visualization failed: %s", result.get("error"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""In-memory knowledge graph built from level2_text triples."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from aimemory.memory.graph_store import GraphMemoryStore, MemoryNode

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """In-memory knowledge graph built from level2_text triples.

    Each edge stores: predicate (relation type) and memory_id (source).
    Supports multi-hop traversal, path queries, and entity context.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    def add_triple(self, subject: str, predicate: str, object_: str, memory_id: str) -> None:
        """Add a single (subject, predicate, object) triple with source memory_id."""
        self._graph.add_edge(
            subject,
            object_,
            predicate=predicate,
            memory_id=memory_id,
        )

    def remove_triples_by_memory(self, memory_id: str) -> int:
        """Remove all triples associated with a memory_id. Returns count removed."""
        edges_to_remove = [
            (u, v) for u, v, d in self._graph.edges(data=True) if d.get("memory_id") == memory_id
        ]
        self._graph.remove_edges_from(edges_to_remove)
        # Clean up isolated nodes
        isolated = list(nx.isolates(self._graph))
        self._graph.remove_nodes_from(isolated)
        return len(edges_to_remove)

    def add_from_memory(self, node: MemoryNode) -> int:
        """Parse level2_text from a MemoryNode and add triples. Returns count added."""
        if not node.level2_text:
            return 0

        count = 0
        # level2_text can contain multiple triples separated by newlines or semicolons
        lines = node.level2_text.replace(";", "\n").split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                subject, predicate, object_ = parts[0], parts[1], parts[2]
                if subject and predicate and object_:
                    self.add_triple(subject, predicate, object_, node.memory_id)
                    count += 1
        return count

    def get_related_entities(self, entity: str, depth: int = 2) -> list[tuple[str, str, str]]:
        """Get all triples reachable from entity within depth hops.

        Returns list of (subject, predicate, object) tuples.
        """
        if entity not in self._graph:
            return []

        result: list[tuple[str, str, str]] = []
        visited_edges: set[tuple[str, str]] = set()

        # BFS from entity
        current_level = {entity}
        for _ in range(depth):
            next_level: set[str] = set()
            for node in current_level:
                # Outgoing edges
                for _, target, data in self._graph.out_edges(node, data=True):
                    edge_key = (node, target)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        result.append((node, data.get("predicate", ""), target))
                        next_level.add(target)
                # Incoming edges
                for source, _, data in self._graph.in_edges(node, data=True):
                    edge_key = (source, node)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        result.append((source, data.get("predicate", ""), node))
                        next_level.add(source)
            current_level = next_level

        return result

    def get_memory_ids_for_entity(self, entity: str) -> set[str]:
        """Get all memory IDs associated with an entity (as subject or object)."""
        if entity not in self._graph:
            return set()

        ids: set[str] = set()
        for _, _, data in self._graph.out_edges(entity, data=True):
            mid = data.get("memory_id")
            if mid:
                ids.add(mid)
        for _, _, data in self._graph.in_edges(entity, data=True):
            mid = data.get("memory_id")
            if mid:
                ids.add(mid)
        return ids

    def query_path(self, start: str, end: str) -> list[tuple[str, str, str]] | None:
        """Find shortest path between two entities. Returns list of triples or None."""
        if start not in self._graph or end not in self._graph:
            return None

        try:
            # Use undirected view for path finding
            undirected = self._graph.to_undirected()
            path_nodes = nx.shortest_path(undirected, start, end)
        except nx.NetworkXNoPath:
            return None

        result: list[tuple[str, str, str]] = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            # Check directed edges in both directions
            if self._graph.has_edge(u, v):
                data = self._graph.edges[u, v]
                result.append((u, data.get("predicate", ""), v))
            elif self._graph.has_edge(v, u):
                data = self._graph.edges[v, u]
                result.append((v, data.get("predicate", ""), u))

        return result

    def get_entity_context(self, entity: str) -> dict:
        """Get summary of entity's relationships."""
        if entity not in self._graph:
            return {"entity": entity, "relations": [], "neighbor_count": 0}

        relations: list[dict] = []
        for _, target, data in self._graph.out_edges(entity, data=True):
            relations.append(
                {
                    "direction": "outgoing",
                    "predicate": data.get("predicate", ""),
                    "target": target,
                    "memory_id": data.get("memory_id", ""),
                }
            )
        for source, _, data in self._graph.in_edges(entity, data=True):
            relations.append(
                {
                    "direction": "incoming",
                    "predicate": data.get("predicate", ""),
                    "source": source,
                    "memory_id": data.get("memory_id", ""),
                }
            )

        neighbors = set(self._graph.successors(entity)) | set(self._graph.predecessors(entity))

        return {
            "entity": entity,
            "relations": relations,
            "neighbor_count": len(neighbors),
        }

    def rebuild_from_store(self, store: GraphMemoryStore) -> int:
        """Rebuild entire graph from a GraphMemoryStore. Returns total triples added."""
        self._graph.clear()
        total = 0
        for node in store.get_all_memories(include_inactive=True):
            total += self.add_from_memory(node)
        return total

    def stats(self) -> dict:
        """Return graph statistics."""
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "components": nx.number_weakly_connected_components(self._graph),
        }

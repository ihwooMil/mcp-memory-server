"""ChromaDB-based graph memory store.

Provides semantic search via sentence-transformer embeddings
and graph traversal via metadata-stored edges (related_ids).
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"fact", "preference", "experience", "emotion", "technical", "core_principle"}


class ImmutableMemoryError(Exception):
    """Raised when attempting to modify or delete an immutable memory node."""

# Default Korean-optimized embedding model
DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"


@dataclass
class MemoryNode:
    """A single memory node in the graph."""

    memory_id: str
    content: str
    keywords: list[str] = field(default_factory=list)
    category: str = "fact"
    related_ids: list[str] = field(default_factory=list)
    created_at: str = ""  # ISO format string for serialization
    similarity_score: Optional[float] = None
    immutable: bool = False
    conversation_id: str = ""
    access_count: int = 0
    level1_text: str = ""
    level2_text: str = ""
    active: bool = True
    pinned: bool = False


class GraphMemoryStore:
    """ChromaDB-backed memory store with graph relationship support.

    Each memory is stored as a ChromaDB document with:
    - document: the memory content text
    - metadata: keywords (comma-sep), category, related_ids (comma-sep), created_at
    - embedding: auto-generated via SentenceTransformerEmbeddingFunction
    """

    def __init__(
        self,
        persist_directory: str = "./memory_db",
        collection_name: str = "memories",
        embedding_model: str = DEFAULT_MODEL,
    ) -> None:
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write operations ──────────────────────────────────────────

    def add_memory(
        self,
        content: str,
        keywords: list[str] | None = None,
        category: str = "fact",
        source_turn_id: int | None = None,
        related_ids: list[str] | None = None,
        immutable: bool = False,
        conversation_id: str = "",
        level1_text: str = "",
        level2_text: str = "",
        pinned: bool = False,
    ) -> str:
        """Add a new memory node. Returns the generated memory_id.

        Args:
            immutable: If True, this memory cannot be updated or deleted.
                       Use for core principles or confirmed facts.
            conversation_id: Episode / conversation ID for linking.
            level1_text: Summary text (Level 1 resolution).
            level2_text: Entity triple text (Level 2 resolution).
            pinned: If True, protected from forgetting pipeline.
        """
        memory_id = uuid.uuid4().hex[:12]
        keywords = keywords or []
        related_ids = related_ids or []
        now = datetime.now().isoformat()

        if category not in VALID_CATEGORIES:
            category = "fact"

        metadata: dict = {
            "keywords": ",".join(keywords),
            "category": category,
            "related_ids": ",".join(related_ids),
            "created_at": now,
            "immutable": "true" if immutable else "false",
            "conversation_id": conversation_id,
            "access_count": 0,
            "level1_text": level1_text,
            "level2_text": level2_text,
            "active": "true",
            "pinned": "true" if pinned else "false",
        }
        if source_turn_id is not None:
            metadata["source_turn_id"] = source_turn_id

        self._collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata],
        )

        # Establish bidirectional edges: update related nodes to include this node
        for rid in related_ids:
            self._add_edge(rid, memory_id)

        return memory_id

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        keywords: list[str] | None = None,
    ) -> bool:
        """Update content and/or keywords of an existing memory. Returns True on success.

        Raises:
            ImmutableMemoryError: If the memory is marked as immutable.
        """
        existing = self._collection.get(ids=[memory_id], include=["documents", "metadatas"])
        if not existing["ids"]:
            return False

        meta = existing["metadatas"][0]
        if meta.get("immutable") == "true":
            raise ImmutableMemoryError(
                f"Cannot update immutable memory: {memory_id}"
            )

        doc = content if content is not None else existing["documents"][0]
        if keywords is not None:
            meta["keywords"] = ",".join(keywords)

        self._collection.update(
            ids=[memory_id],
            documents=[doc],
            metadatas=[meta],
        )
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory node and clean up edges pointing to it.

        Raises:
            ImmutableMemoryError: If the memory is marked as immutable.
        """
        existing = self._collection.get(ids=[memory_id], include=["metadatas"])
        if not existing["ids"]:
            return False

        if existing["metadatas"][0].get("immutable") == "true":
            raise ImmutableMemoryError(
                f"Cannot delete immutable memory: {memory_id}"
            )

        # Remove this node from related nodes' related_ids
        meta = existing["metadatas"][0]
        related_ids = _parse_csv(meta.get("related_ids", ""))
        for rid in related_ids:
            self._remove_edge(rid, memory_id)

        # Also remove references to this node from any node that points to it
        # (scan all nodes that might reference this memory_id)
        all_nodes = self._collection.get(
            where={"related_ids": {"$ne": ""}},
            include=["metadatas"],
        )
        for nid, nmeta in zip(all_nodes["ids"], all_nodes["metadatas"]):
            if nid == memory_id:
                continue
            rids = _parse_csv(nmeta.get("related_ids", ""))
            if memory_id in rids:
                self._remove_edge(nid, memory_id)

        self._collection.delete(ids=[memory_id])
        return True

    # ── Read operations ───────────────────────────────────────────

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        category_filter: str | None = None,
        include_inactive: bool = False,
    ) -> list[MemoryNode]:
        """Semantic search for memories similar to query_text.

        Args:
            include_inactive: If False (default), only returns active memories.
        """
        where_conditions: list[dict] = []
        if not include_inactive:
            where_conditions.append({"active": "true"})
        if category_filter and category_filter in VALID_CATEGORIES:
            where_conditions.append({"category": category_filter})

        where: dict | None = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        results = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        nodes: list[MemoryNode] = []
        if not results["ids"] or not results["ids"][0]:
            return nodes

        for mid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            similarity = 1.0 - (dist / 2.0)
            nodes.append(_meta_to_node(mid, doc, meta, similarity))

            # Auto-increment access_count
            current_count = int(meta.get("access_count", 0))
            meta["access_count"] = current_count + 1
            self._collection.update(ids=[mid], metadatas=[meta])

        return nodes

    def get_related(self, memory_id: str, depth: int = 1) -> list[MemoryNode]:
        """BFS traversal of graph edges up to `depth` hops from memory_id."""
        visited: set[str] = {memory_id}
        queue: deque[tuple[str, int]] = deque()
        result_nodes: list[MemoryNode] = []

        # Seed the BFS with direct neighbors
        seed = self._collection.get(ids=[memory_id], include=["metadatas"])
        if not seed["ids"]:
            return []

        for rid in _parse_csv(seed["metadatas"][0].get("related_ids", "")):
            if rid not in visited:
                queue.append((rid, 1))
                visited.add(rid)

        while queue:
            current_id, current_depth = queue.popleft()
            fetched = self._collection.get(
                ids=[current_id], include=["documents", "metadatas"]
            )
            if not fetched["ids"]:
                continue

            node = _meta_to_node(current_id, fetched["documents"][0], fetched["metadatas"][0])
            result_nodes.append(node)

            if current_depth < depth:
                for rid in node.related_ids:
                    if rid not in visited:
                        queue.append((rid, current_depth + 1))
                        visited.add(rid)

        return result_nodes

    def get_stats(self) -> dict:
        """Return statistics about the memory store."""
        count = self._collection.count()
        if count == 0:
            return {"total": 0, "categories": {}}

        all_data = self._collection.get(include=["metadatas"])
        cat_counts: dict[str, int] = {}
        for meta in all_data["metadatas"]:
            cat = meta.get("category", "fact")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {"total": count, "categories": cat_counts}

    def pin_memory(self, memory_id: str) -> bool:
        """Mark a memory as pinned (protected from forgetting)."""
        existing = self._collection.get(ids=[memory_id], include=["metadatas"])
        if not existing["ids"]:
            return False
        meta = existing["metadatas"][0]
        meta["pinned"] = "true"
        self._collection.update(ids=[memory_id], metadatas=[meta])
        return True

    def unpin_memory(self, memory_id: str) -> bool:
        """Remove pin protection from a memory."""
        existing = self._collection.get(ids=[memory_id], include=["metadatas"])
        if not existing["ids"]:
            return False
        meta = existing["metadatas"][0]
        meta["pinned"] = "false"
        self._collection.update(ids=[memory_id], metadatas=[meta])
        return True

    def deactivate_memory(self, memory_id: str) -> bool:
        """Set a memory as inactive (excluded from search)."""
        existing = self._collection.get(ids=[memory_id], include=["metadatas"])
        if not existing["ids"]:
            return False
        meta = existing["metadatas"][0]
        meta["active"] = "false"
        self._collection.update(ids=[memory_id], metadatas=[meta])
        return True

    def get_all_memories(self, include_inactive: bool = False) -> list[MemoryNode]:
        """Return all memory nodes in the store."""
        all_data = self._collection.get(include=["documents", "metadatas"])
        nodes: list[MemoryNode] = []
        for mid, doc, meta in zip(
            all_data["ids"], all_data["documents"], all_data["metadatas"]
        ):
            node = _meta_to_node(mid, doc, meta)
            if not include_inactive and not node.active:
                continue
            nodes.append(node)
        return nodes

    def compress_memory(self, memory_id: str, level2_text: str) -> bool:
        """Compress a memory to Level 2 (entity triple) only."""
        existing = self._collection.get(ids=[memory_id], include=["documents", "metadatas"])
        if not existing["ids"]:
            return False
        meta = existing["metadatas"][0]
        meta["level2_text"] = level2_text
        self._collection.update(ids=[memory_id], metadatas=[meta])
        return True

    # ── Migration ─────────────────────────────────────────────────

    @classmethod
    def from_legacy_store(
        cls,
        old_store,
        persist_directory: str = "./memory_db",
        collection_name: str = "memories",
        embedding_model: str = DEFAULT_MODEL,
    ) -> GraphMemoryStore:
        """Migrate entries from a legacy MemoryStore (flat list) to GraphMemoryStore.

        Args:
            old_store: An instance of aimemory.selfplay.memory_agent.MemoryStore
            persist_directory: ChromaDB persistence path
            collection_name: ChromaDB collection name
            embedding_model: SentenceTransformer model name

        Returns:
            A new GraphMemoryStore populated with the migrated entries.
        """
        new_store = cls(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )

        # Map legacy categories to valid graph categories
        category_map = {
            "general": "fact",
            "personal": "fact",
            "technical": "technical",
            "preference": "preference",
            "emotion": "emotion",
            "experience": "experience",
            "fact": "fact",
        }

        for entry in old_store.entries:
            category = category_map.get(entry.category, "fact")
            new_store.add_memory(
                content=entry.content,
                keywords=entry.keywords,
                category=category,
                source_turn_id=entry.source_turn_id,
            )

        return new_store

    # ── Private helpers ───────────────────────────────────────────

    def _add_edge(self, target_id: str, new_related_id: str) -> None:
        """Add new_related_id to target_id's related_ids list."""
        existing = self._collection.get(ids=[target_id], include=["metadatas"])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        rids = _parse_csv(meta.get("related_ids", ""))
        if new_related_id not in rids:
            rids.append(new_related_id)
            meta["related_ids"] = ",".join(rids)
            self._collection.update(ids=[target_id], metadatas=[meta])

    def _remove_edge(self, target_id: str, remove_id: str) -> None:
        """Remove remove_id from target_id's related_ids list."""
        existing = self._collection.get(ids=[target_id], include=["metadatas"])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        rids = _parse_csv(meta.get("related_ids", ""))
        if remove_id in rids:
            rids.remove(remove_id)
            meta["related_ids"] = ",".join(rids)
            self._collection.update(ids=[target_id], metadatas=[meta])


def _parse_csv(value: str) -> list[str]:
    """Parse a comma-separated string into a list, filtering empty strings."""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _meta_to_node(
    memory_id: str,
    document: str,
    metadata: dict,
    similarity: float | None = None,
) -> MemoryNode:
    """Convert ChromaDB metadata dict to a MemoryNode."""
    return MemoryNode(
        memory_id=memory_id,
        content=document,
        keywords=_parse_csv(metadata.get("keywords", "")),
        category=metadata.get("category", "fact"),
        related_ids=_parse_csv(metadata.get("related_ids", "")),
        created_at=metadata.get("created_at", ""),
        similarity_score=similarity,
        immutable=metadata.get("immutable") == "true",
        conversation_id=metadata.get("conversation_id", ""),
        access_count=int(metadata.get("access_count", 0)),
        level1_text=metadata.get("level1_text", ""),
        level2_text=metadata.get("level2_text", ""),
        active=metadata.get("active", "true") == "true",
        pinned=metadata.get("pinned", "false") == "true",
    )

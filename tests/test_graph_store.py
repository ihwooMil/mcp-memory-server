"""Tests for GraphMemoryStore (ChromaDB-based graph memory)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aimemory.memory.graph_store import GraphMemoryStore, ImmutableMemoryError
from aimemory.schemas import MemoryEntry
from aimemory.selfplay.memory_agent import MemoryStore


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    """Return a temporary directory for ChromaDB persistence."""
    return str(tmp_path / "test_memory_db")


@pytest.fixture()
def store(tmp_db: str) -> GraphMemoryStore:
    """Create a fresh GraphMemoryStore for each test."""
    return GraphMemoryStore(persist_directory=tmp_db)


# ── Add and Search ────────────────────────────────────────────


class TestAddAndSearch:
    def test_add_returns_memory_id(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(
            content="파이썬은 인터프리터 언어입니다.",
            keywords=["파이썬", "인터프리터"],
            category="technical",
        )
        assert isinstance(mid, str)
        assert len(mid) == 12

    def test_search_finds_relevant_memory(self, store: GraphMemoryStore) -> None:
        store.add_memory(
            content="저는 커피를 좋아합니다.",
            keywords=["커피", "좋아"],
            category="preference",
        )
        store.add_memory(
            content="오늘 날씨가 좋습니다.",
            keywords=["날씨"],
            category="fact",
        )
        results = store.search("커피 맛있어요", top_k=2)
        assert len(results) >= 1
        assert any("커피" in n.content for n in results)

    def test_search_returns_similarity_scores(self, store: GraphMemoryStore) -> None:
        store.add_memory(content="Django는 웹 프레임워크입니다.", keywords=["Django"])
        results = store.search("웹 프레임워크")
        assert len(results) >= 1
        assert results[0].similarity_score is not None
        assert 0.0 <= results[0].similarity_score <= 1.0

    def test_search_empty_store(self, store: GraphMemoryStore) -> None:
        results = store.search("아무거나")
        assert results == []

    def test_search_category_filter(self, store: GraphMemoryStore) -> None:
        store.add_memory(content="React 컴포넌트 만들기", category="technical")
        store.add_memory(content="초콜릿 좋아해요", category="preference")
        results = store.search("좋아하는 것", category_filter="preference")
        assert all(n.category == "preference" for n in results)

    def test_invalid_category_defaults_to_fact(self, store: GraphMemoryStore) -> None:
        store.add_memory(content="테스트", category="invalid_cat")
        results = store.search("테스트")
        assert results[0].category == "fact"


# ── Graph Traversal ───────────────────────────────────────────


class TestGetRelated:
    def test_direct_neighbors(self, store: GraphMemoryStore) -> None:
        m1 = store.add_memory(content="파이썬 기초", keywords=["파이썬"])
        m2 = store.add_memory(
            content="파이썬 데코레이터", keywords=["파이썬", "데코레이터"], related_ids=[m1]
        )
        related = store.get_related(m1, depth=1)
        assert len(related) == 1
        assert related[0].memory_id == m2

    def test_bidirectional_edges(self, store: GraphMemoryStore) -> None:
        m1 = store.add_memory(content="노드 A")
        m2 = store.add_memory(content="노드 B", related_ids=[m1])
        # m1 -> m2 and m2 -> m1 should both work
        assert any(n.memory_id == m2 for n in store.get_related(m1))
        assert any(n.memory_id == m1 for n in store.get_related(m2))

    def test_depth_two_traversal(self, store: GraphMemoryStore) -> None:
        m1 = store.add_memory(content="루트 노드")
        m2 = store.add_memory(content="중간 노드", related_ids=[m1])
        m3 = store.add_memory(content="말단 노드", related_ids=[m2])

        # depth=1: only m2
        depth1 = store.get_related(m1, depth=1)
        assert len(depth1) == 1
        assert depth1[0].memory_id == m2

        # depth=2: m2 and m3
        depth2 = store.get_related(m1, depth=2)
        ids = {n.memory_id for n in depth2}
        assert m2 in ids
        assert m3 in ids

    def test_get_related_nonexistent(self, store: GraphMemoryStore) -> None:
        result = store.get_related("nonexistent_id")
        assert result == []


# ── Update and Delete ─────────────────────────────────────────


class TestUpdateAndDelete:
    def test_update_content(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(content="원래 내용", keywords=["원래"])
        success = store.update_memory(mid, content="수정된 내용")
        assert success is True
        results = store.search("수정된 내용")
        assert any(n.content == "수정된 내용" for n in results)

    def test_update_keywords(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(content="테스트 내용", keywords=["old"])
        store.update_memory(mid, keywords=["new", "updated"])
        results = store.search("테스트 내용")
        node = next(n for n in results if n.memory_id == mid)
        assert "new" in node.keywords
        assert "updated" in node.keywords

    def test_update_nonexistent_returns_false(self, store: GraphMemoryStore) -> None:
        assert store.update_memory("nonexistent") is False

    def test_delete_memory(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(content="삭제할 메모리")
        assert store.delete_memory(mid) is True
        stats = store.get_stats()
        assert stats["total"] == 0

    def test_delete_cleans_edges(self, store: GraphMemoryStore) -> None:
        m1 = store.add_memory(content="노드 1")
        m2 = store.add_memory(content="노드 2", related_ids=[m1])
        store.delete_memory(m2)
        related = store.get_related(m1)
        assert len(related) == 0

    def test_delete_nonexistent_returns_false(self, store: GraphMemoryStore) -> None:
        assert store.delete_memory("nonexistent") is False


# ── Stats ─────────────────────────────────────────────────────


class TestStats:
    def test_empty_stats(self, store: GraphMemoryStore) -> None:
        stats = store.get_stats()
        assert stats["total"] == 0
        assert stats["categories"] == {}

    def test_stats_with_data(self, store: GraphMemoryStore) -> None:
        store.add_memory(content="기술 메모 1", category="technical")
        store.add_memory(content="기술 메모 2", category="technical")
        store.add_memory(content="선호 메모", category="preference")
        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["categories"]["technical"] == 2
        assert stats["categories"]["preference"] == 1


# ── Persistence ───────────────────────────────────────────────


class TestPersistence:
    def test_data_survives_reload(self, tmp_db: str) -> None:
        store1 = GraphMemoryStore(persist_directory=tmp_db)
        store1.add_memory(content="영속성 테스트 데이터", keywords=["영속성"])
        del store1

        store2 = GraphMemoryStore(persist_directory=tmp_db)
        results = store2.search("영속성 테스트")
        assert len(results) >= 1
        assert "영속성" in results[0].content


# ── Migration from Legacy MemoryStore ─────────────────────────


class TestMigration:
    def test_from_legacy_store(self, tmp_db: str) -> None:
        legacy = MemoryStore()
        legacy.add(
            MemoryEntry(
                content="사용자는 파이썬을 좋아합니다",
                source_turn_id=1,
                keywords=["파이썬", "좋아"],
                category="preference",
            )
        )
        legacy.add(
            MemoryEntry(
                content="Docker 컨테이너 사용법",
                source_turn_id=3,
                keywords=["Docker", "컨테이너"],
                category="technical",
            )
        )

        new_store = GraphMemoryStore.from_legacy_store(legacy, persist_directory=tmp_db)
        stats = new_store.get_stats()
        assert stats["total"] == 2
        assert "preference" in stats["categories"]
        assert "technical" in stats["categories"]

    def test_legacy_general_category_maps_to_fact(self, tmp_db: str) -> None:
        legacy = MemoryStore()
        legacy.add(
            MemoryEntry(
                content="일반 정보",
                source_turn_id=0,
                keywords=["일반"],
                category="general",
            )
        )
        new_store = GraphMemoryStore.from_legacy_store(legacy, persist_directory=tmp_db)
        results = new_store.search("일반 정보")
        assert results[0].category == "fact"

    def test_migrated_data_is_searchable(self, tmp_db: str) -> None:
        legacy = MemoryStore()
        legacy.add(
            MemoryEntry(
                content="React와 TypeScript로 프론트엔드 개발",
                source_turn_id=5,
                keywords=["React", "TypeScript", "프론트엔드"],
                category="technical",
            )
        )
        new_store = GraphMemoryStore.from_legacy_store(legacy, persist_directory=tmp_db)
        results = new_store.search("React 프론트엔드 개발")
        assert len(results) >= 1
        assert "React" in results[0].content


# ── Immutable Memory ─────────────────────────────────────────


class TestImmutableMemory:
    def test_add_immutable_memory(self, store: GraphMemoryStore) -> None:
        store.add_memory(
            content="사용자 개인정보를 외부에 공유하지 않는다",
            keywords=["원칙", "개인정보"],
            category="core_principle",
            immutable=True,
        )
        results = store.search("개인정보 보호 원칙")
        assert len(results) >= 1
        assert results[0].immutable is True

    def test_cannot_update_immutable(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(
            content="변경불가 원칙",
            category="core_principle",
            immutable=True,
        )
        with pytest.raises(ImmutableMemoryError):
            store.update_memory(mid, content="변경 시도")

    def test_cannot_delete_immutable(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(
            content="삭제불가 원칙",
            category="core_principle",
            immutable=True,
        )
        with pytest.raises(ImmutableMemoryError):
            store.delete_memory(mid)

    def test_mutable_still_works(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(content="수정 가능한 기억", immutable=False)
        assert store.update_memory(mid, content="수정된 기억") is True
        assert store.delete_memory(mid) is True


# ── Extended Schema Fields ────────────────────────────────────


class TestExtendedSchema:
    def test_new_fields_stored_and_retrieved(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(
            content="확장 필드 테스트",
            keywords=["확장"],
            conversation_id="ep_001",
            level1_text="확장 테스트 요약",
            level2_text="확장,테스트,필드",
            pinned=True,
        )
        results = store.search("확장 필드 테스트")
        node = next(n for n in results if n.memory_id == mid)
        assert node.conversation_id == "ep_001"
        assert node.level1_text == "확장 테스트 요약"
        assert node.level2_text == "확장,테스트,필드"
        assert node.pinned is True
        assert node.active is True
        # access_count in the returned node reflects the pre-increment value;
        # the DB is updated after node creation. Verify via a second search.
        results2 = store.search("확장 필드 테스트")
        node2 = next(n for n in results2 if n.memory_id == mid)
        assert node2.access_count >= 1  # incremented from first search

    def test_active_filter_excludes_inactive(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(content="비활성 메모리 테스트")
        store.deactivate_memory(mid)
        results = store.search("비활성 메모리 테스트")
        assert not any(n.memory_id == mid for n in results)

    def test_include_inactive_search(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(content="비활성 포함 검색 테스트")
        store.deactivate_memory(mid)
        results = store.search("비활성 포함 검색 테스트", include_inactive=True)
        assert any(n.memory_id == mid for n in results)

    def test_pin_and_unpin(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(content="핀 테스트", pinned=False)
        assert store.pin_memory(mid) is True
        results = store.search("핀 테스트")
        node = next(n for n in results if n.memory_id == mid)
        assert node.pinned is True

        assert store.unpin_memory(mid) is True
        results = store.search("핀 테스트")
        node = next(n for n in results if n.memory_id == mid)
        assert node.pinned is False

    def test_access_count_increments(self, store: GraphMemoryStore) -> None:
        mid = store.add_memory(content="접근 카운트 테스트")
        store.search("접근 카운트 테스트")  # 1st search
        store.search("접근 카운트 테스트")  # 2nd search
        results = store.search("접근 카운트 테스트")  # 3rd search
        node = next(n for n in results if n.memory_id == mid)
        assert node.access_count >= 2  # at least 2 prior increments

    def test_get_all_memories(self, store: GraphMemoryStore) -> None:
        store.add_memory(content="활성 메모리")
        mid2 = store.add_memory(content="비활성 메모리")
        store.deactivate_memory(mid2)

        active_only = store.get_all_memories(include_inactive=False)
        all_memories = store.get_all_memories(include_inactive=True)
        assert len(active_only) == 1
        assert len(all_memories) == 2


# ── E2E Integration ──────────────────────────────────────────


class TestE2EIntegration:
    def test_full_lifecycle(self, store: GraphMemoryStore) -> None:
        """E2E: add with multi-resolution → search → pin → deactivate → search again."""
        mid = store.add_memory(
            content="저는 매일 아침 조깅을 합니다.",
            keywords=["조깅", "아침"],
            category="experience",
            conversation_id="ep_100",
            level1_text="매일 아침 조깅",
            level2_text="저,합니다,조깅",
        )

        # Search finds it
        results = store.search("아침 운동")
        assert any(n.memory_id == mid for n in results)

        # Pin it
        store.pin_memory(mid)
        results = store.search("아침 운동")
        pinned_node = next(n for n in results if n.memory_id == mid)
        assert pinned_node.pinned is True

        # Deactivate it
        store.deactivate_memory(mid)
        results = store.search("아침 운동")
        assert not any(n.memory_id == mid for n in results)

        # Include inactive finds it
        results = store.search("아침 운동", include_inactive=True)
        assert any(n.memory_id == mid for n in results)


# ── KnowledgeGraph Integration ────────────────────────────────


class TestKnowledgeGraphIntegration:
    """KG auto-build integration tests."""

    def test_add_memory_auto_builds_kg(self, tmp_db: str) -> None:
        from aimemory.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        store = GraphMemoryStore(persist_directory=tmp_db, knowledge_graph=kg)
        store.add_memory(
            content="저는 봉골레를 싫어해요",
            keywords=["봉골레"],
            level2_text="사용자,싫어함,봉골레",
        )
        assert kg.stats()["edges"] == 1
        ids = kg.get_memory_ids_for_entity("사용자")
        assert len(ids) == 1

    def test_delete_memory_removes_kg_triples(self, tmp_db: str) -> None:
        from aimemory.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        store = GraphMemoryStore(persist_directory=tmp_db, knowledge_graph=kg)
        mid = store.add_memory(
            content="테스트 내용",
            level2_text="A,관계,B",
        )
        assert kg.stats()["edges"] == 1
        store.delete_memory(mid)
        assert kg.stats()["edges"] == 0

    def test_kg_none_no_error(self, tmp_db: str) -> None:
        """Without KG, add/delete work normally (no error)."""
        store = GraphMemoryStore(persist_directory=tmp_db)
        mid = store.add_memory(content="일반 메모리", level2_text="A,B,C")
        assert store.delete_memory(mid) is True

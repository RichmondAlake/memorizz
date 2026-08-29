import uuid
from typing import Any, Dict, List, Optional

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.memagent.managers.entity_memory_manager import EntityMemoryManager

_UNSET = object()


class InMemoryEntityProvider:
    """Minimal provider that mimics the entity-memory interface."""

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}
        self.vector_available = True
        self.semantic_query_count = 0
        self.store_count = 0
        self.last_filter_limit: Optional[int] = None

    def supports_entity_memory(self) -> bool:
        return True

    def store(self, data: Dict[str, Any], memory_store_type: MemoryType, **_) -> str:
        assert memory_store_type == MemoryType.ENTITY_MEMORY
        record = dict(data)
        record.setdefault("_id", record.get("entity_id", str(uuid.uuid4())))
        entity_id = record["entity_id"]

        self.records[entity_id] = record
        self.store_count += 1

        return record["_id"]

    def retrieve_by_query(
        self,
        query: Any,
        memory_type: MemoryType,
        limit: int = 5,
        memory_id: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        assert memory_type == MemoryType.ENTITY_MEMORY
        user_id = kwargs.get("user_id", _UNSET)

        if isinstance(query, dict):
            self.last_filter_limit = limit
            candidates = [
                rec
                for rec in self.records.values()
                if all(rec.get(key) == value for key, value in query.items())
                and (memory_id is None or rec.get("memory_id") == memory_id)
            ]
        else:
            self.semantic_query_count += 1
            if not self.vector_available:
                return []
            candidates = [
                rec
                for rec in self.records.values()
                if memory_id is None or rec.get("memory_id") == memory_id
            ]
        if user_id is not _UNSET:
            candidates = [rec for rec in candidates if rec.get("user_id") == user_id]
        return candidates[:limit]

    def list_all(
        self, memory_store_type: MemoryType, user_id: Any = _UNSET
    ) -> List[Dict[str, Any]]:
        assert memory_store_type == MemoryType.ENTITY_MEMORY
        records = [dict(rec) for rec in self.records.values()]
        if user_id is not _UNSET:
            records = [rec for rec in records if rec.get("user_id") == user_id]
        return records

    def migrate_entity_memory_user_scope(self, *, memory_id: str, user_id: str) -> int:
        migrated = 0
        for record in self.records.values():
            if record.get("memory_id") != memory_id:
                continue
            if record.get("user_id") is not None:
                continue
            record["user_id"] = user_id
            migrated += 1
        return migrated

    def get_vector_search_status(self, memory_store_type: MemoryType):
        assert memory_store_type == MemoryType.ENTITY_MEMORY
        if self.vector_available:
            return {"available": True, "queryable": True, "status": "READY"}
        return {
            "available": False,
            "queryable": False,
            "reason": "vector_index_missing",
        }


@pytest.fixture()
def provider() -> InMemoryEntityProvider:
    return InMemoryEntityProvider()


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """Use deterministic embeddings so tests don't hit external services."""

    def _fake_embedding(text: str) -> List[float]:
        return [float(len(text or ""))]

    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        _fake_embedding,
    )


@pytest.fixture()
def entity_store(provider: InMemoryEntityProvider) -> EntityMemory:
    return EntityMemory(provider)


def test_upsert_merges_attributes(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_id = entity_store.upsert_entity(
        name="Avery",
        entity_type="customer",
        attributes=[{"name": "language", "value": "English"}],
        memory_id="tenant-1",
    )

    entity_store.upsert_entity(
        entity_id=entity_id,
        attributes=[{"name": "timezone", "value": "PST"}],
        memory_id="tenant-1",
    )

    assert len(provider.records) == 1
    first_record = next(iter(provider.records.values()))
    stored_attrs = {attr["name"]: attr["value"] for attr in first_record["attributes"]}
    assert stored_attrs == {"language": "English", "timezone": "PST"}


def test_upsert_canonicalizes_common_attribute_name_aliases(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_store.upsert_entity(
        name="Richmond Alake",
        entity_type="person",
        attributes=[
            {"attribute": "name", "value": "Richmond Alake"},
            {"attribute_name": "role", "value": "AI Memory Engineer"},
        ],
        memory_id="primary-user",
    )

    stored = next(iter(provider.records.values()))
    assert stored["attributes"] == [
        {
            "name": "name",
            "value": "Richmond Alake",
            "confidence": 0.8,
            "source": None,
            "created_at": stored["attributes"][0]["created_at"],
            "updated_at": stored["attributes"][0]["updated_at"],
        },
        {
            "name": "role",
            "value": "AI Memory Engineer",
            "confidence": 0.8,
            "source": None,
            "created_at": stored["attributes"][1]["created_at"],
            "updated_at": stored["attributes"][1]["updated_at"],
        },
    ]


def test_identity_key_reuses_one_canonical_entity_across_name_changes(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    first_id = entity_store.upsert_entity(
        identity_key="authenticated_user",
        name="user",
        entity_type="person",
        attributes=[{"name": "role", "value": "AI Memory Engineer"}],
        memory_id="primary-user",
        user_id="user-1",
    )
    second_id = entity_store.upsert_entity(
        identity_key="authenticated_user",
        name="Richmond Alake",
        entity_type="person",
        attributes=[{"name": "timezone", "value": "Europe/London"}],
        memory_id="primary-user",
        user_id="user-1",
    )

    assert second_id == first_id
    assert len(provider.records) == 1
    stored = provider.records[first_id]
    assert stored["_id"] == first_id
    assert stored["metadata"]["identity_key"] == "authenticated_user"
    assert {item["name"] for item in stored["attributes"]} == {"role", "timezone"}


def test_identity_key_overrides_a_stale_duplicate_entity_id(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    canonical_id = entity_store.upsert_entity(
        identity_key="authenticated_user",
        name="Richmond Alake",
        attributes=[{"name": "role", "value": "AI Memory Engineer"}],
        memory_id="primary-user",
        user_id="user-1",
    )
    duplicate_id = entity_store.upsert_entity(
        name="user",
        attributes=[{"name": "timezone", "value": "Europe/London"}],
        memory_id="primary-user",
        user_id="user-1",
    )

    resolved_id = entity_store.upsert_entity(
        entity_id=duplicate_id,
        identity_key="authenticated_user",
        attributes=[{"name": "audience", "value": "agent engineers"}],
        memory_id="primary-user",
        user_id="user-1",
    )

    assert resolved_id == canonical_id
    assert {item["name"] for item in provider.records[canonical_id]["attributes"]} == {
        "role",
        "audience",
    }
    assert {item["name"] for item in provider.records[duplicate_id]["attributes"]} == {
        "timezone"
    }


def test_duplicate_consolidation_is_dry_run_first_and_soft_supersedes(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    first_id = entity_store.upsert_entity(
        name="Richmond Alake",
        entity_type="person",
        attributes=[{"name": "role", "value": "Engineer", "confidence": 0.7}],
        memory_id="primary-user",
        user_id="user-1",
    )
    second_id = entity_store.upsert_entity(
        name="  richmond   alake  ",
        entity_type="PERSON",
        attributes=[
            {
                "name": "role",
                "value": "AI Memory Engineer",
                "confidence": 0.95,
            },
            {"name": "timezone", "value": "Europe/London"},
        ],
        memory_id="primary-user",
        user_id="user-1",
    )
    assert first_id != second_id

    preview = entity_store.consolidate_duplicate_entities(
        memory_id="primary-user", user_id="user-1"
    )

    assert preview["dry_run"] is True
    assert preview["duplicate_group_count"] == 1
    assert preview["records_to_supersede"] == 1
    assert preview["deleted_count"] == 0
    assert (
        len(entity_store.list_entities(memory_id="primary-user", user_id="user-1")) == 2
    )

    applied = entity_store.consolidate_duplicate_entities(
        memory_id="primary-user", user_id="user-1", apply=True
    )

    assert applied["dry_run"] is False
    assert applied["groups"][0]["applied"] is True
    active = entity_store.list_entities(memory_id="primary-user", user_id="user-1")
    all_rows = entity_store.list_entities(
        memory_id="primary-user", user_id="user-1", include_superseded=True
    )
    assert len(active) == 1
    assert len(all_rows) == 2
    attributes = {item["name"]: item["value"] for item in active[0]["attributes"]}
    assert attributes == {
        "role": "AI Memory Engineer",
        "timezone": "Europe/London",
    }
    superseded = [
        row for row in all_rows if (row.get("metadata") or {}).get("superseded_by")
    ]
    assert len(superseded) == 1


def test_duplicate_consolidation_requires_explicit_legacy_scope_opt_in(
    entity_store: EntityMemory,
):
    with pytest.raises(ValueError, match="allow_legacy_scope"):
        entity_store.consolidate_duplicate_entities(
            memory_id="legacy-memory", user_id=None
        )


def test_cross_name_consolidation_requires_explicit_ids_and_is_reversible(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    generic_id = entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[{"name": "role", "value": "AI Memory Engineer"}],
        memory_id="primary-user",
        user_id="user-1",
    )
    named_id = entity_store.upsert_entity(
        name="Richmond Alake",
        entity_type="person",
        attributes=[{"name": "timezone", "value": "Europe/London"}],
        memory_id="primary-user",
        user_id="user-1",
    )

    automatic = entity_store.consolidate_duplicate_entities(
        memory_id="primary-user", user_id="user-1"
    )
    assert automatic["duplicate_group_count"] == 0

    preview = entity_store.consolidate_duplicate_entities(
        memory_id="primary-user",
        user_id="user-1",
        entity_ids=[generic_id, named_id],
        canonical_identity_key="authenticated_user",
    )
    assert preview["dry_run"] is True
    assert preview["explicit_selection"] is True
    assert preview["groups"][0]["canonical_identity_bound"] is True
    assert (
        len(entity_store.list_entities(memory_id="primary-user", user_id="user-1")) == 2
    )

    applied = entity_store.consolidate_duplicate_entities(
        memory_id="primary-user",
        user_id="user-1",
        entity_ids=[generic_id, named_id],
        canonical_identity_key="authenticated_user",
        apply=True,
    )
    assert applied["records_to_supersede"] == 1
    active = entity_store.list_entities(memory_id="primary-user", user_id="user-1")
    assert len(active) == 1
    assert active[0]["metadata"]["identity_key"] == "authenticated_user"
    assert (
        len(
            entity_store.list_entities(
                memory_id="primary-user",
                user_id="user-1",
                include_superseded=True,
            )
        )
        == 2
    )


def test_explicit_consolidation_fails_closed_for_out_of_scope_id(
    entity_store: EntityMemory,
):
    entity_store.upsert_entity(
        name="user",
        memory_id="primary-user",
        user_id="user-1",
    )
    with pytest.raises(ValueError, match="exact memory/user scope"):
        entity_store.consolidate_duplicate_entities(
            memory_id="primary-user",
            user_id="user-1",
            entity_ids=["missing", "also-missing"],
            canonical_identity_key="authenticated_user",
            apply=True,
        )


def test_record_attribute_creates_entity(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_id = entity_store.record_attribute(
        entity_name="Nova",
        attribute_name="favorite_product",
        attribute_value="Nebula Drone",
        memory_id="tenant-2",
    )

    stored = next(iter(provider.records.values()))
    assert stored["entity_id"] == entity_id
    assert stored["attributes"][0]["name"] == "favorite_product"
    assert stored["attributes"][0]["value"] == "Nebula Drone"


def test_manager_build_context_returns_profiles(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_store.upsert_entity(
        name="Taylor",
        entity_type="analyst",
        attributes=[{"name": "role", "value": "Analyst"}],
        memory_id="team-7",
    )

    manager = EntityMemoryManager(provider)
    profiles = manager.build_context("analyst", memory_id="team-7")

    assert profiles and profiles[0]["attributes"]["role"] == "Analyst"
    summary = manager.summarize_for_prompt(profiles)
    assert "Taylor" in summary
    assert "role: Analyst" in summary


def test_manager_lookup_filters_by_memory_id(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_store.upsert_entity(
        name="Jordan",
        entity_type="user",
        attributes=[{"name": "tier", "value": "gold"}],
        memory_id="org-a",
    )
    entity_store.upsert_entity(
        name="Riley",
        entity_type="user",
        attributes=[{"name": "tier", "value": "silver"}],
        memory_id="org-b",
    )

    assert len(provider.records) == 2
    manager = EntityMemoryManager(provider)
    raw_matches = provider.retrieve_by_query(
        "user", memory_type=MemoryType.ENTITY_MEMORY, memory_id="org-a"
    )
    assert len(raw_matches) == 1
    matches = manager.lookup_entities(query="user", memory_id="org-a")

    assert len(matches) == 1
    assert matches[0]["name"] == "Jordan"


def test_same_generic_entity_name_is_isolated_by_user_with_shared_memory_id(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    first_id = entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[{"name": "tier", "value": "gold"}],
        memory_id="shared-memory",
        user_id="user-a",
    )
    second_id = entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[{"name": "tier", "value": "silver"}],
        memory_id="shared-memory",
        user_id="user-b",
    )

    assert first_id != second_id
    assert len(provider.records) == 2
    manager = EntityMemoryManager(provider)
    first = manager.lookup_entities(
        name="user", memory_id="shared-memory", user_id="user-a"
    )
    second = manager.lookup_entities(
        name="user", memory_id="shared-memory", user_id="user-b"
    )

    assert first[0]["attributes"]["tier"] == "gold"
    assert second[0]["attributes"]["tier"] == "silver"


def test_name_lookup_cannot_cross_memory_scope(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[{"name": "secret", "value": "tenant-a-only"}],
        memory_id="primary-user-a",
        user_id="user-a",
    )

    manager = EntityMemoryManager(provider)
    matches = manager.lookup_entities(
        name="user", memory_id="primary-user-b", user_id="user-b"
    )

    assert matches == []


def test_legacy_null_user_record_requires_explicit_operator_migration(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_id = entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[{"name": "role", "value": "developer advocate"}],
        memory_id="primary-user-a",
        user_id=None,
    )
    manager = EntityMemoryManager(provider)

    authenticated_before = manager.lookup_entities(
        name="user", memory_id="primary-user-a", user_id="user-a"
    )
    anonymous_before = manager.lookup_entities(
        name="user", memory_id="primary-user-a", user_id=None
    )
    migrated = entity_store.migrate_legacy_scope(
        memory_id="primary-user-a", user_id="user-a"
    )
    authenticated_after = manager.lookup_entities(
        name="user", memory_id="primary-user-a", user_id="user-a"
    )
    anonymous_after = manager.lookup_entities(
        name="user", memory_id="primary-user-a", user_id=None
    )

    assert authenticated_before == []
    assert anonymous_before[0]["attributes"]["role"] == "developer advocate"
    assert migrated == 1
    assert authenticated_after[0]["entity_id"] == entity_id
    assert anonymous_after == []
    assert provider.records[entity_id]["user_id"] == "user-a"
    assert len(provider.records) == 1


def test_repeated_relation_upsert_is_a_noop(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    relation = {
        "entity_id": "company-1",
        "relation_type": "works_at",
        "confidence": 0.9,
    }
    entity_id = entity_store.upsert_entity(
        name="Avery",
        entity_type="person",
        relations=[relation],
        memory_id="shared-memory",
        user_id="user-a",
    )
    entity_store.upsert_entity(
        entity_id=entity_id,
        relations=[relation],
        memory_id="shared-memory",
        user_id="user-a",
    )

    assert len(provider.records[entity_id]["relations"]) == 1
    assert provider.store_count == 1


def test_missing_vector_index_uses_exact_scoped_profile_fallback(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[
            {"name": "role", "value": "developer experience leader"},
            {"name": "response_preference", "value": "no em dashes"},
        ],
        memory_id="primary-user-a",
        user_id="user-a",
    )
    provider.vector_available = False
    manager = EntityMemoryManager(provider)

    result = manager.lookup_entities_with_diagnostics(
        query="user profile role preferences audience goals",
        memory_id="primary-user-a",
        user_id="user-a",
    )

    assert result["matches"][0]["attributes"]["role"] == ("developer experience leader")
    assert result["retrieval"]["retrieval_mode"] == "exact_fallback"
    assert result["retrieval"]["fallback_used"] is True
    assert result["retrieval"]["degraded"] is True
    assert result["retrieval"]["degraded_reason"] == "vector_index_missing"
    assert provider.semantic_query_count == 0
    assert provider.last_filter_limit == 100


def test_semantic_zero_match_falls_back_without_crossing_scope(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[{"name": "role", "value": "engineer"}],
        memory_id="primary-user-a",
        user_id="user-a",
    )
    # Simulate a queryable vector service returning no candidates rather than
    # an unavailable index.
    provider.retrieve_by_query = lambda query, memory_type, limit=5, **kwargs: (
        []
        if isinstance(query, str)
        else [
            rec
            for rec in provider.records.values()
            if all(rec.get(key) == value for key, value in query.items())
        ][:limit]
    )
    manager = EntityMemoryManager(provider)

    result = manager.lookup_entities_with_diagnostics(
        query="what do you know about me",
        memory_id="primary-user-a",
        user_id="user-a",
    )

    assert len(result["matches"]) == 1
    assert result["retrieval"]["degraded"] is False
    assert result["retrieval"]["degraded_reason"] == "semantic_empty"


def test_semantic_query_failure_is_reported_as_degraded(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[{"name": "role", "value": "engineer"}],
        memory_id="shared-memory",
        user_id="user-a",
    )
    original_retrieve = provider.retrieve_by_query

    def retrieve(query, memory_type, limit=5, **kwargs):
        if isinstance(query, str):
            raise RuntimeError("semantic backend failed")
        return original_retrieve(query, memory_type, limit=limit, **kwargs)

    provider.retrieve_by_query = retrieve
    manager = EntityMemoryManager(provider)

    result = manager.lookup_entities_with_diagnostics(
        query="what do you know about me",
        memory_id="shared-memory",
        user_id="user-a",
    )

    assert len(result["matches"]) == 1
    assert result["retrieval"]["degraded"] is True
    assert result["retrieval"]["degraded_reason"] == "semantic_error"
    assert result["retrieval"]["semantic_error_type"] == "RuntimeError"


def test_tool_update_rejects_entity_id_outside_active_scope(
    provider: InMemoryEntityProvider, entity_store: EntityMemory
):
    entity_id = entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[{"name": "secret", "value": "tenant-a-only"}],
        memory_id="primary-user-a",
        user_id="user-a",
    )
    manager = EntityMemoryManager(provider)

    with pytest.raises(ValueError, match="not found in the active entity scope"):
        manager.upsert_entity_from_tool(
            entity_id=entity_id,
            name="user",
            entity_type="person",
            attributes=[{"name": "secret", "value": "overwritten"}],
            relations=None,
            metadata=None,
            memory_id="primary-user-b",
            user_id="user-b",
        )

    assert provider.records[entity_id]["attributes"][0]["value"] == "tenant-a-only"

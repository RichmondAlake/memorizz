import uuid
from typing import Any, Dict, List, Optional

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.memagent.managers.entity_memory_manager import EntityMemoryManager


class InMemoryEntityProvider:
    """Minimal provider that mimics the entity-memory interface."""

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}

    def supports_entity_memory(self) -> bool:
        return True

    def store(self, data: Dict[str, Any], memory_store_type: MemoryType, **_) -> str:
        assert memory_store_type == MemoryType.ENTITY_MEMORY
        record = dict(data)
        record.setdefault("_id", record.get("entity_id", str(uuid.uuid4())))
        entity_id = record["entity_id"]

        self.records[entity_id] = record

        return record["_id"]

    def retrieve_by_query(
        self,
        query: Any,
        memory_type: MemoryType,
        limit: int = 5,
        memory_id: Optional[str] = None,
        **__,
    ) -> List[Dict[str, Any]]:
        assert memory_type == MemoryType.ENTITY_MEMORY

        if isinstance(query, dict):
            candidates = [
                rec
                for rec in self.records.values()
                if all(rec.get(key) == value for key, value in query.items())
                and (memory_id is None or rec.get("memory_id") == memory_id)
            ]
        else:
            candidates = [
                rec
                for rec in self.records.values()
                if memory_id is None or rec.get("memory_id") == memory_id
            ]
        return candidates[:limit]

    def list_all(self, memory_store_type: MemoryType) -> List[Dict[str, Any]]:
        assert memory_store_type == MemoryType.ENTITY_MEMORY
        return [dict(rec) for rec in self.records.values()]


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

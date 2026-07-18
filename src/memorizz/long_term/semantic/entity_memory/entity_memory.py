# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....memory_provider import MemoryProvider


class EntityAttribute(BaseModel):
    """Represents a single attribute associated with an entity."""

    name: str
    value: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class EntityRelation(BaseModel):
    """Represents a labeled relationship between two entities."""

    entity_id: str
    relation_type: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class EntityMemoryRecord(BaseModel):
    """Structured entity record persisted in the memory provider."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str
    name: Optional[str] = None
    entity_type: Optional[str] = None
    attributes: List[EntityAttribute] = Field(default_factory=list)
    relations: List[EntityRelation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_id: Optional[str] = None
    user_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: str
    updated_at: str


class EntityMemory:
    """High-level helper for storing and retrieving structured entity facts."""

    def __init__(self, memory_provider: MemoryProvider):
        if memory_provider is None:
            raise ValueError("EntityMemory requires a MemoryProvider instance")
        self.memory_provider = memory_provider

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def upsert_entity(
        self,
        *,
        entity_id: Optional[str] = None,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        attributes: Optional[Sequence[Union[EntityAttribute, Dict[str, Any]]]] = None,
        relations: Optional[Sequence[Union[EntityRelation, Dict[str, Any]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Create or update an entity record.

        Returns the entity_id used to store the record (generated when absent).
        """
        now = self._timestamp()
        entity_id = entity_id or str(uuid.uuid4())

        existing = self._fetch_one({"entity_id": entity_id}, user_id=user_id)
        if not existing and name:
            existing = self._fetch_one({"name": name}, user_id=user_id)

        record = self._merge_record(
            existing=existing,
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes,
            relations=relations,
            metadata=metadata,
            memory_id=memory_id,
            user_id=user_id,
            timestamp=now,
        )

        # Write-path dedup (NOOP guard): when the merge produced no new
        # information — the agent re-asserted facts already stored verbatim —
        # skip the re-embed (an embedding API call) and the re-store
        # entirely. Timestamp-only churn does not count as new information.
        if existing and self._stable_snapshot(record) == self._stable_snapshot(
            existing
        ):
            return record["entity_id"]

        embedding_payload = self._build_embedding_text(record)
        if embedding_payload:
            record["embedding"] = get_embedding(embedding_payload)

        if existing and existing.get("_id") is not None:
            record["_id"] = existing["_id"]
        self.memory_provider.store(
            data=record,
            memory_store_type=MemoryType.ENTITY_MEMORY,
        )
        # Mongo returns _id for updates; we prefer our stable entity_id
        return record["entity_id"]

    def record_attribute(
        self,
        *,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        attribute_name: str,
        attribute_value: str,
        confidence: float = 0.85,
        source: Optional[str] = None,
        entity_type: Optional[str] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Convenience helper to upsert a single attribute for an entity."""
        timestamp = self._timestamp()
        attribute = EntityAttribute(
            name=attribute_name,
            value=attribute_value,
            confidence=confidence,
            source=source,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return self.upsert_entity(
            entity_id=entity_id,
            name=entity_name,
            entity_type=entity_type,
            attributes=[attribute],
            memory_id=memory_id,
        )

    def get_entity(
        self, entity_id: str, *, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the stored record for a specific entity_id."""
        return self._fetch_one({"entity_id": entity_id}, user_id=user_id)

    def get_entity_by_name(
        self, name: str, *, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the stored record matching a given name."""
        return self._fetch_one({"name": name}, user_id=user_id)

    def list_entities(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all stored entities, optionally filtered by memory_id."""
        entities = self.memory_provider.list_all(
            MemoryType.ENTITY_MEMORY, user_id=user_id
        )
        filtered = entities
        if memory_id:
            filtered = [
                entity for entity in filtered if entity.get("memory_id") == memory_id
            ]
        # Strict tenant isolation client-side in case the provider didn't push
        # the filter down.
        filtered = [entity for entity in filtered if entity.get("user_id") == user_id]
        return filtered

    def search_entities(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over entity attributes."""
        results = self.memory_provider.retrieve_by_query(
            query,
            memory_type=MemoryType.ENTITY_MEMORY,
            limit=limit,
            memory_id=memory_id,
            user_id=user_id,
        )
        return self._ensure_list(results)

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _fetch_one(
        self,
        query: Dict[str, Any],
        *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a single record from the provider using a filter query."""
        scoped_query = dict(query)
        scoped_query["user_id"] = user_id
        results = self.memory_provider.retrieve_by_query(
            scoped_query,
            memory_type=MemoryType.ENTITY_MEMORY,
            limit=1,
            include_embedding=True,
            user_id=user_id,
        )
        items = self._ensure_list(results)
        return items[0] if items else None

    def _ensure_list(self, result: Any) -> List[Dict[str, Any]]:
        """Normalize Mongo cursors or single dicts into a list."""
        if result is None:
            return []
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        try:
            return list(result)
        except TypeError:
            return []

    @staticmethod
    def _stable_snapshot(record: Dict[str, Any]) -> str:
        """Serialize a record with volatile fields stripped, for NOOP checks.

        Timestamps, embeddings, and storage ids churn on every write even
        when the facts themselves are unchanged; they must not defeat the
        duplicate-write guard in :meth:`upsert_entity`.
        """
        volatile = {"embedding", "_id", "id", "created_at", "updated_at", "score"}

        def _strip(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: _strip(val)
                    for key, val in sorted(value.items())
                    if key not in volatile
                }
            if isinstance(value, (list, tuple)):
                return [_strip(item) for item in value]
            return value

        try:
            return json.dumps(_strip(record), sort_keys=True, default=str)
        except Exception:
            return str(record)

    def _merge_record(
        self,
        *,
        existing: Optional[Dict[str, Any]],
        entity_id: str,
        name: Optional[str],
        entity_type: Optional[str],
        attributes: Optional[Sequence[Union[EntityAttribute, Dict[str, Any]]]],
        relations: Optional[Sequence[Union[EntityRelation, Dict[str, Any]]]],
        metadata: Optional[Dict[str, Any]],
        memory_id: Optional[str],
        timestamp: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge dictionaries for upsert operations."""
        record = existing.copy() if existing else {}
        record["entity_id"] = record.get("entity_id") or entity_id
        if name:
            record["name"] = name
        if entity_type:
            record["entity_type"] = entity_type
        if memory_id:
            record["memory_id"] = memory_id
        # Persist user_id explicitly (including None, to preserve strict scoping).
        record["user_id"] = user_id if user_id is not None else record.get("user_id")
        if metadata:
            record["metadata"] = {**record.get("metadata", {}), **metadata}

        record_attributes = self._merge_attributes(
            record.get("attributes", []), attributes, timestamp
        )
        record_relations = self._merge_relations(
            record.get("relations", []), relations, timestamp
        )

        record["attributes"] = record_attributes
        record["relations"] = record_relations
        record["created_at"] = record.get("created_at", timestamp)
        record["updated_at"] = timestamp
        return record

    def _merge_attributes(
        self,
        existing: Iterable[Dict[str, Any]],
        updates: Optional[Sequence[Union[EntityAttribute, Dict[str, Any]]]],
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        attributes = {attr["name"].lower(): attr for attr in existing if "name" in attr}
        for attr in updates or []:
            normalized = self._to_attribute(attr, timestamp)
            key = normalized["name"].lower()
            attributes[key] = normalized
        return list(attributes.values())

    def _merge_relations(
        self,
        existing: Iterable[Dict[str, Any]],
        updates: Optional[Sequence[Union[EntityRelation, Dict[str, Any]]]],
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        relations = [relation for relation in existing if relation.get("entity_id")]
        for relation in updates or []:
            normalized = self._to_relation(relation, timestamp)
            relations.append(normalized)
        return relations

    def _to_attribute(
        self,
        attribute: Union[EntityAttribute, Dict[str, Any]],
        timestamp: str,
    ) -> Dict[str, Any]:
        if isinstance(attribute, EntityAttribute):
            payload = attribute.model_dump()
        else:
            payload = EntityAttribute(**attribute).model_dump()
        payload.setdefault("created_at", timestamp)
        payload["updated_at"] = timestamp
        return payload

    def _to_relation(
        self,
        relation: Union[EntityRelation, Dict[str, Any]],
        timestamp: str,
    ) -> Dict[str, Any]:
        if isinstance(relation, EntityRelation):
            payload = relation.model_dump()
        else:
            payload = EntityRelation(**relation).model_dump()
        payload.setdefault("created_at", timestamp)
        payload["updated_at"] = timestamp
        return payload

    def _build_embedding_text(self, record: Dict[str, Any]) -> str:
        """Create a summary string describing the entity for embeddings."""
        lines = []
        if record.get("name"):
            lines.append(f"Name: {record['name']}")
        if record.get("entity_type"):
            lines.append(f"Type: {record['entity_type']}")
        for attribute in record.get("attributes", []):
            lines.append(f"{attribute.get('name')}: {attribute.get('value')}")
        for relation in record.get("relations", []):
            lines.append(
                f"Relation ({relation.get('relation_type')})->{relation.get('entity_id')}"
            )
        if record.get("metadata"):
            metadata_parts = [
                f"{key}: {value}" for key, value in record["metadata"].items()
            ]
            lines.extend(metadata_parts)
        return "\n".join(lines)

    @staticmethod
    def _timestamp() -> str:
        return datetime.utcnow().isoformat()

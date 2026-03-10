# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

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

    entity_id: str
    name: Optional[str] = None
    entity_type: Optional[str] = None
    attributes: List[EntityAttribute] = Field(default_factory=list)
    relations: List[EntityRelation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: str
    updated_at: str

    class Config:
        arbitrary_types_allowed = True


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
    ) -> str:
        """
        Create or update an entity record.

        Returns the entity_id used to store the record (generated when absent).
        """
        now = self._timestamp()
        entity_id = entity_id or str(uuid.uuid4())

        existing = self._fetch_one({"entity_id": entity_id})
        if not existing and name:
            existing = self._fetch_one({"name": name})

        record = self._merge_record(
            existing=existing,
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes,
            relations=relations,
            metadata=metadata,
            memory_id=memory_id,
            timestamp=now,
        )

        embedding_payload = self._build_embedding_text(record)
        if embedding_payload:
            record["embedding"] = get_embedding(embedding_payload)

        if existing and existing.get("_id") is not None:
            record["_id"] = existing["_id"]
        stored_id = self.memory_provider.store(
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

    def link_entities(
        self,
        *,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        confidence: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Attach a relation between two entities."""
        relation = EntityRelation(
            entity_id=target_entity_id,
            relation_type=relation_type,
            confidence=confidence,
            metadata=metadata or {},
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
        )
        self.upsert_entity(
            entity_id=source_entity_id,
            relations=[relation],
        )
        return True

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored record for a specific entity_id."""
        return self._fetch_one({"entity_id": entity_id})

    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the stored record matching a given name."""
        return self._fetch_one({"name": name})

    def list_entities(self, *, memory_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all stored entities, optionally filtered by memory_id."""
        entities = self.memory_provider.list_all(MemoryType.ENTITY_MEMORY)
        if memory_id:
            return [
                entity for entity in entities if entity.get("memory_id") == memory_id
            ]
        return entities

    def search_entities(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over entity attributes."""
        results = self.memory_provider.retrieve_by_query(
            query,
            memory_type=MemoryType.ENTITY_MEMORY,
            limit=limit,
            memory_id=memory_id,
        )
        return self._ensure_list(results)

    def get_entity_profile(
        self, entity_id: str, *, include_relations: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Return a simplified persona/profile-style view for prompting."""
        record = self.get_entity(entity_id)
        if not record:
            return None

        profile = {
            "entity_id": entity_id,
            "name": record.get("name"),
            "entity_type": record.get("entity_type"),
            "attributes": {
                attr["name"]: attr["value"] for attr in record.get("attributes", [])
            },
            "updated_at": record.get("updated_at"),
        }
        if include_relations:
            profile["relations"] = [
                {
                    "entity_id": rel["entity_id"],
                    "relation_type": rel["relation_type"],
                    "confidence": rel.get("confidence"),
                }
                for rel in record.get("relations", [])
            ]
        return profile

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _fetch_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a single record from the provider using a filter query."""
        results = self.memory_provider.retrieve_by_query(
            query,
            memory_type=MemoryType.ENTITY_MEMORY,
            limit=1,
            include_embedding=True,
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
            payload = attribute.dict()
        else:
            payload = EntityAttribute(**attribute).dict()
        payload.setdefault("created_at", timestamp)
        payload["updated_at"] = timestamp
        return payload

    def _to_relation(
        self,
        relation: Union[EntityRelation, Dict[str, Any]],
        timestamp: str,
    ) -> Dict[str, Any]:
        if isinstance(relation, EntityRelation):
            payload = relation.dict()
        else:
            payload = EntityRelation(**relation).dict()
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

# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Entity memory management for MemAgent."""

import logging
from typing import Any, Dict, List, Optional, Sequence

from ...long_term_memory.semantic.entity_memory import EntityMemory

logger = logging.getLogger(__name__)


class EntityMemoryManager:
    """Simple wrapper that exposes entity memory operations to MemAgent."""

    def __init__(self, memory_provider=None):
        self.memory_provider = memory_provider
        self._entity_memory: Optional[EntityMemory] = None

        if memory_provider and self._provider_supports_entity_memory(memory_provider):
            try:
                self._entity_memory = EntityMemory(memory_provider)
                logger.debug("Entity memory manager initialized")
            except Exception as exc:
                logger.warning(f"Failed to initialize entity memory: {exc}")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        return self._entity_memory is not None

    def build_context(
        self, query: str, memory_id: Optional[str], limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Return simplified entity profiles relevant to the query."""
        if not self.is_enabled() or not query:
            return []

        records = self._entity_memory.search_entities(
            query, limit=limit, memory_id=memory_id
        )
        return [
            self._simplify_record(record)
            for record in records
            if record and record.get("entity_id")
        ]

    def lookup_entities(
        self,
        *,
        entity_id: Optional[str] = None,
        name: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
        memory_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lookup utility exposed via agent tools."""
        if not self.is_enabled():
            return []

        if entity_id:
            record = self._entity_memory.get_entity(entity_id)
            return [self._simplify_record(record)] if record else []

        if name:
            record = self._entity_memory.get_entity_by_name(name)
            return [self._simplify_record(record)] if record else []

        if query:
            return self.build_context(query, memory_id, limit=limit)

        # No selector provided – list scoped entities
        entities = self._entity_memory.list_entities(memory_id=memory_id)
        return [
            self._simplify_record(entity)
            for entity in entities[:limit]
            if entity and entity.get("entity_id")
        ]

    def upsert_entity_from_tool(
        self,
        *,
        entity_id: Optional[str],
        name: Optional[str],
        entity_type: Optional[str],
        attributes: Optional[Sequence[Dict[str, Any]]],
        relations: Optional[Sequence[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]],
        memory_id: str,
    ) -> str:
        """Persist entity updates triggered via the built-in tool."""
        if not self.is_enabled():
            raise RuntimeError("Entity memory is not enabled for this provider.")
        if not memory_id:
            raise ValueError("memory_id is required to store entity updates.")

        return self._entity_memory.upsert_entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes,
            relations=relations,
            metadata=metadata,
            memory_id=memory_id,
        )

    def summarize_for_prompt(self, profiles: List[Dict[str, Any]]) -> str:
        """Format entity profiles for inclusion in the system prompt."""
        if not profiles:
            return ""

        lines: List[str] = []
        for profile in profiles:
            attr_pairs = [
                f"{key}: {value}"
                for key, value in profile.get("attributes", {}).items()
            ]
            attr_text = (
                "; ".join(attr_pairs) if attr_pairs else "No attributes recorded"
            )
            entity_line = (
                f"- {profile.get('name') or profile.get('entity_id')}: {attr_text}"
            )
            if profile.get("relations"):
                relations = ", ".join(
                    f"{rel.get('relation_type')}->{rel.get('entity_id')}"
                    for rel in profile["relations"]
                    if rel.get("entity_id")
                )
                if relations:
                    entity_line += f" | Relations: {relations}"
            lines.append(entity_line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _simplify_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not record:
            return {}

        attribute_map = {
            attr.get("name"): attr.get("value")
            for attr in record.get("attributes", [])
            if attr.get("name")
        }
        relations = [
            {
                "entity_id": rel.get("entity_id"),
                "relation_type": rel.get("relation_type"),
                "confidence": rel.get("confidence"),
            }
            for rel in record.get("relations", [])
            if rel.get("entity_id") and rel.get("relation_type")
        ]

        simplified = {
            "entity_id": record.get("entity_id"),
            "name": record.get("name"),
            "entity_type": record.get("entity_type"),
            "attributes": attribute_map,
            "relations": relations,
            "updated_at": record.get("updated_at"),
        }
        if "score" in record:
            simplified["score"] = record["score"]
        return simplified

    def _provider_supports_entity_memory(self, provider: Any) -> bool:
        """Best-effort detection for providers that expose entity memory."""
        if provider is None:
            return False

        support_fn = getattr(provider, "supports_entity_memory", None)
        if callable(support_fn):
            try:
                return bool(support_fn())
            except Exception as exc:
                logger.debug(f"supports_entity_memory check failed: {exc}")

        return hasattr(provider, "entity_memory_collection")

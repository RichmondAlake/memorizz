# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Entity memory management for MemAgent."""

import logging
from typing import Any, Dict, List, Optional, Sequence

from ...long_term.semantic.entity_memory import EntityMemory

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
        self,
        query: str,
        memory_id: Optional[str],
        limit: int = 3,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return simplified entity profiles relevant to the query."""
        return self.build_context_with_diagnostics(
            query=query,
            memory_id=memory_id,
            limit=limit,
            user_id=user_id,
        )["profiles"]

    def build_context_with_diagnostics(
        self,
        query: str,
        memory_id: Optional[str],
        limit: int = 3,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return prompt profiles plus content-free retrieval diagnostics."""
        if not self.is_enabled() or not query:
            return {
                "profiles": [],
                "retrieval": {
                    "retrieval_mode": "disabled" if not self.is_enabled() else "exact",
                    "match_count": 0,
                    "degraded": not self.is_enabled(),
                    "degraded_reason": (
                        "entity_memory_disabled"
                        if not self.is_enabled()
                        else "empty_query"
                    ),
                },
            }

        records, diagnostics = self._entity_memory.search_entities_with_diagnostics(
            query, limit=limit, memory_id=memory_id, user_id=user_id
        )
        if diagnostics.get("fallback_used"):
            logger.info(
                "Entity context retrieval fallback " "(mode=%s, reason=%s, matches=%s)",
                diagnostics.get("retrieval_mode"),
                diagnostics.get("degraded_reason"),
                diagnostics.get("match_count"),
            )
        profiles = [
            self._simplify_record(record)
            for record in records
            if record and record.get("entity_id")
        ]
        diagnostics["match_count"] = len(profiles)
        return {"profiles": profiles, "retrieval": diagnostics}

    def lookup_entities(
        self,
        *,
        entity_id: Optional[str] = None,
        name: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lookup utility exposed via agent tools."""
        return self.lookup_entities_with_diagnostics(
            entity_id=entity_id,
            name=name,
            query=query,
            limit=limit,
            memory_id=memory_id,
            user_id=user_id,
        )["matches"]

    def lookup_entities_with_diagnostics(
        self,
        *,
        entity_id: Optional[str] = None,
        name: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Lookup entities and return safe metadata for traces/observability."""
        safe_limit = max(1, min(int(limit or 5), 100))
        base_diagnostics: Dict[str, Any] = {
            "selector": "list",
            "retrieval_mode": "exact",
            "fallback_used": False,
            "match_count": 0,
            "legacy_scope_match_count": 0,
            "degraded": False,
            "degraded_reason": None,
            "scope": {
                "memory_id_bound": memory_id is not None,
                "user_id_bound": user_id is not None,
            },
        }
        if not self.is_enabled():
            base_diagnostics.update(
                {
                    "degraded": True,
                    "degraded_reason": "entity_memory_disabled",
                }
            )
            return {"matches": [], "retrieval": base_diagnostics}

        if entity_id:
            base_diagnostics["selector"] = "entity_id"
            record = self._entity_memory.get_entity(
                entity_id, memory_id=memory_id, user_id=user_id
            )
            records = [record] if record else []

        elif name:
            base_diagnostics["selector"] = "name"
            record = self._entity_memory.get_entity_by_name(
                name, memory_id=memory_id, user_id=user_id
            )
            records = [record] if record else []

        elif query:
            (
                records,
                query_diagnostics,
            ) = self._entity_memory.search_entities_with_diagnostics(
                query,
                limit=safe_limit,
                memory_id=memory_id,
                user_id=user_id,
            )
            matches = [
                self._simplify_record(record)
                for record in records
                if record and record.get("entity_id")
            ]
            query_diagnostics["match_count"] = len(matches)
            return {"matches": matches, "retrieval": query_diagnostics}

        else:
            records = self._entity_memory.list_entities(
                memory_id=memory_id, user_id=user_id
            )[:safe_limit]

        matches = [
            self._simplify_record(record)
            for record in records
            if record and record.get("entity_id")
        ]
        base_diagnostics.update(
            {
                "match_count": len(matches),
                "legacy_scope_match_count": 0,
            }
        )
        return {"matches": matches, "retrieval": base_diagnostics}

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
        user_id: Optional[str] = None,
    ) -> str:
        """Persist entity updates triggered via the built-in tool."""
        if not self.is_enabled():
            raise RuntimeError("Entity memory is not enabled for this provider.")
        if not memory_id:
            raise ValueError("memory_id is required to store entity updates.")
        if entity_id and not self._entity_memory.get_entity(
            entity_id, memory_id=memory_id, user_id=user_id
        ):
            # An LLM may only update an ID that it could resolve inside the
            # server-bound tenant scope. This prevents an attacker-supplied or
            # hallucinated ID from overwriting a different tenant in providers
            # that use entity_id as their physical upsert key.
            raise ValueError("entity_id was not found in the active entity scope.")

        return self._entity_memory.upsert_entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes,
            relations=relations,
            metadata=metadata,
            memory_id=memory_id,
            user_id=user_id,
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

# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_ENTITY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SELF_ENTITY_ALIASES = frozenset(
    {"user", "current user", "self", "profile", "user profile"}
)
_SELF_IDENTITY_KEYS = frozenset(
    {"authenticated_user", "current_user", "user", "self", "user_profile"}
)
_PROFILE_QUERY_TERMS = frozenset(
    {
        "about",
        "audience",
        "goal",
        "know",
        "me",
        "my",
        "name",
        "preference",
        "profile",
        "project",
        "role",
        "user",
    }
)
_IDENTITY_KEY_RE = re.compile(r"[^a-z0-9_.:-]+")


class EntityAttributeInput(BaseModel):
    """Model-facing contract for a single entity-memory fact."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        validation_alias=AliasChoices("name", "attribute", "attribute_name", "key"),
        description="Stable attribute name, for example role or preferred_language.",
    )
    value: str = Field(description="The factual value to store for this attribute.")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence from 0.0 to 1.0.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Optional non-secret provenance for the fact.",
    )


class EntityRelationInput(BaseModel):
    """Model-facing contract for a relation between two entities."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(description="Target entity identifier.")
    relation_type: str = Field(description="Relationship label, for example works_on.")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityAttribute(BaseModel):
    """Represents a single attribute associated with an entity."""

    name: str = Field(
        validation_alias=AliasChoices("name", "attribute", "attribute_name", "key")
    )
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
        identity_key: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Create or update an entity record.

        Returns the entity_id used to store the record (generated when absent).
        """
        now = self._timestamp()
        normalized_identity = self._normalize_identity_key(identity_key)
        supplied_metadata = dict(metadata or {})
        if normalized_identity:
            supplied_metadata["identity_key"] = normalized_identity

        existing = None
        # A canonical identity is host/application authority. When both an
        # identity key and a stale/model-supplied entity_id are present, reuse
        # the canonical identity rather than reviving a duplicate record.
        if normalized_identity:
            existing = self._find_by_identity_key(
                normalized_identity, memory_id=memory_id, user_id=user_id
            )
        if not existing and entity_id:
            existing = self._fetch_one(
                {"entity_id": entity_id}, memory_id=memory_id, user_id=user_id
            )
        if not existing and name:
            existing = self._fetch_one(
                {"name": name}, memory_id=memory_id, user_id=user_id
            )
        if existing:
            entity_id = str(existing.get("entity_id") or entity_id or uuid.uuid4())
        elif normalized_identity:
            entity_id = self._deterministic_entity_id(
                normalized_identity, memory_id=memory_id, user_id=user_id
            )
        else:
            entity_id = entity_id or str(uuid.uuid4())

        record = self._merge_record(
            existing=existing,
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes,
            relations=relations,
            metadata=supplied_metadata,
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
        elif normalized_identity:
            # Canonical identities need a deterministic storage key as well as
            # a deterministic logical entity_id. Without this, two concurrent
            # first writes can both miss the lookup and MongoDB can insert two
            # documents with different generated _id values. Filesystem uses
            # this as its document path and Oracle accepts the UUID as its row
            # id, so retries become idempotent across all built-in providers.
            record["_id"] = record["entity_id"]
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
        identity_key: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
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
            identity_key=identity_key,
            attributes=[attribute],
            memory_id=memory_id,
            user_id=user_id,
        )

    def get_entity(
        self,
        entity_id: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the stored record for a specific entity_id."""
        return self._fetch_one(
            {"entity_id": entity_id}, memory_id=memory_id, user_id=user_id
        )

    def get_entity_by_name(
        self,
        name: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the stored record matching a given name."""
        return self._fetch_one({"name": name}, memory_id=memory_id, user_id=user_id)

    def list_entities(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_superseded: bool = False,
    ) -> List[Dict[str, Any]]:
        """List entities in exactly one memory/user scope.

        ``user_id=None`` is the anonymous/legacy scope. Authenticated callers
        never inherit those rows implicitly; legacy adoption is an explicit
        administrative migration so request-serving reads stay fail-closed.
        """
        try:
            rows = self.memory_provider.list_all(
                MemoryType.ENTITY_MEMORY, user_id=user_id
            )
        except TypeError:
            # Backward compatibility for third-party providers written before
            # list_all accepted a user_id keyword. Isolation is still enforced
            # below against every returned document.
            rows = self.memory_provider.list_all(MemoryType.ENTITY_MEMORY)

        filtered: List[Dict[str, Any]] = []
        seen = set()
        for entity in self._ensure_list(rows):
            if not self._record_in_scope(
                entity,
                memory_id=memory_id,
                user_id=user_id,
            ):
                continue
            if not include_superseded and self._is_superseded(entity):
                continue
            key = str(
                entity.get("entity_id")
                or entity.get("_id")
                or self._stable_snapshot(entity)
            )
            if key in seen:
                continue
            seen.add(key)
            filtered.append(entity)
        return filtered

    def consolidate_duplicate_entities(
        self,
        *,
        memory_id: str,
        user_id: Optional[str],
        apply: bool = False,
        allow_legacy_scope: bool = False,
        entity_ids: Optional[Sequence[str]] = None,
        canonical_identity_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plan or safely apply duplicate consolidation inside one exact scope.

        The default is a read-only plan.  ``apply=True`` first writes a merged
        canonical record and then marks duplicate rows as ``superseded_by``;
        it never deletes data.  Superseded rows stay recoverable for audit and
        rollback but are excluded from normal lookup and prompt retrieval.
        Anonymous legacy scope requires an additional explicit opt-in.
        """
        resolved_memory_id = str(memory_id or "").strip()
        if not resolved_memory_id:
            raise ValueError("memory_id is required for entity consolidation")
        if user_id is None and not allow_legacy_scope:
            raise ValueError(
                "user_id is required; set allow_legacy_scope=True to inspect "
                "anonymous legacy rows"
            )

        rows = self.list_entities(
            memory_id=resolved_memory_id,
            user_id=user_id,
            include_superseded=False,
        )
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        requested_ids = {
            str(value).strip() for value in (entity_ids or []) if str(value).strip()
        }
        normalized_identity = self._normalize_identity_key(canonical_identity_key)
        if requested_ids:
            selected = [
                row
                for row in rows
                if str(row.get("entity_id") or "").strip() in requested_ids
            ]
            found_ids = {str(row.get("entity_id") or "").strip() for row in selected}
            missing_ids = sorted(requested_ids - found_ids)
            if missing_ids:
                raise ValueError(
                    "Every entity_id must exist in the exact memory/user scope; "
                    f"missing {len(missing_ids)} requested record(s)"
                )
            if len(selected) < 2:
                raise ValueError(
                    "Explicit consolidation requires at least two entity_ids"
                )
            grouped["explicit_selection"] = selected
        else:
            for row in rows:
                key = self._duplicate_group_key(row)
                if key:
                    grouped.setdefault(key, []).append(row)

        plans: List[Dict[str, Any]] = []
        for group_key, members in sorted(grouped.items()):
            if len(members) < 2:
                continue
            canonical = self._choose_canonical(members)
            duplicate_rows = [row for row in members if row is not canonical]
            merged = self._merge_duplicate_group(canonical, duplicate_rows)
            if normalized_identity:
                merged_metadata = dict(merged.get("metadata") or {})
                merged_metadata["identity_key"] = normalized_identity
                merged["metadata"] = merged_metadata
            canonical_id = str(merged.get("entity_id") or "")
            duplicate_ids = sorted(
                str(row.get("entity_id") or "")
                for row in duplicate_rows
                if row.get("entity_id")
            )
            attribute_names = sorted(
                {
                    str(item.get("name"))
                    for item in merged.get("attributes", [])
                    if item.get("name")
                }
            )
            plan = {
                "group_key": group_key,
                "canonical_entity_id": canonical_id,
                "duplicate_entity_ids": duplicate_ids,
                "record_count": len(members),
                "attribute_names": attribute_names,
                "relation_count": len(merged.get("relations") or []),
                "conflicting_attribute_names": self._attribute_conflicts(members),
                "canonical_identity_bound": bool(normalized_identity),
                "applied": False,
            }
            if apply:
                self._store_consolidated_group(
                    merged,
                    duplicate_rows,
                    memory_id=resolved_memory_id,
                    user_id=user_id,
                )
                plan["applied"] = True
            plans.append(plan)

        return {
            "schema_version": 1,
            "dry_run": not apply,
            "memory_id": resolved_memory_id,
            "user_scope_bound": user_id is not None,
            "groups": plans,
            "duplicate_group_count": len(plans),
            "records_to_supersede": sum(
                len(item["duplicate_entity_ids"]) for item in plans
            ),
            "deleted_count": 0,
            "strategy": "merge_then_soft_supersede",
            "explicit_selection": bool(requested_ids),
        }

    def migrate_legacy_scope(self, *, memory_id: str, user_id: str) -> int:
        """Explicitly assign anonymous entity rows to one authenticated user.

        This is an operator-only migration primitive and is never registered as
        an agent tool. Runtime reads remain strictly scoped before and after the
        migration.
        """
        if not str(memory_id or "").strip():
            raise ValueError("memory_id is required for legacy entity migration")
        if not str(user_id or "").strip():
            raise ValueError("A non-empty target user_id is required")
        migrate = getattr(
            self.memory_provider, "migrate_entity_memory_user_scope", None
        )
        if not callable(migrate):
            raise NotImplementedError(
                "This memory provider does not support managed legacy entity migration"
            )
        return int(migrate(memory_id=str(memory_id), user_id=str(user_id)) or 0)

    def search_entities(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search entity attributes with an exact scoped fallback.

        Vector search is an optimization, not a correctness requirement. On
        lower-tier MongoDB deployments without Atlas Search, or whenever the
        semantic path yields no scoped result, a bounded lexical/profile
        fallback searches only entities inside the same tenant scope.
        """
        records, _diagnostics = self.search_entities_with_diagnostics(
            query,
            limit=limit,
            memory_id=memory_id,
            user_id=user_id,
        )
        return records

    def search_entities_with_diagnostics(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return scoped entity matches plus content-free retrieval metadata."""
        safe_limit = max(1, min(int(limit or 5), 100))
        diagnostics: Dict[str, Any] = {
            "selector": "query",
            "retrieval_mode": "semantic",
            "semantic_attempted": False,
            "semantic_match_count": 0,
            "fallback_used": False,
            "fallback_candidate_count": 0,
            "legacy_scope_match_count": 0,
            "match_count": 0,
            "degraded": False,
            "degraded_reason": None,
            "scope": {
                "memory_id_bound": memory_id is not None,
                "user_id_bound": user_id is not None,
            },
        }
        if not str(query or "").strip():
            diagnostics.update(
                {
                    "retrieval_mode": "exact_fallback",
                    "fallback_used": True,
                    "degraded": True,
                    "degraded_reason": "empty_query",
                }
            )
            return [], diagnostics

        vector_status = self._provider_vector_search_status()
        if vector_status:
            diagnostics["vector_search"] = vector_status

        semantic_results: List[Dict[str, Any]] = []
        semantic_available = vector_status.get("available") if vector_status else None
        if semantic_available is not False:
            diagnostics["semantic_attempted"] = True
            try:
                raw_results = self.memory_provider.retrieve_by_query(
                    query,
                    memory_type=MemoryType.ENTITY_MEMORY,
                    limit=safe_limit,
                    memory_id=memory_id,
                    user_id=user_id,
                )
                semantic_results = [
                    record
                    for record in self._ensure_list(raw_results)
                    if self._record_in_scope(
                        record,
                        memory_id=memory_id,
                        user_id=user_id,
                    )
                    and not self._is_superseded(record)
                ]
            except Exception as exc:
                diagnostics["semantic_error_type"] = type(exc).__name__
                semantic_error_reason = getattr(exc, "reason", None)
                if semantic_error_reason in {
                    "semantic_embedding_failed",
                    "vector_query_failed",
                    "vector_search_unavailable",
                }:
                    diagnostics["semantic_error_reason"] = semantic_error_reason
                logger.warning(
                    "Entity semantic search failed; using exact scoped fallback (%s)",
                    type(exc).__name__,
                )

        diagnostics["semantic_match_count"] = len(semantic_results)
        if semantic_results:
            matches = semantic_results[:safe_limit]
            diagnostics["match_count"] = len(matches)
            return matches, diagnostics

        fallback_candidate_limit = min(500, max(50, safe_limit * 20))
        scoped_entities = self._fallback_candidates(
            memory_id=memory_id,
            user_id=user_id,
            limit=fallback_candidate_limit,
        )
        fallback_matches = self._rank_exact_fallback(
            scoped_entities, query=query, limit=safe_limit
        )
        reason = "semantic_empty"
        if semantic_available is False:
            reason = str(vector_status.get("reason") or "vector_search_unavailable")
        elif diagnostics.get("semantic_error_type"):
            reason = str(diagnostics.get("semantic_error_reason") or "semantic_error")

        diagnostics.update(
            {
                "retrieval_mode": "exact_fallback",
                "fallback_used": True,
                "fallback_candidate_count": len(scoped_entities),
                "fallback_candidate_limit": fallback_candidate_limit,
                "fallback_candidate_limit_reached": len(scoped_entities)
                >= fallback_candidate_limit,
                "legacy_scope_match_count": 0,
                "match_count": len(fallback_matches),
                "degraded": semantic_available is False
                or bool(diagnostics.get("semantic_error_type")),
                "degraded_reason": reason,
            }
        )
        logger.info(
            "Entity memory retrieval used exact fallback "
            "(reason=%s, candidates=%s, matches=%s, legacy=%s)",
            reason,
            len(scoped_entities),
            len(fallback_matches),
            0,
        )
        return fallback_matches, diagnostics

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _normalize_identity_key(value: Optional[str]) -> str:
        text = str(value or "").strip().casefold()
        if not text:
            return ""
        normalized = _IDENTITY_KEY_RE.sub("-", text).strip("-._:")
        if not normalized:
            raise ValueError("identity_key must contain a letter or number")
        if len(normalized) > 160:
            raise ValueError("identity_key must be 160 characters or fewer")
        return normalized

    @staticmethod
    def _deterministic_entity_id(
        identity_key: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> str:
        scope = json.dumps(
            {
                "identity_key": identity_key,
                "memory_id": memory_id,
                "user_id": user_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"memorizz:entity:{scope}"))

    def _find_by_identity_key(
        self,
        identity_key: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        for row in self.list_entities(memory_id=memory_id, user_id=user_id):
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            if (
                self._normalize_identity_key(metadata.get("identity_key"))
                == identity_key
            ):
                return row
        return None

    @staticmethod
    def _is_superseded(record: Dict[str, Any]) -> bool:
        metadata = record.get("metadata") or {}
        return bool(
            isinstance(metadata, dict)
            and str(metadata.get("superseded_by") or "").strip()
        )

    @classmethod
    def _duplicate_group_key(cls, record: Dict[str, Any]) -> str:
        metadata = record.get("metadata") or {}
        identity_key = (
            cls._normalize_identity_key(metadata.get("identity_key"))
            if isinstance(metadata, dict)
            else ""
        )
        if identity_key:
            return f"identity:{identity_key}"
        name = " ".join(str(record.get("name") or "").casefold().split())
        entity_type = " ".join(
            str(record.get("entity_type") or "unknown").casefold().split()
        )
        return f"name:{entity_type}:{name}" if name else ""

    @staticmethod
    def _choose_canonical(members: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        def _rank(record: Dict[str, Any]) -> Tuple[int, int, str, str]:
            metadata = record.get("metadata") or {}
            identity_bound = int(
                isinstance(metadata, dict) and bool(metadata.get("identity_key"))
            )
            return (
                identity_bound,
                len(record.get("attributes") or []),
                str(record.get("updated_at") or ""),
                str(record.get("entity_id") or ""),
            )

        return max(members, key=_rank)

    @classmethod
    def _attribute_conflicts(cls, members: Sequence[Dict[str, Any]]) -> List[str]:
        values: Dict[str, set[str]] = {}
        for record in members:
            for item in record.get("attributes") or []:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                name = str(item["name"]).casefold()
                values.setdefault(name, set()).add(str(item.get("value") or ""))
        return sorted(
            name for name, candidates in values.items() if len(candidates) > 1
        )

    @classmethod
    def _merge_duplicate_group(
        cls,
        canonical: Dict[str, Any],
        duplicates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = dict(canonical)
        members = [canonical, *duplicates]
        attributes: Dict[str, Dict[str, Any]] = {}
        for record in members:
            for raw in record.get("attributes") or []:
                if not isinstance(raw, dict) or not raw.get("name"):
                    continue
                item = dict(raw)
                key = str(item["name"]).casefold()
                current = attributes.get(key)
                try:
                    confidence = float(item.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                try:
                    current_confidence = float((current or {}).get("confidence") or 0.0)
                except (TypeError, ValueError):
                    current_confidence = 0.0
                item_rank = (confidence, str(item.get("updated_at") or ""))
                current_rank = (
                    current_confidence,
                    str((current or {}).get("updated_at") or ""),
                )
                if current is None or item_rank > current_rank:
                    attributes[key] = item

        relations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in members:
            for raw in record.get("relations") or []:
                if not isinstance(raw, dict):
                    continue
                target = str(raw.get("entity_id") or "").strip()
                relation_type = str(raw.get("relation_type") or "").strip()
                if target and relation_type:
                    relations[(target, relation_type.casefold())] = dict(raw)

        metadata: Dict[str, Any] = {}
        for record in reversed(members):
            raw_metadata = record.get("metadata") or {}
            if isinstance(raw_metadata, dict):
                metadata.update(
                    {
                        key: value
                        for key, value in raw_metadata.items()
                        if key not in {"superseded_by", "superseded_at"}
                    }
                )
        alias_ids = sorted(
            {
                str(record.get("entity_id"))
                for record in duplicates
                if record.get("entity_id")
            }
        )
        metadata["merged_entity_ids"] = sorted(
            set(metadata.get("merged_entity_ids") or []).union(alias_ids)
        )
        metadata["consolidated_at"] = cls._timestamp()
        merged["attributes"] = list(attributes.values())
        merged["relations"] = list(relations.values())
        merged["metadata"] = metadata
        merged["updated_at"] = cls._timestamp()
        return merged

    def _store_consolidated_group(
        self,
        canonical: Dict[str, Any],
        duplicates: Sequence[Dict[str, Any]],
        *,
        memory_id: str,
        user_id: Optional[str],
    ) -> None:
        if not self._record_in_scope(canonical, memory_id=memory_id, user_id=user_id):
            raise ValueError("Canonical entity escaped the requested scope")
        canonical_payload = dict(canonical)
        embedding_text = self._build_embedding_text(canonical_payload)
        if embedding_text:
            canonical_payload["embedding"] = get_embedding(embedding_text)
        self.memory_provider.store(
            data=canonical_payload,
            memory_store_type=MemoryType.ENTITY_MEMORY,
        )

        canonical_id = str(canonical_payload.get("entity_id") or "")
        verified = self._fetch_one_in_scope(
            {"entity_id": canonical_id},
            memory_id=memory_id,
            user_id=user_id,
            include_superseded=True,
        )
        if not verified:
            raise RuntimeError("Canonical entity verification failed; duplicates kept")

        now = self._timestamp()
        for duplicate in duplicates:
            if not self._record_in_scope(
                duplicate, memory_id=memory_id, user_id=user_id
            ):
                raise ValueError("Duplicate entity escaped the requested scope")
            archived = dict(duplicate)
            metadata = dict(archived.get("metadata") or {})
            metadata.update({"superseded_by": canonical_id, "superseded_at": now})
            archived["metadata"] = metadata
            archived["updated_at"] = now
            self.memory_provider.store(
                data=archived,
                memory_store_type=MemoryType.ENTITY_MEMORY,
            )

    def _fetch_one(
        self,
        query: Dict[str, Any],
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve one record without ever crossing a memory/user boundary."""
        return self._fetch_one_in_scope(query, memory_id=memory_id, user_id=user_id)

    def _fetch_one_in_scope(
        self,
        query: Dict[str, Any],
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
        include_superseded: bool = False,
    ) -> Optional[Dict[str, Any]]:
        scoped_query = dict(query)
        if memory_id is not None:
            scoped_query["memory_id"] = memory_id
        scoped_query["user_id"] = user_id
        try:
            results = self.memory_provider.retrieve_by_query(
                scoped_query,
                memory_type=MemoryType.ENTITY_MEMORY,
                limit=5,
                include_embedding=True,
                memory_id=memory_id,
                user_id=user_id,
            )
        except TypeError:
            results = self.memory_provider.retrieve_by_query(
                scoped_query,
                memory_type=MemoryType.ENTITY_MEMORY,
                limit=5,
                memory_id=memory_id,
            )
        for item in self._ensure_list(results):
            if self._record_in_scope(
                item,
                memory_id=memory_id,
                user_id=user_id,
            ) and (include_superseded or not self._is_superseded(item)):
                return item
        return None

    def _fallback_candidates(
        self,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Load a bounded candidate window for exact fallback ranking.

        All built-in providers support an empty filter query with a server-side
        limit. Third-party providers retain a compatibility path, followed by
        the same strict client-side scope check.
        """
        safe_limit = max(1, min(int(limit), 500))
        try:
            rows = self.memory_provider.retrieve_by_query(
                {},
                memory_type=MemoryType.ENTITY_MEMORY,
                limit=safe_limit,
                memory_id=memory_id,
                user_id=user_id,
            )
        except TypeError:
            try:
                rows = self.memory_provider.retrieve_by_query(
                    {},
                    memory_store_type=MemoryType.ENTITY_MEMORY,
                    limit=safe_limit,
                    memory_id=memory_id,
                )
            except Exception as exc:
                logger.warning(
                    "Legacy entity fallback query failed; using scoped list (%s)",
                    type(exc).__name__,
                )
                rows = self.list_entities(memory_id=memory_id, user_id=user_id)
        except Exception as exc:
            logger.warning(
                "Bounded entity fallback query failed; using scoped list (%s)",
                type(exc).__name__,
            )
            rows = self.list_entities(memory_id=memory_id, user_id=user_id)

        candidates: List[Dict[str, Any]] = []
        for row in self._ensure_list(rows):
            if not self._record_in_scope(row, memory_id=memory_id, user_id=user_id):
                continue
            if self._is_superseded(row):
                continue
            candidates.append(row)
            if len(candidates) >= safe_limit:
                break
        return candidates

    @staticmethod
    def _record_in_scope(
        record: Dict[str, Any],
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> bool:
        if not isinstance(record, dict):
            return False
        if memory_id is not None and record.get("memory_id") != memory_id:
            return False
        return record.get("user_id") == user_id

    def _provider_vector_search_status(self) -> Dict[str, Any]:
        status_fn = getattr(self.memory_provider, "get_vector_search_status", None)
        if not callable(status_fn):
            return {}
        try:
            raw = status_fn(MemoryType.ENTITY_MEMORY)
        except Exception as exc:
            return {
                "available": None,
                "reason": "status_check_failed",
                "error_type": type(exc).__name__,
            }
        if not isinstance(raw, dict):
            return {}
        # Provider errors and raw index definitions can contain credentials or
        # topology details. Only persist this bounded diagnostic allowlist.
        return {
            key: raw.get(key)
            for key in ("available", "queryable", "status", "reason")
            if raw.get(key) is not None
        }

    @classmethod
    def _rank_exact_fallback(
        cls,
        records: List[Dict[str, Any]],
        *,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        query_tokens = cls._tokens(query)
        profile_intent = bool(query_tokens.intersection(_PROFILE_QUERY_TERMS))
        ranked = []
        for record in records:
            record_tokens = cls._tokens(cls._build_embedding_text(record))
            overlap = len(query_tokens.intersection(record_tokens))
            name = str(record.get("name") or "").strip().casefold()
            metadata = record.get("metadata") or {}
            explicit_self = bool(
                name in _SELF_ENTITY_ALIASES
                or metadata.get("is_self") is True
                or cls._normalize_identity_key(metadata.get("identity_key"))
                in _SELF_IDENTITY_KEYS
                or str(metadata.get("subject") or "").casefold()
                in {"current_user", "user", "self"}
            )
            person_profile = bool(
                profile_intent
                and str(record.get("entity_type") or "").casefold()
                in {"person", "user"}
            )
            if not (explicit_self or person_profile or overlap):
                continue
            priority = 2 if explicit_self else (1 if person_profile else 0)
            ranked.append(
                (
                    priority,
                    overlap,
                    str(record.get("updated_at") or ""),
                    record,
                )
            )
        ranked.sort(key=lambda item: item[:3], reverse=True)
        return [item[3] for item in ranked[:limit]]

    @staticmethod
    def _tokens(value: Any) -> set:
        tokens = set(_ENTITY_TOKEN_RE.findall(str(value or "").casefold()))
        normalized = set(tokens)
        for token in tokens:
            if len(token) > 3 and token.endswith("s"):
                normalized.add(token[:-1])
        return normalized

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
        # Persist user_id explicitly, including the anonymous ``None`` scope.
        record["user_id"] = user_id
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
        relations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for relation in existing:
            entity_id = str(relation.get("entity_id") or "").strip()
            relation_type = str(relation.get("relation_type") or "").strip()
            if not entity_id or not relation_type:
                continue
            relations[(entity_id, relation_type.casefold())] = dict(relation)
        for relation in updates or []:
            normalized = self._to_relation(relation, timestamp)
            key = (
                str(normalized["entity_id"]).strip(),
                str(normalized["relation_type"]).strip().casefold(),
            )
            previous = relations.get(key)
            if previous and not normalized.get("created_at"):
                normalized["created_at"] = previous.get("created_at")
            relations[key] = normalized
        return list(relations.values())

    def _to_attribute(
        self,
        attribute: Union[EntityAttribute, Dict[str, Any]],
        timestamp: str,
    ) -> Dict[str, Any]:
        if isinstance(attribute, EntityAttribute):
            payload = attribute.model_dump()
        else:
            payload = EntityAttribute(**attribute).model_dump()
        payload["created_at"] = payload.get("created_at") or timestamp
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
        payload["created_at"] = payload.get("created_at") or timestamp
        payload["updated_at"] = timestamp
        return payload

    @staticmethod
    def _build_embedding_text(record: Dict[str, Any]) -> str:
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

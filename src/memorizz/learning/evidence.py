"""Explainable, scoped and token-budgeted retrieval for the control plane."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .._time import to_utc
from ..enums.memory_type import MemoryType
from ..long_term.semantic.entity_memory import EntityMemory
from ..memagent.utils.context_dedup import is_suppressed, naive_time_convention
from ..memory_provider.base import user_scope_matches
from ..retrieval import sealed_elsewhere
from .models import (
    EvidenceItem,
    EvidencePack,
    LearningControlPlaneConfig,
    canonical_json,
    content_digest,
    estimate_tokens,
    utc_now,
)
from .store import LearningControlPlaneStore

_SOURCE_TYPES: Dict[str, MemoryType] = {
    "knowledge_base": MemoryType.KNOWLEDGE_BASE,
    "conversation_memory": MemoryType.CONVERSATION_MEMORY,
    "summaries": MemoryType.SUMMARIES,
    "entity_memory": MemoryType.ENTITY_MEMORY,
    "workflow_memory": MemoryType.WORKFLOW_MEMORY,
    "skillbox": MemoryType.SKILLBOX,
}

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _tokens(value: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(str(value))}


def _lexical_score(query: str, content: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    content_tokens = _tokens(content)
    if not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens)
    precision = overlap / len(query_tokens)
    coverage = overlap / max(1, min(len(content_tokens), len(query_tokens) * 4))
    return min(1.0, precision * 0.8 + coverage * 0.2)


_TIME_FIELDS = ("timestamp", "updated_at", "created_at", "period_end")


def _coerce_timestamp(
    value: Any, naive: str = "utc"
) -> Tuple[Optional[str], Optional[float]]:
    """``value`` as UTC ISO-8601 and its age in seconds.

    ``naive`` says how a value without an offset was written.
    """
    if value is None:
        return None, None
    parsed = to_utc(value, naive=naive)
    if parsed is None:
        return str(value), None
    age = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
    return parsed.isoformat(), age


def _row_id(row: Mapping[str, Any]) -> str:
    for key in (
        "_id",
        "id",
        "summary_id",
        "skill_id",
        "workflow_id",
        "entity_id",
        "memory_unit_id",
        "memory_id",
    ):
        if row.get(key) is not None:
            return str(row[key])
    return content_digest(row)[:32]


def _entity_facts(attributes: Any) -> List[Tuple[str, str]]:
    """``(name, value)`` per attribute, the most confident (then newest) kept.

    Confidence, source and timestamps rank the facts but are not shown: they
    cost several times the facts' own tokens.
    """
    if isinstance(attributes, Mapping):
        items = [
            {"name": key, **value}
            if isinstance(value, Mapping)
            else {"name": key, "value": value}
            for key, value in attributes.items()
        ]
    elif isinstance(attributes, (list, tuple)):
        items = [item for item in attributes if isinstance(item, Mapping)]
    else:
        return []
    best: Dict[str, Tuple[Tuple[float, str], str]] = {}
    for item in items:
        name = str(
            item.get("name") or item.get("attribute") or item.get("key") or ""
        ).strip()
        value = item.get("value")
        if not name or value in (None, ""):
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        rank = (confidence, str(item.get("updated_at") or item.get("created_at") or ""))
        if name not in best or rank >= best[name][0]:
            best[name] = (rank, str(value))
    return [(name, value) for name, (_rank, value) in best.items()]


def _entity_text(row: Mapping[str, Any]) -> str:
    """One entity as ``Ada (person): role = CTO; city = Lagos``."""
    name = str(row.get("name") or row.get("entity_id") or "entity")
    head = f"{name} ({row['entity_type']})" if row.get("entity_type") else name
    facts = "; ".join(
        f"{key} = {value}" for key, value in _entity_facts(row.get("attributes"))
    )
    lines = [f"{head}: {facts}" if facts else head]
    relations = ", ".join(
        f"{rel.get('relation_type') or 'related'} → {rel['entity_id']}"
        for rel in row.get("relations") or []
        if isinstance(rel, Mapping) and rel.get("entity_id")
    )
    if relations:
        lines.append(f"relations: {relations}")
    if isinstance(row.get("content"), str) and row["content"].strip():
        lines.append(row["content"].strip())
    return "\n".join(lines)


def _row_content(source_type: str, row: Mapping[str, Any]) -> str:
    if source_type == "workflow_memory":
        return "\n".join(
            str(item)
            for item in (
                row.get("name"),
                row.get("description"),
                row.get("user_query"),
                row.get("canonical_signature"),
                row.get("outcome"),
            )
            if item not in (None, "", [], {})
        )
    if source_type == "skillbox":
        return "\n".join(
            str(item)
            for item in (
                row.get("name"),
                row.get("description"),
                row.get("preconditions"),
                row.get("content"),
            )
            if item not in (None, "", [], {})
        )
    if source_type == "entity_memory":
        return _entity_text(row)
    value = row.get("content")
    if hasattr(value, "read"):
        try:
            value = value.read()
        except Exception:
            value = None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return str(value.get("content") or canonical_json(value))
    if value not in (None, ""):
        return str(value)
    return str(row.get("summary") or row.get("text") or "")


def _provider_score(
    row: Mapping[str, Any], rank: int, query: str, content: str
) -> float:
    for key in ("similarity", "score", "relevance_score"):
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
            if math.isfinite(numeric):
                return max(0.0, min(1.0, numeric))
        except (TypeError, ValueError):
            pass
    for key in ("distance", "vector_distance"):
        value = row.get(key)
        if value is None:
            continue
        try:
            distance = max(0.0, float(value))
            if math.isfinite(distance):
                return 1.0 / (1.0 + distance)
        except (TypeError, ValueError):
            pass
    lexical = _lexical_score(query, content)
    reciprocal_rank = 1.0 / max(1, rank)
    return min(1.0, lexical * 0.8 + reciprocal_rank * 0.2)


def _trust(source_type: str, row: Mapping[str, Any]) -> float:
    if row.get("verified") is True:
        return 1.0
    if source_type == "skillbox":
        return 0.85 if str(row.get("status")) == "active" else 0.45
    if source_type in {"knowledge_base", "summaries"}:
        return 0.75
    if source_type == "learning_artifacts":
        return 0.9 if row.get("verified") is True else 0.65
    return 0.6


def _scope_matches(
    row: Mapping[str, Any],
    *,
    source_type: str,
    agent_id: str,
    memory_id: Optional[str],
    user_id: Optional[str],
    thread_id: Optional[str],
    knowledge_base_ids: frozenset = frozenset(),
    recall: Optional["_RecallScope"] = None,
) -> bool:
    row_agent = row.get("agent_id") or row.get("owner_agent_id")
    # Knowledge/conversation/summary rows are shared through an explicit
    # memory_id + tenant grant; their agent_id is provenance, not ownership.
    # Procedural projections remain agent-owned.
    if (
        source_type
        in {"skillbox", "workflow_memory", "entity_memory", "learning_artifacts"}
        and row_agent is not None
        and str(row_agent) != str(agent_id)
    ):
        return False
    row_memory = row.get("trace_memory_id") or row.get("memory_id")
    if source_type == "entity_memory" and recall is not None:
        memory_id = recall.entity_memory_id
    if source_type == "knowledge_base":
        # Ingested chunks are keyed by their knowledge base, not by an agent's
        # memory: in scope when attached to the agent or carrying memory_id.
        attached = str(row.get("knowledge_base_id") or "") in knowledge_base_ids
        if not attached and str(row_memory or "") != str(memory_id or ""):
            return False
        if recall is not None and recall.namespaces:
            content = row.get("content")
            namespace = row.get("namespace") or (
                content.get("namespace") if isinstance(content, Mapping) else None
            )
            # The provider already filtered by namespace; a row that names a
            # different one is dropped, one that names none is kept.
            if namespace is not None and str(namespace) not in recall.namespaces:
                return False
    elif (
        memory_id is not None
        and row_memory is not None
        and str(row_memory) != str(memory_id)
    ):
        return False
    # Tenant isolation is fail-closed.  Explicit None selects anonymous rows;
    # a user also sees the agent-wide (no user) knowledge and skills.
    if not user_scope_matches(row.get("user_id"), user_id, source_type):
        return False
    thread_sources = {"conversation_memory"}
    if recall is not None:
        thread_id = recall.conversation_thread_id
        if recall.summaries_by_thread:
            thread_sources.add("summaries")
    if source_type in thread_sources and thread_id is not None:
        row_thread = row.get("thread_id") or row.get("conversation_id")
        if str(row_thread or "") != str(thread_id):
            return False
    if source_type == "skillbox" and str(row.get("status") or "active") != "active":
        return False
    if source_type == "workflow_memory" and row.get("promoted_skill_id"):
        return False
    return True


@dataclass(frozen=True)
class _RecallScope:
    """An agent's ``RetrievalPolicy`` applied to one pack."""

    conversation_thread_id: Optional[str]
    summaries_by_thread: bool
    skipped_sources: frozenset
    namespaces: Tuple[str, ...]
    entity_memory_id: Optional[str]

    @classmethod
    def build(
        cls,
        policy: Any,
        *,
        memory_id: Optional[str],
        thread_id: Optional[str],
        entity_memory_id: Optional[str],
    ) -> Optional["_RecallScope"]:
        if policy is None:
            return None
        skipped = set()
        if policy.conversation_scope == "disabled":
            skipped.add("conversation_memory")
        if policy.knowledge_base_scope == "disabled":
            skipped.add("knowledge_base")
        if policy.entity_scope == "disabled":
            skipped.add("entity_memory")
        by_thread = policy.conversation_scope == "thread"
        return cls(
            conversation_thread_id=thread_id if by_thread else None,
            summaries_by_thread=by_thread,
            skipped_sources=frozenset(skipped),
            namespaces=(
                tuple(policy.knowledge_base_namespaces)
                if policy.knowledge_base_scope == "namespace"
                else ()
            ),
            entity_memory_id=entity_memory_id or memory_id,
        )


class EvidencePlanner:
    """Create a bounded evidence pack and retain every selection decision."""

    def __init__(
        self,
        provider: Any,
        *,
        agent_id: str,
        config: LearningControlPlaneConfig,
        store: Optional[LearningControlPlaneStore] = None,
    ) -> None:
        self.provider = provider
        self.agent_id = str(agent_id)
        self.config = config
        self.store = store

    def _attached_knowledge_base_ids(self) -> frozenset:
        """The knowledge bases attached to the agent, from its stored record."""
        retrieve = getattr(self.provider, "retrieve_memagent", None)
        if not callable(retrieve):
            return frozenset()
        try:
            record = retrieve(self.agent_id)
        except Exception:
            return frozenset()
        if isinstance(record, Mapping):
            ids = record.get("knowledge_base_ids")
        else:
            ids = getattr(record, "knowledge_base_ids", None)
        return frozenset(str(value) for value in ids or () if value)

    def _provider_candidates(
        self,
        query: str,
        *,
        source_type: str,
        memory_id: Optional[str],
        user_id: Optional[str],
        thread_id: Optional[str],
        knowledge_base_ids: frozenset = frozenset(),
        recall: Optional[_RecallScope] = None,
    ) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
        memory_type = _SOURCE_TYPES[source_type]
        if recall is not None:
            thread_id = recall.conversation_thread_id
            if source_type == "entity_memory":
                memory_id = recall.entity_memory_id
            if source_type == "knowledge_base" and len(recall.namespaces) > 1:
                rows: List[Mapping[str, Any]] = []
                for namespace in recall.namespaces:
                    found, _ = self._provider_candidates(
                        query,
                        source_type=source_type,
                        memory_id=memory_id,
                        user_id=user_id,
                        thread_id=thread_id,
                        knowledge_base_ids=knowledge_base_ids,
                        recall=replace(recall, namespaces=(namespace,)),
                    )
                    rows.extend(found)
                return rows, {}
        if source_type == "entity_memory":
            supports = getattr(self.provider, "supports_entity_memory", None)
            if callable(supports) and not supports():
                return [], {
                    "retrieval_mode": "disabled",
                    "degraded": True,
                    "degraded_reason": "entity_memory_unsupported",
                }
            rows, diagnostics = EntityMemory(
                self.provider
            ).search_entities_with_diagnostics(
                query,
                limit=self.config.evidence_candidates_per_source,
                memory_id=memory_id,
                user_id=user_id,
            )
            annotated: List[Mapping[str, Any]] = []
            for row in rows:
                candidate = dict(row)
                candidate["retrieval_mode"] = diagnostics.get("retrieval_mode")
                candidate["retrieval_degraded"] = bool(diagnostics.get("degraded"))
                candidate["retrieval_reason"] = diagnostics.get("degraded_reason")
                annotated.append(candidate)
            return annotated, diagnostics

        # Skills belong to the agent, not to a memory: scope them by agent and
        # user (agent-wide skills included), never by memory_id.
        skill_candidates = getattr(self.provider, "retrieve_skillbox_candidates", None)
        if source_type == "skillbox" and callable(skill_candidates):
            rows = skill_candidates(
                query,
                limit=self.config.evidence_candidates_per_source,
                statuses=["active"],
                agent_id=self.agent_id,
                user_id=user_id,
            )
            return [item for item in rows or [] if isinstance(item, Mapping)], {}

        kwargs: Dict[str, Any] = {
            "user_id": user_id,
            "limit": self.config.evidence_candidates_per_source,
        }
        if source_type != "skillbox":
            kwargs["memory_id"] = memory_id
        if source_type == "knowledge_base" and knowledge_base_ids:
            kwargs["knowledge_base_ids"] = sorted(knowledge_base_ids)
        if source_type == "conversation_memory" and thread_id is not None:
            kwargs["thread_id"] = thread_id
        if source_type == "knowledge_base" and recall is not None and recall.namespaces:
            kwargs["namespace"] = recall.namespaces[0]
        try:
            result = self.provider.retrieve_by_query(
                query,
                memory_store_type=memory_type,
                **kwargs,
            )
        except TypeError:
            result = self.provider.retrieve_by_query(
                query,
                memory_store_type=memory_type,
                limit=self.config.evidence_candidates_per_source,
            )
        if result is None:
            return [], {}
        if isinstance(result, Mapping):
            return [result], {}
        return [item for item in result if isinstance(item, Mapping)], {}

    def _artifact_candidates(
        self,
        query: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
        thread_id: Optional[str],
    ) -> List[Mapping[str, Any]]:
        if self.store is None:
            return []
        rows = self.store.list_artifacts(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            limit=max(100, self.config.evidence_candidates_per_source * 20),
        )
        rows.sort(
            key=lambda row: _lexical_score(
                query, str(row.get("content") or row.get("summary") or "")
            ),
            reverse=True,
        )
        return rows[: self.config.evidence_candidates_per_source]

    def _collect_source_rows(
        self,
        query: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
        thread_id: Optional[str],
        knowledge_base_ids: frozenset = frozenset(),
        recall: Optional[_RecallScope] = None,
    ) -> Tuple[List[Tuple[str, Mapping[str, Any]]], List[str]]:
        rows_by_source: List[Tuple[str, Mapping[str, Any]]] = []
        warnings: List[str] = []
        for source_type in self.config.evidence_sources:
            if recall is not None and source_type in recall.skipped_sources:
                continue
            try:
                if source_type == "learning_artifacts":
                    rows = self._artifact_candidates(
                        query,
                        memory_id=memory_id,
                        user_id=user_id,
                        thread_id=thread_id,
                    )
                    diagnostics: Mapping[str, Any] = {}
                elif source_type in _SOURCE_TYPES:
                    rows, diagnostics = self._provider_candidates(
                        query,
                        source_type=source_type,
                        memory_id=memory_id,
                        user_id=user_id,
                        thread_id=thread_id,
                        knowledge_base_ids=knowledge_base_ids,
                        recall=recall,
                    )
                else:
                    warnings.append(f"unknown evidence source: {source_type}")
                    continue
            except Exception as exc:
                warnings.append(f"{source_type} retrieval failed: {type(exc).__name__}")
                continue
            if diagnostics.get("degraded"):
                reason = diagnostics.get("degraded_reason") or "unknown"
                warnings.append(f"{source_type} retrieval degraded: {reason}")
            rows_by_source.extend((source_type, row) for row in rows)
        return rows_by_source, warnings

    def _candidate_from_row(
        self,
        *,
        query: str,
        rank: int,
        source_type: str,
        row: Mapping[str, Any],
        memory_id: Optional[str],
        user_id: Optional[str],
        thread_id: Optional[str],
        tombstones: set[str],
        history_hashes: set[str],
        freshness: Mapping[str, int],
        knowledge_base_ids: frozenset = frozenset(),
        recall: Optional[_RecallScope] = None,
    ) -> Optional[EvidenceItem]:
        if source_type in {"conversation_memory", "summaries"} and sealed_elsewhere(
            row, thread_id
        ):
            # Another thread's sealed conversation stays there.
            return None
        if not _scope_matches(
            row,
            source_type=source_type,
            agent_id=self.agent_id,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            knowledge_base_ids=knowledge_base_ids,
            recall=recall,
        ):
            return None
        source_id = _row_id(row)
        if source_id in tombstones:
            return None
        if is_suppressed(row):
            return None
        content = _row_content(source_type, row).strip()
        if not content:
            return None
        content_hash = content_digest(" ".join(content.split()).lower())
        if content_hash in history_hashes:
            return None

        time_field = next((key for key in _TIME_FIELDS if row.get(key)), "timestamp")
        timestamp, age_seconds = _coerce_timestamp(
            row.get(time_field), naive_time_convention(source_type, time_field)
        )
        max_age = freshness.get(source_type)
        stale = bool(max_age and age_seconds is not None and age_seconds > max_age)
        trust = _trust(source_type, row)
        relevance = min(
            1.0,
            _provider_score(row, rank, query, content) * 0.72
            + _lexical_score(query, content) * 0.18
            + trust * 0.10,
        ) * (0.7 if stale else 1.0)
        if relevance < self.config.evidence_min_relevance:
            return None

        metadata_keys = (
            "namespace",
            "source_type",
            "source_key",
            "status",
            "artifact_kind",
            "outcome",
            "verified",
            "memory_id",
            "thread_id",
            "user_id",
            "agent_id",
            "claim_key",
            "retrieval_mode",
            "retrieval_degraded",
            "retrieval_reason",
        )
        metadata = {
            key: row.get(key) for key in metadata_keys if row.get(key) is not None
        }
        claim_key = row.get("claim_key")
        return EvidenceItem(
            evidence_id="evidence-"
            + content_digest(
                {"source": source_type, "id": source_id, "hash": content_hash}
            )[:28],
            source_type=source_type,
            source_id=source_id,
            content=content,
            relevance=relevance,
            token_estimate=estimate_tokens(content),
            content_hash=content_hash,
            reason=(
                "hybrid semantic/lexical match with scoped provenance"
                + ("; freshness penalty applied" if stale else "")
            ),
            timestamp=timestamp,
            age_seconds=age_seconds,
            trust=trust,
            stale=stale,
            contradiction_group=str(claim_key) if claim_key else None,
            metadata=metadata,
        )

    @staticmethod
    def _mark_contradictions(candidates: List[EvidenceItem]) -> List[EvidenceItem]:
        group_hashes: Dict[str, set[str]] = {}
        for item in candidates:
            if item.contradiction_group:
                group_hashes.setdefault(item.contradiction_group, set()).add(
                    item.content_hash
                )
        contradictions = {
            key for key, hashes in group_hashes.items() if len(hashes) > 1
        }
        if not contradictions:
            return candidates
        return [
            replace(
                item,
                contradiction_group=(
                    item.contradiction_group
                    if item.contradiction_group in contradictions
                    else None
                ),
            )
            for item in candidates
        ]

    def _select_candidates(
        self, candidates: Sequence[EvidenceItem]
    ) -> Tuple[List[EvidenceItem], int]:
        selected: List[EvidenceItem] = []
        selected_hashes: set[str] = set()
        per_source: Counter[str] = Counter()
        tokens_used = 0
        for candidate in candidates:
            if len(selected) >= self.config.evidence_max_items:
                break
            if candidate.content_hash in selected_hashes:
                continue
            if per_source[candidate.source_type] >= self.config.evidence_max_per_source:
                continue
            if (
                selected
                and tokens_used + candidate.token_estimate
                > self.config.evidence_token_budget
            ):
                continue
            item = candidate
            if not selected and item.token_estimate > self.config.evidence_token_budget:
                trimmed = item.content[: max(80, self.config.evidence_token_budget * 4)]
                item = replace(
                    item,
                    content=trimmed,
                    token_estimate=estimate_tokens(trimmed),
                    reason=item.reason + "; truncated to evidence budget",
                )
            selected.append(item)
            selected_hashes.add(item.content_hash)
            per_source[item.source_type] += 1
            tokens_used += item.token_estimate
        return selected, tokens_used

    def build(
        self,
        query: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        history_texts: Optional[Sequence[str]] = None,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        retrieval_policy: Any = None,
        entity_memory_id: Optional[str] = None,
    ) -> EvidencePack:
        """Build the pack. ``knowledge_base_ids`` are the knowledge bases
        attached to the agent; omitted, they are read from its stored record.

        ``retrieval_policy`` is the agent's ``RetrievalPolicy`` for the turn:
        its conversation, knowledge-base and entity scopes decide which
        sources are read and how. Without one, conversation evidence comes
        from the current thread only. ``entity_memory_id`` is where the
        turn's entity facts live when they are kept per thread."""
        started = time.perf_counter()
        query_text = str(query or "").strip()
        if not query_text:
            raise ValueError("EvidencePlanner query is required")
        if knowledge_base_ids is not None:
            attached = frozenset(str(value) for value in knowledge_base_ids if value)
        elif "knowledge_base" in self.config.evidence_sources:
            attached = self._attached_knowledge_base_ids()
        else:
            attached = frozenset()
        tombstones = (
            self.store.tombstoned_ids(memory_id=memory_id, user_id=user_id)
            if self.store is not None
            else set()
        )
        history_hashes = {
            content_digest(" ".join(str(text).split()).lower())
            for text in (history_texts or [])
            if str(text).strip()
        }
        recall = _RecallScope.build(
            retrieval_policy,
            memory_id=memory_id,
            thread_id=thread_id,
            entity_memory_id=entity_memory_id,
        )
        source_rows, warnings = self._collect_source_rows(
            query_text,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            knowledge_base_ids=attached,
            recall=recall,
        )
        freshness = self.config.freshness_map()
        candidates: List[EvidenceItem] = []
        for rank, (source_type, row) in enumerate(source_rows, start=1):
            candidate = self._candidate_from_row(
                query=query_text,
                rank=rank,
                source_type=source_type,
                row=row,
                memory_id=memory_id,
                user_id=user_id,
                thread_id=thread_id,
                tombstones=tombstones,
                history_hashes=history_hashes,
                freshness=freshness,
                knowledge_base_ids=attached,
                recall=recall,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates = self._mark_contradictions(candidates)
        candidates.sort(
            key=lambda item: (item.relevance, item.trust, -(item.age_seconds or 0.0)),
            reverse=True,
        )
        candidate_tokens = sum(item.token_estimate for item in candidates)
        selected, used = self._select_candidates(candidates)

        created_at = utc_now()
        scope = {
            "agent_id": self.agent_id,
            "memory_id": memory_id,
            "user_id": user_id,
            "thread_id": thread_id,
        }
        policy_hash = content_digest(
            self.config.to_dict()
            if retrieval_policy is None
            else {
                **self.config.to_dict(),
                "retrieval_policy": retrieval_policy.to_dict(),
            }
        )
        pack_id = (
            "evidence-pack-"
            + hashlib.sha256(
                canonical_json(
                    {
                        "query": query_text,
                        "scope": scope,
                        "policy": policy_hash,
                        "created_at": created_at,
                    }
                ).encode()
            ).hexdigest()[:28]
        )
        return EvidencePack(
            pack_id=pack_id,
            query=query_text,
            items=tuple(selected),
            created_at=created_at,
            token_budget=self.config.evidence_token_budget,
            tokens_used=used,
            candidate_tokens=candidate_tokens,
            candidate_count=len(candidates),
            rejected_count=max(0, len(candidates) - len(selected)),
            scope=scope,
            query_hash=content_digest(query_text),
            policy_hash=policy_hash,
            latency_ms=(time.perf_counter() - started) * 1000,
            warnings=tuple(warnings),
        )


__all__ = ["EvidencePlanner"]

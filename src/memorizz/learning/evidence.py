"""Explainable, scoped and token-budgeted retrieval for the control plane."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..enums.memory_type import MemoryType
from ..long_term.semantic.entity_memory import EntityMemory
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


def _coerce_timestamp(value: Any) -> Tuple[Optional[str], Optional[float]]:
    if value is None:
        return None, None
    if hasattr(value, "isoformat"):
        try:
            text = value.isoformat()
        except Exception:
            text = str(value)
    else:
        text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
        return parsed.isoformat(), age
    except (TypeError, ValueError):
        return text, None


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
        return "\n".join(
            str(item)
            for item in (
                row.get("name"),
                row.get("entity_type"),
                row.get("attributes"),
                row.get("relations"),
                row.get("content"),
            )
            if item not in (None, "", [], {})
        )
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
    if (
        memory_id is not None
        and row_memory is not None
        and str(row_memory) != str(memory_id)
    ):
        return False
    # Tenant isolation is fail-closed.  Explicit None selects anonymous rows.
    if row.get("user_id") != user_id:
        return False
    if source_type == "conversation_memory" and thread_id is not None:
        row_thread = row.get("thread_id") or row.get("conversation_id")
        if str(row_thread or "") != str(thread_id):
            return False
    if source_type == "skillbox" and str(row.get("status") or "active") != "active":
        return False
    if source_type == "workflow_memory" and row.get("promoted_skill_id"):
        return False
    return True


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

    def _provider_candidates(
        self,
        query: str,
        *,
        source_type: str,
        memory_id: Optional[str],
        user_id: Optional[str],
        thread_id: Optional[str],
    ) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
        memory_type = _SOURCE_TYPES[source_type]
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

        kwargs: Dict[str, Any] = {
            "memory_id": memory_id,
            "user_id": user_id,
            "limit": self.config.evidence_candidates_per_source,
        }
        if source_type == "conversation_memory" and thread_id is not None:
            kwargs["thread_id"] = thread_id
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
    ) -> Tuple[List[Tuple[str, Mapping[str, Any]]], List[str]]:
        rows_by_source: List[Tuple[str, Mapping[str, Any]]] = []
        warnings: List[str] = []
        for source_type in self.config.evidence_sources:
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
    ) -> Optional[EvidenceItem]:
        if not _scope_matches(
            row,
            source_type=source_type,
            agent_id=self.agent_id,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
        ):
            return None
        source_id = _row_id(row)
        if source_id in tombstones:
            return None
        content = _row_content(source_type, row).strip()
        if not content:
            return None
        content_hash = content_digest(" ".join(content.split()).lower())
        if content_hash in history_hashes:
            return None

        timestamp, age_seconds = _coerce_timestamp(
            row.get("timestamp")
            or row.get("updated_at")
            or row.get("created_at")
            or row.get("period_end")
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
    ) -> EvidencePack:
        started = time.perf_counter()
        query_text = str(query or "").strip()
        if not query_text:
            raise ValueError("EvidencePlanner query is required")
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
        source_rows, warnings = self._collect_source_rows(
            query_text,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
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
        policy_hash = content_digest(self.config.to_dict())
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

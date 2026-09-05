# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Pre-inference deduplication and selection for retrieved memory candidates.

Everything retrieved for a turn (episodic recall, knowledge-base chunks,
entity facts) passes through :func:`dedupe_and_select` before it is rendered
into the context window. The pipeline is the standard assembly-time stack:

1. Exact dedup — normalized-content fingerprints, cheapest first.
2. History dedup — drop candidates already present (verbatim or contained)
   in the conversation window going into this prompt, so the model never
   pays twice for the same text.
3. Near-dup dedup — cosine similarity >= ``similarity_threshold`` between
   candidate embeddings (vectors come back from the store with the rows, so
   this costs no extra embedding calls).
4. MMR selection — maximal marginal relevance balances query relevance
   against diversity; near-identical survivors score ~zero marginal
   relevance, making this an implicit second dedup stage. Relevance blends
   cosine-to-query with a recency decay when timestamps are available.
5. Stable ordering — selected items are emitted oldest-first, ordered by
   (timestamp, id) rather than relevance score, so the rendered block is
   deterministic across turns (re-ranking retrieved chunks between turns is
   a documented prompt-cache killer).

All functions are dependency-free (no numpy) — candidate sets are tiny
(tens of items), so pure-Python cosine is more than fast enough.
"""

import hashlib
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")

# Cosine similarity at/above which two memories are treated as the same fact.
# Published guidance puts the "same fact" band at 0.95–0.99; agent memories
# tolerate the looser end because paraphrased repeats are common.
DEFAULT_SIMILARITY_THRESHOLD = 0.95

# MMR relevance/diversity balance (0.5 = equal weight, the common default).
DEFAULT_MMR_LAMBDA = 0.5

# Blend weight of recency into MMR relevance when timestamps exist.
DEFAULT_RECENCY_WEIGHT = 0.25

# Exponential recency decay per hour since the memory was written
# (Generative-Agents style scoring, adapted from per-game-hour to wall-clock).
_RECENCY_DECAY_PER_HOUR = 0.995

_WORKFLOW_SOURCE_LABELS = frozenset(
    {
        "workflow",
        "workflows",
        "workflow_memory",
        "procedural_workflow",
    }
)
_PROVENANCE_FIELDS = (
    "source_id",
    "parent_source_id",
    "linked_source_ids",
    "memory_type",
    "workflow_id",
    "canonical_hash",
    "promoted_skill_id",
)


def normalize_text(text: Any) -> str:
    """Lowercase and collapse whitespace so cosmetic diffs don't defeat dedup."""
    return _WHITESPACE_RE.sub(" ", str(text or "").strip().lower())


def content_fingerprint(text: Any) -> str:
    """Stable fingerprint of normalized content for exact-duplicate detection."""
    normalized = normalize_text(text)
    return hashlib.sha1(normalized.encode("utf-8", errors="replace")).hexdigest()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Pure-Python cosine similarity; None when either vector is unusable."""
    if not a or not b or len(a) != len(b):
        return None
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return dot / math.sqrt(norm_a * norm_b)


def _parse_timestamp(value: Any) -> Optional[float]:
    """Best-effort conversion of stored timestamp shapes to epoch seconds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _recency_score(timestamp: Optional[float], now: Optional[float] = None) -> float:
    """Exponential decay in [0, 1]; 1.0 for missing timestamps (no penalty)."""
    if timestamp is None:
        return 1.0
    if now is None:
        now = datetime.now(timezone.utc).timestamp()
    hours = max(0.0, (now - timestamp) / 3600.0)
    return _RECENCY_DECAY_PER_HOUR**hours


def candidate_text(row: Any) -> str:
    """Extract display text from the row shapes memory providers return.

    Handles conversation rows (``{"content": {"role", "content"}}``),
    knowledge-base rows (``{"content": "..."}``), and pre-shaped candidates
    (``{"text": "..."}``).
    """
    if row is None:
        return ""
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return str(row)
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text
    content = row.get("content")
    if isinstance(content, dict):
        nested = content.get("content") or content.get("text")
        if nested is not None:
            return str(nested)
        return ""
    if content is not None:
        return str(content)
    return ""


def candidate_embedding(row: Any) -> Optional[List[float]]:
    """Extract the stored embedding vector from a provider row, if present."""
    if not isinstance(row, dict):
        return None
    for key in ("embedding", "vector"):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and value:
            try:
                return [float(x) for x in value]
            except (TypeError, ValueError):
                return None
    content = row.get("content")
    if isinstance(content, dict):
        value = content.get("embedding")
        if isinstance(value, (list, tuple)) and value:
            try:
                return [float(x) for x in value]
            except (TypeError, ValueError):
                return None
    return None


def _candidate_timestamp(row: Any) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    for key in ("timestamp", "created_at", "createdAt", "period_end"):
        parsed = _parse_timestamp(row.get(key))
        if parsed is not None:
            return parsed
    content = row.get("content")
    if isinstance(content, dict):
        return _parse_timestamp(content.get("timestamp"))
    return None


def _candidate_id(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in (
        "parent_source_id",
        "source_id",
        "_id",
        "id",
        "memory_id",
        "summary_id",
    ):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _candidate_score(row: Any) -> float:
    """Provider similarity score (vector-search rank), 0.0 when absent."""
    if not isinstance(row, dict):
        return 0.0
    try:
        return float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _provenance_width(row: Any) -> int:
    linked = _field(row, "linked_source_ids")
    if isinstance(linked, (list, tuple, set)):
        values = {str(item) for item in linked if str(item)}
        if values:
            return len(values)
    return int(bool(_field(row, "parent_source_id") or _field(row, "source_id")))


def _field(value: Any, name: str) -> Any:
    """Read a field from an object or provider-shaped mapping."""
    if isinstance(value, dict):
        direct = value.get(name)
        if direct is not None:
            return direct
        for container_name in ("content", "metadata"):
            nested = value.get(container_name)
            if isinstance(nested, dict) and nested.get(name) is not None:
                return nested.get(name)
        return None
    return getattr(value, name, None)


def _candidate_provenance(row: Any) -> Dict[str, Any]:
    """Keep only the fields needed for downstream context suppression."""
    return {
        field: value
        for field in _PROVENANCE_FIELDS
        if (value := _field(row, field)) is not None
    }


def _identity_values(value: Any) -> set[str]:
    if value is None:
        return set()
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    return {str(item) for item in values if item is not None and str(item)}


def _is_workflow_memory(item: Any) -> bool:
    """Return whether a selected context item represents a raw workflow."""
    source = _field(item, "source")
    memory_type = _field(item, "memory_type")
    for value in (source, memory_type):
        if value is None:
            continue
        label = getattr(value, "value", value)
        normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
        if (
            normalized in _WORKFLOW_SOURCE_LABELS
            or normalized.rsplit(".", 1)[-1] in _WORKFLOW_SOURCE_LABELS
        ):
            return True
    # Provider workflow documents are self-identifying even if a direct
    # caller omitted the source label while shaping retrieved context.
    return _field(item, "workflow_id") is not None


def filter_skill_covered_workflows(
    retrieved_memories: Iterable[Dict[str, Any]],
    activated_skills: Iterable[Any],
) -> List[Dict[str, Any]]:
    """Prevent a learned skill and its source workflows sharing one prompt.

    Promotion stamps source workflows, but this assembly-time filter is the
    final correctness boundary for callers that manually supply retrieved
    context or bypass :meth:`Workflow.retrieve_workflows_by_query`.
    Workflow items without enough provenance to prove they are distinct are
    dropped while a learned skill is being rendered.
    """
    memories = list(retrieved_memories or [])
    scored_skills = list(activated_skills or [])
    if not memories or not scored_skills:
        return memories

    skill_ids: set[str] = set()
    canonical_hashes: set[str] = set()
    workflow_ids: set[str] = set()
    for scored in scored_skills:
        skill = _field(scored, "skill") or scored
        skill_ids.update(_identity_values(_field(skill, "skill_id")))
        canonical_hashes.update(
            _identity_values(_field(skill, "source_canonical_hash"))
        )
        workflow_ids.update(_identity_values(_field(skill, "source_workflow_ids")))
        workflow_ids.update(_identity_values(_field(skill, "exemplar_workflow_id")))

    kept: List[Dict[str, Any]] = []
    suppressed = 0
    for item in memories:
        if not _is_workflow_memory(item):
            kept.append(item)
            continue

        promoted_ids = _identity_values(_field(item, "promoted_skill_id"))
        item_hashes = _identity_values(_field(item, "canonical_hash"))
        item_workflow_ids = _identity_values(_field(item, "workflow_id"))
        item_workflow_ids.update(_identity_values(_field(item, "id")))
        has_provenance = bool(promoted_ids or item_hashes or item_workflow_ids)
        covered = bool(
            promoted_ids.intersection(skill_ids)
            or item_hashes.intersection(canonical_hashes)
            or item_workflow_ids.intersection(workflow_ids)
        )
        if covered or not has_provenance:
            suppressed += 1
            continue
        kept.append(item)

    if suppressed:
        logger.debug(
            "Suppressed %d raw workflow context item(s) covered by rendered skills",
            suppressed,
        )
    return kept


def dedupe_and_select(
    candidates: Iterable[Tuple[str, Any]],
    *,
    history_texts: Iterable[Any] = (),
    query_embedding: Optional[Sequence[float]] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
    max_items: int = 5,
    max_chars_per_item: int = 600,
    dedupe_parent_sources: bool = False,
    selection_ledger: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Dedupe retrieved memory rows and select a diverse, budgeted subset.

    Parameters
    ----------
    candidates : iterable of (source, row)
        ``source`` labels where the row came from (``"episodic"``,
        ``"knowledge_base"``, ...); ``row`` is the raw provider dict.
    history_texts : iterable
        Message texts already going into this prompt's conversation window.
        Candidates whose normalized text matches — or is contained in — any
        of these are dropped (the model already sees them).
    query_embedding : sequence of float, optional
        Embedding of the user query; enables MMR relevance ranking.
    max_items : int
        Hard cap on selected memories (post-dedup).
    max_chars_per_item : int
        Rendered text is truncated to this many characters per memory.
    dedupe_parent_sources : bool
        Keep only the highest-scoring candidate for a shared parent source.
        This is useful when an original record and a derived semantic record
        are retrieved together. It is opt-in because independent chunks from
        one long source can both be useful in general agent workloads.

    Returns
    -------
    list of dict
        Selected candidates as ``{"source", "text", "id", "timestamp"}``
        plus any workflow provenance required for prompt-time suppression,
        ordered oldest-first with deterministic tie-breaks.
    """
    history_fingerprints = set()
    history_normalized: List[str] = []
    for text in history_texts:
        normalized = normalize_text(text)
        if not normalized:
            continue
        history_fingerprints.add(content_fingerprint(normalized))
        history_normalized.append(normalized)

    fingerprint_indexes: Dict[str, int] = {}
    pool: List[Dict[str, Any]] = []

    def mark(candidate, reason, selected=False):
        entry = candidate.get("_decision")
        if entry is not None:
            entry.update(selection_reason=reason, selected=selected)

    for rank, (source, row) in enumerate(candidates, 1):
        decision = None
        if selection_ledger is not None and len(selection_ledger) < 64:
            from ...observability.privacy import pseudonym

            score = _candidate_score(row)
            decision = {
                "resource": {
                    "resource_type": "memory",
                    "ref": pseudonym(
                        _candidate_id(row) or content_fingerprint(candidate_text(row)),
                        scope="memory-ref",
                    ),
                    "role": "retrieved_supporting_evidence",
                },
                "candidate_rank": rank,
                "relevance_score": score if math.isfinite(score) else None,
                "selected": False,
                "selection_reason": "selection_not_observed",
            }
            selection_ledger.append(decision)
        text = candidate_text(row).strip()
        if not text:
            if decision is not None:
                decision["selection_reason"] = "not_ready"
            continue
        normalized = normalize_text(text)
        fingerprint = content_fingerprint(normalized)

        # 2. Already in the conversation window (verbatim or contained).
        if fingerprint in history_fingerprints:
            if decision is not None:
                decision["selection_reason"] = "already_in_history"
            continue
        if any(normalized in h for h in history_normalized):
            if decision is not None:
                decision["selection_reason"] = "already_in_history"
            continue

        shaped = {
            "_decision": decision,
            "source": source,
            "text": text,
            "normalized": normalized,
            "embedding": candidate_embedding(row),
            "timestamp": _candidate_timestamp(row),
            "id": _candidate_id(row),
            "score": _candidate_score(row),
            **_candidate_provenance(row),
        }

        # 1. Exact dedup across sources and query variants. A later expanded
        # query can return the same row with a stronger score; retain that
        # evidence instead of freezing the first variant's weaker score.
        existing_index = fingerprint_indexes.get(fingerprint)
        if existing_index is not None:
            current = pool[existing_index]
            if shaped["score"] > current["score"]:
                mark(current, "exact_duplicate")
                pool[existing_index] = shaped
            else:
                mark(shaped, "exact_duplicate")
            continue
        fingerprint_indexes[fingerprint] = len(pool)
        pool.append(shaped)

    if not pool:
        return []

    # 3. Near-duplicate dedup via stored embeddings (order: higher provider
    # score first so the better-ranked copy of a near-dup pair survives).
    if dedupe_parent_sources:
        representatives: Dict[str, Dict[str, Any]] = {}
        ungrouped: List[Dict[str, Any]] = []
        for candidate in pool:
            identity = str(
                candidate.get("parent_source_id") or candidate.get("source_id") or ""
            ).strip()
            if not identity:
                ungrouped.append(candidate)
                continue
            current = representatives.get(identity)
            if current is None or (_provenance_width(candidate), candidate["score"]) > (
                _provenance_width(current),
                current["score"],
            ):
                if current is not None:
                    mark(current, "parent_source_duplicate")
                representatives[identity] = candidate
            else:
                mark(candidate, "parent_source_duplicate")
        pool = [*representatives.values(), *ungrouped]
    pool.sort(key=lambda c: (-c["score"], c["id"]))
    kept: List[Dict[str, Any]] = []
    seen_parent_sources: set[str] = set()
    for cand in pool:
        parent_source = str(
            cand.get("parent_source_id") or cand.get("source_id") or ""
        ).strip()
        if dedupe_parent_sources and parent_source in seen_parent_sources:
            mark(cand, "parent_source_duplicate")
            continue
        is_dup = False
        if cand["embedding"] is not None:
            for other in kept:
                if other["embedding"] is None:
                    continue
                sim = cosine_similarity(cand["embedding"], other["embedding"])
                if sim is not None and sim >= similarity_threshold:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(cand)
            if dedupe_parent_sources and parent_source:
                seen_parent_sources.add(parent_source)
        else:
            mark(cand, "near_duplicate")

    # 4. Selection. With a query embedding, run MMR over relevance
    # (cosine-to-query blended with recency); otherwise fall back to the
    # provider score order already established above.
    now = datetime.now(timezone.utc).timestamp()
    if query_embedding is not None:
        query_vec = list(query_embedding)
        for cand in kept:
            relevance = None
            if cand["embedding"] is not None:
                relevance = cosine_similarity(query_vec, cand["embedding"])
            if relevance is None:
                relevance = cand["score"]
            else:
                # A candidate may have been found by a label-blind expanded
                # query. Preserve that provider evidence instead of scoring it
                # only against the less-specific original query.
                relevance = max(relevance, cand["score"])
            recency = _recency_score(cand["timestamp"], now)
            cand["_relevance"] = (
                1.0 - recency_weight
            ) * relevance + recency_weight * recency

        selected: List[Dict[str, Any]] = []
        remaining = list(kept)
        while remaining and len(selected) < max_items:
            best = None
            best_value = -math.inf
            for cand in remaining:
                max_sim_to_selected = 0.0
                if cand["embedding"] is not None:
                    for chosen in selected:
                        if chosen["embedding"] is None:
                            continue
                        sim = cosine_similarity(cand["embedding"], chosen["embedding"])
                        if sim is not None:
                            max_sim_to_selected = max(max_sim_to_selected, sim)
                value = (
                    mmr_lambda * cand["_relevance"]
                    - (1.0 - mmr_lambda) * max_sim_to_selected
                )
                if value > best_value:
                    best_value = value
                    best = cand
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)
    else:
        selected = kept[:max_items]

    for candidate in kept:
        mark(candidate, "budget_exceeded")
    for candidate in selected:
        mark(
            candidate,
            "mmr_selected" if query_embedding is not None else "provider_rank_selected",
            selected=True,
        )

    # 5. Deterministic, chronology-preserving render order.
    selected.sort(key=lambda c: (c["timestamp"] or 0.0, c["id"], c["normalized"]))

    results: List[Dict[str, Any]] = []
    for cand in selected:
        text = cand["text"]
        if max_chars_per_item and len(text) > max_chars_per_item:
            text = text[: max_chars_per_item - 1] + "…"
        result = {
            "source": cand["source"],
            "text": text,
            "id": cand["id"],
            "timestamp": cand["timestamp"],
        }
        for field in _PROVENANCE_FIELDS:
            if cand.get(field) is not None:
                result[field] = cand[field]
        results.append(result)
    return results

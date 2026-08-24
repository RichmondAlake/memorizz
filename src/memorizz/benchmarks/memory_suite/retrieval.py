"""Deterministic, label-blind retrieval fusion for memory evaluations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from ...retrieval_concepts import concept_labels, expand_query_concepts

QueryExpander = Callable[[str], Sequence[str]]


@dataclass(frozen=True)
class FusionConfig:
    """Small set of calibrated retrieval controls."""

    top_k: int = 8
    candidate_pool_size: int = 256
    semantic_weight: float = 1.0
    lexical_weight: float = 0.35
    entity_weight: float = 0.05
    temporal_weight: float = 0.05
    rerank_weight: float = 0.15
    concept_weight: float = 0.20
    diversity_lambda: float = 0.9
    rrf_constant: int = 60
    max_query_variants: int = 3

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.candidate_pool_size < self.top_k:
            raise ValueError("candidate_pool_size must be at least top_k")
        if not 0.0 <= self.diversity_lambda <= 1.0:
            raise ValueError("diversity_lambda must be between 0 and 1")
        if self.rrf_constant < 1:
            raise ValueError("rrf_constant must be positive")
        if self.rerank_weight < 0:
            raise ValueError("rerank_weight cannot be negative")
        if self.concept_weight < 0:
            raise ValueError("concept_weight cannot be negative")


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+", flags=re.IGNORECASE)


def _tokens(value: Any) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(str(value or ""))}


def _entity_tokens(value: Any) -> set[str]:
    text = str(value or "")
    quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", text)
    capitalized = re.findall(r"\b[A-Z][A-Za-z0-9_-]{1,}\b", text)
    numbers = re.findall(r"\b\d[\d._:-]*\b", text)
    return _tokens(" ".join([*quoted, *capitalized, *numbers]))


def _row_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    parent = str(row.get("parent_source_id") or "").strip()
    if parent:
        return ("parent", parent)
    source = str(row.get("source_id") or row.get("_id") or row.get("id") or "")
    return ("source", source)


def _provenance_width(row: Mapping[str, Any]) -> int:
    linked = {str(item) for item in row.get("linked_source_ids") or [] if str(item)}
    if linked:
        return len(linked)
    return int(bool(row.get("parent_source_id") or row.get("source_id")))


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _temporal_score(row: Mapping[str, Any], query_time: datetime | None) -> float:
    if query_time is None:
        return 0.0
    metadata = row.get("benchmark_metadata") or row.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        return 0.0
    event_time = _parse_time(metadata.get("event_time") or metadata.get("timestamp"))
    if event_time is None:
        return 0.0
    try:
        days = abs((query_time - event_time).total_seconds()) / 86_400
    except TypeError:
        return 0.0
    return 1.0 / (1.0 + days / 30.0)


def _similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_tokens = _tokens(left.get("content"))
    right_tokens = _tokens(right.get("content"))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def build_query_variants(
    query: str,
    *,
    expander: QueryExpander | None = None,
    max_variants: int = 3,
    expand_concepts: bool = True,
) -> List[str]:
    """Return stable, de-duplicated variants without consulting gold labels."""

    candidates = [str(query or "").strip()]
    if expand_concepts:
        candidates.extend(expand_query_concepts(query))
    if expander is not None:
        candidates.extend(str(item).strip() for item in expander(query) or [])
    output: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = " ".join(candidate.split()).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(candidate)
        if len(output) >= max(1, int(max_variants)):
            break
    return output


def fuse_rankings(
    rankings: Sequence[Tuple[str, Sequence[Mapping[str, Any]], float]],
    *,
    question: str,
    config: FusionConfig,
    query_time: Any = None,
    query_variants: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Fuse ranked lanes with weighted RRF, light signals, and MMR diversity."""

    rows_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    representative_scores: Dict[Tuple[str, str], float] = {}
    lane_ranks: Dict[Tuple[str, str], Dict[str, int]] = {}
    group_scores: Dict[Tuple[str, str], Dict[str, float]] = {}
    group_weights: Dict[str, float] = {}
    for lane_name, rows, weight in rankings:
        lane_weight = max(0.0, float(weight))
        if lane_weight <= 0:
            continue
        lane_group = "semantic" if lane_name.startswith("semantic:") else lane_name
        group_weights[lane_group] = max(group_weights.get(lane_group, 0.0), lane_weight)
        seen_in_lane = set()
        for rank, row in enumerate(rows, start=1):
            key = _row_key(row)
            if key in seen_in_lane:
                # Original, derived, and grouped semantic records can share a
                # parent. Keep one rank contribution, but retain the version
                # with the broadest explicit source provenance for grounding.
                current = rows_by_key.get(key) or {}
                if _provenance_width(row) > _provenance_width(current):
                    rows_by_key[key] = dict(row)
                continue
            seen_in_lane.add(key)
            lane_ranks.setdefault(key, {})[lane_name] = rank
            contribution = lane_weight / (config.rrf_constant + rank)
            if contribution > representative_scores.get(key, -math.inf):
                rows_by_key[key] = dict(row)
                representative_scores[key] = contribution
            grouped = group_scores.setdefault(key, {})
            grouped[lane_group] = max(grouped.get(lane_group, 0.0), contribution)

    if not rows_by_key:
        return []
    maximum_rrf = sum(group_weights.values()) / (config.rrf_constant + 1)
    query_entities = _entity_tokens(question)
    query_concepts = set(concept_labels(" ".join([question, *query_variants])))
    rerank_tokens = _tokens(" ".join([question, *query_variants]))
    parsed_query_time = _parse_time(query_time)
    candidates: List[Dict[str, Any]] = []
    for key, row in rows_by_key.items():
        row_entities = _entity_tokens(row.get("content"))
        entity_score = (
            len(query_entities & row_entities) / len(query_entities)
            if query_entities
            else 0.0
        )
        temporal_score = _temporal_score(row, parsed_query_time)
        metadata = row.get("benchmark_metadata") or row.get("metadata") or {}
        stored_concepts = (
            metadata.get("semantic_concepts") if isinstance(metadata, Mapping) else ()
        )
        row_concepts = set(stored_concepts or ()) | set(
            concept_labels(str(row.get("content") or ""))
        )
        concept_score = (
            len(query_concepts & row_concepts) / len(query_concepts)
            if query_concepts
            else 0.0
        )
        row_tokens = _tokens(row.get("content"))
        rerank_score = (
            len(rerank_tokens & row_tokens)
            / math.sqrt(len(rerank_tokens) * len(row_tokens))
            if rerank_tokens and row_tokens
            else 0.0
        )
        raw_rrf = sum(group_scores.get(key, {}).values())
        normalized_rrf = raw_rrf / maximum_rrf if maximum_rrf else 0.0
        fused_score = (
            normalized_rrf
            + config.entity_weight * entity_score
            + config.temporal_weight * temporal_score
            + config.rerank_weight * rerank_score
            + config.concept_weight * concept_score
        )
        row["_retrieval"] = {
            "fusion_score": fused_score,
            "rrf_score": normalized_rrf,
            "entity_score": entity_score,
            "temporal_score": temporal_score,
            "rerank_score": rerank_score,
            "concept_score": concept_score,
            "lane_ranks": lane_ranks.get(key, {}),
        }
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            -float((row.get("_retrieval") or {}).get("fusion_score") or 0.0),
            _row_key(row),
        )
    )

    selected: List[Dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < config.top_k:
        best_index = 0
        best_value = -math.inf
        for index, row in enumerate(remaining):
            relevance = float((row.get("_retrieval") or {}).get("fusion_score") or 0.0)
            redundancy = max(
                (_similarity(row, selected_row) for selected_row in selected),
                default=0.0,
            )
            mmr_score = (
                config.diversity_lambda * relevance
                - (1.0 - config.diversity_lambda) * redundancy
            )
            if mmr_score > best_value:
                best_index = index
                best_value = mmr_score
        chosen = remaining.pop(best_index)
        chosen["_retrieval"]["mmr_score"] = best_value
        chosen["_retrieval"]["final_rank"] = len(selected) + 1
        selected.append(chosen)
    return selected


__all__ = [
    "FusionConfig",
    "QueryExpander",
    "build_query_variants",
    "fuse_rankings",
]

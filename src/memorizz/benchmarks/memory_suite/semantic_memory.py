"""Small, query-independent extraction of source-linked semantic memories."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from ...retrieval_concepts import concept_labels
from .models import MemoryDocument

_PATTERNS: Tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "constraint",
        re.compile(
            r"\b(?:must|cannot|can't|never|avoid|do not|don't|should not|"
            r"boundary|overwhelmed|say(?:ing)?\s+['\"’]?no['\"’]?|"
            r"protect(?:s|ed|ing)? (?:my |your |their |our )?time)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "preference",
        re.compile(
            r"\b(?:prefer|favo(?:u)?rite|dislike|would rather|do not like|don't like)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "goal",
        re.compile(
            r"\b(?:my goal|want to|plan to|trying to|aim to|hope to)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "commitment",
        re.compile(
            r"\b(?:promised|committed|agreed to|I will|we will)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "state-update",
        re.compile(
            r"\b(?:now|no longer|used to|moved to|changed to|started|stopped)\b",
            re.IGNORECASE,
        ),
    ),
)


def _causal_fields(sentence: str) -> List[tuple[str, str]]:
    """Extract small, source-faithful causal fields without generation."""

    for marker in (" because ", " so that ", " therefore ", " which means "):
        if marker in sentence.casefold():
            index = sentence.casefold().index(marker)
            before = sentence[:index].strip(" ,.;")
            after = sentence[index + len(marker) :].strip(" ,.;")
            if before and after:
                return [
                    ("Situation or action", before),
                    ("Reason or consequence", after),
                ]
    match = re.match(
        r"(?P<action>.+?)\b(?P<link>protects?|prevents?|reduces?|causes?|"
        r"helps?|leads? to)\b(?P<effect>.+)",
        sentence,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    action = match.group("action").strip(" ,.;")
    effect = f"{match.group('link')} {match.group('effect')}".strip(" ,.;")
    return [("Situation or action", action), ("Consequence", effect)]


def derive_semantic_memories(
    documents: Sequence[MemoryDocument],
) -> List[MemoryDocument]:
    """Extract generic constraints/preferences/state without reading queries.

    This deliberately conservative extractor is not intended to replace a
    domain model.  It creates independently retrievable labels while keeping
    the original statement and parent source ID intact for provenance.
    """

    derived: List[MemoryDocument] = []
    for document in documents:
        if document.metadata.get("derived_semantic_memory"):
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", document.content)
            if len(sentence.strip()) >= 8
        ]
        seen = set()
        index = 0
        for sentence in sentences:
            kinds = [name for name, pattern in _PATTERNS if pattern.search(sentence)]
            concepts = concept_labels(sentence)
            for kind in kinds:
                key = (kind, sentence.casefold())
                if key in seen:
                    continue
                seen.add(key)
                index += 1
                parent = document.parent_source_id or document.source_id
                fields = [
                    f"Semantic memory type: {kind}",
                    f"Concepts: {', '.join(concepts)}" if concepts else "",
                    f"Source statement: {sentence}",
                    *(f"{label}: {value}" for label, value in _causal_fields(sentence)),
                ]
                derived.append(
                    MemoryDocument(
                        source_id=f"{document.source_id}#semantic-{index}",
                        parent_source_id=parent,
                        content="\n".join(field for field in fields if field),
                        metadata={
                            **dict(document.metadata),
                            "derived_semantic_memory": True,
                            "semantic_memory_type": kind,
                            "semantic_concepts": concepts,
                            "source_document_id": document.source_id,
                        },
                    )
                )

    # Some sources explicitly mark several adjacent records as one semantic
    # event (for example, a statement followed by advice). Preserve that
    # relationship in one source-linked memory without reading the query.
    groups: Dict[str, List[MemoryDocument]] = defaultdict(list)
    for document in documents:
        group_id = str(document.metadata.get("semantic_group_id") or "").strip()
        if group_id:
            groups[group_id].append(document)
    for group_id, group_documents in groups.items():
        if len(group_documents) < 2:
            continue
        concepts = sorted(
            {
                concept
                for document in group_documents
                for concept in concept_labels(document.content)
            }
        )
        if not concepts:
            continue
        linked_sources = tuple(
            dict.fromkeys(
                document.parent_source_id or document.source_id
                for document in group_documents
            )
        )
        statements = "\n".join(f"- {document.content}" for document in group_documents)
        event_time = next(
            (
                document.metadata.get("event_time")
                for document in group_documents
                if document.metadata.get("event_time")
            ),
            None,
        )
        derived.append(
            MemoryDocument(
                source_id=f"{group_id}#semantic-summary",
                parent_source_id=linked_sources[0],
                linked_source_ids=linked_sources,
                content=(
                    "Semantic memory type: event-summary\n"
                    f"Concepts: {', '.join(concepts)}\n"
                    f"Source statements:\n{statements}"
                ),
                metadata={
                    "derived_semantic_memory": True,
                    "semantic_memory_type": "event-summary",
                    "semantic_concepts": concepts,
                    "semantic_group_id": group_id,
                    "source_document_ids": list(linked_sources),
                    **({"event_time": event_time} if event_time else {}),
                },
            )
        )
    return derived


__all__ = ["derive_semantic_memories"]

"""Deterministic, label-blind concepts shared by memory ingestion and search."""

from __future__ import annotations

import re
from typing import List, Tuple

_CONCEPTS: Tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "capacity",
        re.compile(
            r"\b(?:overwhelm(?:ed|ing)?|overload(?:ed)?|burnout|too much|"
            r"capacity|workload|busy|commitments?|stress(?:ed|ful)?)\b",
            re.IGNORECASE,
        ),
        "capacity, workload, commitments, overwhelm, stress, boundaries, and priorities",
    ),
    (
        "boundaries",
        re.compile(
            r"\b(?:boundar(?:y|ies)|say(?:ing)?\s+['\"’]?no['\"’]?|"
            r"declin(?:e|ing)|refus(?:e|ing)|protect(?:s|ed|ing)? "
            r"(?:my |your |their |our )?time)\b",
            re.IGNORECASE,
        ),
        "boundaries, declining obligations, saying no, protecting time, capacity, and reducing stress",
    ),
    (
        "preference",
        re.compile(
            r"\b(?:prefer|favo(?:u)?rite|dislike|avoid|would rather|"
            r"do not like|don't like|recommend|choice)\b",
            re.IGNORECASE,
        ),
        "preference likes dislikes avoid recommendation choice",
    ),
    (
        "goal",
        re.compile(
            r"\b(?:goal|want to|plan to|trying to|aim to|hope to|objective)\b",
            re.IGNORECASE,
        ),
        "goal objective plan intention desired outcome",
    ),
    (
        "commitment",
        re.compile(
            r"\b(?:promised|committed|commitment|agreed to|I will|we will|"
            r"deadline|obligation)\b",
            re.IGNORECASE,
        ),
        "commitment promise agreement obligation deadline",
    ),
    (
        "state-change",
        re.compile(
            r"\b(?:now|no longer|used to|moved to|changed to|started|stopped|"
            r"current|latest|updated)\b",
            re.IGNORECASE,
        ),
        "current state update changed superseded previous latest",
    ),
    (
        "temporal",
        re.compile(
            r"\b(?:when|date|day|week|month|year|yesterday|tomorrow|today|"
            r"before|after|ago|last|next)\b",
            re.IGNORECASE,
        ),
        "date time timeline before after relative absolute chronology",
    ),
)


def concept_labels(text: str) -> List[str]:
    """Return stable concept labels inferred without queries or gold evidence."""

    value = str(text or "")
    return [name for name, pattern, _ in _CONCEPTS if pattern.search(value)]


def expand_query_concepts(query: str) -> List[str]:
    """Build one generic semantic-search variant from concepts in ``query``."""

    labels = set(concept_labels(query))
    if "capacity" in labels:
        labels.add("boundaries")
    if "boundaries" in labels:
        labels.add("capacity")
    phrases = [phrase for name, _, phrase in _CONCEPTS if name in labels]
    if not phrases:
        return []
    return ["Retrieve prior memory about " + "; ".join(phrases) + "."]


__all__ = ["concept_labels", "expand_query_concepts"]

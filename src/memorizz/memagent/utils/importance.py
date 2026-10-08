# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Importance ratings for stored memories (Generative Agents "poignancy").

Park et al. (2023) ask a language model, at creation time, to rate each
memory from 1 (purely mundane) to 10 (extremely poignant). MemoRizz stores the
rating as ``importance`` in [0, 1] and uses it as one of the three retrieval
signals (recency, importance, relevance). Three rater modes exist:

``off``
    Store nothing; retrieval falls back to ``RetrievalScoring.default_importance``.
``heuristic``
    Deterministic, free: role/content signals (host-asserted facts and
    verified outcomes high, tool chatter low).
``llm``
    The paper's prompt, answered by a small or local model, batched and
    meant to run off the user path (the embedding backfill worker). Falls
    back to the heuristic when the model is unavailable or answers badly.

Ratings are immutable once stored; only a verified outcome raises one.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

IMPORTANCE_MODES = ("off", "heuristic", "llm")

# Park et al. 2023, section 4.1 ("Importance"), with the agent's purpose added
# so ratings are relative to what this agent is for rather than a human life.
IMPORTANCE_PROMPT = (
    "On the scale of 1 to 10, where 1 is purely mundane (e.g., small talk, "
    "a routine acknowledgement) and 10 is extremely important (e.g., a "
    "decision, a correction of a fact, a commitment, a user's preference), "
    "rate the likely importance of the following memory for an assistant "
    "whose purpose is: {purpose}\n\nMemory ({kind}):\n{memory}\n\n"
    "Answer with a single integer from 1 to 10."
)

BATCH_PROMPT = (
    "Rate the importance of each memory below for an assistant whose purpose "
    "is: {purpose}. Use the scale 1 (purely mundane) to 10 (extremely "
    "important: decisions, corrections of facts, commitments, user "
    "preferences). Reply with a JSON object mapping each memory number to its "
    "integer rating and nothing else.\n\n{items}"
)

_RATING_RE = re.compile(r"\b(10|[1-9])\b")

# Heuristic defaults by memory role / kind (0..1).
_HEURISTIC_BY_KIND: Dict[str, float] = {
    "verified_outcome": 1.0,
    "host_fact": 0.8,
    "knowledge_base": 0.5,
    "entity": 0.6,
    "summary": 0.6,
    "user": 0.5,
    "assistant": 0.4,
    "tool": 0.2,
    "tool_log": 0.2,
    "system": 0.3,
}

_DECISION_WORDS = (
    "decided",
    "decision",
    "must",
    "always",
    "never",
    "deadline",
    "prefer",
    "preference",
    "remember",
    "correct",
    "actually",
    "instead",
    "owner",
    "launch",
    "budget",
    "password",  # a mention is important even though the value is redacted
)


def clamp_importance(value: Any) -> Optional[float]:
    """Normalise a rating to [0, 1]; integers 1..10 are mapped to the GA scale."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    if 1.0 < number <= 10.0:
        number = number / 10.0
    return min(1.0, max(0.0, number))


def parse_rating(text: Any) -> Optional[float]:
    """Extract the first integer 1..10 from a model answer and map it to 0..1."""
    match = _RATING_RE.search(str(text or ""))
    if not match:
        return None
    return clamp_importance(int(match.group(1)))


def heuristic_importance(
    text: Any,
    *,
    role: Optional[str] = None,
    kind: Optional[str] = None,
    host_asserted: bool = False,
    verified: bool = False,
) -> float:
    """Deterministic 0..1 rating from role, kind and simple content signals."""
    if verified:
        return 1.0
    if host_asserted:
        return _HEURISTIC_BY_KIND["host_fact"]
    key = str(kind or role or "").strip().lower()
    base = _HEURISTIC_BY_KIND.get(key)
    if base is None:
        base = 0.5
    body = str(text or "")
    lowered = body.lower()
    if any(word in lowered for word in _DECISION_WORDS):
        base += 0.15
    if len(body) < 24:
        base -= 0.1
    elif len(body) > 400:
        base += 0.05
    return min(1.0, max(0.05, round(base, 3)))


@dataclass(frozen=True)
class ImportanceConfig:
    mode: str = "heuristic"
    model: Optional[str] = None
    batch_size: int = 8
    purpose: str = "a helpful assistant with persistent memory"

    def __post_init__(self) -> None:
        mode = str(self.mode or "heuristic").strip().lower()
        if mode not in IMPORTANCE_MODES:
            mode = "heuristic"
        object.__setattr__(self, "mode", mode)
        try:
            size = int(self.batch_size)
        except (TypeError, ValueError):
            size = 8
        object.__setattr__(self, "batch_size", max(1, min(size, 50)))

    @classmethod
    def from_mapping(
        cls, value: Any, *, base: Optional["ImportanceConfig"] = None
    ) -> "ImportanceConfig":
        current = base or cls()
        if not isinstance(value, dict):
            return current
        updates: Dict[str, Any] = {}
        for name in ("mode", "model", "purpose"):
            if value.get(name) not in (None, ""):
                updates[name] = str(value[name])
        if value.get("batch_size") not in (None, ""):
            updates["batch_size"] = value["batch_size"]
        return replace(current, **updates) if updates else current

    @classmethod
    def from_env(cls, environ: Optional[Dict[str, str]] = None) -> "ImportanceConfig":
        env = os.environ if environ is None else environ
        return cls.from_mapping(
            {
                "mode": env.get("MEMORIZZ_IMPORTANCE_RATER"),
                "model": env.get("MEMORIZZ_IMPORTANCE_MODEL"),
                "batch_size": env.get("MEMORIZZ_IMPORTANCE_BATCH_SIZE"),
                "purpose": env.get("MEMORIZZ_IMPORTANCE_PURPOSE"),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model,
            "batch_size": self.batch_size,
            "purpose": self.purpose,
        }


class ImportanceRater:
    """Rate memories at store time; cheap by default, GA-faithful when asked.

    ``generate`` is any ``callable(prompt: str) -> str``; the caller wires it
    to a small or local model so ratings never touch the user-facing turn.
    """

    def __init__(
        self,
        config: Optional[ImportanceConfig] = None,
        *,
        generate: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config or ImportanceConfig.from_env()
        self._generate = generate

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    def rate(
        self,
        text: Any,
        *,
        role: Optional[str] = None,
        kind: Optional[str] = None,
        host_asserted: bool = False,
        verified: bool = False,
    ) -> Tuple[Optional[float], str]:
        """Return ``(importance, source)``; ``(None, "off")`` when disabled."""
        if self.config.mode == "off":
            return None, "off"
        if verified:
            return 1.0, "verified"
        if host_asserted:
            return heuristic_importance(text, host_asserted=True), "host"
        if self.config.mode == "llm" and self._generate is not None:
            rating = self._rate_with_model(text, kind=kind or role)
            if rating is not None:
                return rating, "llm"
        return heuristic_importance(text, role=role, kind=kind), "heuristic"

    def rate_many(
        self, items: Sequence[Dict[str, Any]]
    ) -> List[Tuple[Optional[float], str]]:
        """Rate several memories; LLM mode batches them into one prompt per chunk.

        Each item is ``{"text", "role"?, "kind"?, "host_asserted"?, "verified"?}``.
        """
        results: List[Tuple[Optional[float], str]] = [(None, "pending")] * len(items)
        pending: List[int] = []
        for index, item in enumerate(items):
            if (
                self.config.mode != "llm"
                or self._generate is None
                or item.get("verified")
                or item.get("host_asserted")
            ):
                results[index] = self.rate(
                    item.get("text"),
                    role=item.get("role"),
                    kind=item.get("kind"),
                    host_asserted=bool(item.get("host_asserted")),
                    verified=bool(item.get("verified")),
                )
            else:
                pending.append(index)
        for start in range(0, len(pending), self.config.batch_size):
            chunk = pending[start : start + self.config.batch_size]
            ratings = self._rate_batch_with_model([items[i] for i in chunk])
            for position, index in enumerate(chunk):
                rating = ratings[position] if position < len(ratings) else None
                item = items[index]
                if rating is None:
                    results[index] = (
                        heuristic_importance(
                            item.get("text"),
                            role=item.get("role"),
                            kind=item.get("kind"),
                        ),
                        "heuristic",
                    )
                else:
                    results[index] = (rating, "llm")
        return results

    # -- model calls ---------------------------------------------------------

    def _rate_with_model(self, text: Any, *, kind: Optional[str]) -> Optional[float]:
        prompt = IMPORTANCE_PROMPT.format(
            purpose=self.config.purpose,
            kind=kind or "memory",
            memory=str(text or "")[:2000],
        )
        try:
            answer = self._generate(prompt)
        except Exception as exc:
            logger.warning("Importance rating failed; using heuristic: %s", exc)
            return None
        return parse_rating(answer)

    def _rate_batch_with_model(
        self, items: Sequence[Dict[str, Any]]
    ) -> List[Optional[float]]:
        if not items:
            return []
        if len(items) == 1:
            return [
                self._rate_with_model(items[0].get("text"), kind=items[0].get("kind"))
            ]
        lines = [
            f"{index + 1}. ({item.get('kind') or item.get('role') or 'memory'}) "
            f"{str(item.get('text') or '')[:600]}"
            for index, item in enumerate(items)
        ]
        prompt = BATCH_PROMPT.format(
            purpose=self.config.purpose, items="\n".join(lines)
        )
        try:
            answer = str(self._generate(prompt) or "")
        except Exception as exc:
            logger.warning("Batch importance rating failed; using heuristic: %s", exc)
            return [None] * len(items)
        ratings: List[Optional[float]] = [None] * len(items)
        match = re.search(r"\{.*\}", answer, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    try:
                        position = int(str(key).strip().rstrip(".")) - 1
                    except ValueError:
                        continue
                    if 0 <= position < len(items):
                        ratings[position] = clamp_importance(value)
                return ratings
        # Fallback: one rating per line in order.
        found = [clamp_importance(int(m)) for m in _RATING_RE.findall(answer)]
        for index in range(min(len(found), len(items))):
            ratings[index] = found[index]
        return ratings


__all__ = [
    "IMPORTANCE_MODES",
    "IMPORTANCE_PROMPT",
    "ImportanceConfig",
    "ImportanceRater",
    "clamp_importance",
    "heuristic_importance",
    "parse_rating",
]

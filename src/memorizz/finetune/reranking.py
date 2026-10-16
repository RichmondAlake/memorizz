"""Rerank retrieved memories with a model fine-tuned in MemoRizz.

``MEMORIZZ_RERANKER_MODEL`` names a model folder: a cross-encoder reranker,
or a relevance decision model (System One ``noul`` question, scored by
P(relevant)). When it is set, an agent's recall and the MCP memory search
rescore the candidates their vector search found, so the best evidence
comes first. Heavy libraries load on first use only.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RERANKER_ENV = "MEMORIZZ_RERANKER_MODEL"

Scorer = Callable[[str, List[str]], List[float]]

_lock = threading.Lock()
_loaded: Dict[str, Tuple[float, Scorer]] = {}


def configured_reranker() -> Optional[str]:
    """The reranker folder named by ``MEMORIZZ_RERANKER_MODEL``, if usable."""
    path = os.getenv(RERANKER_ENV, "").strip()
    if not path:
        return None
    if not Path(path).expanduser().is_dir():
        logger.warning("%s=%s is not a model folder; not reranking", RERANKER_ENV, path)
        return None
    return str(Path(path).expanduser())


def load_scorer(path: str) -> Scorer:
    """A ``(query, passages) -> relevance in [0, 1]`` function for a model
    folder (cross-encoder or relevance decision model)."""
    folder = Path(path)
    if (folder / "memorizz_decision.json").is_file():
        from .inference import DecisionModel
        from .tasks import options

        model = DecisionModel(folder)
        question = model.question
        if question.get("type") != "noul":
            raise ValueError(
                f"{folder} answers a {question.get('type')} question; a reranker "
                "needs a yes/no relevance model"
            )
        positive = [key for key, _ in options(question)].index("true")

        def decide(query: str, passages: List[str]) -> List[float]:
            states = [{"query": query, "candidate": text} for text in passages]
            return [row[positive] for row in model.probabilities_many(states, question)]

        return decide

    from sentence_transformers.cross_encoder import CrossEncoder

    from .training.common import pick_device

    encoder = CrossEncoder(str(folder), device=pick_device(None))

    def cross(query: str, passages: List[str]) -> List[float]:
        raw = [float(v) for v in encoder.predict([(query, p) for p in passages])]
        if any(v < 0.0 or v > 1.0 for v in raw):
            raw = [1.0 / (1.0 + math.exp(-v)) for v in raw]
        return raw

    return cross


def active_scorer() -> Optional[Scorer]:
    """The store's reranker (``MEMORIZZ_RERANKER_MODEL``)."""
    path = configured_reranker()
    return scorer_for(path) if path else None


def scorer_for(path: str) -> Optional[Scorer]:
    """The reranker in ``path``, loaded once per folder (reloaded when the
    folder changes on disk, e.g. after a new swap)."""
    if not path or not Path(path).is_dir():
        return None
    stamp = Path(path).stat().st_mtime
    with _lock:
        cached = _loaded.get(path)
        if cached and cached[0] == stamp:
            return cached[1]
    try:
        scorer = load_scorer(path)
    except Exception as exc:
        logger.warning("Could not load the reranker at %s: %s", path, exc)
        return None
    with _lock:
        _loaded[path] = (stamp, scorer)
    return scorer


def rerank_for(
    query: str,
    *,
    provider: Any = None,
    agent_id: Optional[str] = None,
    memory_id: Optional[str] = None,
) -> Optional[Callable[[List[str]], List[float]]]:
    """``texts -> scores`` for one query, or None without a reranker. An
    agent (named, or owning ``memory_id``) with its own choice uses that
    version, or none; everything else uses the store's."""
    if not query or not str(query).strip():
        return None
    own = None
    if provider is not None and (agent_id or memory_id):
        try:
            from . import agent_models

            own = agent_models.resolve(
                provider, "reranking", agent_id=agent_id, memory_id=memory_id
            )
        except Exception as exc:  # a lookup failure never breaks recall
            logger.warning("Could not read the agent's reranker choice: %s", exc)
    if own and own.get("mode") == "off":
        return None
    scorer = (
        scorer_for(own.get("model_dir", ""))
        if own and own.get("mode") == "run"
        else active_scorer()
    )
    if scorer is None:
        return None
    return lambda texts: scorer(str(query), [str(t) for t in texts])


def rerank_rows(
    query: str,
    rows: List[Dict[str, Any]],
    text_of: Callable[[Dict[str, Any]], str],
    **context: Any,
) -> List[Dict[str, Any]]:
    """Reorder provider rows by the reranker (unchanged without one); see
    ``rerank_for`` for ``context``. Each row gains ``rerank_score``."""
    score = rerank_for(query, **context)
    if score is None or len(rows) < 2:
        return rows
    try:
        values = score([text_of(row) for row in rows])
    except Exception as exc:
        logger.warning("Reranker failed; keeping retrieval order: %s", exc)
        return rows
    for row, value in zip(rows, values):
        row["rerank_score"] = round(float(value), 4)
    return sorted(rows, key=lambda row: -row["rerank_score"])


__all__ = [
    "RERANKER_ENV",
    "active_scorer",
    "configured_reranker",
    "load_scorer",
    "rerank_for",
    "rerank_rows",
    "scorer_for",
]

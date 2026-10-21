# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Rerank what an agent recalls, with a hosted reranker or one fine-tuned in
MemoRizz.

An agent's ``reranker`` setting is one of:

- ``None``: follow the store. The agent's choice on the fine-tuning page,
  else the store's fine-tuned reranker (``MEMORIZZ_RERANKER_MODEL``), else
  no reranking.
- ``"off"``.
- ``"voyage"`` (``VOYAGE_API_KEY``): ``rerank-2.5`` or ``rerank-2.5-lite``.
- ``"cohere"`` (``COHERE_API_KEY``): ``rerank-v3.5``.
- ``"openai_decisions"`` (``OPENAI_API_KEY``): ``gpt-6-luna`` answers "is
  this relevant?" for every candidate in one request.
- ``"finetuned"``: a reranker or relevance model trained in MemoRizz, by run
  id.

Write it as ``"voyage"``, ``"voyage:rerank-2.5-lite"``, ``"finetuned:<run
id>"`` or a mapping. A hosted reranker adds one request to each turn that
recalls memories; when it fails, recall keeps its own order.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "voyage": {
        "title": "Voyage",
        "env": "VOYAGE_API_KEY",
        "models": ("rerank-2.5", "rerank-2.5-lite"),
    },
    "cohere": {"title": "Cohere", "env": "COHERE_API_KEY", "models": ("rerank-v3.5",)},
    "openai_decisions": {
        "title": "OpenAI Decisions",
        "env": "OPENAI_API_KEY",
        "models": ("gpt-6-luna",),
    },
}
KINDS = ("off", "finetuned", *PROVIDERS)
TIMEOUT_SECONDS = 15

Scorer = Callable[[str, List[str]], List[float]]


@dataclass(frozen=True)
class RerankerConfig:
    """An agent's own reranker; ``from_value(None)`` is None (follow the store)."""

    provider: str
    model: Optional[str] = None
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.provider not in KINDS:
            raise ValueError(
                f"Unknown reranker {self.provider!r}; use one of {', '.join(KINDS)}"
            )
        if self.provider == "finetuned" and not self.run_id:
            raise ValueError("A fine-tuned reranker needs its run_id")

    @classmethod
    def from_value(cls, value: Any) -> Optional["RerankerConfig"]:
        if value is None or isinstance(value, cls):
            return value
        if value is False:
            return cls("off")
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"default", "store", "follow"}:
                return None
            kind, _, detail = text.partition(":")
            kind = kind.strip().lower()
            if kind in {"off", "none", "false"}:
                return cls("off")
            if kind == "finetuned":
                return cls(kind, run_id=detail.strip() or None)
            return cls(kind, model=detail.strip() or None)
        if not isinstance(value, Mapping):
            raise TypeError("reranker must be None, a name or a mapping")
        kind = str(value.get("provider") or value.get("mode") or "").strip().lower()
        if kind in {"", "default", "store"}:
            return None
        if kind == "run":  # the fine-tuning page's per-agent record
            kind = "finetuned"
        return cls(
            kind,
            model=value.get("model") or None,
            run_id=value.get("run_id") or None,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"provider": self.provider}
        if self.model:
            out["model"] = self.model
        if self.run_id:
            out["run_id"] = self.run_id
        return out

    @property
    def model_name(self) -> Optional[str]:
        if self.provider in PROVIDERS:
            return self.model or PROVIDERS[self.provider]["models"][0]
        return None

    @property
    def label(self) -> str:
        if self.provider == "off":
            return "off"
        if self.provider == "finetuned":
            return f"fine-tuned · {self.run_id}"
        return f"{PROVIDERS[self.provider]['title']} · {self.model_name}"


def _key(provider: str) -> str:
    env = PROVIDERS[provider]["env"]
    key = os.environ.get(env)
    if not key:
        raise ValueError(
            f"{env} is required for the {PROVIDERS[provider]['title']} reranker"
        )
    return key


def _from_results(results: List[Mapping[str, Any]], count: int) -> List[float]:
    scores = [0.0] * count
    seen = set()
    for row in results:
        index = int(row["index"])
        if not 0 <= index < count or index in seen:
            raise ValueError("The reranker returned an unknown candidate")
        seen.add(index)
        scores[index] = float(row["relevance_score"])
    if len(seen) != count:
        raise ValueError("The reranker left candidates out")
    return scores


def _voyage(model: str) -> Scorer:
    def score(query: str, texts: List[str]) -> List[float]:
        import requests

        response = requests.post(
            "https://api.voyageai.com/v1/rerank",
            headers={"Authorization": f"Bearer {_key('voyage')}"},
            json={
                "model": model,
                "query": query,
                "documents": texts,
                "truncation": True,
            },
            timeout=TIMEOUT_SECONDS,
        )
        if not response.ok:
            raise RuntimeError(f"Voyage reranker returned HTTP {response.status_code}")
        return _from_results(response.json().get("data") or [], len(texts))

    return score


def _cohere(model: str) -> Scorer:
    def score(query: str, texts: List[str]) -> List[float]:
        import requests

        response = requests.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {_key('cohere')}"},
            json={"model": model, "query": query, "documents": texts},
            timeout=TIMEOUT_SECONDS,
        )
        if not response.ok:
            raise RuntimeError(f"Cohere reranker returned HTTP {response.status_code}")
        return _from_results(response.json().get("results") or [], len(texts))

    return score


def _decisions(model: str) -> Scorer:
    def score(query: str, texts: List[str]) -> List[float]:
        _key("openai_decisions")
        from .benchmarks.openai_decisions import rank_decisions

        scores, _usage = rank_decisions(query, texts, "noul", model)
        return [float(value) for value in scores]

    return score


def scorer(config: RerankerConfig, provider: Any = None) -> Optional[Scorer]:
    """``(query, texts) -> scores`` for a reranker setting, or None (off, or
    a fine-tuned model that cannot be loaded here)."""
    if config.provider == "off":
        return None
    if config.provider == "voyage":
        return _voyage(config.model_name)
    if config.provider == "cohere":
        return _cohere(config.model_name)
    if config.provider == "openai_decisions":
        return _decisions(config.model_name)
    if provider is None:
        return None
    from .finetune import reranking, routing

    try:
        path = routing._service(provider).model_dir(config.run_id)
    except Exception as exc:
        logger.warning("Fine-tuned reranker %s is unavailable: %s", config.run_id, exc)
        return None
    return reranking.scorer_for(path)


_FOLLOW = object()


def reranker_for(
    query: str,
    *,
    config: Any = _FOLLOW,
    provider: Any = None,
    agent_id: Optional[str] = None,
    memory_id: Optional[str] = None,
) -> Optional[Callable[[List[str]], List[float]]]:
    """``texts -> scores`` for one query, or None without a reranker.

    ``config`` is the agent's own setting. Without one, the setting saved on
    the agent (named, or owning ``memory_id``) is read; when that follows
    the store, so does this (``finetune.reranking.rerank_for``)."""
    if not query or not str(query).strip():
        return None
    if config is _FOLLOW:
        config = None
        if provider is not None and (agent_id or memory_id):
            try:
                from .finetune import agent_models

                owner = agent_id or agent_models.agent_for_memory(provider, memory_id)
                config = agent_models.record_setting(provider, owner, "reranker")
            except Exception as exc:
                logger.warning("Could not read the agent's reranker: %s", exc)
    try:
        config = RerankerConfig.from_value(config)
    except (TypeError, ValueError) as exc:
        logger.warning("Ignoring an invalid reranker setting: %s", exc)
        config = None
    if config is None:
        try:
            from .finetune.reranking import rerank_for
        except ImportError:
            return None
        return rerank_for(
            query, provider=provider, agent_id=agent_id, memory_id=memory_id
        )
    score = scorer(config, provider)
    if score is None:
        return None
    return lambda texts: score(str(query), [str(t) for t in texts])


def available(environ: Optional[Mapping[str, str]] = None) -> Dict[str, bool]:
    """Which hosted rerankers have a key set."""
    env = os.environ if environ is None else environ
    return {name: bool(env.get(spec["env"])) for name, spec in PROVIDERS.items()}


__all__ = [
    "KINDS",
    "PROVIDERS",
    "RerankerConfig",
    "available",
    "reranker_for",
    "scorer",
]

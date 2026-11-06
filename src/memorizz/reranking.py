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
- ``"huggingface"`` (``HF_TOKEN``): a cross-encoder served by Hugging Face
  Inference Providers, ``cross-encoder/ms-marco-MiniLM-L6-v2`` unless another
  model id follows the colon.
- ``"d1"`` (on this machine), ``"liquid"`` (``LIQUID_API_KEY``) or ``"jev"``
  (``TYPESAFE_API_KEY``): a decision model asked whether each memory gives
  useful evidence for the request (see :mod:`memorizz.decision_models`).
- ``"finetuned"``: a reranker or relevance model trained in MemoRizz, by run
  id.
- Your own reranker: a function ``(query, texts) -> scores``, or an object
  with ``predict(pairs)`` such as sentence-transformers' ``CrossEncoder``.
  Scores are 0 to 1, higher is more relevant; any other range is squashed
  with a sigmoid. It lives in this process, so a saved agent loaded
  elsewhere has none until it is passed again.

Write it as ``"voyage"``, ``"voyage:rerank-2.5-lite"``,
``"huggingface:<model id>"``, ``"finetuned:<run id>"``, a mapping or the
reranker itself. A hosted reranker adds one request to each turn that
recalls memories; when it fails, recall keeps its own order.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

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
    "huggingface": {
        "title": "Hugging Face",
        "env": "HF_TOKEN",
        "models": ("cross-encoder/ms-marco-MiniLM-L6-v2",),
    },
    # Decision models asked "is this relevant?" (memorizz.decision_models).
    "d1": {"title": "Liquid d1", "env": None, "models": ("LiquidAI/d1-2B",)},
    "liquid": {
        "title": "Liquid d1 (hosted)",
        "env": "LIQUID_API_KEY",
        "models": ("d1.1",),
    },
    "jev": {
        "title": "TypeSafe Jev",
        "env": "TYPESAFE_API_KEY",
        "models": ("jev-1.14.0",),
    },
}
#: Rerankers that are decision models, by their decision-model backend.
DECISION_RERANKERS = ("d1", "liquid", "jev")
KINDS = ("off", "finetuned", "custom", *PROVIDERS)
TIMEOUT_SECONDS = 15
# Hugging Face loads a model on its first request after a while idle; the
# request waits for it (x-wait-for-model) rather than failing with a 503.
HF_TIMEOUT_SECONDS = 30
HF_INFERENCE_URL = "https://router.huggingface.co/hf-inference/models/{model}"
# Texts per Hugging Face request; longer lists are sent in batches.
HF_BATCH = 32

Scorer = Callable[[str, List[str]], List[float]]


@dataclass(frozen=True)
class RerankerConfig:
    """An agent's own reranker; ``from_value(None)`` is None (follow the store)."""

    provider: str
    model: Optional[str] = None
    run_id: Optional[str] = None
    # A custom reranker's scorer, in this process only (never saved).
    function: Optional[Callable[..., Any]] = field(
        default=None, compare=False, repr=False
    )

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
        custom = _custom_scorer(value)
        if custom is not None:
            return cls("custom", model=_custom_name(value), function=custom)
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
        if self.provider == "custom":
            return f"custom · {self.model or 'reranker'}"
        return f"{PROVIDERS[self.provider]['title']} · {self.model_name}"


def _custom_scorer(value: Any) -> Optional[Scorer]:
    """``(query, texts) -> scores`` for a reranker passed as an object, or None
    for a setting (a name, a mapping)."""
    if isinstance(value, (str, bytes, Mapping, bool)) or value is None:
        return None
    predict = getattr(value, "predict", None)
    if callable(predict):  # sentence-transformers CrossEncoder and alike
        return lambda query, texts: predict([(query, text) for text in texts])
    if callable(value):
        return value
    return None


def _custom_name(value: Any) -> str:
    name = getattr(value, "__name__", None)
    if not name or name == "<lambda>":
        name = getattr(value, "model_name", None) or type(value).__name__
    return str(name)


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        try:  # a token saved by `hf auth login`
            from huggingface_hub import get_token

            token = (get_token() or "").strip()
        except Exception:
            token = ""
    if not token:
        raise ValueError("HF_TOKEN is required for the Hugging Face reranker")
    return token


def _key(provider: str) -> str:
    if provider == "huggingface":
        return _hf_token()
    env = PROVIDERS[provider]["env"]
    key = os.environ.get(env)
    if not key:
        raise ValueError(
            f"{env} is required for the {PROVIDERS[provider]['title']} reranker"
        )
    return key


def _from_results(
    results: List[Mapping[str, Any]], count: int, key: str = "relevance_score"
) -> List[float]:
    scores = [0.0] * count
    seen = set()
    for row in results:
        index = int(row["index"])
        if not 0 <= index < count or index in seen:
            raise ValueError("The reranker returned an unknown candidate")
        seen.add(index)
        scores[index] = float(row[key])
    if len(seen) != count:
        raise ValueError("The reranker left candidates out")
    return scores


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _unit_scores(values: Sequence[Any], count: int) -> List[float]:
    """Scores from 0 to 1: a list outside that range is squashed with a
    sigmoid, which keeps its order (recall blends relevance with recency and
    importance, so the scale matters)."""
    scores = [float(value) for value in values]
    if len(scores) != count:
        raise ValueError("The reranker returned the wrong number of scores")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("The reranker returned a score that is not a number")
    if any(score < 0.0 or score > 1.0 for score in scores):
        scores = [_sigmoid(score) for score in scores]
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


def _huggingface(model: str) -> Scorer:
    """A Hugging Face ``text-ranking`` model on its hf-inference provider. It
    returns cross-encoder logits, which become 0 to 1 with a sigmoid, the
    reading these models are trained for."""

    def score(query: str, texts: List[str]) -> List[float]:
        import requests

        token = _key("huggingface")
        url = HF_INFERENCE_URL.format(model=model)
        scores: List[float] = []
        for start in range(0, len(texts), HF_BATCH):
            batch = texts[start : start + HF_BATCH]
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "x-wait-for-model": "true",
                },
                json={"inputs": {"query": query, "texts": batch}},
                timeout=HF_TIMEOUT_SECONDS,
            )
            if not response.ok:
                raise RuntimeError(
                    f"Hugging Face reranker returned HTTP {response.status_code}"
                    + _hf_error(response)
                )
            rows = response.json()
            if not isinstance(rows, list):
                raise ValueError("The Hugging Face reranker returned no scores")
            logits = _from_results(rows, len(batch), key="score")
            scores.extend(_sigmoid(value) for value in logits)
        return scores

    return score


def _hf_error(response: Any) -> str:
    """Hugging Face's own reason, such as an unknown model, kept short."""
    try:
        message = str((response.json() or {}).get("error") or "").strip()
    except Exception:
        return ""
    return f": {message[:160]}" if message else ""


def _decision_model(backend: str, model: Optional[str]) -> Scorer:
    """A decision model asked, for each text, whether it gives useful
    evidence for the query; P(yes) is the score."""

    def score(query: str, texts: List[str]) -> List[float]:
        from .decision_models import (
            RELEVANT,
            Decider,
            DecisionModelConfig,
            probability_yes,
            relevance_state,
        )

        decider = Decider(DecisionModelConfig(backend, model))
        answers = decider.decide_many(
            [
                (relevance_state(query, str(text)[:4000]), {"relevant": RELEVANT})
                for text in texts
            ]
        )
        scores = []
        for answer in answers:
            value = probability_yes(answer.get("relevant"))
            if value is None:
                raise ValueError("The decision model gave no relevance answer")
            scores.append(value)
        return scores

    return score


def _custom(function: Callable[..., Any]) -> Scorer:
    def score(query: str, texts: List[str]) -> List[float]:
        return _unit_scores(list(function(query, list(texts))), len(texts))

    return score


def scorer(config: RerankerConfig, provider: Any = None) -> Optional[Scorer]:
    """``(query, texts) -> scores`` for a reranker setting, or None (off, a
    custom reranker passed in another process, or a fine-tuned model that
    cannot be loaded here)."""
    if config.provider == "off":
        return None
    if config.provider == "custom":
        if config.function is None:
            logger.warning(
                "The custom reranker %r is not in this process; pass it again "
                "with reranker= to rerank",
                config.model,
            )
            return None
        return _custom(config.function)
    if config.provider == "voyage":
        return _voyage(config.model_name)
    if config.provider == "cohere":
        return _cohere(config.model_name)
    if config.provider == "openai_decisions":
        return _decisions(config.model_name)
    if config.provider == "huggingface":
        return _huggingface(config.model_name)
    if config.provider in DECISION_RERANKERS:
        return _decision_model(config.provider, config.model)
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


def _local_ready() -> bool:
    """torch and transformers are installed, so d1 can run here."""
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
    )


def available(environ: Optional[Mapping[str, str]] = None) -> Dict[str, bool]:
    """Which hosted rerankers have a key set."""
    env = os.environ if environ is None else environ
    keys = {
        name: (bool(env.get(spec["env"])) if spec["env"] else _local_ready())
        for name, spec in PROVIDERS.items()
    }
    if environ is None and not keys["huggingface"]:
        try:  # or a token saved by `hf auth login`
            keys["huggingface"] = bool(_hf_token())
        except ValueError:
            pass
    return keys


def rerank_rows(
    rows: Sequence[Any],
    rerank: Optional[Callable[[List[str]], List[float]]],
    *,
    text_of: Callable[[Any], str],
) -> Optional[List[tuple]]:
    """``(score, row)`` pairs, most relevant first, or None without a
    reranker or when it fails (callers keep their own order)."""
    if rerank is None or not rows:
        return None
    try:
        scores = [float(value) for value in rerank([text_of(row) for row in rows])]
    except Exception as exc:  # a broken reranker never breaks the search
        logger.warning("Reranker failed; keeping the search order: %s", exc)
        return None
    if len(scores) != len(rows):
        logger.warning(
            "Reranker returned %d scores for %d rows", len(scores), len(rows)
        )
        return None
    order = sorted(range(len(rows)), key=lambda index: (-scores[index], index))
    return [(scores[index], rows[index]) for index in order]


__all__ = [
    "KINDS",
    "PROVIDERS",
    "RerankerConfig",
    "available",
    "rerank_rows",
    "reranker_for",
    "scorer",
]

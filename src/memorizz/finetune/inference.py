"""Use a fine-tuned model: decisions in the System One answer shape, and
retrieval scores from embedding models and rerankers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluate import softmax
from .tasks import options, validate_question


def answer_payload(
    question: Dict[str, Any], probabilities: List[float]
) -> Dict[str, Any]:
    """One answer as a ``/v1/systemone`` response carries it."""
    keys = [key for key, _ in options(question)]
    best = max(range(len(keys)), key=probabilities.__getitem__)
    distribution = {key: round(p, 6) for key, p in zip(keys, probabilities)}
    kind = question["type"]
    if kind == "choice":
        return {
            "type": "choice",
            "choice": keys[best],
            "probabilities": distribution,
            "confidence": round(probabilities[best], 6),
        }
    if kind == "noul":
        p_true = distribution["true"]
        return {"type": "noul", "noul": p_true, "answer": p_true >= 0.5}
    expected = sum(i * p for i, p in enumerate(probabilities))
    return {
        "type": "score",
        "level": best,
        "score": round(expected, 4),
        "probabilities": distribution,
        "confidence": round(probabilities[best], 6),
    }


class DecisionModel:
    """A trained decision model. ``decide(state)`` answers the question it
    was trained on; pass ``questions`` to ask others (their options are
    inputs, so new ones work, without the trained calibration)."""

    def __init__(self, path: str | Path, *, device: Optional[str] = None):
        from .training.common import pick_device

        self.path = Path(path)
        meta_file = self.path / "memorizz_decision.json"
        if not meta_file.exists():
            raise ValueError(f"{self.path} is not a MemoRizz decision model")
        self.meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.backbone = self.meta["backbone"]
        self.question_id = self.meta["question_id"]
        self.question = self.meta["question"]
        self.temperature = float(self.meta.get("temperature") or 1.0)
        self.device = pick_device(device)
        self._scorers: Dict[str, Any] = {}
        if self.backbone == "embedding":
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(str(self.path), device=self.device)
        elif self.backbone == "lm":
            from .training.decision_lm import load_lm

            self.tokenizer, self.model = load_lm(str(self.path), self.device)
            self.model.eval()
        else:
            raise ValueError(
                f"MemoRizz cannot run a {self.backbone} decision model in-process yet"
            )

    def _scorer(self, question: Dict[str, Any]):
        """One scorer per question, its option embeddings computed once:
        options do not change between calls, so routing a memory embeds only
        the memory."""
        from .training.decision_embed import DecisionScorer

        key = json.dumps(question, sort_keys=True)
        scorer = self._scorers.get(key)
        if scorer is None:
            tau = float(self.meta.get("tau") or 0.1)
            scorer = DecisionScorer(
                self.encoder, self.meta["family"], question, tau, cache_options=True
            )
            if len(self._scorers) >= 8:
                self._scorers.pop(next(iter(self._scorers)))
            self._scorers[key] = scorer
        return scorer

    def _logits(self, state: Any, question: Dict[str, Any]) -> List[float]:
        if self.backbone == "embedding":
            scorer = self._scorer(question)
            return [c / scorer.tau for c in scorer.cosines([state])[0]]
        from .training.decision_lm import LMScorer

        scorer = LMScorer(
            self.tokenizer,
            self.model,
            question,
            self.device,
            int(self.meta.get("max_length") or 1024),
        )
        return scorer.logits([state])[0]

    def probabilities_many(
        self, states: List[Any], question: Dict[str, Any]
    ) -> List[List[float]]:
        """Option probabilities for many states at once (calibrated when the
        question is the one the model was trained on)."""
        temperature = self.temperature if question == self.question else 1.0
        return [softmax(row, temperature) for row in self.logits_many(states, question)]

    def logits_many(
        self, states: List[Any], question: Dict[str, Any]
    ) -> List[List[float]]:
        if self.backbone == "embedding":
            scorer = self._scorer(question)
            return [[c / scorer.tau for c in row] for row in scorer.cosines(states)]
        from .training.decision_lm import LMScorer

        scorer = LMScorer(
            self.tokenizer,
            self.model,
            question,
            self.device,
            int(self.meta.get("max_length") or 1024),
        )
        return scorer.logits(states)

    def decide(
        self, state: Any, questions: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        questions = questions or {self.question_id: self.question}
        answers = {}
        for qid, question in questions.items():
            validate_question(qid, question)
            trained = question == self.question
            probabilities = softmax(
                self._logits(state, question), self.temperature if trained else 1.0
            )
            answers[qid] = answer_payload(question, probabilities)
        return answers


def rank_with_embedding(
    path: str | Path, query: str, passages: List[str], *, device: Optional[str] = None
) -> List[float]:
    from sentence_transformers import SentenceTransformer

    from .training.common import pick_device

    model = SentenceTransformer(str(path), device=pick_device(device))
    prompts = getattr(model, "prompts", {}) or {}
    q = model.encode(
        [query],
        prompt=prompts.get("query") or None,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    d = model.encode(
        passages,
        prompt=prompts.get("document") or None,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    return (q @ d.T)[0].float().cpu().tolist()


def rank_with_reranker(
    path: str | Path, query: str, passages: List[str], *, device: Optional[str] = None
) -> List[float]:
    from sentence_transformers.cross_encoder import CrossEncoder

    from .training.common import pick_device

    model = CrossEncoder(str(path), device=pick_device(device))
    return [float(s) for s in model.predict([(query, p) for p in passages])]


def base_decision_logits(
    base_model: str,
    family: str,
    backbone: str,
    question: Dict[str, Any],
    states: List[Any],
    *,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
) -> List[List[float]]:
    """Zero-shot option logits from an untrained base model, scored exactly
    the way training scores it (the "before" of every comparison); it reads
    at most ``max_length`` tokens, the trainer's default unless given."""
    from .training.common import pick_device

    device = pick_device(device)
    if backbone == "embedding":
        from .training.decision_embed import DEFAULTS, DecisionScorer, load_encoder

        encoder = load_encoder(base_model, family, device)
        encoder.max_seq_length = int(max_length or DEFAULTS["max_length"])
        tau = float(DEFAULTS["tau"])
        scorer = DecisionScorer(encoder, family, question, tau)
        return [[c / tau for c in row] for row in scorer.cosines(states)]
    from .training.decision_lm import DEFAULTS as LM_DEFAULTS
    from .training.decision_lm import LMScorer, load_lm

    tokenizer, model = load_lm(base_model, device)
    model.eval()
    length = int(max_length or LM_DEFAULTS["max_length"])
    return LMScorer(tokenizer, model, question, device, length).logits(states)


__all__ = [
    "DecisionModel",
    "answer_payload",
    "base_decision_logits",
    "rank_with_embedding",
    "rank_with_reranker",
]

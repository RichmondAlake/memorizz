"""Deterministic metrics and a small local-judge contract."""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from typing import Any, Callable, Dict, Mapping, Sequence

from .models import MemoryBenchmarkCase

Judge = Callable[[MemoryBenchmarkCase, str], Mapping[str, Any]]

_MONTH_NUMBERS = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}
_MONTH_PATTERN = "|".join(_MONTH_NUMBERS)


def _normalize_unambiguous_dates(text: str) -> str:
    """Canonicalize ISO and named-month dates before token scoring."""

    def canonical(year: str, month: str | int, day: str) -> str:
        month_number = (
            _MONTH_NUMBERS[str(month).casefold()]
            if not str(month).isdigit()
            else int(month)
        )
        return f"{int(year):04d} {month_number:02d} {int(day):02d}"

    value = re.sub(
        r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b",
        lambda match: canonical(
            match.group("year"), match.group("month"), match.group("day")
        ),
        text,
    )
    value = re.sub(
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"(?P<month>{_MONTH_PATTERN})[,]?\s+(?P<year>\d{{4}})\b",
        lambda match: canonical(
            match.group("year"), match.group("month"), match.group("day")
        ),
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(
        rf"\b(?P<month>{_MONTH_PATTERN})\s+"
        rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{{4}})\b",
        lambda match: canonical(
            match.group("year"), match.group("month"), match.group("day")
        ),
        value,
        flags=re.IGNORECASE,
    )


def normalize_answer(value: Any) -> str:
    text = _normalize_unambiguous_dates(str(value or "").lower())
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(prediction: str, answer: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not predicted_tokens and not answer_tokens:
        return 1.0
    if not predicted_tokens or not answer_tokens:
        return 0.0
    common = Counter(predicted_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def substring_exact_match(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction)
    expected = normalize_answer(answer)
    return float(bool(expected) and expected in predicted)


def _predicted_top_five(value: str) -> Sequence[str]:
    text = str(value or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        parsed = (
            parsed.get("items") or parsed.get("recommendations") or parsed.get("answer")
        )
    if isinstance(parsed, list):
        values = [str(item) for item in parsed]
    else:
        values = re.split(r"[,;\n|]+", text)
    return [normalize_answer(item) for item in values if normalize_answer(item)][:5]


def recall_at_five(prediction: str, answers: Sequence[str]) -> float:
    expected = {
        normalize_answer(answer) for answer in answers if normalize_answer(answer)
    }
    if not expected:
        return 0.0
    predicted = set(_predicted_top_five(prediction))
    return len(expected.intersection(predicted)) / len(expected)


def score_answer(
    case: MemoryBenchmarkCase,
    prediction: str,
    *,
    judge: Judge | None = None,
) -> Dict[str, Any]:
    """Apply the benchmark record's upstream scoring family."""
    if case.scorer == "llm_judge":
        if judge is None:
            raise ValueError("An llm_judge case requires a configured local judge")
        judged = dict(judge(case, prediction))
        score = max(0.0, min(float(judged.get("score", 0.0)), 1.0))
        return {
            "score": score,
            "correct": score >= 0.999,
            "metric": "local_llm_judge",
            "judge": judged,
        }
    if case.scorer == "recall_at_5":
        score = recall_at_five(prediction, case.answers)
    else:
        metric = {
            "answer_f1": answer_f1,
            "exact_match": exact_match,
            "substring_exact_match": substring_exact_match,
        }[case.scorer]
        score = max(
            (metric(prediction, answer) for answer in case.answers), default=0.0
        )
    return {
        "score": score,
        "correct": score >= 0.999,
        "metric": case.scorer,
    }


def retrieval_metrics(
    retrieved: Sequence[Mapping[str, Any]], relevant_source_ids: Sequence[str]
) -> Dict[str, float | None]:
    """Compute Recall@k, MRR, and binary nDCG over upstream source IDs."""
    relevant = {str(item) for item in relevant_source_ids if str(item).strip()}
    if not relevant:
        return {"recall_at_k": None, "mrr": None, "ndcg_at_k": None}
    ranked: list[bool] = []
    matched: set[str] = set()
    for row in retrieved:
        identifiers = {
            str(row.get("source_id") or ""),
            str(row.get("parent_source_id") or ""),
            *(str(item) for item in row.get("linked_source_ids") or []),
        }
        hits = relevant.intersection(identifiers)
        ranked.append(bool(hits - matched))
        matched.update(hits)
    recall = len(matched) / len(relevant)
    first_rank = next((index for index, hit in enumerate(ranked, start=1) if hit), None)
    mrr = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(
        (1.0 / math.log2(index + 1)) for index, hit in enumerate(ranked, start=1) if hit
    )
    ideal_hits = min(len(relevant), len(retrieved))
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return {
        "recall_at_k": recall,
        "mrr": mrr,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
    }


def build_judge_prompt(case: MemoryBenchmarkCase, prediction: str) -> str:
    references = (
        "\n".join(f"- {answer}" for answer in case.answers)
        or "- No fixed reference answer."
    )
    rubric = (
        "\n".join(f"- {item}" for item in case.rubric)
        or "- Be correct and grounded in the supplied memory."
    )
    return (
        "Evaluate the candidate response for a long-term-memory benchmark. "
        "Use only the question, references, and rubric below. Award 1 for fully "
        "correct, 0.5 for partially correct, and 0 for incorrect, unsupported, or "
        "a failed abstention. For cognitive-constraint cases, consistency with the "
        "rubric evidence matters even when there is no fixed answer.\n\n"
        f"Question:\n{case.question}\n\n"
        f"Reference answer(s):\n{references}\n\n"
        f"Rubric/evidence:\n{rubric}\n\n"
        f"Candidate response:\n{prediction}\n\n"
        'Return JSON only: {"score": 0|0.5|1, "reason": "brief explanation"}'
    )


def parse_judge_response(value: str) -> Dict[str, Any]:
    text = str(value or "").strip()
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and "score" in payload:
            try:
                score = float(payload["score"])
            except (TypeError, ValueError):
                continue
            return {
                "score": min(1.0, max(0.0, score)),
                "reason": str(payload.get("reason") or ""),
                "raw": text,
            }
    score_match = re.search(r"(?<!\d)(0(?:\.5)?|1(?:\.0)?)(?!\d)", text)
    score = float(score_match.group(1)) if score_match else 0.0
    return {"score": score, "reason": "Unable to parse strict judge JSON.", "raw": text}


def parse_reader_response(value: str) -> Dict[str, Any]:
    """Parse the grounded reader contract while preserving text-model fallback."""

    text = str(value or "").strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping) or "answer" not in payload:
            continue
        source_ids = payload.get("source_ids") or []
        if not isinstance(source_ids, list):
            source_ids = [source_ids]
        return {
            "answer": str(payload.get("answer") or "").strip(),
            "source_ids": [str(item) for item in source_ids if str(item).strip()],
            "abstained": bool(payload.get("abstained")),
            "structured": True,
            "raw": text,
        }
    return {
        "answer": text,
        "source_ids": [],
        "abstained": bool(
            re.search(
                r"\b(?:do not know|don't know|insufficient evidence)\b", text, re.I
            )
        ),
        "structured": False,
        "raw": text,
    }


def reader_response_needs_repair(value: str) -> bool:
    """Return whether a JSON-like reader response failed its output contract."""

    text = str(value or "").strip()
    if not text:
        return False
    parsed = parse_reader_response(text)
    return not parsed["structured"] and (
        "{" in text or "```json" in text.casefold() or '"answer"' in text
    )


def citation_metrics(
    cited_source_ids: Sequence[str],
    retrieved: Sequence[Mapping[str, Any]],
    relevant_source_ids: Sequence[str],
) -> Dict[str, Any]:
    """Measure citation validity separately from answer plausibility."""

    retrieved_ids = {
        str(identifier)
        for row in retrieved
        for identifier in (
            row.get("source_id"),
            row.get("parent_source_id"),
            *(row.get("linked_source_ids") or []),
        )
        if identifier
    }
    relevant = {str(item) for item in relevant_source_ids if str(item).strip()}
    cited = [str(item) for item in cited_source_ids if str(item).strip()]
    valid = [item for item in cited if item in retrieved_ids]
    relevant_cited = {item for item in valid if item in relevant}
    return {
        "cited_source_ids": cited,
        "valid_source_ids": valid,
        "invalid_source_ids": [item for item in cited if item not in retrieved_ids],
        "citation_precision": len(valid) / len(cited) if cited else None,
        "gold_citation_recall": (
            len(relevant_cited) / len(relevant) if relevant else None
        ),
        "grounding_status": (
            "grounded"
            if cited and len(valid) == len(cited)
            else "invalid-citation"
            if cited
            else "uncited"
        ),
    }


__all__ = [
    "answer_f1",
    "build_judge_prompt",
    "exact_match",
    "normalize_answer",
    "citation_metrics",
    "parse_judge_response",
    "parse_reader_response",
    "reader_response_needs_repair",
    "recall_at_five",
    "retrieval_metrics",
    "score_answer",
    "substring_exact_match",
]

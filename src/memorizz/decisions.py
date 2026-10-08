"""Typed OpenAI Decisions API client; decisions never execute actions or tools."""

from __future__ import annotations

import math
import os
import time

import requests


def _probability(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Invalid decision probability")
    return value


def _typed_key(value):
    if not isinstance(value, (str, bool)):
        raise ValueError("Decision choices must be strings or booleans")
    return type(value).__name__, value


class OpenAIDecisions:
    """Predicate, choice and score evaluations using POST /v1/decisions.

    Uses the documented HTTP contract, including with older installed OpenAI
    SDKs. No chat/completions fallback and no execution of returned choices.
    """

    def __init__(self, *, api_key=None, model="gpt-6-luna", timeout=60):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.model = model
        self.timeout = timeout

    def evaluate(self, input, questions, *, safety_identifier=None):
        if not isinstance(questions, list) or not questions:
            raise ValueError("questions must be a non-empty ordered list")
        if any(not isinstance(q, dict) for q in questions):
            raise ValueError("Every question must be an object")
        names = [q.get("name") for q in questions]
        if any(not isinstance(name, str) for name in names) or len(set(names)) != len(
            names
        ):
            raise ValueError("Question names must be unique and present")
        for question in questions:
            if question.get("type") not in {"predicate", "choice", "score"}:
                raise ValueError("Unsupported decision question type")
            if not isinstance(question.get("instructions"), str):
                raise ValueError("Every question requires instructions")
            if question["type"] in {"choice", "score"}:
                field = "choices" if question["type"] == "choice" else "levels"
                key = "value" if question["type"] == "choice" else "label"
                options = question.get(field)
                if (
                    not isinstance(options, list)
                    or not options
                    or any(
                        not isinstance(option, dict) or key not in option
                        for option in options
                    )
                ):
                    raise ValueError("Decision options must be a non-empty list")
                if key == "label" and any(
                    not isinstance(option[key], str) for option in options
                ):
                    raise ValueError("Score labels must be strings")
                keys = [_typed_key(option[key]) for option in options]
                if len(set(keys)) != len(keys):
                    raise ValueError("Decision options must be unique")
        payload = {"model": self.model, "input": input, "questions": questions}
        if safety_identifier is not None:
            payload["safety_identifier"] = safety_identifier
        started = time.perf_counter()
        response = requests.post(
            "https://api.openai.com/v1/decisions",
            headers={"Authorization": "Bearer " + self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(f"OpenAI Decisions returned HTTP {response.status_code}")
        result = response.json()
        answers = result.get("answers")
        if (
            not isinstance(answers, list)
            or any(not isinstance(a, dict) for a in answers)
            or [a.get("name") for a in answers] != names
        ):
            raise ValueError(
                "Decision answer order or coverage does not match the request"
            )
        for question, answer in zip(questions, answers):
            if answer.get("type") == "refusal":
                continue
            if answer.get("type") != question["type"]:
                raise ValueError("Decision answer type does not match the question")
            if question["type"] == "predicate":
                _probability(answer.get("probability"))
            else:
                _probability(answer.get("confidence"))
                field = "value" if question["type"] == "choice" else "label"
                options = question[
                    "choices" if question["type"] == "choice" else "levels"
                ]
                expected = {_typed_key(option[field]) for option in options}
                probabilities = answer.get("probabilities")
                if not isinstance(probabilities, list) or any(
                    not isinstance(p, dict) or field not in p for p in probabilities
                ):
                    raise ValueError("Invalid decision distribution")
                keys = [_typed_key(p[field]) for p in probabilities]
                if len(keys) != len(set(keys)) or set(keys) != expected:
                    raise ValueError(
                        "Decision distribution coverage does not match the options"
                    )
                if not math.isclose(
                    sum(_probability(p.get("probability")) for p in probabilities),
                    1,
                    abs_tol=0.01,
                ):
                    raise ValueError("Decision probabilities must sum to one")
                if (
                    question["type"] == "choice"
                    and _typed_key(answer.get("choice")) not in expected
                ):
                    raise ValueError("Decision choice is not a supplied option")
                if question["type"] == "score":
                    score = answer.get("score")
                    if (
                        isinstance(score, bool)
                        or not isinstance(score, (float, int))
                        or not math.isfinite(score)
                        or not 0 <= score <= len(options) - 1
                    ):
                        raise ValueError("Invalid decision score")
        return {
            **result,
            "latency_seconds": time.perf_counter() - started,
            "endpoint": "/v1/decisions",
        }

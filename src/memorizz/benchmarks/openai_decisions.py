"""Semantically matched Jev/OpenAI question recipes for paired reranking."""

import json

from ..decisions import OpenAIDecisions
from .jev_reranking import question_payload, scores_from_answers


def rank_decisions(query, texts, method, model):
    state, jev_questions = question_payload(query, texts, method)
    questions = []
    for name, question in jev_questions.items():
        value = {
            "name": name,
            "instructions": question["instructions"],
            "type": "predicate" if method == "noul" else method,
        }
        if method == "choice":
            value["choices"] = [
                {"value": key, "description": description}
                for key, description in question["criteria"].items()
            ]
        elif method == "score":
            value["levels"] = [
                {"label": str(i), "description": description}
                for i, description in enumerate(question["criteria"])
            ]
        questions.append(value)
    result = OpenAIDecisions(model=model).evaluate(
        json.dumps(state, ensure_ascii=False, sort_keys=True), questions
    )
    converted = {}
    for answer in result["answers"]:
        if answer["type"] == "refusal":
            raise RuntimeError("OpenAI Decisions declined a relevance question")
        converted[answer["name"]] = {"type": method}
        if method == "noul":
            converted[answer["name"]]["noul"] = answer["probability"]
        else:
            field = "value" if method == "choice" else "label"
            converted[answer["name"]]["probabilities"] = {
                str(item[field]): item["probability"]
                for item in answer["probabilities"]
            }
    return scores_from_answers(converted, len(texts), method), result.get("usage") or {}

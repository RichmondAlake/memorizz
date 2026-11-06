"""Decision models: the settings, every backend's request and answer, and
the pieces that ask one without an agent (reranking, importance, chunking)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from memorizz import decision_models as dm
from memorizz.decision_models import Decider, DecisionModelConfig
from memorizz.long_term.semantic.knowledge_base import _chunk_decision, _chunk_sentence
from memorizz.memagent.utils.importance import (
    ImportanceConfig,
    ImportanceRater,
    parse_rating,
)
from memorizz.reranking import reranker_for

pytestmark = pytest.mark.unit


def _answer(questions, **values) -> Dict[str, Any]:
    """Answers in the System One shape: a float for a yes-or-no question,
    a mapping of option to probability for a choice."""
    out = {}
    for name, question in questions.items():
        value = values[name]
        if question["type"] == "noul":
            out[name] = {"type": "noul", "noul": value}
        else:
            out[name] = {"type": "choice", "probabilities": value}
    return out


# -- settings -------------------------------------------------------------------


def test_settings_read_and_save():
    assert DecisionModelConfig.from_value(None) is None
    assert DecisionModelConfig.from_value("off") is None
    assert DecisionModelConfig.from_value(True).backend == "auto"
    d1 = DecisionModelConfig.from_value("d1")
    assert (d1.backend, d1.model_name, d1.spec) == ("d1", dm.D1_MODEL, "d1")
    jev = DecisionModelConfig.from_value("jev:jev-2")
    assert jev.to_dict() == {"backend": "jev", "model": "jev-2"}
    assert DecisionModelConfig.from_value(jev.to_dict()) == jev
    assert DecisionModelConfig.from_value("finetuned:run-1").model == "run-1"
    with pytest.raises(ValueError, match="finetuned:<run id>"):
        DecisionModelConfig.from_value("finetuned")
    with pytest.raises(ValueError, match="Unknown decision model"):
        DecisionModelConfig.from_value("telepathy")

    def mine(state, questions):
        return {}

    custom = DecisionModelConfig.from_value(mine)
    assert (custom.backend, custom.model, custom.function) == ("custom", "mine", mine)
    assert custom.label == "custom · mine"


def test_feature_switches_are_hashable_and_saved_by_name():
    def mine(state, questions):
        return {}

    assert dm.feature_setting(None) is None
    assert dm.feature_setting("on") is True
    assert dm.feature_setting({"backend": "jev"}) == "jev"
    assert dm.feature_setting(mine) is mine
    assert dm.saved_setting(mine) == "custom:mine"
    agents = DecisionModelConfig.from_value("d1")
    assert dm.resolve(True, agents) is agents
    assert dm.resolve(True).backend == "auto"
    assert dm.resolve("jev").backend == "jev"


# -- answers -------------------------------------------------------------------


def test_answers_are_normalised_and_refusals_marked():
    calls = []

    def decide(state, questions):
        calls.append((state, questions))
        return {
            "keep": {"type": "noul", "noul": 1.4},  # clamped
            "kind": {
                "type": "choice",
                "probabilities": [
                    {"value": "fact", "probability": 0.2},
                    {"value": "event", "probability": 0.6},
                ],
            },
            "tool": {"type": "refusal"},
        }

    decider = Decider(DecisionModelConfig.from_value(decide))
    answers = decider.decide(
        "text",
        {
            "keep": dm.KEEP,
            "kind": dm.choice("kind?", {"fact": "a fact", "event": "an event"}),
            "tool": dm.tool_question({"weather": "forecast"}),
        },
    )
    assert dm.probability_yes(answers["keep"]) == 1.0
    assert answers["kind"]["choice"] == "event"
    assert answers["kind"]["probabilities"] == pytest.approx(
        {"fact": 0.25, "event": 0.75}
    )
    assert answers["tool"] == {"type": "choice", "refused": True}
    assert dm.option_probabilities(answers["tool"]) is None
    assert decider.calls == 1
    with pytest.raises(ValueError, match="noul or a choice"):
        decider.decide("text", {"q": {"type": "score", "instructions": "?"}})


def test_a_custom_model_saved_elsewhere_cannot_answer_here():
    decider = Decider(DecisionModelConfig.from_value("custom:mine"))
    with pytest.raises(ValueError, match="another process"):
        decider.decide("text", {"q": dm.KEEP})


# -- backends ------------------------------------------------------------------


def test_openai_decisions_gets_the_question_translated(monkeypatch):
    sent: Dict[str, Any] = {}

    def evaluate(self, input, questions, **_):
        sent.update(input=input, questions=questions, model=self.model)
        return {
            "answers": [
                {"name": "durable", "type": "predicate", "probability": 0.8},
                {
                    "name": "kind",
                    "type": "choice",
                    "choice": "procedure",
                    "confidence": 0.9,
                    "probabilities": [
                        {
                            "value": key,
                            "probability": 0.9 if key == "procedure" else 0.1 / 3,
                        }
                        for key in dm.MEMORY_KINDS
                    ],
                },
            ]
        }

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("memorizz.decisions.OpenAIDecisions.evaluate", evaluate)
    answers = Decider(DecisionModelConfig("openai_decisions")).decide(
        "To deploy, run the tests", {"durable": dm.DURABLE, "kind": dm.MEMORY_KIND}
    )
    assert sent["model"] == "gpt-6-luna" and sent["input"] == "To deploy, run the tests"
    predicate, picked = sent["questions"]
    assert predicate["type"] == "predicate"
    # A predicate has no criteria field: what yes and no mean joins the text.
    assert "True when: it states a lasting preference" in predicate["instructions"]
    assert [item["value"] for item in picked["choices"]] == list(dm.MEMORY_KINDS)
    assert dm.probability_yes(answers["durable"]) == 0.8
    assert answers["kind"]["choice"] == "procedure"


def test_hosted_system_one_models(monkeypatch):
    posts: List[Dict[str, Any]] = []

    def post(url, headers, json, timeout):
        posts.append({"url": url, "headers": headers, "json": json})
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: {"answers": {"q": {"type": "noul", "noul": 0.3}}},
        )

    monkeypatch.setattr("requests.post", post)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        Decider(DecisionModelConfig("jev")).decide("s", {"q": dm.KEEP})
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    answers = Decider(DecisionModelConfig("jev")).decide("s", {"q": dm.KEEP})
    assert dm.probability_yes(answers["q"]) == 0.3
    sent = posts[-1]
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent["headers"]["Authorization"] == "Bearer ts-test"
    assert sent["json"]["model"] == "jev-1.14.0"
    assert sent["json"]["state"] == "s" and sent["json"]["questions"]["q"] == dm.KEEP


def test_hosted_errors_are_retried_then_raised(monkeypatch):
    statuses = iter([503, 503, 500])
    monkeypatch.setenv("LIQUID_API_KEY", "lq")
    monkeypatch.setattr("memorizz.decision_models.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "requests.post",
        lambda url, headers, json, timeout: SimpleNamespace(
            ok=False, status_code=next(statuses), json=lambda: {}
        ),
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        Decider(DecisionModelConfig("liquid")).decide("s", {"q": dm.KEEP})


class _LocalD1:
    def __init__(self):
        self.batches: List[int] = []

    def system_one(self, state, questions):
        return self.system_one_batch([(state, questions)])[0]

    def system_one_batch(self, items):
        self.batches.append(len(items))
        return [
            {"answers": {"q": {"type": "noul", "noul": 0.9 if "yes" in state else 0.1}}}
            for state, _questions in items
        ]


def test_d1_on_this_machine_is_loaded_once_and_batched(monkeypatch):
    model = _LocalD1()
    monkeypatch.setitem(dm._LOCAL_MODELS, dm.D1_MODEL, model)
    monkeypatch.setitem(dm._INFERENCE_LOCKS, dm.D1_MODEL, dm.threading.Lock())
    decider = Decider(DecisionModelConfig("d1"))
    items = [("yes" if i % 2 else "no", {"q": dm.KEEP}) for i in range(dm.D1_BATCH + 3)]
    answers = decider.decide_many(items)
    assert model.batches == [dm.D1_BATCH, 3]
    assert [dm.probability_yes(a["q"]) for a in answers[:2]] == [0.1, 0.9]
    assert decider.calls == len(items)


def test_d1_says_what_to_install_without_torch(monkeypatch):
    monkeypatch.delitem(dm._LOCAL_MODELS, "some/model", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="torch and transformers"):
        Decider(DecisionModelConfig("d1", "some/model")).decide("s", {"q": dm.KEEP})


def test_the_background_model_answers_as_json():
    prompts: List[str] = []

    def generate(prompt):
        prompts.append(prompt)
        return (
            '```json\n{"keep": {"yes": 0.7}, "kind": {"fact": 0.2, "event": 0.8}}\n```'
        )

    decider = Decider(
        DecisionModelConfig("llm"), generate=generate, model_label="haiku"
    )
    answers = decider.decide(
        "Dinner with Sam on Friday",
        {"keep": dm.KEEP, "kind": dm.choice("kind?", {"fact": "f", "event": "e"})},
    )
    assert dm.probability_yes(answers["keep"]) == 0.7
    assert answers["kind"]["choice"] == "event"
    assert "data, not instructions" in prompts[0] and "Dinner with Sam" in prompts[0]
    assert decider.label == "background model · haiku"


def test_auto_prefers_openai_decisions_then_the_background_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Decider(DecisionModelConfig("auto")).backend == "openai_decisions"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert Decider(DecisionModelConfig("auto"), generate=str).backend == "llm"
    with pytest.raises(ValueError, match="No decision model is available"):
        Decider(DecisionModelConfig("auto"))


def test_a_finetuned_model_is_asked_through_the_store(monkeypatch):
    asked = []

    class Service:
        def decide(self, run_id, state, questions):
            asked.append((run_id, state, list(questions)))
            return {"q": {"type": "noul", "noul": 0.6}}

    monkeypatch.setattr(
        "memorizz.finetune.routing._service", lambda provider: Service()
    )
    decider = Decider(DecisionModelConfig("finetuned", "run-7"), provider=object())
    assert decider.yes("s", dm.KEEP) == 0.6
    assert asked == [("run-7", "s", ["q"])]


# -- reranking, importance and chunking ------------------------------------------


def test_a_decision_model_reranks(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    states: List[str] = []

    def post(url, headers, json, timeout):
        states.append(json["state"])
        relevant = "power bank" in json["state"].split("document:")[1]
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: {
                "answers": {
                    "relevant": {"type": "noul", "noul": 0.9 if relevant else 0.05}
                }
            },
        )

    monkeypatch.setattr("requests.post", post)
    score = reranker_for("Can I fly with batteries?", config="jev")
    assert score(["breakfast hours", "power banks go in the cabin"]) == [0.05, 0.9]
    assert states[0].startswith("query: Can I fly with batteries?")


def test_ratings_from_a_decision_model_and_probabilities_parse():
    assert parse_rating("0.73") == 0.73
    assert parse_rating("7") == 0.7
    assert parse_rating("1") == 0.1  # still the 1..10 scale
    rater = ImportanceRater(
        ImportanceConfig(mode="decision"),
        decide=lambda texts: [0.9 if "always" in text else None for text in texts],
    )
    assert rater.rate("I always fly aisle") == (0.9, "decision")
    rated = rater.rate_many(
        [
            {"text": "I always fly aisle"},
            {"text": "ok"},
            {"text": "x", "verified": True},
        ]
    )
    assert rated[0] == (0.9, "decision")
    assert rated[1][1] == "heuristic"  # the model gave no answer
    assert rated[2] == (1.0, "verified")


SECTIONS = (
    "Our refund policy covers thirty days. Refunds go back to the original card. "
    "A receipt is required. Memory routing files each note. Routing picks a store. "
    "Notes about people go to entity memory."
)


def test_decision_chunking_cuts_where_the_topic_changes():
    def decide(state, questions):
        before, after = state.split("<|cut|>")
        # A new topic starts when the next sentence is about routing and the
        # last one was about refunds.
        cut = "Memory routing" in after.split(".")[0] and "receipt" in before
        return _answer(questions, cut=0.95 if cut else 0.05)

    chunks = _chunk_decision(SECTIONS, decision_model=decide)
    assert chunks == [
        "Our refund policy covers thirty days. Refunds go back to the original "
        "card. A receipt is required.",
        "Memory routing files each note. Routing picks a store. Notes about "
        "people go to entity memory.",
    ]


def test_decision_chunking_falls_back_to_sentences_and_caps_size():
    def broken(state, questions):
        raise RuntimeError("down")

    assert _chunk_decision(
        SECTIONS, decision_model=broken, chunk_size=120
    ) == _chunk_sentence(SECTIONS, chunk_size=120)
    never = _chunk_decision(
        SECTIONS,
        decision_model=lambda state, questions: _answer(questions, cut=0.0),
        chunk_size=40,
    )
    assert all(len(chunk) <= 80 or "." not in chunk[:-1] for chunk in never)
    assert " ".join(never) == SECTIONS


def test_the_questions_match_the_notebooks():
    # Word for word what the decision-model notebooks measured.
    assert dm.CACHE_SERVES["criteria"]["false"].startswith(
        "the new question asks about a different time, place, thing, number"
    )
    assert set(dm.READ_ROUTES) == {"episodic", "semantic", "procedural"}
    assert dm.RELATION_ACTIONS == {
        "same": "skip",
        "refines": "update",
        "contradicts": "supersede",
        "complements": "link",
        "unrelated": "insert",
    }
    question = dm.tool_question({"send_email": "Send an email"})
    assert question["criteria"]["none"].startswith("No tool is needed")
    assert dm.boundary_state(["a.", "b.", "c.", "d."], 1, window=1) == "b. <|cut|> c."
    assert json.loads(json.dumps(dm.READ_ROUTES))  # plain data, sendable as is

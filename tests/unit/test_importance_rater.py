"""Importance ratings: heuristic ordering, prompt parsing and batched LLM mode."""

from __future__ import annotations

import json

import pytest

from memorizz.memagent.utils.importance import (
    ImportanceConfig,
    ImportanceRater,
    clamp_importance,
    heuristic_importance,
    parse_rating,
)


@pytest.mark.unit
def test_clamp_and_parse_map_the_paper_scale_to_unit_interval():
    assert clamp_importance(7) == 0.7
    assert clamp_importance(0.42) == 0.42
    assert clamp_importance("not a number") is None
    assert parse_rating("Rating: 8") == 0.8
    assert parse_rating("I would say 10 out of 10") == 1.0
    assert parse_rating("no digits here") is None


@pytest.mark.unit
def test_heuristic_orders_kinds_sensibly():
    assert heuristic_importance("x", verified=True) == 1.0
    host = heuristic_importance("The owner is Mira", host_asserted=True)
    user = heuristic_importance("We decided the launch moves to Friday", role="user")
    tool = heuristic_importance("{...3000 bytes of json...}", kind="tool")
    chat = heuristic_importance("ok", role="assistant")
    assert host > user > tool
    assert user > chat


@pytest.mark.unit
def test_off_mode_stores_nothing():
    rater = ImportanceRater(ImportanceConfig(mode="off"))
    assert rater.rate("anything") == (None, "off")
    assert rater.enabled is False


@pytest.mark.unit
def test_llm_mode_uses_the_model_and_falls_back_to_heuristic():
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return "7"

    rater = ImportanceRater(
        ImportanceConfig(mode="llm", purpose="tracking a release"), generate=generate
    )
    assert rater.rate("The release owner changed to Mira", role="user") == (0.7, "llm")
    assert "tracking a release" in prompts[-1]

    def broken(prompt):
        raise RuntimeError("model down")

    rater = ImportanceRater(ImportanceConfig(mode="llm"), generate=broken)
    rating, source = rater.rate("The release owner changed to Mira", role="user")
    assert source == "heuristic" and 0.0 < rating <= 1.0


@pytest.mark.unit
def test_batch_rating_parses_json_and_keeps_verified_items_out_of_the_model():
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return json.dumps({"1": 9, "2": 2})

    rater = ImportanceRater(
        ImportanceConfig(mode="llm", batch_size=8), generate=generate
    )
    results = rater.rate_many(
        [
            {"text": "We must ship by Friday", "role": "user"},
            {"text": "ok", "role": "assistant"},
            {"text": "verified outcome", "verified": True},
        ]
    )
    assert results[0] == (0.9, "llm")
    assert results[1] == (0.2, "llm")
    assert results[2] == (1.0, "verified")
    assert len(calls) == 1  # one batched prompt for the two model-rated items


@pytest.mark.unit
def test_config_from_env(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "llm")
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_BATCH_SIZE", "4")
    config = ImportanceConfig.from_env()
    assert (config.mode, config.model, config.batch_size) == ("llm", "qwen2.5:3b", 4)
    assert ImportanceConfig.from_mapping({"mode": "bogus"}).mode == "heuristic"

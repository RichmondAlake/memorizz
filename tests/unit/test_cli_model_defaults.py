"""The CLI's default models match the UI's, and /models offers current ones."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from memorizz.cli import agent_factory, commands
from memorizz.llms.model_catalog import DEFAULT_LLM_MODEL_BY_PROVIDER

pytestmark = pytest.mark.unit


def test_cli_defaults_are_the_ui_defaults(monkeypatch):
    monkeypatch.delenv("MEMORIZZ_DEFAULT_LLM_MODEL", raising=False)
    assert agent_factory._default_model("openai") == "gpt-6.1-sol"
    assert agent_factory._default_model("anthropic") == "claude-opus-5-5"
    for provider in ("openai", "anthropic", "azure"):
        assert (
            agent_factory._default_model(provider)
            == DEFAULT_LLM_MODEL_BY_PROVIDER[provider]
        )
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_MODEL", "gpt-6-luna")
    assert agent_factory._default_model("openai") == "gpt-6-luna"


def test_models_lists_the_catalog_then_newer_models_for_the_key(monkeypatch):
    monkeypatch.setattr(
        "memorizz.llms.model_lists.latest_models",
        lambda provider: ["gpt-7-preview", "gpt-6.1-sol"],
    )
    out = io.StringIO()
    session = SimpleNamespace(
        console=Console(file=out, force_terminal=False, width=120),
        provider_name="openai",
        model_name="gpt-6.1-sol",
        llm_config={"provider": "openai", "model": "gpt-6.1-sol"},
    )
    commands.cmd_model(session, "")
    text = out.getvalue()
    assert "Models for openai:" in text
    listed = [line.strip("• ").strip() for line in text.splitlines() if "•" in line]
    assert listed[:3] == ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"]
    assert listed[-1] == "gpt-7-preview" and listed.count("gpt-6.1-sol") == 1
    assert "gpt-4o" not in listed


def test_anthropic_choices_lead_with_the_5_5_models(monkeypatch):
    monkeypatch.setattr("memorizz.llms.model_lists.latest_models", lambda provider: [])
    values = [row["value"] for row in commands._model_choices("anthropic")]
    assert values[:4] == [
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-fable-5-1",
    ]

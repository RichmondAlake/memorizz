# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Ollama model selection for the CLI agent factory."""

import pytest

from memorizz.cli.agent_factory import _pick_ollama_model

LOCAL = ["qwen2.5:7b", "llama3.1:8b", "nomic-embed-text:latest"]


@pytest.mark.unit
def test_pick_ollama_honors_cloud_model(monkeypatch):
    # Ollama Cloud models (":cloud") never appear in local /api/tags but must
    # still be honored instead of silently substituted with a local model.
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_MODEL", "glm-5.2:cloud")
    assert _pick_ollama_model(LOCAL) == "glm-5.2:cloud"


@pytest.mark.unit
def test_pick_ollama_exact_local_match(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_MODEL", "qwen2.5:7b")
    assert _pick_ollama_model(LOCAL) == "qwen2.5:7b"


@pytest.mark.unit
def test_pick_ollama_base_name_match(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_MODEL", "qwen2.5")
    assert _pick_ollama_model(LOCAL) == "qwen2.5:7b"


@pytest.mark.unit
def test_pick_ollama_auto_rank_skips_embeddings(monkeypatch):
    monkeypatch.delenv("MEMORIZZ_DEFAULT_LLM_MODEL", raising=False)
    # llama3.1 ranks above qwen2.5 as the tool-agent default; embed models skipped
    assert _pick_ollama_model(LOCAL) == "llama3.1:8b"

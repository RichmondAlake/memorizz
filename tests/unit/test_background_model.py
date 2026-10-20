"""A second model for memory upkeep, and upkeep that runs off the turn."""

from __future__ import annotations

import contextvars
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from memorizz.memagent import MemAgent
from memorizz.memagent.managers.memory_manager import MemoryManager
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-background-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda text: list(VECTOR))
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")


class _Embedder:
    def get_embedding(self, text: str) -> List[float]:
        return list(VECTOR)


class _Model:
    def __init__(self, name: str = "main", **config: Any) -> None:
        self.model = name
        self.config = {"provider": "openai", "model": name, **config}

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        return "ok"

    def get_config(self) -> Dict[str, Any]:
        return {k: v for k, v in self.config.items() if k != "api_key"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


@pytest.fixture()
def built(monkeypatch):
    """create_llm_provider builds a _Model named after the config."""
    made: List[Dict[str, Any]] = []

    def create(config):
        made.append(dict(config))
        return _Model(
            config["model"], **{k: v for k, v in config.items() if k != "model"}
        )

    monkeypatch.setattr("memorizz.memagent.core.create_llm_provider", create)
    return made


def _agent(work_dir: Path, **kwargs) -> MemAgent:
    kwargs.setdefault("model", _Model())
    kwargs.setdefault(
        "memory_provider",
        FileSystemProvider(
            FileSystemConfig(
                root_path=work_dir / "memory",
                embedding_provider=_Embedder(),
                use_faiss=False,
                lazy_vector_indexes=True,
            )
        ),
    )
    kwargs.setdefault("memory_ids", ["memory-1"])
    kwargs.setdefault("auto_register", False)
    kwargs.setdefault("agent_id", "agent-1")
    return MemAgent(**kwargs)


def test_summaries_run_on_the_background_model(work_dir, built, monkeypatch):
    agent = _agent(
        work_dir, background_llm_config={"provider": "openai", "model": "cheap"}
    )
    used: List[Any] = []
    monkeypatch.setattr(
        MemoryManager,
        "generate_summaries",
        lambda self, **kw: used.append(kw["model"]) or [],
    )
    agent.generate_summaries(memory_id="memory-1", thread_id="t1")
    assert [model.model for model in used] == ["cheap"]
    assert agent.model.model == "main"
    assert agent.describe_memory()["background_model"] == {
        "provider": "openai",
        "model": "cheap",
    }

    agent.with_background_model(None)
    agent.generate_summaries(memory_id="memory-1", thread_id="t1")
    assert used[-1] is agent.model
    agent.close()


def test_the_background_model_is_saved_without_its_key(work_dir, built):
    agent = _agent(
        work_dir,
        background_llm_config={
            "provider": "openai",
            "model": "cheap",
            "api_key": "sk-secret",
        },
    )
    agent.save()
    stored = agent.memory_provider.retrieve_memagent("agent-1")
    stored = stored if isinstance(stored, dict) else stored.model_dump()
    assert stored["background_llm_config"] == {"provider": "openai", "model": "cheap"}
    assert "sk-secret" not in str(stored)

    loaded = MemAgent.load(
        "agent-1", memory_provider=agent.memory_provider, model=_Model()
    )
    assert loaded.background_model.model == "cheap"
    assert built[-1] == {"provider": "openai", "model": "cheap"}
    loaded.close()
    agent.close()


def test_a_model_that_cannot_be_built_falls_back_to_the_agents(work_dir, monkeypatch):
    def fail(config):
        raise RuntimeError("no such model")

    monkeypatch.setattr("memorizz.memagent.core.create_llm_provider", fail)
    agent = _agent(
        work_dir, background_llm_config={"provider": "openai", "model": "gone"}
    )
    assert agent.background_model is agent.model
    assert agent.background_llm_config == {"provider": "openai", "model": "gone"}
    agent.close()


def test_skill_distillation_follows_a_swapped_model_unless_it_has_its_own(work_dir):
    agent = _agent(work_dir)
    calls: List[Any] = []
    agent.continual_learning_manager = SimpleNamespace(set_llm_provider=calls.append)
    swapped = _Model("swapped")
    agent.model = swapped
    assert calls[-1] is swapped
    cheap = _Model("cheap")
    agent.with_background_model(cheap)
    agent.model = _Model("again")
    assert calls[-1] is cheap
    agent.continual_learning_manager = None
    agent.close()


def test_the_next_turns_compaction_runs_between_turns(work_dir, monkeypatch):
    agent = _agent(work_dir)
    started, release = threading.Event(), threading.Event()
    calls: List[Dict[str, Any]] = []

    def summarize(**kwargs):
        calls.append(kwargs)
        started.set()
        release.wait(5)
        return ["summary-1"]

    monkeypatch.setattr(agent, "generate_summaries", summarize)
    agent._compaction_outlook = {
        "estimate": 900,
        "window": 1000,
        "threshold": 80,
        "keep": 2,
        "messages": 6,
        "scope": ("memory-1", None, "t1"),
    }
    # 900 tokens plus the answer passes 80% of the window.
    agent._schedule_background_compaction("x" * 400)
    assert started.wait(5)
    assert calls[0]["thread_id"] == "t1" and calls[0]["keep_recent"] == 2
    assert calls[0]["max_memories_per_summary"] == 6

    # The next turn of that thread waits for it before loading history.
    agent._current_thread_id = "t1"
    joined = threading.Thread(
        target=contextvars.copy_context().run,
        args=(agent._join_background_compaction, "memory-1", None),
    )
    joined.start()
    joined.join(0.2)
    assert joined.is_alive()
    release.set()
    joined.join(5)
    assert not joined.is_alive()
    assert agent.wait_for_memory_upkeep(timeout=5)
    agent.close()


def test_a_turn_well_under_the_limit_schedules_nothing(work_dir, monkeypatch):
    agent = _agent(work_dir)
    monkeypatch.setattr(
        agent, "generate_summaries", lambda **kw: pytest.fail("compacted")
    )
    agent._compaction_outlook = {
        "estimate": 100,
        "window": 1000,
        "threshold": 80,
        "keep": 2,
        "messages": 6,
        "scope": ("memory-1", None, "t1"),
    }
    agent._schedule_background_compaction("short answer")
    assert agent._upkeep_worker is None
    assert "memory_upkeep" not in agent.close()["closed"]

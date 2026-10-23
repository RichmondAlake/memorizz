"""One agent serving many users (a server's shared agent): no user's memory,
history or facts reach another, under concurrency and on reused threads."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.long_term.semantic.entity_memory import entity_memory as entity_module
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-tenants-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding", lambda text, **k: list(VECTOR)
    )
    monkeypatch.setattr(entity_module, "get_embedding", lambda text, **k: list(VECTOR))
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")


class _Embedder:
    def get_embedding(self, text: str) -> List[float]:
        return list(VECTOR)


def _provider(work_dir: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=work_dir / "memory",
            embedding_provider=_Embedder(),
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )


class _PerUserModel:
    """Saves each user's secret through the tool, then answers; records
    every prompt it was shown, per user, so a test can look for leaks."""

    model = "stub"

    def __init__(self):
        self.seen: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
        self.barrier = None

    def generate(self, messages, tools=None, **kwargs):
        text = json.dumps(messages, default=str)
        user = "alice" if "ALICE-SECRET" in text.split("[latest]")[-1] else "bob"
        with self.lock:
            self.seen.setdefault(user, []).append(text)
        if self.barrier is not None:
            try:
                self.barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass
        if not any(m.get("role") == "tool" for m in messages if isinstance(m, dict)):
            secret = "ALICE-SECRET" if user == "alice" else "BOB-SECRET"
            call = SimpleNamespace(
                id="c1",
                function=SimpleNamespace(
                    name="entity_memory_upsert",
                    arguments=json.dumps(
                        {
                            "name": "user",
                            "attributes": [{"name": "secret", "value": secret}],
                        }
                    ),
                ),
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=[call])
                    )
                ]
            )
        return "Noted."

    def get_config(self):
        return {"provider": "openai", "model": "stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


def _agent(work_dir, model) -> MemAgent:
    return MemAgent(
        model=model,
        memory_provider=_provider(work_dir),
        auto_register=False,
        agent_id="shared",
        application_mode="assistant",
        multi_tenant=True,
        entity_policy={"self_identity_key": "authenticated_user", "subjects": "self"},
    ).with_entity_memory(True, capture=False)


def _run(agent, user):
    secret = "ALICE-SECRET" if user == "alice" else "BOB-SECRET"
    return agent.run(
        f"[latest] My code word is {secret}.",
        memory_id=f"primary_{user}",
        thread_id=f"t-{user}",
        user_id=user,
    )


def _facts(agent, user) -> Dict[str, str]:
    return {
        a["name"]: a["value"]
        for row in EntityMemory(agent.memory_provider).list_entities(
            memory_id=f"primary_{user}", user_id=user
        )
        for a in row.get("attributes") or []
    }


def test_concurrent_users_on_one_agent_never_see_each_other(work_dir):
    model = _PerUserModel()
    model.barrier = threading.Barrier(2)
    agent = _agent(work_dir, model)
    errors = []

    def turn(user):
        try:
            _run(agent, user)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=turn, args=(u,)) for u in ("alice", "bob")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert errors == []
    assert _facts(agent, "alice") == {"secret": "ALICE-SECRET"}
    assert _facts(agent, "bob") == {"secret": "BOB-SECRET"}
    assert not any("BOB-SECRET" in text for text in model.seen["alice"])
    assert not any("ALICE-SECRET" in text for text in model.seen["bob"])
    # No user's memory is registered on the shared agent.
    assert agent.memory_ids == []
    agent.close()


def test_a_reused_server_thread_carries_nothing_to_the_next_user(work_dir):
    model = _PerUserModel()
    agent = _agent(work_dir, model)
    _run(agent, "alice")
    _run(agent, "bob")  # same thread, as a pooled server worker would
    assert not any("ALICE-SECRET" in text for text in model.seen["bob"])
    assert _facts(agent, "bob") == {"secret": "BOB-SECRET"}
    # Host calls on the same thread must name their memory.
    with pytest.raises(ValueError, match="pass memory_id"):
        agent.capture_entity_facts("I teach biology to teenagers.", user_id="bob")
    with pytest.raises(ValueError, match="pass memory_id"):
        agent.build_personalization_context("biology", user_id="bob")
    # A turn without its memory fails before any model call.
    calls = sum(len(texts) for texts in model.seen.values())
    agent.run("hello", user_id="bob")
    assert sum(len(texts) for texts in model.seen.values()) == calls
    agent.close()


def test_a_shared_agent_refuses_to_collect_or_wipe_users_memories(work_dir):
    agent = _agent(work_dir, _PerUserModel())
    with pytest.raises(ValueError):
        agent.update_memory(["primary_alice"])
    with pytest.raises(ValueError):
        agent.delete_memory()
    assert agent.describe_memory()["multi_tenant"] is True
    agent.close()


def test_multi_tenant_is_saved_with_the_agent(work_dir):
    provider = _provider(work_dir)
    agent = MemAgent(
        model=_PerUserModel(),
        memory_provider=provider,
        auto_register=False,
        agent_id="shared",
        application_mode="assistant",
        multi_tenant=True,
    )
    agent.save()
    loaded = MemAgent.load("shared", memory_provider=provider, model=_PerUserModel())
    assert loaded.multi_tenant is True
    agent.close()
    loaded.close()


def test_a_single_tenant_agent_keeps_its_fallback(work_dir):
    agent = MemAgent(
        model=_PerUserModel(),
        memory_provider=_provider(work_dir),
        memory_ids=["mine"],
        auto_register=False,
        agent_id="solo",
        application_mode="assistant",
    )
    assert agent._default_memory_id() == "mine"
    assert MemAgent._host_memory_id(agent, None) == "mine"
    agent.close()

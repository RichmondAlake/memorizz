"""Memory upkeep under a production workload: off the user's turn, concurrent
across users, skipped for messages with nothing to file, switchable per turn;
built-in memory tools carry honest governance; workflows embed when stored
and keep their conversation; facts can be forgotten by their source."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from memorizz import ContextPolicy
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.long_term.semantic.entity_memory import entity_memory as entity_module
from memorizz.memagent import MemAgent
from memorizz.memagent.memory_routing import worth_capturing
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-upkeep-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def embeds(monkeypatch):
    calls: List[str] = []

    def embed(text, **kwargs):
        calls.append(str(text))
        return list(VECTOR)

    monkeypatch.setattr("memorizz.embeddings.get_embedding", embed)
    monkeypatch.setattr(entity_module, "get_embedding", embed)
    monkeypatch.setattr(
        "memorizz.long_term.procedural.workflow.workflow.get_embedding", embed
    )
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding_dimensions", lambda model=None: len(VECTOR)
    )
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")
    return calls


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


class _Model:
    model = "stub"

    def __init__(self, extract: Optional[Callable[[str], List[Dict]]] = None):
        self.extract = extract or (lambda prompt: [])
        self.prompts: List[str] = []
        self.replies: List[Any] = []

    def generate(self, messages, tools=None, **kwargs):
        return self.replies.pop(0) if self.replies else "ok"

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        self.prompts.append(prompt)
        return json.dumps({"facts": self.extract(prompt)})

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "openai", "model": "stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


def _role_fact(prompt: str) -> List[Dict[str, Any]]:
    if "I teach biology" not in prompt:
        return []
    return [
        {
            "entity": "user",
            "entity_type": "person",
            "attribute": "role",
            "value": "biology teacher",
            "evidence": "I teach biology",
        }
    ]


def _agent(work_dir, chat=None, background=None, **kwargs) -> MemAgent:
    agent = MemAgent(
        model=chat or _Model(),
        memory_provider=kwargs.pop("memory_provider", None) or _provider(work_dir),
        memory_ids=["memory-1"],
        auto_register=False,
        agent_id="agent-1",
        application_mode="assistant",
        entity_policy={"self_identity_key": "authenticated_user", "subjects": "self"},
        **kwargs,
    ).with_entity_memory(True)
    if background is not None:
        agent.with_background_model(background)
    return agent


def _facts(provider, user_id) -> Dict[str, str]:
    return {
        attribute["name"]: attribute["value"]
        for record in EntityMemory(provider).list_entities(
            memory_id="memory-1", user_id=user_id
        )
        for attribute in record.get("attributes") or []
    }


# -- off the user's turn ------------------------------------------------------------


class _Gated(_Model):
    """A background model that waits until the test lets it answer."""

    def __init__(self, extract=None, gate: Optional[threading.Event] = None):
        super().__init__(extract)
        self.gate = gate or threading.Event()
        self.threads: List[str] = []
        self.started: List[float] = []

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        self.threads.append(threading.current_thread().name)
        self.started.append(time.monotonic())
        assert self.gate.wait(10), "the test never released the background model"
        return super().generate_text(prompt, instructions)


def test_capture_runs_after_the_turn_returns_on_an_upkeep_thread(work_dir):
    background = _Gated(_role_fact)
    agent = _agent(work_dir, background=background)
    started = time.monotonic()
    reply = agent.run(
        "I teach biology to teenagers.",
        memory_id="memory-1",
        thread_id="t1",
        user_id="u1",
    )
    returned = time.monotonic() - started
    # The reply came back while capture was still blocked in the background.
    assert reply == "ok" and returned < 5
    assert _facts(agent.memory_provider, "u1") == {}
    background.gate.set()
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _facts(agent.memory_provider, "u1") == {"role": "biology teacher"}
    assert background.threads and all(
        name.startswith("memorizz-memory-upkeep") for name in background.threads
    )
    assert threading.current_thread().name not in background.threads
    agent.close()


def test_users_captures_run_side_by_side(work_dir, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UPKEEP_WORKERS", "4")
    gate = threading.Event()
    background = _Gated(_role_fact, gate)
    agent = _agent(work_dir, background=background)
    for user in ("u1", "u2", "u3"):
        agent.run(
            "I teach biology to teenagers.",
            memory_id="memory-1",
            thread_id=f"t-{user}",
            user_id=user,
        )
    deadline = time.monotonic() + 5
    while len(background.started) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    # All three users' captures are in the model at once, not queued.
    assert len(background.started) == 3
    gate.set()
    assert agent.wait_for_memory_upkeep(timeout=10)
    for user in ("u1", "u2", "u3"):
        assert _facts(agent.memory_provider, user) == {"role": "biology teacher"}
    agent.close()


def test_one_worker_still_works_when_configured(work_dir, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UPKEEP_WORKERS", "1")
    background = _Model(_role_fact)
    agent = _agent(work_dir, background=background)
    for user in ("u1", "u2"):
        agent.run(
            "I teach biology to teenagers.",
            memory_id="memory-1",
            thread_id=f"t-{user}",
            user_id=user,
        )
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert agent._upkeep_worker._max_workers == 1
    assert _facts(agent.memory_provider, "u2") == {"role": "biology teacher"}
    agent.close()


# -- what is worth a background call ---------------------------------------------


@pytest.mark.parametrize(
    "text, worth",
    [
        ("thanks", False),
        ("Ok thanks!", False),
        ("Thank you so much!!", False),
        ("continue", False),
        ("Biology", False),
        ("https://example.com/paper.pdf", False),
        ("I'm Ada", True),
        ("keep it short", True),
        ("call me Rich", True),
        ("I teach biology to teenagers.", True),
    ],
)
def test_worth_capturing(text, worth):
    assert worth_capturing(text) is worth


def test_an_acknowledgement_costs_no_background_call(work_dir):
    background = _Model(_role_fact)
    agent = _agent(work_dir, background=background)
    agent.run("thanks!", memory_id="memory-1", thread_id="t1", user_id="u1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert background.prompts == []
    agent.close()


def test_a_turn_the_user_did_not_write_is_not_captured(work_dir):
    background = _Model(_role_fact)
    agent = _agent(work_dir, background=background)
    agent.run(
        "I teach biology to teenagers.",
        memory_id="memory-1",
        thread_id="t1",
        user_id="u1",
        memory_capture=False,
    )
    list(
        agent.run_stream(
            "I teach biology to teenagers.",
            memory_id="memory-1",
            thread_id="t1",
            user_id="u1",
            memory_capture=False,
        )
    )
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert background.prompts == []
    # The switch lasts one call only.
    agent.run(
        "I teach biology to teenagers.",
        memory_id="memory-1",
        thread_id="t1",
        user_id="u1",
    )
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _facts(agent.memory_provider, "u1") == {"role": "biology teacher"}
    agent.close()


# -- governed built-in memory tools ------------------------------------------------


def test_built_in_memory_tools_say_what_they_do(work_dir):
    agent = _agent(work_dir)
    policy = agent.tool_manager.get_tool_policy
    for name in ("entity_memory_upsert", "summarize_conversation"):
        assert policy(name)["side_effects"] is True, name
        assert policy(name)["requires_approval"] is False, name
    for name in (
        "entity_memory_lookup",
        "retrieve_tool_log_entry",
        "expand_summary",
        "context_window_stats_tool",
        "list_recent_tool_logs",
    ):
        assert policy(name)["deterministic"] is False, name
        assert policy(name)["side_effects"] is False, name
    agent.close()


# -- host tools that every turn offers ----------------------------------------------


def test_always_visible_tools_reach_the_router(work_dir):
    def update_task_progress(task_id: str) -> str:
        """Mark a task done."""
        return task_id

    agent = _agent(
        work_dir,
        context_policy=ContextPolicy(always_visible_tools=("update_task_progress",)),
        tools=[update_task_progress],
    )
    assert "update_task_progress" in agent.semantic_tool_router.always_visible
    assert agent.context_policy.to_dict()["always_visible_tools"] == [
        "update_task_progress"
    ]
    assert "always_visible_tools" not in ContextPolicy().to_dict()
    agent.close()


# -- workflows ------------------------------------------------------------------------


def _tool_turn(name: str, arguments: Dict[str, Any]) -> Any:
    from types import SimpleNamespace

    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))
        ]
    )


def test_a_turn_without_tools_makes_no_workflow_embedding(work_dir, embeds):
    from memorizz.enums import MemoryType

    agent = _agent(work_dir, memory_types=[MemoryType.WORKFLOW_MEMORY])
    agent.run(
        "What is photosynthesis?", memory_id="memory-1", thread_id="t1", user_id="u1"
    )
    assert not [text for text in embeds if text.startswith("Tool Execution for Query")]
    agent.close()


def test_a_stored_workflow_keeps_its_conversation_and_embeds_its_steps(
    work_dir, embeds
):
    from memorizz.enums import MemoryType

    def lookup_word(word: str) -> str:
        """Look a word up."""
        return f"{word}: a meaning"

    chat = _Model()
    chat.replies = [_tool_turn("lookup_word", {"word": "osmosis"}), "Done."]
    agent = _agent(
        work_dir,
        chat,
        memory_types=[MemoryType.WORKFLOW_MEMORY],
        tools=[lookup_word],
        context_policy=ContextPolicy(progressive_tool_disclosure=False),
    )
    agent.run("Define osmosis", memory_id="memory-1", thread_id="t9", user_id="u1")
    rows = agent.memory_provider.list_all(MemoryType.WORKFLOW_MEMORY, user_id="u1")
    assert rows, "the workflow was not stored"
    assert rows[0]["origin_thread_id"] == "t9"
    assert rows[0]["embedding"] == VECTOR
    workflow_texts = [
        text for text in embeds if text.startswith("Tool Execution for Query")
    ]
    assert len(workflow_texts) == 1 and "lookup_word" in workflow_texts[0]
    agent.close()


# -- forgetting by source -----------------------------------------------------------


def test_forget_sources_takes_out_what_one_comment_taught(work_dir):
    entities = EntityMemory(_provider(work_dir))
    scope = dict(memory_id="memory-1", user_id="u1")
    entities.upsert_entity(
        name="user",
        identity_key="authenticated_user",
        attributes=[
            {
                "name": "tone",
                "value": "formal",
                "source": "entity_capture",
                "thread_id": "t1",
            }
        ],
        **scope,
    )
    entities.upsert_entity(
        identity_key="authenticated_user",
        attributes=[
            {
                "name": "tone",
                "value": "casual",
                "source": "doc_comment:c1",
                "thread_id": "doc:7",
            },
            {
                "name": "brand_color",
                "value": "#0E4F2F",
                "source": "doc_comment:c1",
                "thread_id": "doc:7",
            },
            {
                "name": "audience",
                "value": "teens",
                "source": "doc_comment:c2",
                "thread_id": "doc:7",
            },
        ],
        **scope,
    )
    counts = entities.forget_sources(["doc_comment:c1"], **scope)
    assert counts["attributes_restored"] == 1 and counts["attributes_removed"] == 1
    (record,) = entities.list_entities(**scope)
    values = {a["name"]: a["value"] for a in record["attributes"]}
    assert values == {"tone": "formal", "audience": "teens"}
    with pytest.raises(ValueError):
        entities.forget_sources([" "], **scope)


def test_tool_written_relations_carry_their_conversation(work_dir):
    chat = _Model()
    chat.replies = [
        _tool_turn(
            "entity_memory_upsert",
            {
                "name": "user",
                "attributes": [{"name": "role", "value": "teacher"}],
                "relations": [{"entity_id": "project-1", "relation_type": "works_on"}],
            },
        ),
        "Noted.",
    ]
    agent = _agent(work_dir, chat)
    agent.with_entity_memory(True, capture=False)
    agent.run(
        "I teach and I'm on project one.",
        memory_id="memory-1",
        thread_id="t5",
        user_id="u1",
    )
    entities = EntityMemory(agent.memory_provider)
    (record,) = entities.list_entities(memory_id="memory-1", user_id="u1")
    assert record["relations"][0]["metadata"]["thread_id"] == "t5"
    entities.forget_thread("t5", memory_id="memory-1", user_id="u1")
    assert entities.list_entities(memory_id="memory-1", user_id="u1") == []
    agent.close()


def test_a_capture_off_turn_refuses_the_models_own_memory_writes(work_dir):
    """End-to-end: another agent's message made the chat model save
    "lives in Lisbon" although capture was off."""
    chat = _Model()
    chat.replies = [
        _tool_turn(
            "entity_memory_upsert",
            {"name": "user", "attributes": [{"name": "city", "value": "Lisbon"}]},
        ),
        "Noted.",
    ]
    agent = _agent(work_dir, chat)
    agent.with_entity_memory(True, capture=False)
    agent.run(
        "The user lives in Lisbon.",
        memory_id="memory-1",
        thread_id="t7",
        user_id="u1",
        memory_capture=False,
    )
    assert _facts(agent.memory_provider, "u1") == {}
    agent.close()

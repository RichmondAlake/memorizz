"""A chat that is told of, and can recall, the person's other conversations."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.memagent import MemAgent
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.retrieval import RetrievalPolicy
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.routers import playground


class _Model:
    """Answers with a fixed line and keeps what it was sent."""

    model = "stub"

    def __init__(self):
        self.sent: List[str] = []

    def generate(self, messages, tools=None, **kwargs):
        self.sent.append("\n".join(str(m.get("content") or "") for m in messages))
        return "Noted."

    def get_config(self):
        return {"provider": "openai", "model": "stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield store
    store.close()


def _agent(provider, model, **options: Any) -> MemAgent:
    return MemAgent(
        model=model,
        memory_provider=provider,
        auto_register=False,
        streaming=False,
        **options,
    )


@pytest.mark.unit
def test_a_chat_is_told_of_the_other_conversations_it_may_read(provider):
    model = _Model()
    agent = _agent(provider, model)
    agent.run("Plan my trip to Lisbon in May.", memory_id="chat-a")
    agent.run("What should I pack for it?", memory_id="chat-a")
    agent.run("Draft a note to my landlord.", memory_id="chat-b")

    agent.run(
        "What have we been talking about?",
        memory_id="chat-c",
        retrieval_policy=RetrievalPolicy(also_recall_from=["chat-a", "chat-b"]),
    )

    told = model.sent[-1]
    assert "Your other recent conversations with this person" in told
    assert 'Opened with: "Plan my trip to Lisbon in May."' in told
    assert 'Last asked: "What should I pack for it?"' in told
    assert 'Opened with: "Draft a note to my landlord."' in told
    # Nothing was written into the conversations that were read.
    read = provider.retrieve_conversation_history_ordered_by_timestamp
    assert len([r for r in read(memory_id="chat-a") if r.get("role") == "user"]) == 2


@pytest.mark.unit
def test_without_the_policy_or_when_scoped_a_chat_stands_alone(provider):
    model = _Model()
    agent = _agent(provider, model)
    agent.run("Plan my trip to Lisbon in May.", memory_id="chat-a")

    agent.run("What have we been talking about?", memory_id="chat-c")
    assert "Lisbon" not in model.sent[-1]

    for policy in (
        RetrievalPolicy(conversation_scope="thread", also_recall_from=["chat-a"]),
        RetrievalPolicy.sealed(also_recall_from=["chat-a"]),
    ):
        agent.run("And now?", memory_id="chat-d", retrieval_policy=policy)
        assert "Lisbon" not in model.sent[-1]

    # The agent's own saved policy is never changed by a run's.
    assert agent.retrieval_policy.also_recall_from == ()


@pytest.mark.unit
def test_the_playground_reads_an_agents_other_conversations_unless_told_not_to(
    provider,
):
    model = _Model()
    agent = _agent(provider, model)
    agent.memory_ids = ["chat-a", "chat-b", "chat-c"]

    policy = playground._playground_recall_policy(agent, "chat-c")
    assert policy.also_recall_from == ("chat-b", "chat-a")
    assert playground._playground_recall_policy(agent, None).also_recall_from == (
        "chat-c",
        "chat-b",
        "chat-a",
    )

    agent.retrieval_policy = RetrievalPolicy(conversation_scope="thread")
    assert playground._playground_recall_policy(agent, "chat-c") is None
    agent.retrieval_policy = RetrievalPolicy()
    agent.memory_ids = ["chat-c"]
    assert playground._playground_recall_policy(agent, "chat-c") is None


@pytest.mark.unit
def test_the_settings_switch_keeps_conversations_apart_or_lets_them_meet(provider):
    provider.store_memagent(MemAgentModel(agent_id="agent-1", name="Vera"))
    url = "/agents/agent-1/playground"
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {},
            "read_only": False,
        },
    ):
        client = TestClient(create_app(), follow_redirects=False)
        assert 'name="recall_across" type="checkbox" value="1" checked' in (
            client.get(url).text
        )

        client.post(url + "/config", data={"recall_across_present": "1"})
        stored = provider.retrieve_memagent("agent-1")
        assert stored.retrieval_policy == {"conversation_scope": "thread"}
        assert playground._recall_scope(stored) == "separate"

        client.post(
            url + "/config", data={"recall_across_present": "1", "recall_across": "1"}
        )
        stored = provider.retrieve_memagent("agent-1")
        assert playground._recall_scope(stored) == "across"

        # A save that does not carry the switch leaves the choice alone, and
        # a sealed agent is not unsealed from here.
        provider.store_memagent(
            MemAgentModel(
                agent_id="agent-2",
                name="Sealed",
                retrieval_policy={
                    "conversation_scope": "thread",
                    "entity_scope": "thread",
                },
            )
        )
        client.post(
            "/agents/agent-2/playground/config",
            data={"recall_across_present": "1", "recall_across": "1"},
        )
        assert playground._recall_scope(provider.retrieve_memagent("agent-2")) == (
            "sealed"
        )

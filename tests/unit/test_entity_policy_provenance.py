"""One record for the user (EntityPolicy), the conversation each fact came
from, pinned facts, forgetting a conversation, and host capture of facts the
user wrote outside the chat."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest

from memorizz import EntityPolicy, capabilities
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.long_term.semantic.entity_memory import entity_memory as entity_module
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]
SELF = "authenticated_user"


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-entity-policy-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def embeds(monkeypatch):
    """Offline embeddings that count their calls."""
    calls: List[str] = []

    def embed(text):
        calls.append(text)
        return list(VECTOR)

    monkeypatch.setattr("memorizz.embeddings.get_embedding", embed)
    monkeypatch.setattr(entity_module, "get_embedding", embed)
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
    """A model whose fact extraction is scripted per prompt."""

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


def _agent(work_dir: Path, chat=None, **kwargs) -> MemAgent:
    return MemAgent(
        model=chat or _Model(),
        memory_provider=kwargs.pop("memory_provider", None) or _provider(work_dir),
        memory_ids=["memory-1"],
        auto_register=False,
        agent_id="agent-1",
        application_mode="assistant",
        **kwargs,
    )


def _records(provider, user_id="u1") -> List[Dict[str, Any]]:
    return EntityMemory(provider).list_entities(memory_id="memory-1", user_id=user_id)


def _attribute(record: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(a for a in record["attributes"] if a["name"] == name)


def _write(entities, value, *, thread=None, source="chat", pinned=False, name="role"):
    attribute = {"name": name, "value": value, "source": source}
    if thread:
        attribute["thread_id"] = thread
    if pinned:
        attribute["pinned"] = True
    return entities.upsert_entity_with_report(
        name="Rich",
        entity_type="person",
        identity_key=SELF,
        attributes=[attribute],
        memory_id="memory-1",
        user_id="u1",
    )


# -- history, provenance and pinning ------------------------------------------


def test_a_replaced_value_moves_to_history_with_its_conversation(work_dir):
    entities = EntityMemory(_provider(work_dir))
    _write(entities, "engineer", thread="t1")
    _write(entities, "developer advocate", thread="t2")

    (record,) = _records(entities.memory_provider)
    role = _attribute(record, "role")
    assert role["value"] == "developer advocate"
    assert role["thread_id"] == "t2"
    assert [(h["value"], h["thread_id"]) for h in role["history"]] == [
        ("engineer", "t1")
    ]


def test_the_same_fact_from_the_same_place_is_a_no_op(work_dir, embeds):
    entities = EntityMemory(_provider(work_dir))
    _write(entities, "engineer", thread="t1")
    before = len(embeds)
    report = _write(entities, "Engineer", thread="t1")
    assert report["unchanged"] is True
    assert len(embeds) == before


def test_a_provenance_only_change_keeps_the_stored_vector(work_dir, embeds):
    entities = EntityMemory(_provider(work_dir))
    _write(entities, "engineer", thread="t1")
    before = len(embeds)
    report = _write(entities, "engineer", thread="t2")
    assert report["unchanged"] is False
    assert report["embedding"]["status"] == "full"
    assert len(embeds) == before  # same text, no embedding call
    _write(entities, "designer", thread="t3")
    assert len(embeds) == before + 1


def test_records_written_before_the_new_fields_keep_their_shape(work_dir):
    entities = EntityMemory(_provider(work_dir))
    entities.upsert_entity(
        name="Ines",
        attributes=[{"name": "city", "value": "Porto"}],
        memory_id="memory-1",
        user_id="u1",
    )
    (record,) = _records(entities.memory_provider)
    assert set(_attribute(record, "city")) == {
        "name",
        "value",
        "confidence",
        "source",
        "created_at",
        "updated_at",
    }


def test_inferred_writes_never_replace_a_pinned_value(work_dir):
    entities = EntityMemory(_provider(work_dir))
    _write(entities, "founder", source="user", pinned=True)
    report = _write(entities, "intern", thread="t9")
    assert report["kept_pinned"] == ["role"]
    (record,) = _records(entities.memory_provider)
    assert _attribute(record, "role")["value"] == "founder"

    _write(entities, "CEO", source="user", pinned=True)
    (record,) = _records(entities.memory_provider)
    role = _attribute(record, "role")
    assert (role["value"], role["pinned"]) == ("CEO", True)
    assert role["history"][0]["value"] == "founder"


def test_retiring_skips_pinned_facts_unless_the_user_asks(work_dir):
    entities = EntityMemory(_provider(work_dir))
    entity_id = _write(entities, "founder", source="user", pinned=True)["entity_id"]
    args = dict(
        memory_id="memory-1",
        user_id="u1",
        retirements=[{"name": "role", "reason": "older"}],
        retired_by="consolidation:p1",
    )
    assert entities.retire_attributes(entity_id, **args) == 0
    assert entities.retire_attributes(entity_id, **args, include_pinned=True) == 1


# -- forgetting a conversation ------------------------------------------------


def test_forgetting_a_conversation_restores_what_another_one_said(work_dir):
    entities = EntityMemory(_provider(work_dir))
    _write(entities, "engineer", thread="t1")
    _write(entities, "developer advocate", thread="t2")
    _write(entities, "London", thread="t2", name="city")

    counts = entities.forget_thread("t2", memory_id="memory-1", user_id="u1")
    assert counts == {
        "entities_changed": 1,
        "entities_deleted": 0,
        "attributes_removed": 1,
        "attributes_restored": 1,
    }
    (record,) = _records(entities.memory_provider)
    role = _attribute(record, "role")
    assert (role["value"], role["thread_id"]) == ("engineer", "t1")
    assert "history" not in role
    assert [a["name"] for a in record["attributes"]] == ["role"]


def test_a_fact_two_conversations_taught_survives_forgetting_one(work_dir):
    entities = EntityMemory(_provider(work_dir))
    _write(entities, "engineer", thread="t1")
    _write(entities, "engineer", thread="t2")
    entities.forget_thread("t2", memory_id="memory-1", user_id="u1")
    (record,) = _records(entities.memory_provider)
    assert _attribute(record, "role")["thread_id"] == "t1"


def test_an_entity_only_one_conversation_taught_is_deleted(work_dir):
    entities = EntityMemory(_provider(work_dir))
    _write(entities, "engineer", thread="t1")
    counts = entities.forget_thread("t1", memory_id="memory-1", user_id="u1")
    assert counts["entities_deleted"] == 1
    assert _records(entities.memory_provider) == []


def test_forgetting_keeps_the_users_own_facts_and_reads_legacy_sources(work_dir):
    entities = EntityMemory(_provider(work_dir))
    _write(entities, "founder", source="user", pinned=True)
    _write(entities, "Porto", source="assistant_thread:t7", name="city")
    counts = entities.forget_thread(
        "t7", memory_id="memory-1", user_id="u1", sources=["assistant_thread:t7"]
    )
    assert counts["attributes_removed"] == 1
    (record,) = _records(entities.memory_provider)
    assert [a["name"] for a in record["attributes"]] == ["role"]


def test_forget_thread_needs_a_thread(work_dir):
    with pytest.raises(ValueError):
        EntityMemory(_provider(work_dir)).forget_thread(
            " ", memory_id="memory-1", user_id="u1"
        )


# -- EntityPolicy ---------------------------------------------------------------


def test_policy_validation_and_presentation():
    assert EntityPolicy().active is False
    with pytest.raises(ValueError):
        EntityPolicy(subjects="self")
    with pytest.raises(ValueError):
        EntityPolicy(self_identity_key="k", subjects="everyone")
    policy = EntityPolicy.from_value(
        {"self_identity_key": "Authenticated User", "subjects": "self"}
    )
    assert policy.self_identity_key == "authenticated_user"
    me = {"name": "Rich", "metadata": {"identity_key": "authenticated_user"}}
    other = {"name": "Ines", "metadata": {}}
    assert policy.present([me, other]) == [{**me, "name": "user"}]
    assert EntityPolicy(self_identity_key="authenticated_user").present(
        [me, other]
    ) == [{**me, "name": "user"}, other]


def _self_and_ines(prompt: str) -> List[Dict[str, Any]]:
    facts = []
    if "I teach data science" in prompt:
        facts.append(
            {
                "entity": "user",
                "entity_type": "person",
                "attribute": "role",
                "value": "data science teacher",
                "evidence": "I teach data science",
            }
        )
    if "Ines lives in Porto" in prompt:
        facts.append(
            {
                "entity": "Ines",
                "entity_type": "person",
                "attribute": "home_city",
                "value": "Porto",
                "evidence": "Ines lives in Porto",
            }
        )
    return facts


MESSAGE = "I teach data science. My colleague Ines lives in Porto."


def _seed_self(provider) -> str:
    return EntityMemory(provider).upsert_entity(
        name="Rich",
        entity_type="person",
        identity_key=SELF,
        attributes=[{"name": "first_name", "value": "Rich"}],
        memory_id="memory-1",
        user_id="u1",
    )


def test_capture_files_the_users_facts_on_their_one_record(work_dir):
    provider = _provider(work_dir)
    self_id = _seed_self(provider)
    agent = _agent(
        work_dir,
        memory_provider=provider,
        entity_policy={"self_identity_key": SELF},
    ).with_entity_memory(True)
    agent.with_background_model(_Model(_self_and_ines))

    agent.run(MESSAGE, memory_id="memory-1", thread_id="t1", user_id="u1")
    assert agent.wait_for_memory_upkeep(timeout=10)

    records = {r["name"]: r for r in _records(provider)}
    assert set(records) == {"Rich", "Ines"}  # no second record named "user"
    assert records["Rich"]["entity_id"] == self_id
    role = _attribute(records["Rich"], "role")
    assert (role["value"], role["thread_id"]) == ("data science teacher", "t1")
    # The extractor read the user's record under the name it gives the user.
    assert "user.first_name = Rich" in agent.background_model.prompts[-1]
    agent.close()


def test_self_only_capture_drops_facts_about_anyone_else(work_dir):
    provider = _provider(work_dir)
    _seed_self(provider)
    agent = _agent(work_dir, memory_provider=provider).with_entity_memory(
        True, policy={"self_identity_key": SELF, "subjects": "self"}
    )
    agent.with_background_model(_Model(_self_and_ines))
    agent.run(MESSAGE, memory_id="memory-1", thread_id="t1", user_id="u1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert [r["name"] for r in _records(provider)] == ["Rich"]
    agent.close()


def test_capture_keeps_a_pinned_value(work_dir):
    provider = _provider(work_dir)
    EntityMemory(provider).upsert_entity(
        name="Rich",
        identity_key=SELF,
        attributes=[{"name": "role", "value": "founder", "pinned": True}],
        memory_id="memory-1",
        user_id="u1",
    )
    agent = _agent(work_dir, memory_provider=provider).with_entity_memory(
        True, policy={"self_identity_key": SELF, "subjects": "self"}
    )
    agent.with_background_model(_Model(_self_and_ines))
    receipt = agent.capture_entity_facts(MESSAGE, user_id="u1", memory_id="memory-1")
    assert receipt["stored"] is False
    (record,) = _records(provider)
    assert _attribute(record, "role")["value"] == "founder"
    agent.close()


def _tool_turn(name: str, arguments: Dict[str, Any]) -> Any:
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))
        ]
    )


@pytest.mark.parametrize("model_name", ["user", "Rich", "me"])
def test_tool_writes_about_the_user_land_on_their_record(work_dir, model_name):
    provider = _provider(work_dir)
    self_id = _seed_self(provider)
    chat = _Model()
    chat.replies = [
        _tool_turn(
            "entity_memory_upsert",
            {
                "entity_id": "made-up-id",
                "name": model_name,
                "entity_type": "person",
                "attributes": [{"name": "audience", "value": "AI engineers"}],
            },
        ),
        "Noted.",
    ]
    agent = _agent(
        work_dir,
        chat,
        memory_provider=provider,
        entity_policy={"self_identity_key": SELF},
    ).with_entity_memory(True, capture=False)
    agent.run(
        "I write for AI engineers.", memory_id="memory-1", thread_id="t4", user_id="u1"
    )

    (record,) = _records(provider)
    assert record["entity_id"] == self_id
    assert record["name"] == "Rich"
    audience = _attribute(record, "audience")
    assert (audience["value"], audience["thread_id"]) == ("AI engineers", "t4")
    agent.close()


def test_tool_writes_about_someone_else_keep_their_own_record(work_dir):
    provider = _provider(work_dir)
    _seed_self(provider)
    chat = _Model()
    chat.replies = [
        _tool_turn(
            "entity_memory_upsert",
            {"name": "Ines", "attributes": [{"name": "city", "value": "Porto"}]},
        ),
        "Noted.",
    ]
    agent = _agent(
        work_dir,
        chat,
        memory_provider=provider,
        entity_policy={"self_identity_key": SELF},
    ).with_entity_memory(True, capture=False)
    agent.run(
        "Ines lives in Porto.", memory_id="memory-1", thread_id="t5", user_id="u1"
    )
    assert sorted(r["name"] for r in _records(provider)) == ["Ines", "Rich"]
    agent.close()


# -- host capture -------------------------------------------------------------------


def _brand(prompt: str) -> List[Dict[str, Any]]:
    if "our brand green is #0E4F2F" not in prompt:
        return []
    return [
        {
            "entity": "user",
            "entity_type": "person",
            "attribute": "brand_primary_color",
            "value": "#0E4F2F green",
            "evidence": "our brand green is #0E4F2F",
        }
    ]


def test_host_capture_files_a_comment_with_its_source_and_thread(work_dir):
    provider = _provider(work_dir)
    _seed_self(provider)
    background = _Model(_brand)
    agent = _agent(
        work_dir,
        memory_provider=provider,
        entity_policy={"self_identity_key": SELF, "subjects": "self"},
    ).with_entity_memory(True)
    agent.with_background_model(background)

    receipt = agent.capture_entity_facts(
        "Make every chart match: our brand green is #0E4F2F",
        user_id="u1",
        memory_id="memory-1",
        thread_id="deck:42",
        source="slide_comment:c1",
        description="a comment the user left on their slide deck",
    )
    assert receipt["stored"] is True
    assert receipt["facts"][0]["attribute"] == "brand_primary_color"
    assert "a comment the user left on their slide deck" in background.prompts[-1]
    (record,) = _records(provider)
    color = _attribute(record, "brand_primary_color")
    assert (color["source"], color["thread_id"]) == ("slide_comment:c1", "deck:42")

    EntityMemory(provider).forget_thread("deck:42", memory_id="memory-1", user_id="u1")
    (record,) = _records(provider)
    assert [a["name"] for a in record["attributes"]] == ["first_name"]
    agent.close()


def test_host_capture_uses_the_chat_model_without_a_background_one(work_dir):
    provider = _provider(work_dir)
    chat = _Model(_brand)
    agent = _agent(
        work_dir,
        chat,
        memory_provider=provider,
        entity_policy={"self_identity_key": SELF},
    ).with_entity_memory(True)
    receipt = agent.capture_entity_facts(
        "our brand green is #0E4F2F", user_id="u1", memory_id="memory-1"
    )
    assert receipt["stored"] is True
    (record,) = _records(provider)
    assert record["name"] == "user"
    assert record["metadata"]["identity_key"] == SELF
    agent.close()


def test_host_capture_does_nothing_with_entity_memory_off(work_dir):
    agent = _agent(work_dir).with_entity_memory(False)
    receipt = agent.capture_entity_facts("our brand green is #0E4F2F", user_id="u1")
    assert receipt == {
        "stored": False,
        "memory_type": None,
        "memory_id": "memory-1",
        "source": "host",
        "reason": "Entity memory is off.",
    }
    agent.close()


# -- saved with the agent, and announced ----------------------------------------


def test_the_policy_is_saved_and_reloaded_with_the_agent(work_dir):
    provider = _provider(work_dir)
    agent = _agent(
        work_dir,
        memory_provider=provider,
        entity_policy={"self_identity_key": SELF, "subjects": "self"},
    )
    agent.save()
    loaded = MemAgent.load("agent-1", memory_provider=provider, model=_Model())
    assert loaded.entity_policy == EntityPolicy(self_identity_key=SELF, subjects="self")
    assert loaded.describe_memory()["entity_policy"] == {
        "self_identity_key": SELF,
        "subjects": "self",
    }
    agent.close()
    loaded.close()


def test_capabilities_announce_the_new_entity_features():
    features = capabilities()["features"]
    for name in (
        "background_model",
        "entity_capture",
        "canonical_entity_capture",
        "host_entity_capture",
        "entity_attribute_provenance",
        "pinned_entity_facts",
        "memory_routing",
        "recall_budget",
    ):
        assert features[name]["available"] is True, name


# -- self-only refusals -----------------------------------------------------------


def test_self_only_refuses_a_tool_write_about_someone_else(work_dir):
    """Live run: "my colleague Ben teaches physics" became the user's subject."""
    provider = _provider(work_dir)
    _seed_self(provider)
    chat = _Model()
    chat.replies = [
        _tool_turn(
            "entity_memory_upsert",
            {
                "name": "Ben",
                "entity_type": "person",
                "attributes": [{"name": "teaching_subject", "value": "Physics"}],
            },
        ),
        "Noted.",
    ]
    agent = _agent(
        work_dir,
        chat,
        memory_provider=provider,
        entity_policy={"self_identity_key": SELF, "subjects": "self"},
    ).with_entity_memory(True, capture=False)
    agent.run(
        "My colleague Ben teaches physics.",
        memory_id="memory-1",
        thread_id="t6",
        user_id="u1",
    )
    (record,) = _records(provider)
    assert [a["name"] for a in record["attributes"]] == ["first_name"]
    agent.close()


def test_a_name_the_user_goes_by_binds_to_their_record():
    policy = EntityPolicy(self_identity_key=SELF, subjects="self")
    record = {
        "name": "user",
        "metadata": {"identity_key": SELF},
        "attributes": [{"name": "full_name", "value": "Ada Lovelace"}],
    }
    for name in ("user", "me", "Ada", "ada lovelace"):
        assert policy.binds(name=name, self_record=record), name
    assert policy.refuses(name="Ben", self_record=record)
    assert not policy.refuses(
        name="Ben", metadata={"is_self": True}, self_record=record
    )
    assert not EntityPolicy(self_identity_key=SELF).refuses(
        name="Ben", self_record=record
    )


def test_the_prompt_says_entity_memory_is_about_the_user_only(work_dir):
    agent = _agent(
        work_dir, entity_policy={"self_identity_key": SELF, "subjects": "self"}
    ).with_entity_memory(True, capture=False)
    agent.run("hello", memory_id="memory-1", thread_id="t1", user_id="u1")
    agent.close()
    plain = _agent(work_dir).with_entity_memory(True, capture=False)
    assert "holds facts about the user only" not in plain._build_system_prompt()
    assert "holds facts about the user only" in agent._build_system_prompt()
    plain.close()


def test_forget_attributes_deletes_a_fact_with_its_history(work_dir):
    entities = EntityMemory(_provider(work_dir))
    entity_id = _write(entities, "engineer", thread="t1")["entity_id"]
    _write(entities, "designer", thread="t2")
    _write(entities, "London", thread="t2", name="city")
    _write(entities, "concise", thread="t3", name="tone")
    entities.retire_attributes(
        entity_id,
        memory_id="memory-1",
        user_id="u1",
        retirements=[{"name": "city"}],
        retired_by="consolidation:p1",
    )
    scope = dict(memory_id="memory-1", user_id="u1")
    assert entities.forget_attributes(entity_id, ["role"], **scope) == 1
    assert entities.forget_attributes(entity_id, ["city"], **scope) == 0
    (record,) = _records(entities.memory_provider)
    assert [a["name"] for a in record["attributes"]] == ["tone"]
    for gone in ("engineer", "designer", "London"):
        assert gone not in str(record)
    # The last fact takes the record with it.
    assert entities.forget_attributes(entity_id, ["tone"], **scope) == 1
    assert _records(entities.memory_provider) == []


def test_a_self_only_agent_reads_the_profile_by_key_not_by_search(
    work_dir, monkeypatch
):
    provider = _provider(work_dir)
    _seed_self(provider)
    searches = []
    original = EntityMemory.search_entities_with_diagnostics

    def spy(self, *args, **kwargs):
        searches.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EntityMemory, "search_entities_with_diagnostics", spy)

    class _Seeing(_Model):
        seen: List[str] = []

        def generate(self, messages, tools=None, **kwargs):
            self.seen.append(json.dumps(messages, default=str))
            return "ok"

    chat = _Seeing()
    agent = _agent(
        work_dir,
        chat,
        memory_provider=provider,
        entity_policy={"self_identity_key": SELF, "subjects": "self"},
    ).with_entity_memory(True, capture=False)
    agent.run(
        "What should I teach next?", memory_id="memory-1", thread_id="t1", user_id="u1"
    )
    agent.close()
    assert searches == []
    assert "first_name" in chat.seen[-1] and "Rich" in chat.seen[-1]


def test_get_by_identity_key_finds_older_records_too(work_dir):
    provider = _provider(work_dir)
    entities = EntityMemory(provider)
    # A canonical record whose id was not derived from its key.
    entities.memory_provider.store(
        data={
            "_id": "legacy-row",
            "entity_id": "legacy-id",
            "name": "user",
            "memory_id": "memory-1",
            "user_id": "u1",
            "metadata": {"identity_key": SELF},
            "attributes": [{"name": "role", "value": "teacher"}],
        },
        memory_store_type=entity_module.MemoryType.ENTITY_MEMORY,
    )
    found = entities.get_by_identity_key(SELF, memory_id="memory-1", user_id="u1")
    assert found["entity_id"] == "legacy-id"
    assert (
        entities.get_by_identity_key("nobody", memory_id="memory-1", user_id="u1")
        is None
    )

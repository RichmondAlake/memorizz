"""Public agent IDs must round-trip through Oracle's internal foreign keys."""

import uuid
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider.oracle import OracleProvider


class DelegateDatabase:
    def __init__(self):
        # Public IDs intentionally differ from the internal RAW primary keys.
        self.agents = {
            name: uuid.uuid4().bytes for name in ("team", "research", "review")
        }
        self.links = []
        self.rows = []
        self.commits = 0

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        # Every table identifier is schema-qualified; match on the bare name.
        sql = " ".join(sql.split()).replace("MEMORIZZ.", "")
        params = params or {}
        self.rows = []
        if sql.startswith("SELECT id FROM agents"):
            identity = self.agents.get(params["agent_id"])
            self.rows = [(identity,)] if identity else []
        elif sql.startswith("DELETE FROM agent_delegates"):
            self.links = [link for link in self.links if link[0] != params["agent_id"]]
        elif sql.startswith("INSERT INTO agent_delegates"):
            link = (params["agent_id"], params["delegate_agent_id"])
            assert link[0] in self.agents.values() and link[1] in self.agents.values()
            assert link not in self.links
            self.links.append(link)
        elif "FROM agent_delegates link" in sql:
            names = {raw: name for name, raw in self.agents.items()}
            self.rows = [
                (names[parent], names[delegate])
                if "SELECT parent.agent_id" in sql
                else (names[delegate],)
                for parent, delegate in self.links
                if not params or parent == params["agent_id"]
            ]
        elif sql.startswith("SELECT id, agent_id, name"):
            name = params["agent_id"]
            self.rows = [
                (
                    self.agents[name],
                    name,
                    name,
                    "Help.",
                    "assistant",
                    20,
                    "private",
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                )
            ]

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def commit(self):
        self.commits += 1


@pytest.fixture
def database():
    db = DelegateDatabase()
    provider = OracleProvider.__new__(OracleProvider)
    provider.config = SimpleNamespace(schema="MEMORIZZ", user="MEMORIZZ")
    provider._get_connection = lambda: nullcontext(db)
    provider._table_has_column = lambda *_args: True
    provider._embedding_provider = None
    provider.list_all = lambda _kind: [
        {"agent_id": name, "name": name} for name in db.agents
    ]
    return provider, db


def test_oracle_delegates_survive_save_reload_list_replace_and_detach(database):
    provider, db = database
    team = MemAgentModel(agent_id="team", delegates=["research", "review", "research"])
    provider.store_memagent(team)
    assert provider.retrieve_memagent("team").delegates == ["research", "review"]
    listed = {agent.agent_id: agent for agent in provider.list_memagents()}
    assert listed["team"].delegates == ["research", "review"]
    assert listed["research"].delegates is None
    team.delegates = ["review"]
    provider.store_memagent(team)
    assert provider.retrieve_memagent("team").delegates == ["review"]
    team.delegates = []
    provider.store_memagent(team)
    assert provider.retrieve_memagent("team").delegates is None
    assert db.links == []
    assert db.commits == 3


def test_oracle_does_not_silently_drop_an_unsaved_delegate(database):
    provider, db = database
    provider.store_memagent(MemAgentModel(agent_id="team", delegates=["research"]))
    with pytest.raises(ValueError, match="must be saved first"):
        provider.store_memagent(MemAgentModel(agent_id="team", delegates=["missing"]))
    assert provider.retrieve_memagent("team").delegates == ["research"]
    assert db.commits == 1


def test_oracle_loading_persona_does_not_generate_embeddings(database, monkeypatch):
    from memorizz.long_term.semantic.persona import persona as persona_module

    provider, db = database
    execute = db.execute

    def with_persona(sql, params=None):
        execute(sql, params)
        if "FROM MEMORIZZ.personas" in sql:
            db.rows = [
                (
                    "persona-one",
                    "Researcher",
                    "general",
                    "Saved background",
                    None,
                    None,
                    None,
                )
            ]

    db.execute = with_persona
    monkeypatch.setattr(
        persona_module,
        "get_embedding",
        lambda *_: pytest.fail("Read must not invoke embeddings"),
    )
    agent = provider.retrieve_memagent("team")
    assert agent.persona.background == "Saved background"


@pytest.mark.parametrize("hydrated", [False, True])
def test_oracle_agent_listing_accepts_stored_or_hydrated_persona(
    database, monkeypatch, hydrated
):
    from memorizz.long_term.semantic.persona import persona as persona_module

    provider, _db = database
    monkeypatch.setattr(
        persona_module,
        "get_embedding",
        lambda *_: pytest.fail("Listing must not invoke embeddings"),
    )
    persona = {"background": "Saved launch evidence", "goals": ["Keep provenance"]}
    if hydrated:
        persona = persona_module.Persona.from_dict(persona)
    provider.list_all = lambda _kind: [
        {"agent_id": "team", "name": "Coordinator", "persona": persona}
    ]
    listed = provider.list_memagents()
    assert len(listed) == 1
    assert listed[0].persona.background == "Saved launch evidence"
    assert listed[0].persona.goals == ["Keep provenance"]


def test_oracle_archive_agent_storage_skips_embedding_and_tool_sync():
    from types import SimpleNamespace

    from memorizz import MemoryType
    from memorizz.memory_provider.oracle.archive import store_record

    calls = []
    provider = SimpleNamespace(
        store_memagent=lambda model, **options: calls.append(options)
    )
    assert (
        store_record(provider, MemoryType.MEMAGENT, "agent", {"agent_id": "agent"})
        == "agent"
    )
    assert calls == [
        {"generate_embeddings": False, "sync_tools": False, "sync_persona": False}
    ]


def test_oracle_archive_agent_reuses_restored_persona():
    from types import SimpleNamespace

    from memorizz import MemoryType
    from memorizz.memory_provider.oracle.archive import store_record

    calls = []
    provider = SimpleNamespace(
        store_memagent=lambda model, **options: calls.append(options),
        retrieve_by_id=lambda identifier, kind: {"persona_id": identifier},
    )
    store_record(
        provider,
        MemoryType.MEMAGENT,
        "agent",
        {"agent_id": "agent", "persona": {"persona_id": "restored-persona"}},
    )
    assert calls[0]["sync_persona"] is False

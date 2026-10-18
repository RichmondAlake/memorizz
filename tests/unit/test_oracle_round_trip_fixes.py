"""Oracle round-trip regressions (fake connections, no database).

Agent records must reload with everything they were saved with (llm_config,
completion_policy, WhatsApp settings, tool removals, an emptied memory list)
and list the same way; knowledge-base and short-term records keep their
column-less fields; semantic-cache entries keep their admission time;
summaries bind epoch periods; cascades leave memory other agents use; every
TIMESTAMP is written and read as UTC; agent-wide stores show their shared
rows to every user; long knowledge-base scopes stay under Oracle's IN limit.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.procedural.toolbox.indexing import NOT_AGENT_TOOLS
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider.oracle.provider import (
    _AGENT_TOOL_TYPE_FILTER,
    _BY_ID_SPECS,
    _LIST_SPECS,
    OracleProvider,
)
from tests.unit.test_oracle_provider_fixes import _Cursor, _Lob, _provider, _row_for

# ---------------------------------------------------------------------------
# A small in-memory stand-in for the agent tables
# ---------------------------------------------------------------------------


class _AgentTables:
    """Answers the SQL ``store_memagent`` / ``retrieve_memagent`` /
    ``list_memagents`` issue, keeping rows per table."""

    def __init__(self):
        self.agents = {}  # agent_id -> row dict
        self.llm = {}  # agent RAW id -> row dict
        self.memories = []  # (agent RAW id, memory_id)
        self.toolbox = {}  # tool_id -> row dict
        self.statements = []
        self.rows = []
        self.rowcount = 0

    def cursor(self):
        return self

    def commit(self):
        pass

    def rollback(self):
        pass

    def setinputsizes(self, **_kwargs):
        pass

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def __iter__(self):
        return iter(self.fetchall())

    @staticmethod
    def _columns(sql):
        return [
            column.strip().split(".")[-1]
            for column in sql[len("SELECT ") : sql.index(" FROM ")].split(",")
        ]

    def execute(self, sql, params=None):
        sql = " ".join(str(sql).split()).replace("MEMORIZZ.", "")
        params = dict(params or {})
        self.statements.append((sql, params))
        self.rows, self.rowcount = [], 0
        raw_ids = {row["id"]: key for key, row in self.agents.items()}
        if sql.startswith("SELECT id FROM agents WHERE agent_id"):
            row = self.agents.get(params["agent_id"])
            self.rows = [(row["id"],)] if row else []
        elif sql.startswith("INSERT INTO agents"):
            self.agents[params["agent_id"]] = dict(params)
        elif sql.startswith("UPDATE agents"):
            self.agents[params["agent_id"]].update(params)
        elif sql.startswith("INSERT INTO agent_llm_configs"):
            if params["agent_id"] in self.llm:
                raise RuntimeError("ORA-00001: unique constraint violated")
            self.llm[params["agent_id"]] = dict(params)
        elif sql.startswith("UPDATE agent_llm_configs"):
            self.llm[params["agent_id"]] = dict(params)
        elif sql.startswith("DELETE FROM agent_memories"):
            self.memories = [m for m in self.memories if m[0] != params["agent_id"]]
        elif sql.startswith("INSERT INTO agent_memories"):
            self.memories.append((params["agent_id"], params["memory_id"]))
        elif sql.startswith("SELECT memory_id FROM agent_memories"):
            self.rows = [(m[1],) for m in self.memories if m[0] == params["agent_id"]]
        elif "FROM agent_memories link JOIN agents owner" in sql:
            self.rows = [
                (
                    sum(
                        1
                        for owner, memory_id in self.memories
                        if memory_id == params["memory_id"]
                        and raw_ids[owner] != params["agent_id"]
                    ),
                )
            ]
        elif sql.startswith("SELECT id FROM toolbox WHERE tool_id"):
            row = self.toolbox.get(params["tool_id"])
            self.rows = [(b"\x01" * 16,)] if row else []
        elif sql.startswith("INSERT INTO toolbox"):
            self.toolbox[params["tool_id"]] = dict(params)
        elif sql.startswith("UPDATE toolbox"):
            self.toolbox[params["tool_id"]].update(params)
        elif sql.startswith("DELETE FROM toolbox"):
            assert _AGENT_TOOL_TYPE_FILTER in sql
            kept = {v for k, v in params.items() if k.startswith("kept_tool")}
            for tool_id, row in list(self.toolbox.items()):
                if (
                    row.get("agent_id") == params["agent_id"]
                    and str(row.get("tool_type") or "").lower() not in NOT_AGENT_TOOLS
                    and tool_id not in kept
                ):
                    del self.toolbox[tool_id]
        elif sql.startswith("SELECT id, agent_id, name, instruction"):
            row = self.agents.get(params["agent_id"])
            if row:
                self.rows = [
                    (
                        row["id"],
                        row["agent_id"],
                        row.get("name"),
                        row.get("instruction"),
                        row.get("application_mode"),
                        row.get("max_steps"),
                        row.get("tool_access"),
                        row.get("semantic_cache"),
                        row.get("is_favorite", 0),
                        row.get("verbose", 0),
                        None,
                        None,
                        None,
                    )
                ]
        elif sql.startswith("SELECT provider, model"):
            row = self.llm.get(params["agent_id"])
            if row:
                self.rows = [tuple(row.get(c) for c in self._columns(sql))]
        elif sql.startswith("SELECT a.agent_id, c.provider"):
            columns = self._columns(sql)[1:]
            self.rows = [
                (key,) + tuple((self.llm.get(row["id"]) or {}).get(c) for c in columns)
                for key, row in self.agents.items()
            ]
        elif sql.startswith("SELECT tool_id, name") and "FROM toolbox" in sql:
            columns = self._columns(sql)
            self.rows = [
                tuple(row.get(c) for c in columns)
                for row in self.toolbox.values()
                if row.get("agent_id") == params["agent_id"]
                and str(row.get("tool_type") or "").lower() not in NOT_AGENT_TOOLS
            ]
        # Delegates, knowledge bases and personas: no rows.


@pytest.fixture
def agents():
    tables = _AgentTables()
    provider = OracleProvider.__new__(OracleProvider)
    provider.config = SimpleNamespace(schema="MEMORIZZ", user="MEMORIZZ")
    provider._get_connection = lambda: _Context(tables)
    provider._table_has_column = lambda *_args: True
    provider._memory_type_has_column = lambda *_args: True
    provider._table_columns_cache = {}
    provider._archive_types = set()
    provider._embedding_provider = None

    def list_all(kind, *_args, **_kwargs):
        assert kind == MemoryType.MEMAGENT
        return [provider.retrieve_memagent(key).model_dump() for key in tables.agents]

    provider.list_all = list_all
    return provider, tables


class _Context:
    def __init__(self, tables):
        self.tables = tables

    def __enter__(self):
        return self.tables

    def __exit__(self, *_args):
        return False


_LOCAL_OPENAI = {
    "provider": "openai",
    "model": "llama3.1",
    "base_url": "http://localhost:11434/v1",
    "reasoning_effort": "low",
    "api_mode": "responses",
    "prompt_cache_retention": "24h",
    "context_window_tokens": 32768,
    "additional_config": {"seed": 7},
}
_AZURE = {
    "provider": "azure",
    "deployment_name": "prod-gpt",
    "azure_endpoint": "https://example.openai.azure.com",
    "api_version": "2024-06-01",
}
_OLLAMA = {
    "provider": "ollama",
    "model": "qwen3",
    "host": "http://gpu:11434",
    "think": True,
}


@pytest.mark.unit
@pytest.mark.parametrize("llm_config", [_LOCAL_OPENAI, _AZURE, _OLLAMA])
def test_agent_llm_config_round_trips_every_key(agents, llm_config):
    provider, _tables = agents
    provider.store_memagent(MemAgentModel(agent_id="a", llm_config=llm_config))

    assert provider.retrieve_memagent("a").llm_config == llm_config
    assert provider.list_memagents()[0].llm_config == llm_config


@pytest.mark.unit
def test_agent_without_llm_config_reloads_without_an_invented_model(agents):
    provider, tables = agents
    provider.store_memagent(MemAgentModel(agent_id="a"))

    assert provider.retrieve_memagent("a").llm_config is None
    # The NOT NULL columns still get a value, but none a reload would use.
    (row,) = tables.llm.values()
    assert row["model"] != "gpt-4o-mini"


@pytest.mark.unit
def test_rows_saved_before_the_full_llm_config_still_load_from_columns(agents):
    provider, tables = agents
    provider.store_memagent(MemAgentModel(agent_id="a", name="A"))
    (row,) = tables.llm.values()
    row.update(provider="openai", model="gpt-4.1", temperature=0.2)
    row["additional_config"] = json.dumps({"meta_harness": True, "seed": 3})

    agent = provider.retrieve_memagent("a")

    assert agent.llm_config["model"] == "gpt-4.1"
    assert agent.llm_config["temperature"] == 0.2
    assert agent.llm_config["additional_config"] == {"seed": 3}
    assert agent.meta_harness is True


@pytest.mark.unit
def test_agent_completion_policy_and_whatsapp_settings_round_trip(agents):
    provider, _tables = agents
    agent = MemAgentModel(
        agent_id="a",
        completion_policy={"require_tool_evidence": True, "max_retries": 2},
        whatsapp_enabled=True,
        whatsapp_config={"to": "+15550100"},
        embedding_config={"provider": "openai", "dimensions": 256},
    )
    provider.store_memagent(agent)

    for loaded in (provider.retrieve_memagent("a"), provider.list_memagents()[0]):
        assert loaded.completion_policy == agent.completion_policy
        assert loaded.whatsapp_enabled is True
        assert loaded.whatsapp_config == {"to": "+15550100"}
        assert loaded.embedding_config == agent.embedding_config


@pytest.mark.unit
def test_list_memagents_matches_retrieve_memagent(agents):
    provider, _tables = agents
    agent = MemAgentModel(
        agent_id="a",
        name="Agent",
        llm_config=_LOCAL_OPENAI,
        mcp_servers=[{"name": "files", "command": "mcp-files"}],
        skill_paths=["skills/"],
        sandbox_provider="e2b",
        internet_access_provider="tavily",
        internet_access_config={"depth": 2},
        skills_marketplace_provider="hub",
        skills_marketplace_config={"index": "main"},
        completion_policy={"mode": "strict"},
        memory_ids=["m1"],
    )
    provider.store_memagent(agent)

    listed = provider.list_memagents()[0].model_dump()
    loaded = provider.retrieve_memagent("a").model_dump()
    assert listed == loaded
    for field in (
        "llm_config",
        "mcp_servers",
        "skill_paths",
        "sandbox_provider",
        "internet_access_provider",
        "internet_access_config",
        "skills_marketplace_provider",
        "skills_marketplace_config",
        "completion_policy",
    ):
        assert listed[field] == getattr(agent, field), field


def _tool(name):
    return {"name": name, "description": f"{name} tool", "signature": "(q)"}


@pytest.mark.unit
def test_removed_tools_stay_removed_and_mcp_rows_are_kept(agents):
    provider, tables = agents
    provider.store_memagent(MemAgentModel(agent_id="a", tools=[_tool("x"), _tool("y")]))
    for tool_type in NOT_AGENT_TOOLS:
        tables.toolbox[f"a:{tool_type}"] = {
            "tool_id": f"a:{tool_type}",
            "agent_id": "a",
            "tool_type": tool_type,
            "name": tool_type,
        }

    loaded = provider.retrieve_memagent("a")
    loaded.tools = [tool for tool in loaded.tools if tool["name"] == "x"]
    provider.store_memagent(loaded)

    assert [tool["name"] for tool in provider.retrieve_memagent("a").tools] == ["x"]
    assert {f"a:{tool_type}" for tool_type in NOT_AGENT_TOOLS} <= set(tables.toolbox)
    # The reloaded tool keeps its stored id (and row, so its embedding) rather
    # than gaining a second agent prefix.
    assert "a:x" in tables.toolbox and "a:a:x" not in tables.toolbox

    provider.store_memagent(MemAgentModel(agent_id="a", tools=[]))
    assert provider.retrieve_memagent("a").tools is None

    # No tool list in the save: nothing is removed.
    provider.store_memagent(MemAgentModel(agent_id="a", tools=[_tool("z")]))
    provider.store_memagent(MemAgentModel(agent_id="a"))
    assert [tool["name"] for tool in provider.retrieve_memagent("a").tools] == ["z"]


@pytest.mark.unit
def test_emptied_memory_ids_are_saved(agents):
    provider, tables = agents
    agent = MemAgentModel(agent_id="a", memory_ids=["m1", "m2"])
    provider.store_memagent(agent)
    agent.memory_ids = []
    provider.store_memagent(agent)

    assert tables.memories == []
    assert provider.retrieve_memagent("a").memory_ids == []


@pytest.mark.unit
def test_cascade_delete_keeps_memory_another_agent_lists(agents):
    provider, _tables = agents
    provider.store_memagent(MemAgentModel(agent_id="root", memory_ids=["shared"]))
    provider.store_memagent(
        MemAgentModel(agent_id="delegate", memory_ids=["shared", "own"])
    )
    deleted = []
    provider._delete_memory_units_by_memory_id = lambda memory_id, kind: deleted.append(
        (memory_id, kind)
    )
    provider.delete_by_id = lambda *_args: True

    provider.delete_memagent("delegate", cascade=True)

    assert {memory_id for memory_id, _kind in deleted} == {"own"}
    assert (("own", MemoryType.SEMANTIC_CACHE)) in deleted
    assert all(kind != MemoryType.MEMAGENT for _memory_id, kind in deleted)


@pytest.mark.unit
def test_semantic_cache_rows_are_deleted_by_memory_id():
    cursor = _Cursor()
    provider, _connection = _provider(cursor)

    provider._delete_memory_units_by_memory_id("m1", MemoryType.SEMANTIC_CACHE)

    assert cursor.calls == [
        (
            "DELETE FROM MEMORIZZ.semantic_cache WHERE memory_id = :memory_id",
            {"memory_id": "m1"},
        )
    ]


# ---------------------------------------------------------------------------
# Knowledge-base and short-term records keep their column-less fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_knowledge_base_store_keeps_top_level_metadata_and_reads_it_back():
    provider = OracleProvider.__new__(OracleProvider)
    provider._generate_embedding_if_needed = lambda *_args, **_kwargs: None
    provider._memory_type_has_column = lambda *_args: True
    captured = {}
    provider._insert_base_row = lambda *_args, **kwargs: captured.update(kwargs)

    provider._store_knowledge_base(
        {
            "content": "chunk",
            "memory_id": "m",
            "namespace": "m",
            "source_path": "/docs/a.md",
            "source_sha256": "abc",
            "source_name": "a.md",
            "timestamp": "2026-10-01T09:00:00+00:00",
            "metadata": {"source_path": "/old/a.md", "team": "docs"},
            "user_id": "alice",
        }
    )

    stored = json.loads(captured["optional_columns"]["metadata"])
    assert stored == {
        "source_path": "/docs/a.md",
        "source_sha256": "abc",
        "source_name": "a.md",
        "timestamp": "2026-10-01T09:00:00+00:00",
        "team": "docs",
    }

    fields = provider._extended_fields(
        MemoryType.KNOWLEDGE_BASE, _BY_ID_SPECS[MemoryType.KNOWLEDGE_BASE][1]
    )
    cursor = _Cursor(
        rows=[
            _row_for(
                fields,
                {
                    "id": uuid.uuid4().bytes,
                    "memory_id": "m",
                    "content": _Lob("chunk"),
                    "user_id": "alice",
                    "namespace": "m",
                    "metadata": _Lob(json.dumps(stored)),
                },
            )
        ]
    )
    reader, _connection = _provider(cursor)
    row = reader.retrieve_by_id(str(uuid.uuid4()), MemoryType.KNOWLEDGE_BASE)

    assert row["source_path"] == "/docs/a.md" and row["source_sha256"] == "abc"
    assert row["team"] == "docs" and row["metadata"]["source_name"] == "a.md"
    assert row["namespace"] == "m" and row["user_id"] == "alice"


@pytest.mark.unit
def test_knowledge_base_supersede_is_persisted_by_update_by_id():
    cursor = _Cursor(rowcount=1)
    provider, _connection = _provider(cursor)
    row_id = uuid.uuid4()

    assert provider.update_by_id(
        str(row_id),
        {
            "status": "superseded",
            "superseded_by": "new-id",
            "superseded_at": "2026-10-01T09:00:00+02:00",
            "supersede_reason": "corrected",
        },
        MemoryType.KNOWLEDGE_BASE,
    )

    sql, params = cursor.calls[0]
    assert "status = :status" in sql and "superseded_by = :superseded_by" in sql
    assert params["status"] == "superseded" and params["superseded_by"] == "new-id"
    assert params["superseded_at"] == datetime(2026, 10, 1, 7, 0)
    assert json.loads(params["metadata_patch"]) == {"supersede_reason": "corrected"}
    assert "updated_at = SYS_EXTRACT_UTC(SYSTIMESTAMP)" in sql


@pytest.mark.unit
def test_knowledge_base_supersede_lands_in_metadata_before_migration_011():
    cursor = _Cursor(rowcount=1)
    provider, _connection = _provider(cursor)
    provider._memory_type_has_column = lambda _kind, column: column not in {
        "status",
        "superseded_by",
        "superseded_at",
    }

    provider.update_by_id(
        str(uuid.uuid4()),
        {"status": "superseded", "superseded_by": "new-id"},
        MemoryType.KNOWLEDGE_BASE,
    )

    sql, params = cursor.calls[0]
    assert "status = :status" not in sql
    assert json.loads(params["metadata_patch"]) == {
        "status": "superseded",
        "superseded_by": "new-id",
    }


@pytest.mark.unit
def test_short_term_memory_keeps_and_returns_its_extra_fields():
    provider = OracleProvider.__new__(OracleProvider)
    provider._generate_embedding_if_needed = lambda *_args, **_kwargs: None
    provider._memory_type_has_column = lambda *_args: True
    captured = {}
    provider._insert_base_row = lambda *_args, **kwargs: captured.update(kwargs)

    provider._store_short_term_memory(
        {"content": "note", "memory_id": "m", "source": "scratchpad", "priority": 2}
    )
    stored = json.loads(captured["optional_columns"]["metadata"])
    assert stored == {"source": "scratchpad", "priority": 2}

    fields, _order_by = _LIST_SPECS[MemoryType.SHORT_TERM_MEMORY]
    fields = provider._extended_fields(MemoryType.SHORT_TERM_MEMORY, fields)
    assert fields[-1].col == "metadata"
    cursor = _Cursor(
        rows=[
            _row_for(
                fields,
                {
                    "id": uuid.uuid4().bytes,
                    "memory_id": "m",
                    "content": "note",
                    "metadata": json.dumps(stored),
                },
            )
        ]
    )
    reader, _connection = _provider(cursor)
    (row,) = reader.list_all(MemoryType.SHORT_TERM_MEMORY)
    assert row["source"] == "scratchpad" and row["priority"] == 2


# ---------------------------------------------------------------------------
# Semantic cache keeps its admission time
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_semantic_cache_store_binds_the_entry_admission_time_as_utc():
    provider = OracleProvider.__new__(OracleProvider)
    provider._generate_embedding_if_needed = lambda *_args, **_kwargs: None
    refreshes, inserts = [], []
    provider._update_semantic_cache_by_key = (
        lambda key, payload: refreshes.append(payload) and False
    )
    provider._insert_base_row = lambda *_args, **kwargs: inserts.append(kwargs)
    admitted = datetime(2026, 10, 1, 9, 0, tzinfo=timezone(timedelta(hours=2)))

    provider._store_semantic_cache(
        {
            "cache_key": "k",
            "query_text": "q",
            "response": "r",
            "created_at": admitted,
            "last_accessed": admitted.timestamp() + 60,
        }
    )

    required = inserts[0]["required_columns"]
    assert required["created_at"] == datetime(2026, 10, 1, 7, 0)
    assert inserts[0]["optional_columns"]["last_accessed"] == datetime(
        2026, 10, 1, 7, 1
    )
    assert refreshes[0]["created_at"] == datetime(2026, 10, 1, 7, 0)


@pytest.mark.unit
def test_semantic_cache_update_rewrites_created_at_and_last_accessed():
    cursor = _Cursor(rowcount=1)
    provider, _connection = _provider(cursor)

    provider.update_by_id(
        "k",
        {"created_at": "2026-10-01T09:00:00+00:00", "last_accessed": 1_790_000_000},
        MemoryType.SEMANTIC_CACHE,
    )

    sql, params = cursor.calls[0]
    assert "created_at = :created_at" in sql and "last_accessed = :last_accessed" in sql
    assert params["created_at"] == datetime(2026, 10, 1, 9, 0)
    assert params["last_accessed"] == datetime.fromtimestamp(
        1_790_000_000, timezone.utc
    ).replace(tzinfo=None)


@pytest.mark.unit
def test_semantic_cache_rows_read_created_at_as_aware_utc():
    fields, _order_by = _LIST_SPECS[MemoryType.SEMANTIC_CACHE]
    created = datetime(2026, 10, 1, 7, 0)
    cursor = _Cursor(
        rows=[_row_for(fields, {"id": uuid.uuid4().bytes, "created_at": created})]
    )
    provider, _connection = _provider(cursor)

    (row,) = provider.list_all(MemoryType.SEMANTIC_CACHE)

    assert row["created_at"] == created.replace(tzinfo=timezone.utc)
    assert row["timestamp"] == created.replace(tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Summaries, UTC binds and startup defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_summary_periods_bind_as_epoch_seconds():
    cursor = _Cursor(rowcount=1)
    provider, _connection = _provider(cursor)

    provider._store_summary(
        {
            "summary_id": "s",
            "content": "c",
            "period_start": "2026-10-01T09:00:00+00:00",
            "period_end": datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        }
    )
    _sql, params = cursor.calls[0]
    start = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc).timestamp()
    assert params["period_start"] == start
    assert params["period_end"] == start + 3600

    cursor.calls.clear()
    provider.update_by_id(
        "s", {"period_start": "2026-10-01T09:00:00+00:00"}, MemoryType.SUMMARIES
    )
    assert cursor.calls[0][1]["period_start"] == start


@pytest.mark.unit
def test_timestamp_binds_convert_aware_values_to_naive_utc():
    bind = OracleProvider._coerce_utc_timestamp_bind
    plus_two = timezone(timedelta(hours=2))
    assert bind(datetime(2026, 10, 1, 9, 0, tzinfo=plus_two)) == datetime(
        2026, 10, 1, 7
    )
    assert bind("2026-10-01T09:00:00Z") == datetime(2026, 10, 1, 9, 0)
    assert bind(datetime(2026, 10, 1, 9, 0)) == datetime(2026, 10, 1, 9, 0)
    assert bind("not a time") is None

    # Skill promotion stamps come from datetime.now(): naive means local.
    local = datetime(2026, 10, 1, 9, 0)
    cursor = _Cursor(routes=[("SELECT id FROM MEMORIZZ.skillbox", None)])
    provider, _connection = _provider(cursor)
    provider._store_skillbox({"skill_id": "s", "name": "n", "promoted_at": local})
    insert = next(params for sql, params in cursor.calls if sql.startswith("INSERT"))
    assert insert["promoted_at"] == local.astimezone(timezone.utc).replace(tzinfo=None)


@pytest.mark.unit
def test_startup_points_local_clock_defaults_at_utc():
    cursor = _Cursor()
    provider, connection = _provider(cursor)
    catalog = {
        "columns": {},
        "indexes": set(),
        "indexed_columns": set(),
        "timestamp_defaults": {
            ("AGENTS", "CREATED_AT"): "CURRENT_TIMESTAMP ",
            ("TOOL_LOG", "TIMESTAMP"): "systimestamp(6)",
            ("PERSONAS", "UPDATED_AT"): "SYS_EXTRACT_UTC(SYSTIMESTAMP)",
            ("SOMEONE_ELSES_TABLE", "CREATED_AT"): "CURRENT_TIMESTAMP",
        },
    }

    provider._ensure_utc_timestamp_defaults(cursor, connection, catalog)

    assert [sql for sql, _params in cursor.calls] == [
        "ALTER TABLE MEMORIZZ.agents MODIFY (created_at DEFAULT "
        "SYS_EXTRACT_UTC(SYSTIMESTAMP))",
        "ALTER TABLE MEMORIZZ.tool_log MODIFY (timestamp DEFAULT "
        "SYS_EXTRACT_UTC(SYSTIMESTAMP))",
    ]


@pytest.mark.unit
def test_schema_and_migrations_cover_the_new_columns_and_utc_defaults():
    from pathlib import Path

    oracle = Path(__file__).parents[2] / "src/memorizz/memory_provider/oracle"
    schema = (oracle / "schema_relational.sql").read_text(encoding="utf-8")
    assert "DEFAULT CURRENT_TIMESTAMP" not in schema
    assert "superseded_at TIMESTAMP" in schema and "last_accessed TIMESTAMP" in schema
    required = OracleProvider._REQUIRED_COLUMNS
    assert {"status", "superseded_by", "superseded_at"} <= {
        column for column, _type in required[MemoryType.KNOWLEDGE_BASE]
    }
    assert "metadata" in {c for c, _t in required[MemoryType.SHORT_TERM_MEMORY]}
    assert "last_accessed" in {c for c, _t in required[MemoryType.SEMANTIC_CACHE]}
    assert not any(
        "CURRENT_TIMESTAMP" in definition
        for columns in required.values()
        for _column, definition in columns
    )
    lifecycle = (oracle / "migrations/011_record_lifecycle_columns.sql").read_text()
    for table, column in (
        ("knowledge_base", "status"),
        ("knowledge_base", "superseded_at"),
        ("short_term_memory", "metadata"),
        ("semantic_cache", "last_accessed"),
    ):
        assert f"'{table}', '{column}'" in lifecycle
    defaults = (oracle / "migrations/012_utc_timestamp_defaults.sql").read_text()
    assert "DEFAULT SYS_EXTRACT_UTC(SYSTIMESTAMP)" in defaults
    conversion = (oracle / "migrations/013_utc_timestamps.sql").read_text()
    assert "SET_SOURCE_TIME_ZONE" in conversion  # refuses to run unconfigured
    assert "already applied" in conversion


# ---------------------------------------------------------------------------
# Agent-wide stores and long knowledge-base scopes
# ---------------------------------------------------------------------------


def _vector_where(provider, cursor, memory_type, **kwargs):
    provider._vector_search(memory_type, [0.1, 0.2], **kwargs)
    sql = cursor.calls[-1][0]
    return sql[sql.index(" WHERE ") : sql.index(" ORDER BY ")], cursor.calls[-1][1]


@pytest.mark.unit
@pytest.mark.parametrize(
    "memory_type",
    [MemoryType.KNOWLEDGE_BASE, MemoryType.SKILLBOX, MemoryType.TOOLBOX],
)
def test_agent_wide_stores_show_shared_rows_to_every_user(memory_type):
    cursor = _Cursor()
    provider, _connection = _provider(cursor)

    where, params = _vector_where(provider, cursor, memory_type, user_id="alice")
    assert "(user_id = :user_id_scope OR user_id IS NULL)" in where
    assert params["user_id_scope"] == "alice"

    where, _params = _vector_where(provider, cursor, memory_type, user_id=None)
    assert "user_id IS NULL" in where and "OR user_id IS NULL" not in where

    cursor.calls.clear()
    provider._retrieve_by_filter({"user_id": "alice"}, memory_type, 5)
    assert "(user_id = :user_id OR user_id IS NULL)" in cursor.calls[0][0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "memory_type",
    [MemoryType.CONVERSATION_MEMORY, MemoryType.WORKFLOW_MEMORY, MemoryType.SUMMARIES],
)
def test_personal_stores_keep_exact_user_isolation(memory_type):
    cursor = _Cursor()
    provider, _connection = _provider(cursor)

    where, _params = _vector_where(
        provider, cursor, memory_type, user_id="alice", memory_id="m"
    )
    assert "user_id = :user_id_scope" in where and "IS NULL" not in where
    # Scope runs in SQL before top-k.
    assert "memory_id = :memory_id" in where


@pytest.mark.unit
def test_list_all_includes_agent_wide_rows_only_in_agent_wide_stores():
    rows = [{"user_id": "alice"}, {"user_id": "bob"}, {"user_id": None}]
    for memory_type in (
        MemoryType.KNOWLEDGE_BASE,
        MemoryType.SKILLBOX,
        MemoryType.TOOLBOX,
    ):
        kept = OracleProvider._apply_user_id_filter(rows, memory_type, "alice")
        assert kept == [{"user_id": "alice"}, {"user_id": None}]
    kept = OracleProvider._apply_user_id_filter(
        rows, MemoryType.CONVERSATION_MEMORY, "alice"
    )
    assert kept == [{"user_id": "alice"}]
    assert OracleProvider._apply_user_id_filter(
        rows, MemoryType.KNOWLEDGE_BASE, None
    ) == [{"user_id": None}]


@pytest.mark.unit
def test_knowledge_base_scope_beyond_1000_ids_is_split():
    cursor = _Cursor()
    provider, _connection = _provider(cursor)
    ids = [f"kb-{index}" for index in range(2500)]

    where, params = _vector_where(
        provider,
        cursor,
        MemoryType.KNOWLEDGE_BASE,
        memory_id="m",
        knowledge_base_ids=ids,
    )

    lists = re.findall(r"knowledge_base_id IN \(([^)]*)\)", where)
    assert [len(item.split(", ")) for item in lists] == [1000, 1000, 500]
    assert (
        " OR knowledge_base_id IN (" in where and "OR memory_id = :memory_id" in where
    )
    assert sorted(v for k, v in params.items() if k.startswith("kb_")) == sorted(ids)


# ---------------------------------------------------------------------------
# Automation schedule times and archives
# ---------------------------------------------------------------------------


class _Pool:
    def __init__(self, cursor):
        self._connection = _Context(
            SimpleNamespace(cursor=lambda: cursor, commit=lambda: None)
        )

    def acquire(self):
        return self._connection


@pytest.mark.unit
def test_automation_schedule_times_are_written_as_utc_instants():
    from memorizz.automation.store.oracle import OracleAutomationStore

    job_row = (
        ("job", "agent", "n", 1, "interval", None, 60, "UTC")
        + (None, datetime(2026, 10, 1, 7, 0), None)
        + ("skip", 900, 1, 60, "prompt", "{}", None, "{}", None, None, None, None)
    )
    cursor = _Cursor(routes=[("SELECT job_id", job_row)])
    store = OracleAutomationStore(
        SimpleNamespace(config=SimpleNamespace(schema="MEMORIZZ"), pool=_Pool(cursor))
    )
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone(timedelta(hours=2)))

    store.claim_due_jobs("worker", now, limit=5, lease_seconds=60)

    sql, params = cursor.calls[0]
    assert "next_run_at <= FROM_TZ(CAST(:now_utc AS TIMESTAMP), 'UTC')" in sql
    assert params["now_utc"] == datetime(2026, 10, 1, 7, 0)

    cursor.calls.clear()
    store.update_job("job", {"next_run_at": now, "name": "n"})
    sql, params = cursor.calls[0]
    assert "next_run_at = FROM_TZ(CAST(:v_next_run_at AS TIMESTAMP), 'UTC')" in sql
    assert "updated_at = SYS_EXTRACT_UTC(SYSTIMESTAMP)" in sql
    assert params["v_next_run_at"] == datetime(2026, 10, 1, 7, 0)


@pytest.mark.unit
def test_archive_labels_and_binds_timestamps_as_utc():
    from memorizz.memory_provider.oracle import archive

    class _Fetched:
        description = [("ID",), ("CREATED_AT",)]

        def __iter__(self):
            return iter([(b"\x01" * 16, datetime(2026, 10, 1, 7, 0))])

    provider, _connection = _provider(_Cursor())
    (record,) = archive._decode(provider, MemoryType.KNOWLEDGE_BASE, _Fetched())
    assert record["created_at"] == "2026-10-01T07:00:00+00:00"

    insert = _Cursor()
    provider, _connection = _provider(insert)
    provider._archive_schema = {MemoryType.KNOWLEDGE_BASE: _archive_columns()}
    archive.store_record(
        provider,
        MemoryType.KNOWLEDGE_BASE,
        str(uuid.uuid4()),
        {"memory_id": "m", "content": "c", "created_at": "2026-10-01T09:00:00+02:00"},
    )
    params = next(p for sql, p in insert.calls if sql.startswith("INSERT"))
    assert params["created_at"] == datetime(2026, 10, 1, 7, 0)
    with pytest.raises(archive.MemoryArchiveError):
        archive.store_record(
            provider,
            MemoryType.KNOWLEDGE_BASE,
            str(uuid.uuid4()),
            {"memory_id": "m", "content": "c", "created_at": "yesterday"},
        )


def _archive_columns():
    return {
        "id": ("RAW", "N", None, 16),
        "memory_id": ("VARCHAR2", "N", None, 255),
        "content": ("CLOB", "N", None, None),
        "created_at": ("TIMESTAMP(6)", "Y", "SYS_EXTRACT_UTC(SYSTIMESTAMP)", None),
        "archive_extras": ("CLOB", "Y", None, None),
    }


@pytest.mark.unit
def test_archive_reads_spread_metadata_back_like_the_provider():
    """Live on a schema once restored from an archive: knowledge rows came back
    with source_path only inside ``metadata``, so MCP ingest re-ingested
    unchanged files."""
    from memorizz.memory_provider.oracle import archive

    class _Fetched:
        description = [("ID",), ("SOURCE_ID",), ("METADATA",)]

        def __iter__(self):
            meta = '{"source_path": "/notes/plan.md", "source_id": "ignored"}'
            return iter([(b"\x01" * 16, "kept", meta)])

    provider, _connection = _provider(_Cursor())
    (record,) = archive._decode(provider, MemoryType.KNOWLEDGE_BASE, _Fetched())
    assert record["source_path"] == "/notes/plan.md"
    assert record["source_id"] == "kept"  # a column value wins
    (other,) = archive._decode(provider, MemoryType.TOOLBOX, _Fetched())
    assert "source_path" not in other

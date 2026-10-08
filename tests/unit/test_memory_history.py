"""Durable mutation history, scope isolation and request-local attribution."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from memorizz import MemoryHistory, MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit


@pytest.fixture
def history_provider(tmp_path):
    return FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )


def test_history_records_versions_branches_deletions_without_content(history_provider):
    provider = history_provider
    history = MemoryHistory(provider)
    with history.recording(
        actor="operator-a", source="sdk", agent_id="a", memory_id="ns"
    ):
        first = provider.store(
            {"content": "sensitive original", "memory_id": "ns"},
            MemoryType.KNOWLEDGE_BASE,
        )
        assert provider.update_by_id(
            first, {"content": "changed"}, MemoryType.KNOWLEDGE_BASE
        )
        child = provider.store(
            {"content": "derived", "memory_id": "ns", "supersedes": first},
            MemoryType.KNOWLEDGE_BASE,
        )
        provider.delete_by_id(first, MemoryType.KNOWLEDGE_BASE)
    events = history.timeline(agent_id="a")["events"]
    assert [e["action"] for e in events] == ["created", "updated", "created", "deleted"]
    assert events[1]["previous_event_id"] == events[0]["record_id"]
    assert events[3]["previous_event_id"] == events[1]["record_id"]
    assert events[2]["parents"] == [{"record_id": first, "relation": "supersedes"}]
    assert events[2]["target_record_id"] == child
    assert events[1]["changed_fields"] == ["content"]
    assert events[1]["before_hash"] != events[1]["after_hash"]
    assert all(e["actor"] == "operator-a" for e in events)
    assert (
        len(
            history.timeline(agent_id="a", memory_type=MemoryType.KNOWLEDGE_BASE)[
                "events"
            ]
        )
        == 4
    )
    assert "sensitive original" not in json.dumps(
        provider.list_all(MemoryType.SHARED_MEMORY)
    )
    # A new process/provider instance can read the retained history.
    reloaded = FileSystemProvider(
        FileSystemConfig(root_path=provider.root_path, use_faiss=False)
    )
    assert len(MemoryHistory(reloaded).timeline(agent_id="a")["events"]) == 4


def test_noop_failed_and_internal_writes_do_not_create_changes(history_provider):
    history = MemoryHistory(history_provider, record_changes=True)
    record = history_provider.store({"content": "same"}, MemoryType.KNOWLEDGE_BASE)
    history_provider.update_by_id(
        record, {"content": "same"}, MemoryType.KNOWLEDGE_BASE
    )
    assert not history_provider.delete_by_id("missing", MemoryType.KNOWLEDGE_BASE)
    history_provider.store(
        {"record_type": "observability_context_snapshot", "content": "private input"},
        MemoryType.SHARED_MEMORY,
    )
    assert [e["action"] for e in history.timeline()["events"]] == ["created"]


def test_field_operations_and_current_values_do_not_invent_past_content(
    history_provider,
):
    history = MemoryHistory(history_provider)
    with history.recording(actor="author", agent_id="a", memory_id="m"):
        record = history_provider.store(
            {"agent_id": "a", "memory_id": "m", "content": "before"},
            MemoryType.KNOWLEDGE_BASE,
        )
        history_provider.update_by_id(
            record,
            {"content": "after", "description": "new"},
            MemoryType.KNOWLEDGE_BASE,
        )
    first, latest = history.timeline(agent_id="a")["events"]
    assert latest["field_changes"] == {"content": "modified", "description": "added"}
    current = history.current_record(first)
    assert current["record"]["content"] == "after"
    assert current["matches_selected_version"] is False
    assert history.current_record(latest)["matches_selected_version"] is True
    assert history.get_change(latest["record_id"], agent_id="wrong") is None
    assert history.get_change(latest["record_id"], user_id="wrong") is None
    assert history.get_change(latest["record_id"], application_id="wrong") is None
    assert '"before"' not in json.dumps(history.timeline()["events"])
    history_provider.delete_by_id(record, MemoryType.KNOWLEDGE_BASE)
    assert (
        history.current_record(history.timeline()["events"][-1])["state"] == "deleted"
    )


def test_current_record_ui_is_scoped_redacted_bounded_and_metadata_safe(
    history_provider, monkeypatch
):
    import asyncio

    from fastapi import HTTPException

    from memorizz.ui.routers.memory_history import current_memory_record
    from memorizz.ui.state import _state
    from memorizz.ui.trace_access import TracePrincipal, current_principal

    history = MemoryHistory(history_provider)
    with history.recording(
        actor="a", agent_id="a", memory_id="m", user_id="alice", application_id="app"
    ):
        record = history_provider.store(
            {
                "agent_id": "a",
                "memory_id": "m",
                "user_id": "alice",
                "application_id": "app",
                "content": json.dumps(
                    {"fact": "Mira after 19:00", "password": "test-secret"}
                ),
            },
            MemoryType.KNOWLEDGE_BASE,
        )
    event = history.timeline()["events"][-1]
    monkeypatch.setitem(_state, "provider", history_provider)
    token = current_principal.set(
        TracePrincipal(user_id="alice", user_bound=True, application_id="app")
    )
    try:
        response = json.loads(
            asyncio.run(current_memory_record(event["record_id"], agent_id="a")).body
        )
        assert response["matches_selected_version"] is True
        assert "Mira after 19:00" in response["fields"]["content"]
        assert "test-secret" not in json.dumps(response)
        with pytest.raises(HTTPException) as denied:
            asyncio.run(current_memory_record(event["record_id"], agent_id="other"))
        assert denied.value.status_code == 404
        history_provider.update_by_id(
            record, {"content": "x" * 20000}, MemoryType.KNOWLEDGE_BASE
        )
        preview = json.loads(
            asyncio.run(current_memory_record(event["record_id"])).body
        )
        assert preview["truncated"] and len(preview["fields"]["content"]) == 16000
        history_provider.update_by_id(
            record,
            {"user_id": "bob", "content": "bob private"},
            MemoryType.KNOWLEDGE_BASE,
        )
        assert (
            json.loads(asyncio.run(current_memory_record(event["record_id"])).body)[
                "state"
            ]
            == "unavailable"
        )
    finally:
        current_principal.reset(token)
    token = current_principal.set(TracePrincipal(role="viewer"))
    try:
        assert json.loads(
            asyncio.run(current_memory_record(event["record_id"])).body
        ) == {"state": "hidden", "content_mode": "metadata"}
    finally:
        current_principal.reset(token)
    token = current_principal.set(TracePrincipal(user_id="bob", user_bound=True))
    try:
        with pytest.raises(HTTPException) as denied:
            asyncio.run(current_memory_record(event["record_id"]))
        assert denied.value.status_code == 404
    finally:
        current_principal.reset(token)


def test_atomic_shared_updates_record_authors_versions_and_ignore_failed_retries(
    history_provider,
):
    from memorizz.coordination.shared_memory import SharedMemory

    history = MemoryHistory(history_provider, record_changes=True)
    shared = SharedMemory(history_provider)
    with history.recording(actor="coordinator", agent_id="coordinator"):
        session = shared.create_shared_session(
            "coordinator", ["researcher", "reviewer"]
        )
        assert shared.add_blackboard_entry(
            session, "researcher", "private finding", "result", entry_id="finding-1"
        )
        # Idempotent replays and stale CAS attempts must not invent extra versions.
        assert shared.add_blackboard_entry(
            session, "researcher", "private finding", "result", entry_id="finding-1"
        )
        assert not history_provider.compare_and_swap_shared_memory(
            session, "stale", "new"
        )
        assert shared.add_blackboard_entry(
            session, "reviewer", "private review", "result"
        )
    events = history.timeline(
        agent_id="coordinator", memory_type=MemoryType.SHARED_MEMORY
    )["events"]
    assert [e["action"] for e in events] == ["created", "updated", "updated"]
    assert [e["actor"] for e in events] == ["coordinator", "researcher", "reviewer"]
    assert events[1]["initiator"] == events[2]["initiator"] == "coordinator"
    assert events[1]["previous_event_id"] == events[0]["record_id"]
    assert events[2]["previous_event_id"] == events[1]["record_id"]
    assert events[1]["label"] == "Shared result"
    assert events[1]["operation"] == "compare_and_swap_shared_memory"
    assert all(e["changed_fields"] == ["content"] for e in events[1:])
    assert "private finding" not in json.dumps(events)
    assert "private review" not in json.dumps(events)


def test_ui_writer_names_respect_scoped_events_and_application(
    history_provider, monkeypatch
):
    import asyncio

    from memorizz import MemAgentModel
    from memorizz.ui.routers.memory_history import evolution_data
    from memorizz.ui.state import _state
    from memorizz.ui.trace_access import TracePrincipal, current_principal

    for identity, app, name in (
        ("root", "app-a", "Coordinator"),
        ("writer-a", "app-a", "Researcher"),
        ("writer-b", "app-b", "Other application name"),
        ("unrelated", "app-a", "Other tenant name"),
    ):
        history_provider.store_memagent(
            MemAgentModel(
                agent_id=identity,
                application_id=app,
                name=name,
                delegates=["writer-a", "writer-b", "missing"]
                if identity == "root"
                else None,
            )
        )
    history = MemoryHistory(history_provider)
    for actor, tenant in (
        ("writer-a", "alice"),
        ("writer-b", "alice"),
        ("unrelated", "bob"),
    ):
        with history.recording(
            actor=actor,
            agent_id="root",
            memory_id="team",
            user_id=tenant,
            application_id="app-a",
        ):
            history_provider.store(
                {"content": "private finding", "user_id": tenant},
                MemoryType.KNOWLEDGE_BASE,
            )
    monkeypatch.setitem(_state, "provider", history_provider)
    token = current_principal.set(
        TracePrincipal(
            role="viewer", application_id="app-a", user_id="alice", user_bound=True
        )
    )
    try:
        response = asyncio.run(evolution_data(agent_id="root", memory_id="team"))
    finally:
        current_principal.reset(token)
    result = json.loads(response.body)
    assert len(result["events"]) == 2
    assert result["writers"] == {"writer-a": "Researcher", "root": "Coordinator"}
    assert result["playground_available"] is False
    assert result["navigation"] == {
        "agent": {"agent_id": "root", "name": "Coordinator"},
        "delegates": [{"agent_id": "writer-a", "name": "Researcher"}],
    }
    assert "Other application name" not in response.body.decode()
    assert "Other tenant name" not in response.body.decode()
    assert "private finding" not in response.body.decode()
    token = current_principal.set(
        TracePrincipal(
            role="viewer", application_id="app-a", user_id="no-history", user_bound=True
        )
    )
    try:
        response = asyncio.run(evolution_data(agent_id="root", memory_id="team"))
    finally:
        current_principal.reset(token)
    assert json.loads(response.body)["navigation"] == {"agent": None, "delegates": []}


def test_agent_observations_do_not_mix_other_owners_in_a_common_namespace(
    history_provider,
):
    ids = {}
    for owner in ("root", "delegate", None):
        ids[owner] = history_provider.store(
            {"content": "old", "memory_id": "team", "agent_id": owner},
            MemoryType.CONVERSATION_MEMORY,
        )
    history = MemoryHistory(history_provider)
    own = history.observations(agent_id="delegate", memory_ids=["team"])
    assert {e["target_record_id"] for e in own} == {ids["delegate"], ids[None]}
    # An explicit namespace-wide view may still show all its contributors.
    assert {
        e["target_record_id"] for e in history.observations(memory_ids=["team"])
    } == set(ids.values())


def test_agent_navigation_can_descend_through_saved_delegate_configurations(
    history_provider, monkeypatch
):
    import asyncio

    from memorizz import MemAgentModel
    from memorizz.ui.routers.memory_history import evolution_data
    from memorizz.ui.state import _state
    from memorizz.ui.trace_access import TracePrincipal, current_principal

    for identity, children in (
        ("root", ["delegate"]),
        ("delegate", ["nested"]),
        ("nested", []),
    ):
        history_provider.store_memagent(
            MemAgentModel(agent_id=identity, name=identity, delegates=children)
        )
    monkeypatch.setitem(_state, "provider", history_provider)
    token = current_principal.set(TracePrincipal())
    try:
        root = json.loads(asyncio.run(evolution_data(agent_id="root")).body)
        delegate = json.loads(asyncio.run(evolution_data(agent_id="delegate")).body)
    finally:
        current_principal.reset(token)
    assert root["navigation"]["delegates"] == [
        {"agent_id": "delegate", "name": "delegate"}
    ]
    assert delegate["navigation"]["delegates"] == [
        {"agent_id": "nested", "name": "nested"}
    ]


def test_journal_failure_does_not_turn_successful_write_into_failure(
    history_provider, monkeypatch
):
    from memorizz.observability.store import ObservabilityStore

    history = MemoryHistory(history_provider, record_changes=True)
    monkeypatch.setattr(
        ObservabilityStore,
        "_put",
        lambda *_: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    record = history_provider.store({"content": "stored"}, MemoryType.KNOWLEDGE_BASE)
    assert (
        history_provider.retrieve_by_id(record, MemoryType.KNOWLEDGE_BASE)["content"]
        == "stored"
    )
    assert history.timeline()["events"] == []


def test_preparation_failure_preserves_write_and_nested_internal_records_are_excluded(
    history_provider, monkeypatch
):
    import memorizz.memory_history as module

    history = MemoryHistory(history_provider, record_changes=True)
    for content in (
        {"record_type": "observability_context_snapshot"},
        json.dumps({"record_type": "memory_history_head"}),
    ):
        history_provider.store({"content": content}, MemoryType.SHARED_MEMORY)
    assert history.timeline()["events"] == []
    monkeypatch.setattr(
        module,
        "_json",
        lambda *_: (_ for _ in ()).throw(RuntimeError("failed inspection")),
    )
    record = history_provider.store(
        {"content": "write must succeed"}, MemoryType.KNOWLEDGE_BASE
    )
    assert (
        history_provider.retrieve_by_id(record, MemoryType.KNOWLEDGE_BASE)["content"]
        == "write must succeed"
    )


def test_portable_query_accepts_oracle_decoded_json_payload():
    from memorizz.memory_provider.base import MemoryProvider

    class OracleShape:
        query_observability_records = MemoryProvider.query_observability_records

        def list_all(self, kind):
            return [
                {
                    "memory_id": "physical-journal-key",
                    "content": {
                        "record_type": "memory_history_change",
                        "record_id": "e",
                        "agent_id": "a",
                        "memory_id": "thread",
                        "user_id": "alice",
                    },
                }
            ]

    provider = OracleShape()
    assert (
        len(
            provider.query_observability_records(
                MemoryType.SHARED_MEMORY,
                agent_ids=["a"],
                record_type="memory_history_change",
                user_id="alice",
            )["items"]
        )
        == 1
    )
    assert not provider.query_observability_records(
        MemoryType.SHARED_MEMORY, record_type="memory_history_change", user_id="bob"
    )["items"]


def test_exact_tenant_and_agent_namespace_intersection(history_provider):
    history = MemoryHistory(history_provider)
    for agent, tenant, memory in (
        ("a", "alice", "one"),
        ("a", "bob", "one"),
        ("b", "alice", "one"),
        ("a", None, "one"),
        ("a", "alice", "two"),
    ):
        with history.recording(
            actor=tenant or "anonymous", agent_id=agent, user_id=tenant
        ):
            history_provider.store(
                {"content": "x", "memory_id": memory, "user_id": tenant},
                MemoryType.KNOWLEDGE_BASE,
            )
    events = history.timeline(agent_id="a", memory_id="one", user_id="alice")["events"]
    assert len(events) == 1 and events[0]["actor"] == "alice"
    assert len(history.timeline(user_id=None)["events"]) == 1


def test_batch_and_paginated_history_are_complete(history_provider):
    history = MemoryHistory(history_provider, record_changes=True)
    ids = history_provider.store_many(
        [{"content": str(i)} for i in range(5)], MemoryType.KNOWLEDGE_BASE
    )
    events, cursor = [], None
    while True:
        page = history.timeline(limit=2, cursor=cursor)
        events.extend(page["events"])
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert {e["target_record_id"] for e in events} == set(ids)
    assert len(events) == 5


def test_older_records_are_observations_and_updates_report_history_gap(
    history_provider,
):
    record = history_provider.store(
        {
            "content": "old",
            "memory_id": "m",
            "agent_id": "a",
            "timestamp": "2020-01-01T00:00:00Z",
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    history = MemoryHistory(history_provider, record_changes=True)
    observed = history.observations(agent_id="a")
    assert len(observed) == 1 and observed[0]["action"] == "observed"
    assert observed[0]["actor"] == "unknown"
    assert history.timeline()["events"] == []
    history_provider.update_by_id(record, {"content": "new"}, MemoryType.KNOWLEDGE_BASE)
    assert history.timeline()["events"][0]["history_gap"] is True


def test_threaded_attribution_does_not_cross_requests(history_provider):
    history = MemoryHistory(history_provider, record_changes=True)

    def write(actor):
        with history.recording(actor=actor, agent_id=actor):
            return history_provider.store({"content": actor}, MemoryType.KNOWLEDGE_BASE)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, ("alice", "bob")))
    assert {(e["actor"], e["agent_id"]) for e in history.timeline()["events"]} == {
        ("alice", "alice"),
        ("bob", "bob"),
    }


def test_cli_timeline_reports_scoped_changes(history_provider, monkeypatch):
    from typer.testing import CliRunner

    from memorizz.cli.memory_commands import memory_app

    history = MemoryHistory(history_provider)
    with history.recording(actor="operator", agent_id="a"):
        history_provider.store({"content": "private text"}, MemoryType.KNOWLEDGE_BASE)
    monkeypatch.setattr("memorizz.cli.config.load_layered_env", lambda: None)
    monkeypatch.setattr(
        "memorizz.cli.agent_factory.detect_memory_provider", lambda *a: history_provider
    )
    result = CliRunner().invoke(
        memory_app, ["timeline", "--agent-id", "a", "--actor", "operator", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)["events"]) == 1
    assert "private text" not in result.output


def test_nested_provider_metadata_retains_origin_and_actor(history_provider):
    history = MemoryHistory(history_provider, record_changes=True)
    history_provider.store(
        {
            "content": "fictional",
            "metadata": {
                "name": "Launch brief",
                "source_path": "brief.md",
                "created_by": "author",
            },
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    event = history.timeline()["events"][0]
    assert event["label"] == "Launch brief"
    assert event["source_ref"] == "brief.md"
    assert event["actor"] == "author"


def test_memory_unit_dictionary_uses_provider_default_conversation_type(
    history_provider,
):
    history = MemoryHistory(history_provider, record_changes=True)
    record = history_provider.store(
        memory_unit={"role": "user", "content": "question"}, memory_id="m"
    )
    event = history.timeline(memory_id="m")["events"][0]
    assert event["target_record_id"] == record
    assert event["memory_type"] == MemoryType.CONVERSATION_MEMORY.value


def test_specialized_agent_namespace_setters_are_updates(history_provider):
    from memorizz import MemAgentModel

    history = MemoryHistory(history_provider, record_changes=True)
    history_provider.store_memagent(MemAgentModel(agent_id="a", memory_ids=["m"]))
    history_provider.update_memagent_memory_ids("a", ["next"])
    history_provider.delete_memagent_memory_ids("a")
    events = history.timeline(agent_id="a")["events"]
    assert [e["action"] for e in events] == ["created", "updated", "updated"]
    assert events[-1]["changed_fields"] == ["memory_ids"]

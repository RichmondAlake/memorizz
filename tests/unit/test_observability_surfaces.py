"""Observability on the CLI and the MCP server, the per-turn link on the
Traces page, and the rule the surfaces share."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memorizz.approval import SQLiteApprovalStore
from memorizz.cli.app import app
from memorizz.enums import MemoryType
from memorizz.mcp_server import MemorizzMCPServerConfig, MemorizzRuntime
from memorizz.mcp_server.auth import RequestIdentity
from memorizz.mcp_server.config import ALL_SCOPES
from memorizz.mcp_server.runtime import MemorizzServerError
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import ObservabilityProvider
from memorizz.observability.inspection import build_causal_waterfall
from memorizz.observability.normalization import read_payload
from memorizz.observability.providers import observability_change
from memorizz.observability.providers.sentry import SentryObservability

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_sentry_environment(monkeypatch):
    for name in ("MEMORIZZ_SENTRY_DSN", "SENTRY_DSN", "SENTRY_ORG", "SENTRY_SPOTLIGHT"):
        monkeypatch.delenv(name, raising=False)


# ---- the rule the surfaces share -----------------------------------------


def test_naming_a_provider_turns_tracing_on_and_memorizz_clears_it():
    assert observability_change() == (True, None)
    assert observability_change(enabled=False) == (False, None)
    assert observability_change(False, None, provider="Sentry") == (
        True,
        {"provider": "sentry"},
    )
    assert observability_change(provider="sentry", org="acme") == (
        True,
        {"provider": "sentry", "org": "acme"},
    )
    stored = {"provider": "sentry", "org": "acme", "environment": "staging"}
    assert observability_change(True, stored, provider="memorizz") == (True, None)


def test_what_is_not_named_is_kept():
    stored = {"provider": "sentry", "org": "acme", "environment": "staging"}
    assert observability_change(True, stored) == (True, stored)
    assert observability_change(True, stored, enabled=False) == (False, stored)
    assert observability_change(False, stored) == (False, stored)
    assert observability_change(True, stored, org="")[1] == {
        "provider": "sentry",
        "environment": "staging",
    }
    assert observability_change(True, stored, include_content=True)[1] == {
        **stored,
        "include_content": True,
    }


def test_an_organisation_needs_a_provider_and_the_provider_must_exist():
    with pytest.raises(ValueError, match="provider such as sentry"):
        observability_change(org="acme")
    with pytest.raises(ValueError, match="Unknown observability provider"):
        observability_change(provider="carrier-pigeon")


# ---- CLI -------------------------------------------------------------------


def _env(tmp_path):
    return {
        "MEMORIZZ_HOME": str(tmp_path),
        "MEMORIZZ_BACKEND": "",
        "MEMORIZZ_DEFAULT_LLM_PROVIDER": "",
        "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "AZURE_OPENAI_API_KEY": "",
        "OLLAMA_HOST": "http://127.0.0.1:1",
    }


def _cli(tmp_path, *args):
    result = runner.invoke(app, ["agents", *args, "--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    return json.loads(result.output.strip().splitlines()[-1])["agent"]


def _stored(tmp_path, agent_id):
    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    try:
        return store.retrieve_memagent(agent_id)
    finally:
        store.close()


def test_cli_creates_and_changes_an_agents_observability(tmp_path):
    created = _cli(
        tmp_path,
        "create",
        "--name",
        "Ada",
        "--no-llm",
        "--observability-provider",
        "sentry",
        "--observability-org",
        "acme",
    )
    assert (
        created["observability"],
        created["observability_provider"],
        created["observability_org"],
    ) == (True, "sentry", "acme")
    agent_id = created["agent_id"]
    stored = _stored(tmp_path, agent_id)
    assert stored.observability is True
    assert stored.observability_config == {"provider": "sentry", "org": "acme"}

    off = _cli(tmp_path, "update", agent_id, "--no-observability")
    assert off["observability"] is False
    assert off["observability_provider"] == "sentry"  # kept

    back = _cli(tmp_path, "update", agent_id, "--observability-provider", "memorizz")
    assert (back["observability"], back["observability_provider"]) == (
        True,
        "memorizz",
    )
    assert _stored(tmp_path, agent_id).observability_config is None
    assert _cli(tmp_path, "show", agent_id)["observability_provider"] == "memorizz"


def test_cli_agents_are_traced_in_memorizz_unless_told_otherwise(tmp_path):
    plain = _cli(tmp_path, "create", "--name", "Bo", "--no-llm")
    assert (plain["observability"], plain["observability_provider"]) == (
        True,
        "memorizz",
    )
    quiet = _cli(tmp_path, "create", "--name", "Cy", "--no-llm", "--no-observability")
    assert quiet["observability"] is False
    assert _stored(tmp_path, quiet["agent_id"]).observability is False


def test_cli_refuses_an_unknown_observability_provider(tmp_path):
    result = runner.invoke(
        app,
        [
            "agents",
            "create",
            "--name",
            "Di",
            "--no-llm",
            "--observability-provider",
            "pigeon",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Unknown observability provider" in result.output


# ---- MCP -------------------------------------------------------------------


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield MemorizzRuntime(
        MemorizzMCPServerConfig(
            transport="stdio", allow_writes=True, exposed_agent_ids=None
        ),
        provider=store,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    store.close()


def _identity():
    return RequestIdentity(
        principal="alice", scopes=frozenset(ALL_SCOPES), authenticated=True
    )


def test_mcp_creates_and_updates_an_agents_observability(runtime):
    created = runtime.create_agent(
        "Ada",
        _identity(),
        llm_provider="ollama",
        llm_model="qwen2.5:7b",
        observability_provider="sentry",
        observability_org="acme",
    )["agent"]
    assert created["observability"] is True
    assert created["observability_config"] == {"provider": "sentry", "org": "acme"}
    agent_id = created["agent_id"]

    off = runtime.update_agent(agent_id, _identity(), observability=False)["agent"]
    assert off["observability"] is False
    assert off["observability_config"] == {"provider": "sentry", "org": "acme"}

    back = runtime.update_agent(
        agent_id, _identity(), observability_provider="memorizz"
    )["agent"]
    assert back["observability"] is True and not back.get("observability_config")


def test_mcp_agents_default_to_memorizz_and_bad_providers_are_refused(runtime):
    plain = runtime.create_agent(
        "Bo", _identity(), llm_provider="ollama", llm_model="qwen2.5:7b"
    )["agent"]
    assert plain["observability"] is True and not plain.get("observability_config")
    with pytest.raises(MemorizzServerError, match="Unknown observability provider"):
        runtime.update_agent(
            plain["agent_id"], _identity(), observability_provider="pigeon"
        )


# ---- the per-turn link ----------------------------------------------------


def test_sentry_links_one_trace_by_its_id_and_time(monkeypatch):
    root = "689837bf-201e-4086-81a7-6d3894afaaa6"
    provider = SentryObservability({"org": "acme"})
    assert provider.trace_url(root) == (
        "https://acme.sentry.io/explore/traces/trace/"
        "689837bf201e408681a76d3894afaaa6/"
    )
    assert provider.trace_url(root, started_at="2026-11-19T11:00:00.000Z") == (
        "https://acme.sentry.io/explore/traces/trace/"
        "689837bf201e408681a76d3894afaaa6/?timestamp=1795086000"
    )
    assert provider.trace_url("") is None
    # No organisation known: Sentry resolves the signed-in person's.
    assert (
        SentryObservability()
        .trace_url(root)
        .startswith(
            "https://sentry.io/orgredirect/organizations/:orgslug/explore/traces/trace/"
        )
    )
    monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.example.com/7")
    assert (
        SentryObservability({"org": "ops"})
        .trace_url(root)
        .startswith(
            "https://sentry.example.com/organizations/ops/explore/traces/trace/"
        )
    )


class _Model:
    def get_config(self):
        return {"model": "gpt-4o-mini", "provider": "openai"}

    def get_last_usage(self):
        return {"prompt_tokens": 100, "completion_tokens": 10}

    def generate(self, messages, tools=None):
        return "answer"


class _Sends(ObservabilityProvider):
    name = "sends"
    sent = True

    def export_turn(self, trace_context, events, *, agent_name=None):
        return "trace-1" if self.sent else None


def _bundle_events(store):
    return [
        event
        for row in store.list_all(MemoryType.SHARED_MEMORY)
        if row.get("record_type") == "observability_trace_bundle"
        for event in (read_payload(row) or row).get("events") or []
    ]


def test_a_turn_records_where_it_was_sent_and_only_when_it_was(tmp_path):
    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    provider = _Sends()
    agent = MemAgent(
        model=_Model(),
        memory_provider=store,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
        name="Ada",
        context_policy={"progressive_tool_disclosure": False},
        observability_provider=provider,
    )
    agent._schedule_conversation_embedding_backfill = lambda rows: None

    agent.run("one", memory_id="m1")
    marked = [e for e in _bundle_events(store) if e.get("trace_export")]
    assert [(e["trace_kind"], e["trace_export"]) for e in marked] == [
        ("turn_result", "sends")
    ]

    # A turn the provider did not take (sampled out, not ready) is not marked.
    provider.sent = False
    agent.run("two", memory_id="m1")
    assert len([e for e in _bundle_events(store) if e.get("trace_export")]) == 1
    store.close()


def test_the_waterfall_turn_carries_the_mark():
    from memorizz.ui.waterfall_view import waterfall_view

    root = "689837bf-201e-4086-81a7-6d3894afaaa6"
    common = {
        "agent_id": "a1",
        "thread_id": "t1",
        "turn_id": "turn-1",
        "root_trace_id": root,
    }
    events = [
        {
            **common,
            "event_id": "e1",
            "kind": "turn_start",
            "phase": "start",
            "span_id": root,
            "timestamp": "2026-11-19T11:00:00.000Z",
            "status": "started",
        },
        {
            **common,
            "event_id": "e2",
            "kind": "turn_result",
            "phase": "result",
            "span_id": root,
            "timestamp": "2026-11-19T11:00:02.000Z",
            "status": "success",
            "trace_export": "sentry",
        },
    ]
    view = waterfall_view(build_causal_waterfall(events))
    (turn,) = view["turns"]
    assert turn["trace_export"] == "sentry"
    assert turn["root_trace_id"] == root

    plain = [
        {k: v for k, v in event.items() if k != "trace_export"} for event in events
    ]
    assert (
        waterfall_view(build_causal_waterfall(plain))["turns"][0]["trace_export"]
        is None
    )


def test_the_traces_page_link_follows_the_turn_not_the_agents_current_setting():
    pytest.importorskip("fastapi")
    from memorizz.ui.helpers import observability_trace_link

    root = "689837bf-201e-4086-81a7-6d3894afaaa6"
    agent = {"observability_config": {"provider": "sentry", "org": "acme"}}
    link = observability_trace_link(agent, "sentry", root, "2026-11-19T11:00:00.000Z")
    assert link == {
        "label": "Sentry",
        "url": "https://acme.sentry.io/explore/traces/trace/"
        "689837bf201e408681a76d3894afaaa6/?timestamp=1795086000",
    }
    # The agent went back to MemoRizz: the turn it sent earlier still opens.
    moved = observability_trace_link({"observability_config": None}, "sentry", root)
    assert moved["url"].startswith("https://sentry.io/orgredirect/")
    # Not sent anywhere, or to a provider with no page for one trace: no link.
    assert observability_trace_link(agent, None, root) is None
    assert observability_trace_link(agent, "memorizz", root) is None
    assert observability_trace_link(agent, "gone-away", root) is None

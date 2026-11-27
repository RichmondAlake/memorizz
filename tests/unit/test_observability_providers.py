"""MemAgent(observability=..., observability_provider=...): the switch, the
built-in provider, the Sentry provider and what is saved with an agent."""

from __future__ import annotations

import gzip
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest

from memorizz import MemAgent
from memorizz.enums import MemoryType
from memorizz.memagent import persistence
from memorizz.memagent.builders.agent_builder import MemAgentBuilder
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    MemorizzObservability,
    ObservabilityConfig,
    ObservabilityProvider,
    create_observability_provider,
    observability_status,
)
from memorizz.observability.normalization import read_payload
from memorizz.observability.providers import observability_settings
from memorizz.observability.providers import sentry as sentry_provider
from memorizz.observability.providers.sentry import (
    SentryObservability,
    build_sentry_events,
    sentry_trace_id,
)

pytestmark = pytest.mark.unit

ROOT = "689837bf-201e-4086-81a7-6d3894afaaa6"
CONTEXT = {
    "agent_id": "agent-1",
    "run_id": "run-1",
    "turn_id": "turn-1",
    "root_trace_id": ROOT,
    "memory_id": "memory-1",
    "thread_id": "thread-1",
    "user_id": "user-42",
}


def _events(status="success", error_code=None):
    """One turn as MemoRizz records it: a memory read, a model call that asks
    for a tool, the tool, a second model call and the turn's result."""
    common = {"root_trace_id": ROOT, "agent_id": "agent-1", "thread_id": "thread-1"}
    rows = [
        {
            "trace_kind": "turn_start",
            "title": "Agent turn started",
            "span_id": ROOT,
            "timestamp": "2026-11-19T11:00:00.000Z",
            "status": "started",
            "trace_tags": {"experiment": "lme-v2"},
        },
        {
            "trace_kind": "memory_retrieval",
            "title": "Memory retrieval",
            "span_id": "memory-read:1",
            "parent_span_id": ROOT,
            "timestamp": "2026-11-19T11:00:00.100Z",
            "status": "success",
            "duration_ms": 40.0,
            "memory_type": "episodic",
        },
        {
            "trace_kind": "model_call",
            "title": "Model call",
            "span_id": "model-1",
            "parent_span_id": ROOT,
            "timestamp": "2026-11-19T11:00:00.200Z",
            "status": "started",
            "model": "gpt-6-luna",
            "provider": "OpenAI",
        },
        {
            "trace_kind": "model_result",
            "title": "Model result",
            "span_id": "model-1",
            "parent_span_id": ROOT,
            "timestamp": "2026-11-19T11:00:01.200Z",
            "status": "success",
            "success": True,
            "duration_ms": 1000.0,
            "model": "gpt-6-luna",
            "provider": "OpenAI",
            "input_tokens": 40,
            "output_tokens": 8,
            "total_tokens": 48,
            "cached_tokens": 10,
            "finish_reason": "tool_calls",
            "request_id": "req-1",
            "prompt_cache_key": "fingerprint-not-for-export",
        },
        {
            "trace_kind": "tool_call",
            "title": "Tool Call: add",
            "content": '{"a": 2, "b": 3}',
            "span_id": "tool-1",
            "parent_span_id": ROOT,
            "timestamp": "2026-11-19T11:00:01.300Z",
            "status": "started",
            "tool_name": "add",
            "logical_tool_name": "add",
        },
        {
            "trace_kind": "tool_result",
            "title": "Tool Result: add",
            "content": "5",
            "span_id": "tool-1",
            "parent_span_id": ROOT,
            "timestamp": "2026-11-19T11:00:01.400Z",
            "status": "success",
            "success": True,
            "duration_ms": 100.0,
            "tool_name": "add",
            "logical_tool_name": "add",
        },
        {
            "trace_kind": "model_result",
            "title": "Model result",
            "span_id": "model-2",
            "parent_span_id": ROOT,
            "timestamp": "2026-11-19T11:00:02.000Z",
            "status": "success",
            "duration_ms": 500.0,
            "model": "gpt-6-luna",
            "provider": "OpenAI",
            "input_tokens": 60,
            "output_tokens": 6,
            "total_tokens": 66,
        },
        {
            "trace_kind": "turn_result",
            "title": "Agent turn completed",
            "span_id": ROOT,
            "timestamp": "2026-11-19T11:00:02.100Z",
            "status": status,
            "success": status == "success",
            **({"error_code": error_code} if error_code else {}),
        },
    ]
    return [{**common, **row} for row in rows]


class _Model:
    def get_config(self):
        return {"model": "gpt-4o-mini", "provider": "openai"}

    def get_last_usage(self):
        return {"prompt_tokens": 100, "completion_tokens": 10}

    def generate(self, messages, tools=None):
        return "answer"


class _Recording(ObservabilityProvider):
    name = "recording"
    label = "Recording"

    def __init__(self, config=None):
        super().__init__(config)
        self.turns = []

    def export_turn(self, trace_context, events, *, agent_name=None):
        self.turns.append((dict(trace_context), list(events), agent_name))
        return "sent"


class _Broken(ObservabilityProvider):
    name = "broken"

    def export_turn(self, trace_context, events, *, agent_name=None):
        raise RuntimeError("provider is down")


@pytest.fixture()
def store(tmp_path):
    instance = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield instance
    instance.close()


@pytest.fixture(autouse=True)
def _no_sentry_environment(monkeypatch):
    for name in ("MEMORIZZ_SENTRY_DSN", "SENTRY_DSN", "SENTRY_ORG", "SENTRY_SPOTLIGHT"):
        monkeypatch.delenv(name, raising=False)
    yield
    sentry_provider.reset_clients()


def _agent(store=None, **settings):
    agent = MemAgent(
        model=_Model(),
        memory_provider=store,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
        name="Ada",
        context_policy={"progressive_tool_disclosure": False},
        **settings,
    )
    agent._schedule_conversation_embedding_backfill = lambda rows: None
    return agent


def _bundles(store):
    return [
        row
        for row in store.list_all(MemoryType.SHARED_MEMORY)
        if row.get("record_type") == "observability_trace_bundle"
    ]


# ---- settings --------------------------------------------------------------


def test_the_default_and_true_both_mean_memorizz_keeps_the_traces():
    assert observability_settings() == (True, None, None)
    assert observability_settings(True) == (True, None, None)
    assert observability_settings(True, "memorizz") == (True, None, None)
    assert observability_settings(False) == (False, None, None)


def test_a_provider_is_named_by_either_argument_but_not_both():
    assert observability_settings(True, "sentry") == (
        True,
        {"provider": "sentry"},
        None,
    )
    assert observability_settings("sentry") == (True, {"provider": "sentry"}, None)
    enabled, config, _ = observability_settings(
        None, {"provider": "Sentry", "organization": "acme", "include_content": True}
    )
    assert enabled is True
    assert config == {"provider": "sentry", "org": "acme", "include_content": True}
    # Switched off, the provider is kept for when it is switched back on.
    assert observability_settings(False, "sentry") == (
        False,
        {"provider": "sentry"},
        None,
    )
    with pytest.raises(ValueError, match="once"):
        observability_settings("sentry", "memorizz")


def test_settings_refuse_credentials_unknown_providers_and_bad_values():
    with pytest.raises(ValueError, match="credentials"):
        ObservabilityConfig.from_value({"provider": "sentry", "dsn": "https://k@h/1"})
    with pytest.raises(ValueError, match="Unknown observability provider"):
        ObservabilityConfig.from_value("pigeon")
    with pytest.raises(ValueError, match="between 0 and 1"):
        ObservabilityConfig.from_value({"provider": "sentry", "sample_rate": 3})
    with pytest.raises(ValueError, match="https://"):
        ObservabilityConfig.from_value({"provider": "sentry", "url": "javascript:x"})
    half = ObservabilityConfig.from_value({"provider": "sentry", "sample_rate": "0.5"})
    assert half.to_dict() == {"provider": "sentry", "sample_rate": 0.5}


def test_every_provider_reports_whether_it_is_ready(monkeypatch):
    rows = {row["provider"]: row for row in observability_status()}
    assert set(rows) >= {"memorizz", "sentry"}
    assert rows["memorizz"]["ready"] is True and rows["memorizz"]["external"] is False
    assert rows["sentry"]["ready"] is False
    assert "SENTRY_DSN" in rows["sentry"]["missing"]


# ---- the agent --------------------------------------------------------------


def test_an_agent_is_traced_in_memorizz_unless_told_otherwise(store):
    agent = _agent(store)
    assert agent.observability is True and agent.observability_config is None
    assert isinstance(agent.observability_provider, MemorizzObservability)
    agent.run("question", memory_id="m1")
    assert len(_bundles(store)) == 1
    status = agent.observability_status()
    assert (status["provider"], status["enabled"], status["ready"]) == (
        "memorizz",
        True,
        True,
    )
    assert agent.observability_url() == f"/traces?agent_id={agent.agent_id}"


def test_observability_off_leaves_no_trace_anywhere(store):
    provider = _Recording()
    agent = _agent(store, observability=False, observability_provider=provider)
    assert agent.run("question", memory_id="m1") == "answer"
    assert _bundles(store) == []
    assert provider.turns == []
    assert agent.observability_status()["enabled"] is False

    agent.with_observability(True)
    agent.run("again", memory_id="m1")
    assert len(_bundles(store)) == 1 and len(provider.turns) == 1


def test_a_provider_receives_each_turn_and_memorizz_keeps_its_copy(store):
    provider = _Recording()
    agent = _agent(store, observability=True, observability_provider=provider)
    agent.run("question", memory_id="m1", thread_id="t1", user_id="u1")

    assert len(_bundles(store)) == 1
    ((context, events, agent_name),) = provider.turns
    assert agent_name == "Ada"
    assert context["root_trace_id"] == agent.get_trace_context()["root_trace_id"]
    assert (context["memory_id"], context["thread_id"], context["user_id"]) == (
        "m1",
        "t1",
        "u1",
    )
    kinds = [event["trace_kind"] for event in events]
    assert kinds[0] == "turn_start" and kinds[-1] == "turn_result"
    assert "model_result" in kinds


def test_a_provider_still_receives_turns_without_a_memory_provider():
    provider = _Recording()
    agent = _agent(None, observability_provider=provider)
    agent.run("question")
    assert len(provider.turns) == 1


def test_a_failing_provider_never_fails_the_turn(store, caplog):
    agent = _agent(store, observability_provider=_Broken())
    with caplog.at_level(logging.WARNING):
        assert agent.run("question", memory_id="m1") == "answer"
    assert "Sending the trace to broken failed (RuntimeError)" in caplog.text
    assert "provider is down" not in caplog.text
    assert len(_bundles(store)) == 1


def test_the_builder_sets_observability(store):
    builder = MemAgentBuilder().with_observability("sentry", org="acme")
    assert builder._observability is True
    assert builder._observability_provider == {"provider": "sentry", "org": "acme"}
    off = MemAgentBuilder().with_observability(enabled=False)
    assert off._observability is False


# ---- saved with the agent -------------------------------------------------


def test_observability_is_saved_and_restored(store):
    agent = _agent(
        store,
        observability=False,
        observability_provider={"provider": "sentry", "org": "acme"},
    )
    agent.save()
    stored = store.retrieve_memagent(agent.agent_id)
    assert stored.observability is False
    assert stored.observability_config == {"provider": "sentry", "org": "acme"}

    settings = persistence._saved_settings(stored)
    assert settings["observability"] is False
    assert settings["observability_provider"] == {"provider": "sentry", "org": "acme"}


def test_older_records_and_unknown_providers_load_as_traced_in_memorizz(caplog):
    settings = persistence._saved_settings(MemAgentModel())
    assert settings["observability"] is True
    assert settings["observability_provider"] is None

    stale = MemAgentModel(observability_config={"provider": "gone-away"})
    with caplog.at_level(logging.WARNING):
        assert persistence._saved_settings(stale)["observability_provider"] is None
    assert "Ignoring a stored observability configuration" in caplog.text


def test_a_refresh_applies_a_change_made_elsewhere(store):
    agent = _agent(store)
    agent.save()
    stored = store.retrieve_memagent(agent.agent_id)
    stored.observability = False
    stored.observability_config = {"provider": "sentry"}
    store.store_memagent(stored)

    agent.refresh()
    assert agent.observability is False
    assert agent.observability_config == {"provider": "sentry"}
    assert isinstance(agent.observability_provider, SentryObservability)


# ---- Sentry: the events -----------------------------------------------------


def test_a_turn_becomes_one_sentry_transaction_with_agent_spans():
    built = build_sentry_events(CONTEXT, _events(), agent_name="Ada")
    transaction, error = built["transaction"], built["error"]
    assert error is None
    trace = transaction["contexts"]["trace"]

    assert transaction["type"] == "transaction"
    assert transaction["transaction"] == "invoke_agent Ada"
    assert trace["trace_id"] == ROOT.replace("-", "") == sentry_trace_id(ROOT)
    assert trace["op"] == "gen_ai.invoke_agent" and trace["status"] == "ok"
    assert trace["data"]["gen_ai.agent.name"] == "Ada"
    assert trace["data"]["gen_ai.conversation.id"] == "thread-1"
    assert trace["data"]["gen_ai.usage.input_tokens"] == 100
    assert trace["data"]["gen_ai.usage.output_tokens"] == 14
    assert trace["data"]["gen_ai.usage.total_tokens"] == 114
    assert transaction["tags"]["experiment"] == "lme-v2"
    assert transaction["tags"]["memorizz.agent_id"] == "agent-1"
    assert (
        transaction["timestamp"] - transaction["start_timestamp"]
    ).total_seconds() == 2.1

    spans = {span["description"]: span for span in transaction["spans"]}
    assert [span["op"] for span in transaction["spans"]] == [
        "memorizz.memory_retrieval",
        "gen_ai.chat",
        "gen_ai.execute_tool",
        "gen_ai.chat",
    ]
    assert all(span["trace_id"] == trace["trace_id"] for span in transaction["spans"])
    assert all(
        span["parent_span_id"] == trace["span_id"] for span in transaction["spans"]
    )

    chat = transaction["spans"][1]
    assert chat["description"] == "chat gpt-6-luna"
    assert (chat["timestamp"] - chat["start_timestamp"]).total_seconds() == 1.0
    assert chat["data"]["gen_ai.operation.name"] == "chat"
    assert chat["data"]["gen_ai.request.model"] == "gpt-6-luna"
    assert chat["data"]["gen_ai.provider.name"] == "openai"
    assert chat["data"]["gen_ai.usage.input_tokens"] == 40
    assert chat["data"]["gen_ai.usage.cache_read.input_tokens"] == 10
    assert chat["data"]["gen_ai.response.finish_reasons"] == '["tool_calls"]'
    assert chat["data"]["gen_ai.response.id"] == "req-1"

    tool = spans["execute_tool add"]
    assert tool["data"]["gen_ai.tool.name"] == "add"
    assert (tool["timestamp"] - tool["start_timestamp"]).total_seconds() == 0.1

    # A step recorded once, at its end, runs backwards by its duration.
    memory = spans["Memory retrieval (episodic)"]
    assert (memory["timestamp"] - memory["start_timestamp"]).total_seconds() == 0.04


def test_content_and_identity_stay_home_unless_asked_for():
    quiet = build_sentry_events(CONTEXT, _events(), agent_name="Ada")
    sent = json.dumps(quiet, default=str)
    assert '\\"a\\": 2' not in sent  # the tool's arguments
    assert "user-42" not in sent
    assert "fingerprint-not-for-export" not in sent
    assert "user" not in quiet["transaction"]

    loud = build_sentry_events(
        CONTEXT, _events(), agent_name="Ada", include_content=True
    )
    tool = next(
        span for span in loud["transaction"]["spans"] if span["op"].endswith("tool")
    )
    assert tool["data"]["gen_ai.tool.call.arguments"] == '{"a": 2, "b": 3}'
    assert tool["data"]["gen_ai.tool.call.result"] == "5"
    assert loud["transaction"]["user"] == {"id": "user-42"}
    assert "fingerprint-not-for-export" not in json.dumps(loud, default=str)


def test_a_failed_turn_is_also_an_error_on_the_same_trace():
    built = build_sentry_events(
        CONTEXT, _events("error", "provider_timeout"), agent_name="Ada"
    )
    transaction, error = built["transaction"], built["error"]
    assert transaction["contexts"]["trace"]["status"] == "internal_error"
    assert error["level"] == "error"
    assert error["message"] == "Ada: agent turn failed (provider_timeout)"
    assert error["contexts"]["trace"]["trace_id"] == ROOT.replace("-", "")
    assert error["tags"]["memorizz.error_code"] == "provider_timeout"
    assert error["fingerprint"] == [
        "memorizz-agent-turn",
        "agent-1",
        "provider_timeout",
    ]

    # A turn somebody stopped is not an error.
    stopped = build_sentry_events(CONTEXT, _events("cancelled"), agent_name="Ada")
    assert stopped["error"] is None
    assert stopped["transaction"]["contexts"]["trace"]["status"] == "cancelled"


# ---- Sentry: readiness and links -----------------------------------------


def test_sentry_links_follow_the_organisation_the_dsn_and_the_setting(monkeypatch):
    assert SentryObservability().dashboard_url() == (
        "https://sentry.io/orgredirect/organizations/:orgslug/explore/traces/"
    )
    monkeypatch.setenv("SENTRY_DSN", "https://key@o1.ingest.us.sentry.io/4242")
    assert SentryObservability({"org": "acme"}).dashboard_url() == (
        "https://acme.sentry.io/explore/traces/?project=4242"
    )
    monkeypatch.setenv("SENTRY_ORG", "from-env")
    assert SentryObservability().dashboard_url() == (
        "https://from-env.sentry.io/explore/traces/?project=4242"
    )
    monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.example.com:9000/7")
    assert SentryObservability({"org": "ops"}).dashboard_url() == (
        "https://sentry.example.com:9000/organizations/ops/explore/traces/?project=7"
    )
    pinned = SentryObservability({"url": "https://acme.sentry.io/dashboard/9/"})
    assert pinned.dashboard_url() == "https://acme.sentry.io/dashboard/9/"


def test_sentry_without_a_dsn_sends_nothing_and_says_so_once(store, caplog):
    agent = _agent(store, observability="sentry")
    assert agent.observability_status()["ready"] is False
    with caplog.at_level(logging.WARNING):
        assert agent.run("one", memory_id="m1") == "answer"
        assert agent.run("two", memory_id="m1") == "answer"
    assert caplog.text.count("Sentry observability is on but cannot send") == 1
    assert len(_bundles(store)) == 2


# ---- Sentry: end to end through the real SDK -------------------------------


class _Ingest(BaseHTTPRequestHandler):
    """A stand-in for Sentry's ingest endpoint: keeps what it is sent."""

    items: list = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        lines = [line for line in body.split(b"\n") if line.strip()]
        for header, payload in zip(lines[1::2], lines[2::2]):
            type(self).items.append(
                {
                    "path": self.path,
                    "auth": self.headers.get("X-Sentry-Auth", ""),
                    "type": json.loads(header).get("type"),
                    "payload": json.loads(payload),
                }
            )
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture()
def sentry_ingest(monkeypatch):
    pytest.importorskip("sentry_sdk")
    _Ingest.items = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Ingest)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(
        "SENTRY_DSN", f"http://publickey@127.0.0.1:{server.server_address[1]}/4242"
    )
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    yield _Ingest.items
    sentry_provider.reset_clients()
    server.shutdown()
    server.server_close()


def _tool_calling_model():
    from memorizz.llms.openai import OpenAI

    model = OpenAI.__new__(OpenAI)
    model.model = "fixture-model"
    model.base_url = None
    model._request_options = {}
    model._prompt_cache_key = model._prompt_cache_retention = None
    replies = iter(
        [
            NS(
                id="req-1",
                choices=[
                    NS(
                        finish_reason="tool_calls",
                        message=NS(
                            content=None,
                            tool_calls=[
                                NS(
                                    id="call_1",
                                    type="function",
                                    function=NS(
                                        name="add", arguments='{"a": 2, "b": 3}'
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=NS(prompt_tokens=40, completion_tokens=8, total_tokens=48),
            ),
            NS(
                id="req-2",
                choices=[
                    NS(
                        finish_reason="stop",
                        message=NS(content="The answer is 5.", tool_calls=None),
                    )
                ],
                usage=NS(prompt_tokens=60, completion_tokens=6, total_tokens=66),
            ),
        ]
    )
    model._create_chat_completion = lambda kwargs: next(replies)
    return model


def test_a_turn_reaches_sentry_through_the_real_sdk(store, sentry_ingest):
    import sentry_sdk

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = MemAgent(
        model=_tool_calling_model(),
        memory_provider=store,
        tools=[add],
        name="Ada",
        streaming=False,
        auto_register=False,
        observability=True,
        observability_provider={"provider": "sentry", "org": "acme"},
    )
    agent._schedule_conversation_embedding_backfill = lambda rows: None
    assert agent.observability_status()["ready"] is True
    assert agent.run("what is 2+3?", trace_tags={"experiment": "e2e"}).startswith(
        "The answer is 5"
    )
    agent.flush_observability(5)

    (item,) = [row for row in sentry_ingest if row["type"] == "transaction"]
    assert item["path"] == "/api/4242/envelope/"
    assert "sentry_key=publickey" in item["auth"]
    transaction = item["payload"]
    root = agent.get_trace_context()["root_trace_id"]
    assert transaction["contexts"]["trace"]["trace_id"] == root.replace("-", "")
    assert transaction["transaction"] == "invoke_agent Ada"
    assert transaction["tags"]["experiment"] == "e2e"
    assert transaction["contexts"]["trace"]["data"]["gen_ai.usage.total_tokens"] == 114
    ops = [span["op"] for span in transaction["spans"]]
    assert ops.count("gen_ai.chat") == 2 and ops.count("gen_ai.execute_tool") == 1
    tool = next(s for s in transaction["spans"] if s["op"] == "gen_ai.execute_tool")
    assert tool["description"] == "execute_tool add"
    assert "gen_ai.tool.call.arguments" not in tool["data"]

    # MemoRizz kept its own copy, and the host's Sentry was never initialised.
    assert len(_bundles(store)) == 1
    assert sentry_sdk.get_client().is_active() is False


# ---- Sentry's local development server ----------------------------------


def test_the_local_server_makes_sentry_ready_without_a_dsn(monkeypatch):
    assert sentry_provider.local_server_url() is None
    monkeypatch.setenv("SENTRY_SPOTLIGHT", "0")
    assert SentryObservability().status()["ready"] is False
    monkeypatch.setenv("SENTRY_SPOTLIGHT", "1")
    assert sentry_provider.local_server_url() == "http://localhost:8969/stream"

    monkeypatch.setenv("SENTRY_SPOTLIGHT", "http://localhost:9000/stream")
    provider = SentryObservability({"org": "acme"})
    status = provider.status()
    assert (status["ready"], status["local_only"], status["label"]) == (
        True,
        True,
        "Sentry (local)",
    )
    assert status["local"] == "http://localhost:9000/stream"
    # Nothing is on sentry.io: the links are the local page, and no trace page.
    assert provider.dashboard_url() == (
        "https://local.sentry.dev/?url=http%3A%2F%2Flocalhost%3A9000%2Fstream"
    )
    assert provider.trace_url(ROOT) is None
    assert provider.export_mark() == "sentry-local"

    # With a DSN as well, turns go to both and the links are Sentry's.
    monkeypatch.setenv("SENTRY_DSN", "https://key@o1.ingest.us.sentry.io/4242")
    assert provider.status()["local_only"] is False
    assert provider.export_mark() == "sentry"
    assert provider.dashboard_url().startswith("https://acme.sentry.io/")


def test_a_turn_reaches_the_local_server_without_a_dsn(store, monkeypatch):
    pytest.importorskip("sentry_sdk")
    _Ingest.items = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Ingest)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(
        "SENTRY_SPOTLIGHT", f"http://127.0.0.1:{server.server_address[1]}/stream"
    )
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    try:
        agent = _agent(store, observability_provider="sentry")
        assert agent.observability_status()["local_only"] is True
        assert agent.run("question", memory_id="m1") == "answer"
        agent.flush_observability(5)
    finally:
        sentry_provider.reset_clients()
        server.shutdown()
        server.server_close()

    (item,) = [row for row in _Ingest.items if row["type"] == "transaction"]
    assert item["path"] == "/stream"
    assert item["payload"]["transaction"] == "invoke_agent Ada"
    root = agent.get_trace_context()["root_trace_id"]
    assert item["payload"]["contexts"]["trace"]["trace_id"] == root.replace("-", "")
    # MemoRizz's copy says the turn went to the local server, not to Sentry,
    # so the Traces page offers no link to a page that does not exist.
    (bundle,) = _bundles(store)
    marks = [e.get("trace_export") for e in read_payload(bundle)["events"]]
    assert "sentry-local" in marks and "sentry" not in marks


def test_create_observability_provider_names_the_class():
    assert isinstance(create_observability_provider(None), MemorizzObservability)
    assert isinstance(create_observability_provider("sentry"), SentryObservability)
    custom = _Recording({"org": "acme"})
    assert create_observability_provider(custom) is custom
    assert custom.config.to_dict() == {"provider": "recording", "org": "acme"}

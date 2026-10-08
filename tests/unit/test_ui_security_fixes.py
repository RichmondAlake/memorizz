# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Regression tests for UI security and availability fixes.

Each class pins one defect: cross-origin form submissions on every
state-changing route, knowledge-base deletion ownership, blocking calls
kept off the event loop, the Twilio webhook staying reachable behind UI
auth / read-only mode, skill lifecycle error reporting, playground thread
membership and paging, the knowledge-base upload cap, and the playground
stream reading the agent record once per message.

No real provider daemons, Docker, HuggingFace or Twilio are contacted, and
``MEMORIZZ_HOME`` always points at a temporary directory.
"""

import asyncio
import sys
import threading
import time
import types
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.enums.memory_type import MemoryType  # noqa: E402
from memorizz.memagent import MemAgent  # noqa: E402
from memorizz.memagent.models import MemAgentModel  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

EVIL = "https://evil.example"
TOKEN = "burner-operator-token"
VALIDATOR = "memorizz.channels.whatsapp.webhook_validator.validate_twilio_signature"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Keep every test away from ~/.memorizz and from any real UI auth config."""
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    for key in (
        "MEMORIZZ_UI_AUTH_TOKEN",
        "MEMORIZZ_UI_AUTH_ACCOUNTS",
        "MEMORIZZ_UI_READ_ONLY",
        "MEMORIZZ_UI_SESSION_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _connected(provider, provider_type="filesystem"):
    return patch.dict(
        state._state,
        {"provider": provider, "provider_type": provider_type, "connection_info": {}},
    )


def _client():
    return TestClient(create_app(), follow_redirects=False)


class StubAgent:
    """What ``MemAgent.load`` returns here: enough for the knowledge-base routes."""

    def __init__(self, agent_id, knowledge_base_ids, provider):
        self.agent_id = agent_id
        self.knowledge_base_ids = list(knowledge_base_ids)
        self.memory_provider = provider
        self.saves = 0

    def save(self):
        self.saves += 1
        self.memory_provider.agents[self.agent_id] = self


class StubProvider:
    def __init__(self, agents=(), kb_entries=()):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.kb_entries = list(kb_entries)
        self.retrieve_calls = 0

    def retrieve_memagent(self, agent_id):
        self.retrieve_calls += 1
        return self.agents.get(agent_id)

    def list_memagents(self):
        return list(self.agents.values())

    def list_all(self, memory_store_type=None):
        return list(self.kb_entries)

    def delete_by_id(self, id, memory_store_type=None):
        before = len(self.kb_entries)
        self.kb_entries = [entry for entry in self.kb_entries if entry.get("_id") != id]
        return len(self.kb_entries) < before

    def update_memagent(self, agent):
        self.agents[agent.agent_id] = agent
        return True


def _fake_load(provider):
    def load(agent_id, memory_provider=None, **_overrides):
        try:
            return provider.agents[agent_id]
        except KeyError:
            raise ValueError(f"MemAgent {agent_id} not found")

    return patch.object(MemAgent, "load", side_effect=load)


# ---------------------------------------------------------------------------
# Finding 1: cross-origin state changes
# ---------------------------------------------------------------------------


class TestCrossOriginSubmissions:
    @pytest.mark.unit
    @pytest.mark.parametrize("path", ["/settings", "/connect"])
    def test_cross_origin_post_is_denied(self, home, path):
        with _connected(StubProvider()):
            resp = _client().post(path, data={}, headers={"Origin": EVIL})
        assert resp.status_code == 403
        assert "cross-origin" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_referer_is_checked_when_origin_is_absent(self, home):
        with _connected(StubProvider()):
            resp = _client().post(
                "/settings", data={}, headers={"Referer": f"{EVIL}/attack.html"}
            )
        assert resp.status_code == 403

    @pytest.mark.unit
    def test_same_origin_and_headerless_posts_pass(self, home):
        client = _client()
        with _connected(StubProvider()):
            same = client.post(
                "/settings", data={}, headers={"Origin": "http://testserver"}
            )
            bare = client.post("/settings", data={})
        assert same.status_code == 200
        assert bare.status_code == 200

    @pytest.mark.unit
    def test_safe_methods_are_not_affected(self, home):
        with _connected(StubProvider()):
            resp = _client().get("/api/status", headers={"Origin": EVIL})
        assert resp.status_code == 200

    @pytest.mark.unit
    def test_bearer_token_api_clients_are_exempt(self, home, monkeypatch):
        monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", TOKEN)
        client = _client()
        with _connected(StubProvider()):
            api = client.post(
                "/settings",
                data={},
                headers={"Origin": EVIL, "Authorization": f"Bearer {TOKEN}"},
            )
            login = client.post("/login", data={"access_token": TOKEN, "next": "/"})
            browser = client.post("/settings", data={}, headers={"Origin": EVIL})
        assert api.status_code == 200
        assert login.status_code == 303
        # A cookie session is exactly what a cross-site form would ride on.
        assert browser.status_code == 403


# ---------------------------------------------------------------------------
# Finding 2: knowledge-base deletion ownership
# ---------------------------------------------------------------------------


class TestKnowledgeBaseDelete:
    def _world(self):
        provider = StubProvider(
            kb_entries=[
                {"_id": "e1", "knowledge_base_id": "kb-1", "content": "x"},
                {"_id": "e2", "knowledge_base_id": "kb-1", "content": "y"},
            ]
        )
        provider.agents = {
            "agent-a": StubAgent("agent-a", ["kb-1"], provider),
            "agent-b": StubAgent("agent-b", ["kb-1"], provider),
        }
        return provider

    @pytest.mark.unit
    def test_shared_knowledge_survives_one_agent_detaching(self, home):
        provider = self._world()
        with _connected(provider), _fake_load(provider):
            resp = _client().delete("/api/agents/agent-a/knowledge-base/kb-1")
        assert resp.status_code == 409
        body = resp.json()
        assert body["deleted"] is False
        assert body["detached"] is True
        assert len(provider.kb_entries) == 2
        assert "kb-1" not in provider.agents["agent-a"].knowledge_base_ids
        assert "kb-1" in provider.agents["agent-b"].knowledge_base_ids

    @pytest.mark.unit
    def test_unattached_knowledge_is_not_deleted(self, home):
        provider = self._world()
        provider.agents["agent-a"].knowledge_base_ids = []
        with _connected(provider), _fake_load(provider):
            resp = _client().delete("/api/agents/agent-a/knowledge-base/kb-1")
        assert resp.status_code == 404
        assert resp.json()["deleted"] is False
        assert len(provider.kb_entries) == 2

    @pytest.mark.unit
    def test_last_owner_deletes_the_entries(self, home):
        provider = self._world()
        provider.agents["agent-a"].knowledge_base_ids = []
        with _connected(provider), _fake_load(provider):
            resp = _client().delete("/api/agents/agent-b/knowledge-base/kb-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True and body["detached"] is True
        assert body["deleted"] is True
        assert provider.kb_entries == []


# ---------------------------------------------------------------------------
# Finding 3: blocking calls off the event loop
# ---------------------------------------------------------------------------


def _recording(calls, result):
    """A stand-in for a blocking call that notes which kind of thread ran it."""

    def call(*_args, **_kwargs):
        try:
            asyncio.get_running_loop()
            calls.append("event-loop")
        except RuntimeError:
            calls.append("worker")
        return result

    return call


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class TestBlockingCallsLeaveTheEventLoop:
    @pytest.mark.unit
    def test_ollama_pull_runs_off_the_loop(self, home):
        calls = []
        with patch(
            "urllib.request.urlopen",
            _recording(calls, _FakeResp(b'{"status": "success"}')),
        ):
            resp = _client().post("/api/ollama/pull", data={"name": "llama3.1:8b"})
        assert resp.status_code == 200
        assert calls == ["worker"]

    @pytest.mark.unit
    def test_slow_ollama_pull_does_not_stall_other_requests(self, home):
        import httpx

        app = create_app()
        release = threading.Event()

        def slow_urlopen(*_args, **_kwargs):
            release.wait(5)
            return _FakeResp(b'{"status": "success"}')

        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                started = time.monotonic()
                pull = asyncio.create_task(
                    client.post("/api/ollama/pull", data={"name": "m"})
                )
                await asyncio.sleep(0.1)
                status = await client.get("/api/status")
                elapsed = time.monotonic() - started
                release.set()
                return status.status_code, elapsed, (await pull).status_code

        with patch("urllib.request.urlopen", slow_urlopen):
            code, elapsed, pull_code = asyncio.run(scenario())
        assert code == 200 and pull_code == 200
        assert elapsed < 2, f"status request waited {elapsed:.1f}s behind the pull"

    @pytest.mark.unit
    def test_huggingface_pull_runs_off_the_loop(self, home):
        calls = []
        hf = types.ModuleType("huggingface_hub")
        hf.snapshot_download = _recording(calls, "/tmp/hf/models--gpt2")
        with patch.dict(sys.modules, {"huggingface_hub": hf}):
            resp = _client().post("/api/huggingface/pull", data={"repo_id": "gpt2"})
        assert resp.status_code == 200
        assert calls == ["worker"]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "path,data,helper,container_state",
        [
            ("/api/docker/oracle/runtime/start", {}, "start_runtime", None),
            (
                "/api/docker/oracle/start",
                {"container_name": "memorizz_oracle"},
                "start_container",
                "stopped",
            ),
            (
                "/api/docker/oracle/create",
                {
                    "oracle_user": "u",
                    "oracle_password": "p",
                    "oracle_dsn": "localhost:1521/FREEPDB1",
                },
                "create_container",
                "absent",
            ),
        ],
    )
    def test_docker_oracle_actions_run_off_the_loop(
        self, home, path, data, helper, container_state
    ):
        calls = []
        mod = "memorizz.ui.docker_oracle"
        with (
            patch(f"{mod}.docker_available", return_value=True),
            patch(f"{mod}.get_container_state", return_value=container_state),
            patch(f"{mod}.parse_port_from_dsn", return_value=1521),
            patch(f"{mod}.{helper}", _recording(calls, (True, "ok"))),
        ):
            resp = _client().post(path, data=data)
        assert resp.status_code == 200, resp.text
        assert calls == ["worker"]

    @pytest.mark.unit
    def test_evalground_dataset_download_runs_off_the_loop(self, home, tmp_path):
        calls = []
        provider = FileSystemProvider(
            FileSystemConfig(root_path=tmp_path / "fs", lazy_vector_indexes=True)
        )
        mod = "memorizz.ui.routers.evalground"
        with (
            _connected(provider),
            patch(f"{mod}._build_dataset_status", return_value=([], ["oracle"])),
            patch(
                f"{mod}._run_longmemeval_dataset_download",
                _recording(calls, (True, "Dataset download complete.")),
            ),
        ):
            resp = _client().post("/evalground/datasets")
        assert resp.status_code == 200
        assert calls == ["worker"]

    @pytest.mark.unit
    def test_knowledge_base_ingest_runs_off_the_loop(self, home):
        from memorizz.long_term.semantic.knowledge_base import KnowledgeBase

        calls = []
        provider = StubProvider(
            kb_entries=[{"_id": "e1", "knowledge_base_id": "kb-new"}]
        )
        provider.agents = {"agent-a": StubAgent("agent-a", [], provider)}
        with (
            _connected(provider),
            _fake_load(provider),
            patch.object(KnowledgeBase, "ingest_file", _recording(calls, "kb-new")),
        ):
            resp = _client().post(
                "/api/agents/agent-a/knowledge-base/ingest",
                files={"files": ("notes.txt", b"hello world", "text/plain")},
            )
        assert resp.status_code == 200, resp.text
        assert calls == ["worker"]
        assert resp.json()["results"][0]["knowledge_base_id"] == "kb-new"


# ---------------------------------------------------------------------------
# Finding 4: Twilio webhook behind UI auth / read-only
# ---------------------------------------------------------------------------


class TestTwilioWebhook:
    FORM = {"From": "whatsapp:+15550001111", "Body": "hello", "MessageSid": "SM123"}

    @pytest.mark.unit
    def test_signed_webhook_reaches_the_handler(self, home, monkeypatch):
        monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", TOKEN)
        monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "twilio-burner")
        queued = []
        with (
            patch(VALIDATOR, return_value=True) as validator,
            patch(
                "memorizz.channels.whatsapp.queue.enqueue_message",
                side_effect=queued.append,
            ),
        ):
            resp = _client().post(
                "/webhook/whatsapp/incoming",
                data=self.FORM,
                headers={"X-Twilio-Signature": "sig"},
            )
        assert resp.status_code == 200
        assert resp.json() == {"status": "queued"}
        assert validator.call_count == 1
        assert queued == [
            {"from": self.FORM["From"], "body": "hello", "message_sid": "SM123"}
        ]

    @pytest.mark.unit
    def test_unsigned_webhook_is_still_rejected_by_the_handler(self, home, monkeypatch):
        monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", TOKEN)
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "twilio-burner")
        with patch("memorizz.channels.whatsapp.queue.enqueue_message") as enqueue:
            resp = _client().post(
                "/webhook/whatsapp/incoming",
                data=self.FORM,
                headers={"X-Twilio-Signature": "forged"},
            )
        assert resp.status_code == 403
        assert enqueue.call_count == 0


# ---------------------------------------------------------------------------
# Finding 5: skill lifecycle error reporting and by-id lookup
# ---------------------------------------------------------------------------


class TestSkillLifecycleErrors:
    def _provider(self, tmp_path):
        return FileSystemProvider(
            FileSystemConfig(root_path=tmp_path / "cl", lazy_vector_indexes=True)
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("action", ["activate", "demote"])
    def test_unknown_skill_redirects_with_an_error(self, home, tmp_path, action):
        with _connected(self._provider(tmp_path)):
            resp = _client().post(f"/continual-learning/skills/missing-skill/{action}")
        assert resp.status_code == 303
        location = resp.headers["location"]
        assert location.startswith("/memory/skills?error=")
        assert "missing-skill" in unquote(location)

    @pytest.mark.unit
    def test_failed_activation_reports_the_cause(self, home, tmp_path):
        from memorizz.memagent.managers.continual_learning_manager import (
            ContinualLearningManager,
        )

        provider = self._provider(tmp_path)
        provider.store(
            {
                "skill_id": "s-1",
                "agent_id": "agent-1",
                "name": "n",
                "content": "c",
                "status": "shadow",
                "embedding": [0.0],
            },
            memory_store_type=MemoryType.SKILLBOX,
        )
        with (
            _connected(provider),
            patch.object(
                ContinualLearningManager,
                "activate_skill",
                side_effect=RuntimeError("engine exploded"),
            ),
        ):
            resp = _client().post("/continual-learning/skills/s-1/activate")
        assert resp.status_code == 303
        assert "engine exploded" in unquote(resp.headers["location"])

    @pytest.mark.unit
    def test_skill_lookup_is_by_id_not_a_full_scan(self):
        from memorizz.ui.routers.continual_learning import _find_skill

        class ByIdProvider:
            def __init__(self):
                self.queries = []

            def retrieve_by_query(self, query, memory_store_type=None, limit=1, **_):
                self.queries.append((query, memory_store_type, limit))
                if query.get("skill_id") == "s-1":
                    return [{"skill_id": "s-1", "agent_id": "agent-1"}]
                return []

            def list_all(self, *_args, **_kwargs):
                raise AssertionError("skill lookup must not scan the whole skillbox")

        provider = ByIdProvider()
        with _connected(provider):
            assert _find_skill("s-1")["agent_id"] == "agent-1"
            assert _find_skill("nope") is None
        assert provider.queries[0][0] == {"skill_id": "s-1"}
        assert provider.queries[0][1] == MemoryType.SKILLBOX


# ---------------------------------------------------------------------------
# Finding 6: playground thread membership and paging
# ---------------------------------------------------------------------------


class TestPlaygroundThread:
    def _provider(self, tmp_path):
        provider = FileSystemProvider(
            FileSystemConfig(root_path=tmp_path / "pg", lazy_vector_indexes=True)
        )
        provider.store_memagent(
            MemAgentModel(
                agent_id="agent-a", name="A", instruction="x", memory_ids=["m-mine"]
            )
        )
        provider.store_memagent(
            MemAgentModel(
                agent_id="agent-b", name="B", instruction="x", memory_ids=["m-theirs"]
            )
        )
        for memory_id, count in (("m-mine", 5), ("m-theirs", 1)):
            for index in range(count):
                provider.store(
                    {
                        "memory_id": memory_id,
                        "role": "user" if index % 2 == 0 else "assistant",
                        "content": f"{memory_id} message {index}",
                        "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
                        "embedding": [0.0],
                    },
                    memory_store_type=MemoryType.CONVERSATION_MEMORY,
                )
        return provider

    @pytest.mark.unit
    def test_thread_of_another_agent_is_not_served(self, home, tmp_path):
        with _connected(self._provider(tmp_path)):
            resp = _client().get(
                "/agents/agent-a/playground/thread", params={"memory_id": "m-theirs"}
            )
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_thread_history_is_paged(self, home, tmp_path):
        provider = self._provider(tmp_path)
        original = provider.retrieve_conversation_history_ordered_by_timestamp
        limits = []

        def spy(*args, **kwargs):
            limits.append(kwargs.get("limit"))
            return original(*args, **kwargs)

        with (
            _connected(provider),
            patch.object(
                provider,
                "retrieve_conversation_history_ordered_by_timestamp",
                side_effect=spy,
            ),
        ):
            resp = _client().get(
                "/agents/agent-a/playground/thread",
                params={"memory_id": "m-mine", "limit": 2},
            )
        assert resp.status_code == 200, resp.text
        assert limits and None not in limits
        assert 2 in limits
        assert resp.json()["message_count"] == 2


# ---------------------------------------------------------------------------
# Finding 9: knowledge-base upload cap
# ---------------------------------------------------------------------------


class TestUploadCap:
    @pytest.mark.unit
    def test_cap_defaults_to_fifty_mib(self):
        from memorizz.ui.routers import knowledge_base as kb_router

        assert kb_router.MAX_UPLOAD_BYTES == 50 * 1024 * 1024

    @pytest.mark.unit
    def test_oversized_upload_is_rejected_before_ingest(self, home, monkeypatch):
        from memorizz.long_term.semantic.knowledge_base import KnowledgeBase
        from memorizz.ui.routers import knowledge_base as kb_router

        monkeypatch.setattr(kb_router, "MAX_UPLOAD_BYTES", 1024)
        provider = StubProvider()
        provider.agents = {"agent-a": StubAgent("agent-a", [], provider)}
        with (
            _connected(provider),
            _fake_load(provider),
            patch.object(KnowledgeBase, "ingest_file") as ingest,
        ):
            resp = _client().post(
                "/api/agents/agent-a/knowledge-base/ingest",
                files=[("files", ("big.txt", b"x" * 2048, "text/plain"))],
            )
        assert resp.status_code == 400
        result = resp.json()["results"][0]
        assert result["ok"] is False
        assert "exceeds" in result["error"]
        ingest.assert_not_called()


# ---------------------------------------------------------------------------
# Finding 10: the stream reads the agent record once
# ---------------------------------------------------------------------------


class TestPlaygroundStreamPrepare:
    @pytest.mark.unit
    def test_agent_record_is_read_once_per_message(self, home):
        record = SimpleNamespace(
            agent_id="agent-s",
            llm_config={"provider": "openai", "model": "gpt-4.1-mini"},
            memory_ids=[],
            memory_types=None,
            application_mode=None,
            sandbox_provider=None,
            browser_control=None,
            internet_access_provider=None,
            internet_access_config=None,
            persona=None,
        )
        provider = StubProvider(agents=[record])
        runtime = SimpleNamespace(
            agent_id="agent-s",
            memory_ids=[],
            model=None,
            tool_manager=None,
            _llm_init_error=None,
            has_sandbox=lambda: True,
            has_browser_control=lambda: True,
            has_internet_access=lambda: True,
        )
        prepared = {}

        class FakeStream:
            def poll(self, timeout=None):
                raise StopIteration

            def close(self):
                return None

        def fake_event_stream(agent, query, *, _prepare=None, **_kwargs):
            session = SimpleNamespace(stack=ExitStack(), emit=lambda *a, **k: None)
            with session.stack:
                prepared["result"] = _prepare(session)
            return FakeStream()

        with (
            _connected(provider),
            patch.object(MemAgent, "load", return_value=runtime),
            patch("memorizz.streaming.agent_event_stream", fake_event_stream),
            patch(
                "memorizz.llms.llm_factory.create_llm_provider", return_value=object()
            ),
        ):
            resp = _client().post(
                "/agents/agent-s/playground/stream",
                data={"query": "hi", "llm_model": "gpt-4.1"},
            )
        assert resp.status_code == 200
        assert prepared["result"][0] is runtime
        assert provider.retrieve_calls == 1

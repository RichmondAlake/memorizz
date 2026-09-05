# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Whole-app smoke tests for the local UI.

Every user-facing page must render without a server error, both while
disconnected and against a real (filesystem) provider. This is the safety
net that lets app.py be refactored — route extractions and template edits
that break a page fail here instead of in someone's browser.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memagent.models import MemAgentModel  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.routers.traces import (  # noqa: E402
    _RUNTIME_TRACE_AGENT_ID,
    _expand_trace_bundle,
    _thread_row_key,
)

# Pages that must render for a DISCONNECTED session (200, or a redirect to
# /connect — never a 5xx).
PAGES = [
    "/",
    "/connect",
    "/dashboard",
    "/settings",
    "/playground",
    "/agents",
    "/agents/new",
    "/automations",
    "/traces",
    "/observability",
    "/vercel-skills",
    "/evalground",
    "/learning-control-plane",
    "/harnesses",
    "/api/status",
    "/api/personas",
    "/api/persona-presets",
    "/evalground/runs/active",
]

MEMORY_PAGES = [
    "personas",
    "toolbox",
    "skills",
    "conversations",
    "workflows",
    "knowledge-base",
    "short-term",
    "entity",
    "summaries",
    "shared",
    "cache",
]


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def fs_provider(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "ui-smoke", lazy_vector_indexes=True
        )
    )
    yield provider


@pytest.mark.unit
@pytest.mark.parametrize("path", PAGES)
def test_page_renders_disconnected(client, path):
    with patch.dict(
        state._state,
        {"provider": None, "provider_type": None, "connection_info": {}},
    ):
        resp = client.get(path)
    assert resp.status_code < 500, f"{path} -> {resp.status_code}"


@pytest.mark.unit
def test_connect_page_uses_env_secrets_without_rendering_them(client):
    secret_uri = "mongodb://operator:super-secret@db.example.test/memorizz"
    secret_password = "oracle-super-secret"
    with (
        patch.dict(
            os.environ,
            {
                "MONGODB_URI": secret_uri,
                "MONGODB_DB_NAME": "production-memory",
                "ORACLE_PASSWORD": secret_password,
            },
            clear=False,
        ),
        patch.dict(
            state._state,
            {"provider": None, "provider_type": None, "connection_info": {}},
        ),
    ):
        response = client.get("/connect")

    assert response.status_code == 200
    assert secret_uri not in response.text
    assert secret_password not in response.text
    assert "Configured via MONGODB_URI" in response.text
    assert '<option value="mongodb" selected>' in response.text


@pytest.mark.unit
def test_ui_token_auth_read_only_mode_and_trace_audit(tmp_path, fs_provider):
    audit_path = Path(tmp_path) / "trace-audit.jsonl"
    with (
        patch.dict(
            os.environ,
            {
                "MEMORIZZ_UI_AUTH_TOKEN": "burner-operator-token",
                "MEMORIZZ_UI_READ_ONLY": "true",
                "MEMORIZZ_UI_AUDIT_LOG": str(audit_path),
                "MEMORIZZ_UI_TRACE_CONTENT_MODE": "redacted",
            },
            clear=False,
        ),
        patch.dict(
            state._state,
            {
                "provider": fs_provider,
                "provider_type": "filesystem",
                "connection_info": {"root_path": str(fs_provider.root_path)},
                "read_only": True,
            },
        ),
    ):
        secure_client = TestClient(create_app(), follow_redirects=False)
        unauthenticated = secure_client.get("/traces", headers={"accept": "text/html"})
        invalid = secure_client.post(
            "/login",
            data={"access_token": "wrong", "next": "/traces"},
        )
        authenticated = secure_client.post(
            "/login",
            data={"access_token": "burner-operator-token", "next": "/traces"},
        )
        trace_page = secure_client.get("/traces")
        blocked_mutation = secure_client.post("/settings", data={})

    assert unauthenticated.status_code == 303
    assert unauthenticated.headers["location"].startswith("/login")
    assert invalid.status_code == 401
    assert "burner-operator-token" not in invalid.text
    assert authenticated.status_code == 303
    assert "httponly" in authenticated.headers["set-cookie"].lower()
    assert trace_page.status_code == 200
    assert "Read-only provider" in trace_page.text
    assert blocked_mutation.status_code == 403
    assert audit_path.exists()
    audit_text = audit_path.read_text(encoding="utf-8")
    assert "trace_dashboard" in audit_text
    assert "burner-operator-token" not in audit_text


@pytest.mark.unit
@pytest.mark.parametrize("path", PAGES)
def test_page_renders_connected(client, fs_provider, path):
    with patch.dict(
        state._state,
        {
            "provider": fs_provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(fs_provider.root_path)},
        },
    ):
        resp = client.get(path)
    assert resp.status_code < 500, f"{path} -> {resp.status_code}"


@pytest.mark.unit
@pytest.mark.parametrize("slug", MEMORY_PAGES)
def test_memory_pages_render(client, fs_provider, slug):
    with patch.dict(
        state._state,
        {
            "provider": fs_provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(fs_provider.root_path)},
        },
    ):
        resp = client.get(f"/memory/{slug}")
    assert resp.status_code == 200, f"/memory/{slug} -> {resp.status_code}"


@pytest.mark.unit
def test_unknown_memory_type_is_404(client, fs_provider):
    with patch.dict(
        state._state,
        {"provider": fs_provider, "provider_type": "filesystem"},
    ):
        resp = client.get("/memory/nonexistent")
    assert resp.status_code == 404


@pytest.mark.unit
def test_trace_bundle_expands_into_timeline_events():
    events = _expand_trace_bundle(
        {
            "role": "tool",
            "timestamp": "2026-08-12T10:30:00+00:00",
            "content": (
                '{"type":"trace_bundle","version":1,"events":['
                '{"trace_kind":"tool_call","title":"Tool Call · inventory_status",'
                '"content":"{\\"region\\":\\"London\\"}","trace_id":"call-1",'
                '"logical_tool_name":"inventory_status"},'
                '{"trace_kind":"tool_result","title":"Tool Result · inventory_status",'
                '"content":"{\\"units\\":21}","trace_id":"result:call-1",'
                '"logical_tool_name":"inventory_status","success":true,'
                '"outcome":"fallback","outcome_reason_code":"primary_timeout",'
                '"fallback_provider":"replica","duration_ms":18.5}'
                "]}"
            ),
        },
        memory_id="erpa-course",
        thread_id="thread-1",
    )

    assert events is not None
    assert [event["kind"] for event in events] == ["tool_call", "tool_result"]
    assert events[0]["title"] == "Tool Call · inventory_status"
    assert events[1]["trace_id"] == "result:call-1"
    assert events[1]["logical_tool_name"] == "inventory_status"
    assert events[1]["success"] is True
    assert events[1]["outcome"] == "fallback"
    assert events[1]["outcome_reason_code"] == "primary_timeout"
    assert events[1]["fallback_provider"] == "replica"
    assert events[1]["duration_ms"] == 18.5
    assert all(event["thread_id"] == "thread-1" for event in events)


@pytest.mark.unit
def test_trace_bundle_expands_context_and_cache_provenance_metadata():
    events = _expand_trace_bundle(
        {
            "role": "tool",
            "timestamp": "2026-08-20T10:30:00+00:00",
            "content": json.dumps(
                {
                    "type": "trace_bundle",
                    "version": 2,
                    "events": [
                        {
                            "trace_kind": "context_provenance",
                            "title": "Request context provenance",
                            "content": '{"grounding_status":"ready"}',
                            "canonical_page_type": "analysis",
                            "canonical_page_id": "analysis-1",
                            "canonical_title_fingerprint": "sha256:abc123",
                            "thread_binding_status": "matched",
                            "ownership_verified": True,
                            "grounding_status": "ready",
                            "grounding_source": "stored_text",
                        },
                        {
                            "trace_kind": "cache_decision",
                            "title": "Semantic cache · miss",
                            "content": '{"cache_decision":"miss"}',
                            "cache_decision": "miss",
                            "cache_enabled": True,
                        },
                    ],
                }
            ),
        },
        memory_id="primary-user-1",
        thread_id="analysis_analysis-1",
    )

    assert events is not None
    assert events[0]["canonical_page_id"] == "analysis-1"
    assert events[0]["canonical_title_fingerprint"] == "sha256:abc123"
    assert events[0]["grounding_source"] == "stored_text"
    assert events[1]["cache_decision"] == "miss"


@pytest.mark.unit
def test_trace_thread_key_keeps_threads_in_one_memory_distinct():
    assert _thread_row_key("thread-a", "memory-1") != _thread_row_key(
        "thread-b", "memory-1"
    )


class _RuntimeOnlyTraceProvider:
    def __init__(self):
        self.conversations = [
            {
                "memory_id": "primary_user-1",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "role": "user",
                "content": "Remember this exact article URL",
                "timestamp": 1,
            },
            {
                "memory_id": "primary_user-1",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "role": "assistant",
                "content": "I will keep it in this thread.",
                "timestamp": 2,
            },
        ]
        self.tool_logs = [
            {
                "agent_id": "ephemeral-process-agent",
                "memory_id": "primary_user-1",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "tool_name": "ingest_url",
                "success": True,
                "outcome": "fallback",
                "outcome_details": {
                    "status": "fallback",
                    "ok": True,
                    "fallback_used": True,
                    "reason_code": "primary_timeout",
                    "fallback_provider": "replica",
                },
                "timestamp": 3,
            }
        ]

    def list_memagents(self):
        return []

    def retrieve_memagent(self, _agent_id):
        return None

    def list_all(self, memory_type, **_kwargs):
        value = getattr(memory_type, "value", str(memory_type))
        if value == "conversation_memory":
            return list(self.conversations)
        if value == "tool_log":
            return list(self.tool_logs)
        return []


@pytest.mark.unit
def test_traces_discovers_unregistered_runtime_and_renders_its_timeline(client):
    provider = _RuntimeOnlyTraceProvider()
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "mongodb",
            "connection_info": {"db_name": "trace-test"},
        },
    ):
        overview = client.get("/traces")
        timeline = client.get(
            "/traces",
            params={
                "agent_id": _RUNTIME_TRACE_AGENT_ID,
                "thread_id": "thread-1",
                "thread_memory_id": "primary_user-1",
            },
        )
        analysis = client.get(
            "/traces/analysis.json",
            params={
                "agent_id": _RUNTIME_TRACE_AGENT_ID,
                "thread_id": "thread-1",
                "thread_memory_id": "primary_user-1",
            },
        )

    assert overview.status_code == 200
    assert "Unregistered runtime traces" in overview.text
    assert "thread-1" in overview.text
    assert timeline.status_code == 200
    assert "Trace Insights" in timeline.text
    assert "Remember this exact article URL" not in timeline.text
    assert "Reveal trace content" in timeline.text
    assert "Execution Log · ingest_url" in timeline.text
    assert "Completed via fallback" in timeline.text
    assert "Reason primary timeout" in timeline.text
    assert analysis.status_code == 200
    assert analysis.json()["read_only"] is True
    assert any(
        row["id"] == "missing_trace_identity" for row in analysis.json()["insights"]
    )


@pytest.mark.unit
def test_favorite_update_preserves_advanced_agent_configuration(client, fs_provider):
    agent_id = "advanced-policy-agent"
    fs_provider.store_memagent(
        MemAgentModel(
            agent_id=agent_id,
            retrieval_policy={
                "conversation_scope": "thread",
                "knowledge_base_scope": "namespace",
                "knowledge_base_namespaces": ["agents"],
            },
            tool_result_policy={"offload_above_chars": 4096},
            context_policy={"progressive_tool_disclosure": True},
            skill_retrieval=True,
            whatsapp_enabled=True,
            whatsapp_config={"phone_number_id": "configured"},
        )
    )

    with patch.dict(
        state._state,
        {
            "provider": fs_provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(fs_provider.root_path)},
        },
    ):
        response = client.post(
            f"/agents/{agent_id}/favorite",
            data={"is_favorite": "true"},
        )

    assert response.status_code == 302
    stored = fs_provider.retrieve_memagent(agent_id)
    assert stored.is_favorite is True
    assert stored.retrieval_policy["conversation_scope"] == "thread"
    assert stored.tool_result_policy == {"offload_above_chars": 4096}
    assert stored.context_policy == {"progressive_tool_disclosure": True}
    assert stored.skill_retrieval is True
    assert stored.whatsapp_enabled is True
    assert stored.whatsapp_config == {"phone_number_id": "configured"}

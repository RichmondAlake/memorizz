# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Whole-app smoke tests for the local UI.

Every user-facing page must render without a server error, both while
disconnected and against a real (filesystem) provider. This is the safety
net that lets app.py be refactored — route extractions and template edits
that break a page fail here instead of in someone's browser.
"""

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
                '"content":"{\\"region\\":\\"London\\"}","trace_id":"call-1"},'
                '{"trace_kind":"tool_result","title":"Tool Result · inventory_status",'
                '"content":"{\\"units\\":21}","trace_id":"result:call-1"}'
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
    assert all(event["thread_id"] == "thread-1" for event in events)


@pytest.mark.unit
def test_trace_thread_key_keeps_threads_in_one_memory_distinct():
    assert _thread_row_key("thread-a", "memory-1") != _thread_row_key(
        "thread-b", "memory-1"
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

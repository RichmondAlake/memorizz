"""Codex and Claude Code sessions in Observability, the Dashboard and Usage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.approval import SQLiteApprovalStore  # noqa: E402
from memorizz.episodic_capture import record_turn  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.metaharness import MetaHarness, SQLiteHarnessRunStore  # noqa: E402
from memorizz.metaharness.agent_sessions import (  # noqa: E402
    record_session,
    session_run_id,
)
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from tests.unit.test_agent_sessions import (  # noqa: E402
    claude_transcript,
    codex_rollout,
)

pytestmark = pytest.mark.unit

MEMORY = "project-shop-abc123"


@pytest.fixture
def console(tmp_path: Path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[],
        run_store=store,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    # Both agents' turns, in the same project memory, as the plugin saves them.
    with patch("memorizz.embeddings.get_embedding", side_effect=RuntimeError):
        record_turn(
            provider,
            memory_id=MEMORY,
            thread_id="codex-codex-session-1",
            user_message="Fix the failing test",
            assistant_message="Fixed: int() not round().",
            agent_id="codex",
        )
        record_turn(
            provider,
            memory_id=MEMORY,
            thread_id="claude-code-claude-session-1",
            user_message="Why int() and not round()?",
            assistant_message="Points are per whole dollar.",
            agent_id="claude-code",
        )
    # Only the Codex session has a recorded run.
    record_session(codex_rollout(tmp_path / "rollout.jsonl", tmp_path), store=store)
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "meta_harness": meta,
        "meta_harness_provider": provider,
        "read_only": False,
    }
    try:
        with patch.dict(state._state, values):
            yield TestClient(create_app(), follow_redirects=False), tmp_path, store
    finally:
        meta.close()
        provider.close()


def _thread(agent: str, session: str) -> str:
    return (
        f"/traces?agent_id=__memorizz_plugin_{agent}__"
        f"&thread_id={quote(f'{agent}-{session}')}&thread_memory_id={MEMORY}"
    )


def test_each_coding_agent_has_its_own_trace_source(console):
    client, _tmp, _store = console
    page = client.get("/traces")
    assert page.status_code == 200, page.text[:500]
    assert "Codex sessions" in page.text and "Claude Code sessions" in page.text
    assert "Unregistered runtime traces" not in page.text


def test_a_codex_session_links_to_its_run_and_gets_no_memagent_advice(console):
    client, _tmp, _store = console
    page = client.get(_thread("codex", "codex-session-1"))
    assert page.status_code == 200, page.text[:500]
    run_id = session_run_id("codex", "codex-session-1")
    assert f'href="/harnesses?run={run_id}"' in page.text
    assert "A Codex session, saved by the MemoRizz plugin" in page.text
    assert "Trace ownership is not durable" not in page.text
    assert "Model and token economics are not traceable" not in page.text
    # Codex and Claude Code share the memory ID; each source keeps its own turns.
    timeline = page.text.split("Trace Timeline")[-1]
    assert "codex-codex-session-1" in timeline
    assert "claude-code-claude-session-1" not in timeline


def test_a_session_without_a_run_says_how_to_add_it(console):
    client, _tmp, _store = console
    page = client.get(_thread("claude-code", "claude-session-1"))
    assert page.status_code == 200
    assert "No run was recorded for it" in page.text
    assert "memorizz plugin import-session" in page.text
    assert "/harnesses?run=" not in page.text


def test_the_dashboard_and_usage_report_coding_agent_sessions(console):
    client, tmp_path, store = console
    record_session(
        claude_transcript(tmp_path / "claude.jsonl", tmp_path / "shop"), store=store
    )
    from memorizz.ui.dashboard import coding_agent_sessions

    rows = coding_agent_sessions()
    assert [row["label"] for row in rows] == ["Claude Code", "Codex"]
    claude = rows[0]
    assert claude["sessions"] == 1 and claude["priced"] == 1
    assert float(claude["cost_usd"]) == pytest.approx(0.0123)
    # Claude Code reports cache reads apart from input; both are tokens in.
    assert claude["input_tokens"] == 20 + 2_000 + 4_000
    codex = rows[1]
    assert codex["input_tokens"] == 20_000 and codex["cached_tokens"] == 12_000
    by_model = coding_agent_sessions(by_model=True)
    assert {row["model"] for row in by_model} == {
        "claude-haiku-4-5-20251001",
        "gpt-5.1-codex",
    }

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Coding-agent sessions" in dashboard.text and "Claude Code" in dashboard.text

    usage = client.get("/traces/usage")
    assert usage.status_code == 200, usage.text[:500]
    assert "Coding-agent sessions" in usage.text and "gpt-5.1-codex" in usage.text
    narrowed = client.get("/traces/usage?agent_id=someone")
    assert "Coding-agent sessions" not in narrowed.text


def test_the_harnesses_page_opens_a_run_named_in_the_link(console):
    client, _tmp, _store = console
    page = client.get("/harnesses")
    assert "new URLSearchParams(location.search).get('run')" in page.text

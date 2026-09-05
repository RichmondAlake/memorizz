"""Operator routes use real filesystem persistence, tenant scope and audit files."""

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    ObservabilityRecorder,
    ObservabilityStore,
    TraceContext,
)
from memorizz.observability.privacy import pseudonym
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.security import ReadOnlyProviderProxy


@pytest.fixture
def operator_ui(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "false")
    monkeypatch.setenv("MEMORIZZ_UI_TRACE_CONTENT_MODE", "redacted")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    accounts = {
        role: {
            "token": role + "-test-token-123456",
            "role": role,
            "application_id": "app-A",
            "user_id": "alice",
        }
        for role in ("viewer", "analyst", "operator", "admin")
    }
    accounts["operator"].pop("user_id")
    accounts["bob"] = {
        **accounts["viewer"],
        "token": "bob-test-token-123456",
        "user_id": "bob",
    }
    accounts["anonymous"] = {
        **accounts["viewer"],
        "token": "anon-test-token-123456",
        "user_id": None,
    }
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", json.dumps(accounts))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    for app, user in (
        ("app-A", "alice"),
        ("app-A", "bob"),
        ("app-A", None),
        ("app-B", "alice"),
    ):
        context = TraceContext(
            application_id=app,
            user_id=user,
            agent_id="agent",
            thread_id="thread",
            root_trace_id="root",
            run_id="run",
            turn_id="turn",
        )
        recorder = ObservabilityRecorder(provider, context, strict=True)
        recorder.record_event(
            "lookup",
            input_refs=[
                {"resource_type": "analysis", "ref": "source", "version": "v1"}
            ],
        )
        ObservabilityStore(provider).record_trace_bundle(
            trace_context=context.to_carrier(),
            events=[
                {
                    "event_id": "preview",
                    "trace_kind": "model_result",
                    "content": f"PRIVATE-{app}-{user} secret@example.com",
                    "timestamp": "2026-09-04T10:00:00Z",
                }
            ],
        )
    resolver = Mock(return_value={"user_id": "alice", "application_id": "app-A"})
    artifact = Mock(
        return_value={
            "exists": True,
            "ownership_verified": True,
            "version": "v1",
            "title": "PRIVATE TITLE",
        }
    )
    authorizer = Mock(return_value=True)
    app = create_app(
        identity_resolver=resolver,
        artifact_resolver=artifact,
        resource_authorizer=authorizer,
    )
    with patch.dict(
        state._state, provider=provider, provider_type="filesystem", connection_info={}
    ):
        yield {
            "client": TestClient(app),
            "provider": provider,
            "accounts": accounts,
            "resolver": resolver,
            "artifact": artifact,
            "authorizer": authorizer,
            "audit": tmp_path / "audit.jsonl",
            "app": app,
        }
    provider.close()


def headers(ui, role="viewer"):
    return {"Authorization": "Bearer " + ui["accounts"][role]["token"]}


def selection(**extra):
    return {"agent_id": "agent", "root_trace_id": "root", **extra}


def test_tenant_scope_is_conjunctive_and_alternate_routes_are_closed(operator_ui):
    ui = operator_ui
    response = ui["client"].get("/traces/find.json?q=root", headers=headers(ui))
    assert response.status_code == 200
    assert len(response.json()["items"]) == 2
    assert {e["user_id"] for e in response.json()["items"]} == {pseudonym("alice")}
    assert {e["application_id"] for e in response.json()["items"]} == {"app-A"}
    assert "PRIVATE" not in response.text
    for route in (
        "/traces/find.json?user_id=bob",
        "/traces?application_id=app-B",
        "/memory/shared",
        "/agents",
        "/settings",
        "/evalground",
    ):
        assert ui["client"].get(route, headers=headers(ui)).status_code == 403
    anonymous = ui["client"].get(
        "/traces/find.json?q=root", headers=headers(ui, "anonymous")
    )
    assert len(anonymous.json()["items"]) == 2
    assert all(e.get("user_id") is None for e in anonymous.json()["items"])


def test_concurrent_principals_do_not_share_context(operator_ui):
    ui = operator_ui

    def query(role):
        response = ui["client"].get("/traces/find.json", headers=headers(ui, role))
        return {e["user_id"] for e in response.json()["items"]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(query, ["viewer", "bob"] * 8))
    assert results == [{pseudonym("alice")}, {pseudonym("bob")}] * 8


def test_account_resolution_is_transient_permission_gated_and_audited(operator_ui):
    ui = operator_ui
    endpoint = "/traces/account/resolve"
    body = {"email": "person@example.com"}
    assert (
        ui["client"].post(endpoint, json=body, headers=headers(ui)).status_code == 403
    )
    response = ui["client"].post(endpoint, json=body, headers=headers(ui, "operator"))
    assert response.status_code == 200
    assert response.json() == {"application_id": "app-A", "user_id": "alice"}
    ui["resolver"].assert_called_once_with(
        "person@example.com", {"application_id": "app-A"}
    )
    audit = ui["audit"].read_text()
    assert "account_resolution_completed" in audit and "trace_access_denied" in audit
    assert "person@example.com" not in audit
    assert "person@example.com" not in json.dumps(
        ui["provider"].list_all(MemoryType.SHARED_MEMORY)
    )
    ui["resolver"].return_value = {"user_id": "alice", "application_id": "app-B"}
    assert (
        ui["client"]
        .post(endpoint, json=body, headers=headers(ui, "operator"))
        .status_code
        == 403
    )


def test_privileged_hooks_fail_closed_when_audit_is_unavailable(operator_ui):
    ui = operator_ui
    with patch("memorizz.ui.routers.trace_tools.audit_trace_view", return_value=False):
        response = ui["client"].post(
            "/traces/account/resolve",
            json={"email": "person@example.com"},
            headers=headers(ui, "operator"),
        )
    assert response.status_code == 503
    ui["resolver"].assert_not_called()


def test_invalid_inputs_and_hook_errors_do_not_echo_secrets(operator_ui):
    ui = operator_ui
    response = ui["client"].post(
        "/traces/account/resolve",
        json={"email": "person@example.com", "password": "secret"},
        headers=headers(ui, "operator"),
    )
    assert response.status_code == 422
    assert "person@example.com" not in response.text and "secret" not in response.text
    response = ui["client"].get(
        "/traces/find.json?q=person@example.com", headers=headers(ui)
    )
    assert response.status_code == 400 and "person@example.com" not in response.text
    ui["resolver"].side_effect = RuntimeError("person@example.com SECRET")
    response = ui["client"].post(
        "/traces/account/resolve",
        json={"email": "person@example.com"},
        headers=headers(ui, "operator"),
    )
    assert response.status_code == 503 and "SECRET" not in response.text
    response = ui["client"].get(
        "/traces/inspect.json?agent_id=person@example.com&root_trace_id=root",
        headers=headers(ui, "analyst"),
    )
    assert response.status_code == 400 and "person@example.com" not in response.text


def test_content_reveal_requires_privilege_scope_and_explicit_action(operator_ui):
    ui = operator_ui
    body = selection(event_id="preview")
    assert (
        ui["client"].post("/traces/reveal", json=body, headers=headers(ui)).status_code
        == 403
    )
    response = ui["client"].post(
        "/traces/reveal", json=body, headers=headers(ui, "admin")
    )
    assert response.status_code == 200
    assert (
        "PRIVATE-app-A-alice" in response.text
        and "secret@example.com" not in response.text
    )
    assert (
        ui["client"]
        .post(
            "/traces/reveal",
            json={**body, "user_id": "bob"},
            headers=headers(ui, "admin"),
        )
        .status_code
        == 403
    )
    assert "trace_content_revealed" in ui["audit"].read_text()
    page = ui["client"].get("/traces?agent_id=agent", headers=headers(ui, "admin"))
    assert page.status_code == 200 and "PRIVATE" not in page.text


def test_artifact_lookup_does_not_expose_unscoped_content(operator_ui):
    ui = operator_ui
    path = "/traces/artifact.json?resource_type=analysis&ref=source"
    assert ui["client"].get(path, headers=headers(ui)).status_code == 403
    response = ui["client"].get(path, headers=headers(ui, "analyst"))
    assert response.json() == {
        "exists": True,
        "ownership_verified": True,
        "version": "v1",
    }
    assert ui["artifact"].call_args.args[1] == {
        "application_id": "app-A",
        "user_id": "alice",
    }
    ui["artifact"].return_value = {"exists": True, "ownership_verified": False}
    assert ui["client"].get(path, headers=headers(ui, "analyst")).status_code == 404


def test_replay_is_a_frozen_authorized_inert_draft(operator_ui):
    ui = operator_ui
    response = ui["client"].post(
        "/traces/replays", json=selection(), headers=headers(ui, "analyst")
    )
    assert response.status_code == 200
    draft = response.json()
    assert draft["status"] == "draft" and draft["kind"] == "trace_replay"
    assert draft["resource_refs"][0]["version"] == "v1"
    assert draft["replay_policy"]["execution_enabled"] is False
    assert draft["replay_policy"]["network"] is False
    assert "PRIVATE" not in response.text
    assert len(ObservabilityStore(ui["provider"]).list_experiments()) == 1
    ui["authorizer"].return_value = False
    assert (
        ui["client"]
        .post("/traces/replays", json=selection(), headers=headers(ui, "analyst"))
        .status_code
        == 403
    )
    assert len(ObservabilityStore(ui["provider"]).list_experiments()) == 1


def test_readonly_blocks_nested_index_mutations_and_allows_replay_plan(
    operator_ui, monkeypatch
):
    ui = operator_ui
    ui["provider"].get_observability_index().initialize()
    proxy = ReadOnlyProviderProxy(ui["provider"])
    index = proxy.get_observability_index()
    for operation in (
        lambda: index.initialize(),
        lambda: index.put({}, []),
        lambda: index.write_bundle({}),
        lambda: index.retention(dry_run=False),
    ):
        with pytest.raises(PermissionError):
            operation()
    assert index.retention()["dry_run"]
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    client = TestClient(create_app(resource_authorizer=lambda *_: True))
    with patch.dict(state._state, provider=proxy):
        assert (
            client.post(
                "/traces/replays", json=selection(), headers=headers(ui, "analyst")
            ).status_code
            == 403
        )
        result = client.get(
            "/traces/replay-plan.json?agent_id=agent&root_trace_id=root",
            headers=headers(ui, "analyst"),
        )
        assert result.status_code == 200
    assert ObservabilityStore(ui["provider"]).list_experiments() == []


def test_csrf_and_signed_role_sessions(operator_ui):
    ui = operator_ui
    response = ui["client"].post(
        "/traces/replays",
        json=selection(),
        headers={**headers(ui, "analyst"), "Origin": "https://evil.invalid"},
    )
    assert response.status_code == 403
    login = ui["client"].post(
        "/login",
        data={"access_token": ui["accounts"]["viewer"]["token"]},
        follow_redirects=False,
    )
    assert login.status_code == 303
    assert ui["client"].get("/traces/find.json").status_code == 200
    assert ui["client"].post("/traces/replays", json=selection()).status_code == 403


def test_index_mode_serves_same_scoped_inspectors_and_metadata(
    operator_ui, monkeypatch
):
    from memorizz.observability import ObservabilityMaintenance

    ui = operator_ui
    maintenance = ObservabilityMaintenance(ui["provider"])
    maintenance.initialize()
    assert maintenance.backfill(since="2020-01-01", dry_run=False)["failed"] == 0
    assert maintenance.parity(user_id="alice", application_id="app-A")["passed"]
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_READ_PATH", "index")
    for route in (
        "/traces/find.json?q=root",
        "/traces/inspect.json?agent_id=agent&root_trace_id=root",
        "/traces?agent_id=agent",
        "/traces/health.json",
    ):
        response = ui["client"].get(route, headers=headers(ui, "analyst"))
        assert response.status_code == 200, response.text
        assert "PRIVATE" not in response.text

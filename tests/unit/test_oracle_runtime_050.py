"""Local Oracle runtime and bootstrap regressions for MemoRizz 0.5."""

from __future__ import annotations

from pathlib import Path

import pytest

from memorizz.enums import MemoryType
from memorizz.memory_provider.oracle.runtime import LocalOracleRuntime
from memorizz.ui import docker_oracle


@pytest.mark.unit
def test_local_oracle_runtime_respects_configured_container(monkeypatch):
    monkeypatch.setenv("ORACLE_USER", "memorizz_user")
    monkeypatch.setenv("ORACLE_PASSWORD", "not-printed")
    monkeypatch.setenv("ORACLE_DSN", "localhost:1522/FREEPDB1")
    monkeypatch.setenv("MEMORIZZ_ORACLE_CONTAINER", "existing-oracle-26ai")

    runtime = LocalOracleRuntime.from_env(provision_if_missing=True)

    assert runtime.container_name == "existing-oracle-26ai"
    assert runtime.provision_if_missing is True


@pytest.mark.unit
def test_oracle_from_env_honors_index_policy(monkeypatch):
    from memorizz.memory_provider.oracle.provider import OracleProvider

    monkeypatch.setenv("ORACLE_USER", "memorizz_user")
    monkeypatch.setenv("ORACLE_PASSWORD", "not-printed")
    monkeypatch.setenv("ORACLE_DSN", "localhost:1522/FREEPDB1")
    monkeypatch.setenv("MEMORIZZ_ORACLE_INDEX_POLICY", "selected")
    captured = {}

    def fake_init(self, config):
        captured["config"] = config

    monkeypatch.setattr(OracleProvider, "__init__", fake_init)
    OracleProvider.from_env(selected_vector_indexes=[])

    assert captured["config"].index_policy == "selected"


@pytest.mark.unit
def test_local_oracle_runtime_creates_the_requested_container(monkeypatch):
    runtime = LocalOracleRuntime(
        user="memorizz_user",
        password="not-printed",
        dsn="localhost:1522/FREEPDB1",
        container_name="course-oracle-26ai",
        provision_if_missing=True,
    )
    captured = {}
    monkeypatch.setattr(docker_oracle, "docker_available", lambda: True)
    monkeypatch.setattr(docker_oracle, "get_container_state", lambda _name: "absent")

    def create(username, password, port, *, name):
        captured.update(
            username=username,
            password=password,
            port=port,
            name=name,
        )
        return True, "healthy"

    monkeypatch.setattr(docker_oracle, "create_container", create)

    report = runtime.ensure_ready()

    assert report["ok"] is True
    assert report["action"] == "created"
    assert report["container"] == "course-oracle-26ai"
    assert captured == {
        "username": "memorizz_user",
        "password": "not-printed",
        "port": 1522,
        "name": "course-oracle-26ai",
    }


@pytest.mark.unit
def test_docker_creation_uses_custom_name_and_volume(monkeypatch):
    calls = []
    monkeypatch.setattr(docker_oracle, "get_container_state", lambda _name: "absent")

    def run(args, timeout=30):
        calls.append((list(args), timeout))
        return 0, "ok", ""

    monkeypatch.setattr(docker_oracle, "_run", run)
    monkeypatch.setattr(
        docker_oracle, "_wait_for_healthy", lambda name, _timeout: (True, name)
    )

    ok, _detail = docker_oracle.create_container(
        "memorizz_user", "not-printed", 1522, name="course-oracle-26ai"
    )

    assert ok is True
    docker_run = next(args for args, _timeout in calls if args[:2] == ["docker", "run"])
    assert docker_run[docker_run.index("--name") + 1] == "course-oracle-26ai"
    assert "1522:1521" in docker_run
    assert "course-oracle-26ai_data:/opt/oracle/oradata" in docker_run
    assert "--env-file" in docker_run
    assert not any("not-printed" in argument for argument in docker_run)
    assert "-e" not in docker_run


@pytest.mark.unit
def test_oracle_setup_examples_never_interpolate_the_database_password():
    source = (
        Path(__file__).parents[2]
        / "src"
        / "memorizz"
        / "memory_provider"
        / "oracle"
        / "setup.py"
    ).read_text(encoding="utf-8")

    assert 'print(f"  Password: {MEMORIZZ_PASSWORD}")' not in source
    assert "Password: <configured in ORACLE_PASSWORD; redacted>" in source
    assert 'password=os.environ["ORACLE_PASSWORD"]' in source
    assert "MyPassword123" not in source
    assert "SecurePass123" not in source


@pytest.mark.unit
def test_oracle_operational_scripts_have_no_default_or_output_passwords():
    project_root = Path(__file__).parents[2]
    scripts = [
        project_root / "src" / "memorizz" / "scripts" / "install_oracle.sh",
        project_root / "src" / "memorizz" / "scripts" / "teardown_oracle.sh",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in scripts)

    assert "MyPassword123" not in source
    assert "SecurePass123" not in source
    assert "Admin Password: $PASSWORD" not in source
    assert 'export ORACLE_ADMIN_PASSWORD="$PASSWORD"' not in source


@pytest.mark.unit
def test_oracle_vector_retrieval_applies_scope_before_top_k(monkeypatch):
    import memorizz.embeddings as embeddings
    from memorizz.memory_provider.oracle.provider import OracleProvider

    provider = object.__new__(OracleProvider)
    calls = []

    def vector_search(memory_type, embedding, **kwargs):
        calls.append((memory_type, embedding, kwargs))
        return []

    provider._vector_search = vector_search
    monkeypatch.setattr(embeddings, "get_embedding", lambda _query: [1.0, 0.0])

    provider.retrieve_by_query(
        "find policy",
        memory_type=MemoryType.KNOWLEDGE_BASE,
        memory_id="memory-1",
        user_id="alice",
        namespace="agents",
    )
    provider.retrieve_by_query(
        "what happened",
        memory_type=MemoryType.CONVERSATION_MEMORY,
        memory_id="memory-1",
        user_id="alice",
        thread_id="thread-a",
    )

    assert calls[0][0] == MemoryType.KNOWLEDGE_BASE
    assert calls[0][2]["filters"] == {"namespace": "agents"}
    assert calls[1][0] == MemoryType.CONVERSATION_MEMORY
    assert calls[1][2]["filters"] == {"thread_id": "thread-a"}


@pytest.mark.unit
def test_public_capabilities_export_is_a_stable_callable():
    import importlib

    import memorizz
    from memorizz import capabilities

    assert callable(capabilities)
    assert capabilities()["version"] == memorizz.__version__
    importlib.import_module("memorizz.capabilities")
    assert callable(memorizz.capabilities)
    assert memorizz.capabilities()["package"] == "memorizz"

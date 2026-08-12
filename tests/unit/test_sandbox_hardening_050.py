"""Regression tests for the production sandbox provider contracts in 0.5."""

from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz.sandbox.providers.e2b_provider import (
    E2BSandboxProvider,
    _sdk_compatibility_error,
)
from memorizz.sandbox.providers.graalpy_provider import GraalPySandboxProvider
from memorizz.ui.helpers import _build_graalpy_default_sandbox_config


@pytest.mark.unit
def test_e2b_requires_api_key_during_construction(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(ValueError, match="E2B_API_KEY"):
        E2BSandboxProvider()


@pytest.mark.unit
def test_e2b_v2_uses_one_stateful_session_and_normalizes_results(monkeypatch):
    class Files:
        def __init__(self):
            self.values = {}

        def write(self, path, content):
            self.values[path] = content

        def read(self, path):
            return {"text": self.values[path]}

    class FakeSandbox:
        creates = []

        def __init__(self):
            self.files = Files()
            self.killed = False
            self.run_calls = []

        @classmethod
        def create(
            cls,
            template=None,
            timeout=None,
            metadata=None,
            api_key=None,
            allow_internet_access=True,
            **options,
        ):
            instance = cls()
            cls.creates.append(
                {
                    "instance": instance,
                    "template": template,
                    "timeout": timeout,
                    "metadata": metadata,
                    "api_key": api_key,
                    "allow_internet_access": allow_internet_access,
                    "options": options,
                }
            )
            return instance

        def run_code(self, code, language=None, envs=None, timeout=None):
            self.run_calls.append((code, language, envs, timeout))
            return SimpleNamespace(
                logs=SimpleNamespace(
                    stdout=[{"line": "first"}, SimpleNamespace(text="second")],
                    stderr=[SimpleNamespace(value="warning")],
                ),
                results=[{"text": "42"}],
                error=None,
            )

        def kill(self):
            self.killed = True

    module = types.ModuleType("e2b_code_interpreter")
    module.Sandbox = FakeSandbox
    monkeypatch.setitem(sys.modules, "e2b_code_interpreter", module)

    with E2BSandboxProvider(
        api_key="e2b-test",
        template="memorizz-bounded",
        session_timeout=90,
        max_execution_timeout=10,
        allow_internet_access=False,
        cpu_count=2,
        memory_mb=1024,
    ) as provider:
        assert provider.write_file("/workspace/value.txt", "coherent") is True
        result = provider.execute_code("print(42)", timeout=99)
        assert provider.read_file("/workspace/value.txt") == "coherent"
        assert result.stdout == ["first", "second"]
        assert result.stderr == ["warning"]
        assert result.results == ["42"]
        assert result.metadata["provider"] == "e2b"
        assert result.metadata["stateful_session"] is True
        assert result.metadata["execution_timeout"] == 10
        assert result.metadata["egress_allowed"] is False
        assert result.metadata["cpu_count"] == 2
        assert result.metadata["memory_mb"] == 1024
        assert result.metadata["resource_policy_enforcement"] == "e2b_template"
        assert result.metadata["egress_policy_enforcement"] == "e2b_create"
        instance = FakeSandbox.creates[0]["instance"]
        assert instance.run_calls[0][-1] == 10
        assert len(FakeSandbox.creates) == 1

    assert instance.killed is True
    assert FakeSandbox.creates[0]["allow_internet_access"] is False
    # CPU and memory are enforced by the selected E2B template; they are not
    # leaked into the SDK's generic API options where current clients reject
    # unknown keys.
    assert FakeSandbox.creates[0]["options"] == {}


@pytest.mark.unit
def test_e2b_normalizes_legacy_dictionary_result_shapes():
    class LegacySession:
        def run_code(self, _code, **_kwargs):
            return {
                "stdout": "single stdout value",
                "stderr": [{"message": "legacy warning"}],
                "text": "legacy result",
                "error": None,
            }

    provider = E2BSandboxProvider(api_key="e2b-test")
    provider._sandbox = LegacySession()
    result = provider.execute_code("40 + 2")

    assert result.stdout == ["single stdout value"]
    assert result.stderr == ["legacy warning"]
    assert result.results == ["legacy result"]


@pytest.mark.unit
def test_e2b_compute_policy_requires_a_bounded_template():
    provider = E2BSandboxProvider(api_key="e2b-test", cpu_count=2, memory_mb=1024)
    assert "explicit E2B template" in provider.validate_configuration()


@pytest.mark.unit
def test_e2b_known_transport_incompatibility_has_an_actionable_error():
    assert "e2b>=2.26.0,<2.38.0" in _sdk_compatibility_error(
        {"e2b": "2.38.0", "e2b_code_interpreter": "2.9.0"}
    )
    assert (
        _sdk_compatibility_error({"e2b": "2.37.1", "e2b_code_interpreter": "2.9.0"})
        is None
    )


@pytest.mark.unit
def test_graalpy_subprocess_is_labeled_execution_provider_and_confines_files(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-inherited")
    provider = GraalPySandboxProvider(
        graalpy_path="/bin/echo",
        working_dir=str(tmp_path),
        max_memory_mb=64,
        max_cpu_seconds=0,
        max_processes=0,
        max_file_bytes=1,
    )
    try:
        config = provider.get_config()
        assert config["security_boundary"] == (
            "bounded_execution_provider_not_strong_sandbox"
        )
        assert config["allow_network"] is False
        assert "blocked" in config["network_policy"]
        assert config["max_memory_mb"] >= 128
        assert config["max_cpu_seconds"] >= 1
        assert config["max_processes"] >= 1
        assert config["max_file_bytes"] >= 1024

        assert provider.write_file("nested/value.txt", "safe") is True
        assert provider.read_file("nested/value.txt") == "safe"
        assert provider.write_file("../escape.txt", "no") is False
        assert provider.read_file("/etc/passwd") is None
        environment = provider._safe_environment(None)
        assert "OPENAI_API_KEY" not in environment
        assert Path(environment["HOME"]).resolve() == Path(provider.working_dir)
        with pytest.raises(ValueError, match="env_allowlist"):
            provider._safe_environment({"OPENAI_API_KEY": "leak"})
    finally:
        private_dir = Path(provider.working_dir)
        provider.close()
    assert not private_dir.exists()


@pytest.mark.unit
def test_ui_graalpy_defaults_to_network_denied_trusted_execution(monkeypatch):
    monkeypatch.delenv("MEMORIZZ_GRAALPY_MODE", raising=False)
    monkeypatch.delenv("MEMORIZZ_GRAALPY_INTERNET_ACCESS", raising=False)
    monkeypatch.delenv("GRAALPY_JAVA_WRAPPER_JAR", raising=False)

    config = _build_graalpy_default_sandbox_config()

    assert config["mode"] == "subprocess"
    assert config["allow_network"] is False

    monkeypatch.setenv("MEMORIZZ_GRAALPY_INTERNET_ACCESS", "1")
    assert _build_graalpy_default_sandbox_config()["allow_network"] is True

    monkeypatch.setenv("MEMORIZZ_GRAALPY_MODE", "java_wrapper")
    monkeypatch.setenv("GRAALPY_JAVA_WRAPPER_JAR", "/opt/memorizz/wrapper.jar")
    wrapper = _build_graalpy_default_sandbox_config()
    assert wrapper["mode"] == "java_wrapper"
    assert wrapper["sandbox_policy"] == "UNTRUSTED"
    assert wrapper["java_wrapper_jar"] == "/opt/memorizz/wrapper.jar"


@pytest.mark.unit
def test_graalpy_applies_host_limits_and_network_policy_fails_closed(
    tmp_path, monkeypatch
):
    resource = pytest.importorskip("resource")
    provider = GraalPySandboxProvider(
        graalpy_path="/bin/echo",
        working_dir=str(tmp_path),
        max_memory_mb=256,
        max_cpu_seconds=7,
        max_processes=3,
        max_file_bytes=4096,
    )
    calls = []
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda resource_id, limits: calls.append((resource_id, limits)),
    )
    try:
        provider._resource_limiter()()
        resource_ids = {resource_id for resource_id, _limits in calls}
        assert resource.RLIMIT_CPU in resource_ids
        assert resource.RLIMIT_AS in resource_ids
        assert resource.RLIMIT_FSIZE in resource_ids
        if hasattr(resource, "RLIMIT_NPROC"):
            assert resource.RLIMIT_NPROC in resource_ids

        monkeypatch.setattr("platform.system", lambda: "UnsupportedOS")
        with pytest.raises(RuntimeError, match="Network-denied"):
            provider._subprocess_command("print('never runs')")
    finally:
        provider.close()


@pytest.mark.unit
def test_graalpy_untrusted_mode_requires_wrapper_and_wrapper_is_shipped():
    provider = GraalPySandboxProvider(
        graalpy_path="/bin/echo",
        mode="java_wrapper",
        java_wrapper_jar=None,
    )
    try:
        assert "requires `java_wrapper_jar`" in provider.validate_configuration()
    finally:
        provider.close()

    source = (
        Path(__file__).parents[2]
        / "src"
        / "memorizz"
        / "sandbox"
        / "MemorizzGraalSandbox.java"
    ).read_text(encoding="utf-8")
    assert "SandboxPolicy.UNTRUSTED" in source
    assert "allowCreateProcess(false)" in source
    assert "EnvironmentAccess.NONE" in source
    assert "IOAccess.NONE" in source
    for mandatory_limit in (
        "sandbox.MaxCPUTime",
        "sandbox.MaxHeapMemory",
        "sandbox.MaxASTDepth",
        "sandbox.MaxStackFrames",
        "sandbox.MaxThreads",
        "sandbox.MaxOutputStreamSize",
        "sandbox.MaxErrorStreamSize",
    ):
        assert mandatory_limit in source
    assert ".in(InputStream.nullInputStream())" in source
    assert ".out(output)" in source
    assert ".err(error)" in source
    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    assert "MemorizzGraalSandbox.java" in pyproject


@pytest.mark.unit
def test_graalpy_untrusted_mode_validates_compiled_wrapper_identity(
    tmp_path, monkeypatch
):
    wrapper = tmp_path / "memorizz-graal-sandbox.jar"
    with zipfile.ZipFile(wrapper, "w") as archive:
        archive.writestr(
            "META-INF/MANIFEST.MF",
            "Manifest-Version: 1.0\n"
            "Main-Class: memorizz.sandbox.MemorizzGraalSandbox\n",
        )
        archive.writestr("memorizz/sandbox/MemorizzGraalSandbox.class", b"compiled")
    monkeypatch.setattr(
        "memorizz.sandbox.providers.graalpy_provider.shutil.which",
        lambda executable: "/usr/bin/java" if executable == "java" else None,
    )
    provider = GraalPySandboxProvider(
        mode="java_wrapper",
        java_wrapper_jar=str(wrapper),
        sandbox_policy="UNTRUSTED",
    )
    try:
        assert provider.validate_configuration() is None
    finally:
        provider.close()

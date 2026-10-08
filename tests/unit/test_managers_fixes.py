"""Regression tests for the manager-level defect fixes.

Covers:
1. ``SelfAwarenessManager.run_command`` bypasses (PATH-resolved binaries,
   per-command flag deny-lists, path validation for every command).
2. ``CacheManager.cache_response`` never caches error responses / bypasses.
3. ``MemoryManager.retrieve_tool_log`` honours memory_id / thread_id scopes.
4. Conversation cache is invalidated after an atomic summary store.
5. ``normalize_history_item`` is exported by the memory manager module.
6. ``load_summaries_for_thread`` passes ``user_id`` to ``list_all``.
7. Sandbox ``execute_code`` tool description is honest per provider mode.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from memorizz.long_term.episodic.conversational_memory_unit import (
    ConversationMemoryUnit,
)
from memorizz.memagent.managers import (
    CacheManager,
    MemoryManager,
    SandboxManager,
    SelfAwarenessManager,
)
from memorizz.memagent.managers.cache_manager import CACHE_ERROR_RESPONSE_PREFIX
from memorizz.memagent.managers.memory_manager import normalize_history_item
from memorizz.memagent.managers.self_awareness_manager import DANGEROUS_TOOL_NAMES
from memorizz.memory_provider.base import _UNSET

# ---------------------------------------------------------------------------
# Finding 1: run_command hardening
# ---------------------------------------------------------------------------


def _manager(root, **config):
    cfg = {"root_paths": [str(root)]}
    cfg.update(config)
    return SelfAwarenessManager(config=cfg, cwd=str(root))


def _git_init(root) -> bool:
    if not shutil.which("git"):
        return False
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    return True


@pytest.mark.unit
def test_run_command_rejects_find_exec(tmp_path):
    manager = _manager(tmp_path)
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("find . -maxdepth 0 -exec echo PWNED {} +")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("find . -delete")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("find . -fprint out.txt")


@pytest.mark.unit
def test_run_command_rejects_binary_outside_path(tmp_path):
    """A script whose basename is allow-listed must not run from a path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ls"
    script.write_text("#!/bin/sh\necho PWNED\n", encoding="utf-8")
    script.chmod(0o755)
    manager = _manager(tmp_path)

    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("./bin/ls")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("bin/ls")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command(f"{script}")
    # Even the real system binary is rejected when spelled as a path: only
    # the PATH-resolved program may run.
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("/bin/ls")


@pytest.mark.unit
def test_run_command_rejects_git_config_injection(tmp_path):
    manager = _manager(tmp_path)
    if not shutil.which("git"):
        pytest.skip("git not installed")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("git -c alias.zz=!echo PWNED zz")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("git -C /tmp status")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("git --exec-path=/tmp status")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("git --git-dir=/tmp/x status")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("git --work-tree /tmp status")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("git --config-env=alias.zz=PWNED zz")


@pytest.mark.unit
def test_run_command_rejects_git_alias_invocation(tmp_path):
    if not _git_init(tmp_path):
        pytest.skip("git not installed")
    subprocess.run(
        ["git", "config", "alias.zz", "!echo PWNED"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    manager = _manager(tmp_path)
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("git zz")
    # Subcommands that run external programs are never allowed.
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("git difftool")


@pytest.mark.unit
def test_run_command_rejects_rg_external_programs(tmp_path):
    if not shutil.which("rg"):
        pytest.skip("rg not installed")
    manager = _manager(tmp_path)
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("rg --pre cat pattern .")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("rg --pre=cat pattern .")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("rg --pre-glob '*' pattern .")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("rg --hostname-bin /bin/echo pattern .")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("rg -f /etc/passwd .")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("rg --file=/etc/passwd .")
    with pytest.raises((PermissionError, ValueError)):
        manager.run_command("rg -f/etc/passwd .")


@pytest.mark.unit
def test_run_command_validates_paths_for_every_command(tmp_path):
    manager = _manager(tmp_path)
    if shutil.which("rg"):
        with pytest.raises(ValueError):
            manager.run_command("rg -n pattern /etc")
        with pytest.raises(ValueError):
            manager.run_command("rg -n pattern ../")
    if shutil.which("git"):
        with pytest.raises(ValueError):
            manager.run_command("git log -- /etc/passwd")
        with pytest.raises(ValueError):
            manager.run_command("git diff --output=/tmp/leak")
    with pytest.raises(ValueError):
        manager.run_command("cat /etc/passwd")
    with pytest.raises(ValueError):
        manager.run_command("ls ../")


@pytest.mark.unit
def test_run_command_ordinary_usage_still_works(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "data.txt").write_text("hello pattern world\n", encoding="utf-8")
    manager = _manager(tmp_path)

    result = manager.run_command("ls -la src")
    assert result["exit_code"] == 0
    assert "data.txt" in result["stdout"]

    result = manager.run_command("cat src/data.txt")
    assert result["exit_code"] == 0
    assert "hello pattern world" in result["stdout"]

    result = manager.run_command("find . -name *.txt")
    assert result["exit_code"] == 0
    assert "data.txt" in result["stdout"]

    result = manager.run_command("head -n 1 src/data.txt")
    assert result["exit_code"] == 0

    if shutil.which("rg"):
        result = manager.run_command("rg -n pattern src")
        assert result["exit_code"] == 0
        assert "data.txt" in result["stdout"]
        result = manager.run_command("rg -n -g *.txt pattern src")
        assert result["exit_code"] == 0

    if _git_init(tmp_path):
        result = manager.run_command("git status")
        assert result["exit_code"] == 0
        result = manager.run_command("git log --oneline -n 1")
        # Empty repository: git exits non-zero but the command was allowed.
        assert "PWNED" not in result["stdout"]
        result = manager.run_command("git --version")
        assert result["exit_code"] == 0


@pytest.mark.unit
def test_dangerous_tool_names_exposed():
    names = SelfAwarenessManager.dangerous_tool_names()
    assert {
        "self_aware_run_command",
        "self_aware_write_file",
        "self_aware_delete_file",
    } <= set(names)
    assert set(names) == set(DANGEROUS_TOOL_NAMES)
    assert "self_aware_read_file" not in names


# ---------------------------------------------------------------------------
# Finding 2: cache_response guard
# ---------------------------------------------------------------------------


def _cache_manager_with_stub():
    manager = CacheManager(enabled=False)
    manager.enabled = True
    manager.cache_instance = Mock()
    manager.cache_instance.set.return_value = True
    return manager


@pytest.mark.unit
def test_cache_response_skips_error_responses():
    manager = _cache_manager_with_stub()
    assert CACHE_ERROR_RESPONSE_PREFIX == (
        "I encountered an error while processing your request"
    )
    error_text = f"{CACHE_ERROR_RESPONSE_PREFIX}: boom"
    assert manager.cache_response("q", error_text, "s1") is False
    manager.cache_instance.set.assert_not_called()
    # Leading whitespace does not disguise the error prefix.
    assert manager.cache_response("q", f"  {error_text}", "s1") is False
    manager.cache_instance.set.assert_not_called()


@pytest.mark.unit
def test_cache_response_skips_when_bypass_reason_set():
    manager = _cache_manager_with_stub()
    assert manager.cache_response("q", "fine", "s1", bypass_reason="tool") is False
    manager.cache_instance.set.assert_not_called()
    manager.cache_instance.record_bypass.assert_called_once_with("tool")


@pytest.mark.unit
def test_cache_response_still_caches_ordinary_response():
    manager = _cache_manager_with_stub()
    assert manager.cache_response("q", "fine", "s1") is True
    manager.cache_instance.set.assert_called_once()


# ---------------------------------------------------------------------------
# Finding 3: retrieve_tool_log scoping
# ---------------------------------------------------------------------------


class _ToolLogProvider:
    def __init__(self, row):
        self.row = row

    def retrieve_by_id(self, identifier, memory_store_type):
        return dict(self.row)


@pytest.mark.unit
def test_retrieve_tool_log_rejects_other_memory_or_thread():
    row = {
        "id": "log-1",
        "user_id": "alice",
        "memory_id": "mem-1",
        "thread_id": "thread-1",
        "result": "ok",
    }
    manager = MemoryManager(_ToolLogProvider(row))

    assert manager.retrieve_tool_log("log-1", user_id="alice") == row
    assert manager.retrieve_tool_log("log-1", "alice") == row
    assert (
        manager.retrieve_tool_log(
            "log-1", user_id="alice", memory_id="mem-1", thread_id="thread-1"
        )
        == row
    )
    assert (
        manager.retrieve_tool_log("log-1", user_id="alice", memory_id="mem-2") is None
    )
    assert (
        manager.retrieve_tool_log("log-1", user_id="alice", thread_id="thread-2")
        is None
    )
    assert manager.retrieve_tool_log("log-1", user_id="bob") is None
    # Unscoped read keeps working when no scope is given.
    assert manager.retrieve_tool_log("log-1") == row


@pytest.mark.unit
def test_retrieve_tool_log_signature_keywords():
    params = inspect.signature(MemoryManager.retrieve_tool_log).parameters
    assert list(params) == ["self", "tool_log_id", "user_id", "memory_id", "thread_id"]
    assert params["memory_id"].default is None
    assert params["thread_id"].default is None


# ---------------------------------------------------------------------------
# Finding 4: cache invalidation after atomic summary store
# ---------------------------------------------------------------------------


class _AtomicSummaryProvider:
    """Stub provider with Oracle's atomic ``store_summary_with_links``."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
        self.summaries: List[Dict[str, Any]] = []
        self.update_calls: List[str] = []

    def memory_capabilities(self):
        return SimpleNamespace(manages_embeddings=True)

    def retrieve_conversation_history_ordered_by_timestamp(
        self,
        memory_id: str,
        memory_type: Any = None,
        limit: Optional[int] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        include_embedding: bool = True,
    ):
        return [
            dict(row)
            for row in self.rows
            if row["memory_id"] == memory_id
            and row.get("user_id") == user_id
            and (thread_id is None or row.get("thread_id") == thread_id)
        ]

    def store_summary_with_links(self, data: Dict[str, Any]) -> str:
        summary_id = f"summary-{len(self.summaries) + 1}"
        self.summaries.append(dict(data, id=summary_id))
        for row in self.rows:
            if row["id"] in data.get("source_message_ids", []):
                row["summary_id"] = summary_id
        return summary_id

    def update_by_id(self, identifier, data, memory_store_type):
        self.update_calls.append(identifier)
        return True


@pytest.mark.unit
def test_conversation_cache_invalidated_after_atomic_summary_store():
    now = time.time()
    rows = [
        {
            "id": f"msg-{i}",
            "memory_id": "mem-1",
            "thread_id": "thread-1",
            "user_id": None,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"message {i}",
            "timestamp": now - 100 + i,
        }
        for i in range(4)
    ]
    provider = _AtomicSummaryProvider(rows)
    manager = MemoryManager(provider)
    manager.compress_memories_with_llm = lambda *a, **k: "compact summary"

    # Prime the in-process cache with the unsummarized originals.
    history = manager.load_conversation_history(
        "mem-1", limit=0, user_id=None, thread_id="thread-1"
    )
    assert len(history) == 4

    summary_ids = manager.generate_summaries(
        model=object(),
        agent_id="agent-1",
        memory_ids=["mem-1"],
        current_memory_id="mem-1",
        user_id=None,
        thread_id="thread-1",
        days_back=7,
    )
    assert summary_ids == ["summary-1"]
    # The atomic primitive marked the rows itself; no per-row update needed.
    assert provider.update_calls == []

    # The cache must not keep serving the (now summarized) originals.
    reloaded = manager.load_conversation_history(
        "mem-1", limit=0, user_id=None, thread_id="thread-1"
    )
    assert reloaded == []


# ---------------------------------------------------------------------------
# Finding 5: normalize_history_item helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_history_item_handles_cache_and_provider_shapes():
    nested = {"content": {"role": "assistant", "content": "nested reply"}}
    assert normalize_history_item(nested) == {
        "role": "assistant",
        "content": "nested reply",
    }
    flat = {"role": "user", "content": "flat question"}
    assert normalize_history_item(flat) == flat
    assert normalize_history_item({"role": "user", "text": "alt key"}) == {
        "role": "user",
        "content": "alt key",
    }
    assert normalize_history_item({"content": {"content": "no role"}}) == {
        "role": "user",
        "content": "no role",
    }
    assert normalize_history_item({"role": "robot", "content": "x"}) == {
        "role": "user",
        "content": "x",
    }
    assert normalize_history_item({"role": "user", "content": "   "}) is None
    assert normalize_history_item("not a dict") is None
    assert normalize_history_item(None) is None
    assert MemoryManager.normalize_history_item(flat) == flat


@pytest.mark.unit
def test_save_memory_unit_cache_rows_normalize_like_provider_rows():
    provider = Mock()
    provider.store.return_value = "unit-1"
    manager = MemoryManager(provider)
    cache_key = ("mem-1", None, "thread-1")
    manager._conversation_memory_cache[cache_key] = [
        {"role": "user", "content": "from provider"}
    ]
    unit = ConversationMemoryUnit(
        role="assistant",
        content="from session",
        timestamp="2024-01-01T00:00:00",
        memory_id="mem-1",
        thread_id="thread-1",
    )
    assert manager.save_memory_unit(unit, "mem-1") == "unit-1"
    cached = manager._conversation_memory_cache[cache_key]
    assert len(cached) == 2
    assert [normalize_history_item(row) for row in cached] == [
        {"role": "user", "content": "from provider"},
        {"role": "assistant", "content": "from session"},
    ]


# ---------------------------------------------------------------------------
# Finding 6: load_summaries_for_thread user scoping
# ---------------------------------------------------------------------------


class _ListAllProvider:
    def __init__(self, docs):
        self.docs = docs
        self.calls: List[Any] = []

    def list_all(self, memory_store_type, user_id=_UNSET):
        self.calls.append(user_id)
        if user_id is _UNSET:
            return [dict(d) for d in self.docs]
        return [dict(d) for d in self.docs if d.get("user_id") == user_id]


class _LegacyListAllProvider:
    def __init__(self, docs):
        self.docs = docs

    def list_all(self, memory_store_type):
        return [dict(d) for d in self.docs]


_SUMMARY_DOCS = [
    {
        "id": "s-alice",
        "memory_id": "mem-1",
        "thread_id": "thread-1",
        "user_id": "alice",
        "content": "alice summary",
        "period_end": 2,
    },
    {
        "id": "s-bob",
        "memory_id": "mem-1",
        "thread_id": "thread-1",
        "user_id": "bob",
        "content": "bob summary",
        "period_end": 3,
    },
    {
        "id": "s-other-thread",
        "memory_id": "mem-1",
        "thread_id": "thread-2",
        "user_id": "alice",
        "content": "other thread",
        "period_end": 4,
    },
]


@pytest.mark.unit
def test_load_summaries_fallback_passes_user_id_to_list_all():
    provider = _ListAllProvider(_SUMMARY_DOCS)
    manager = MemoryManager(provider)
    result = manager.load_summaries_for_thread(
        "mem-1", user_id="alice", thread_id="thread-1"
    )
    assert provider.calls == ["alice"]
    assert [item["summary_id"] for item in result] == ["s-alice"]


@pytest.mark.unit
def test_load_summaries_fallback_works_with_legacy_list_all():
    manager = MemoryManager(_LegacyListAllProvider(_SUMMARY_DOCS))
    result = manager.load_summaries_for_thread(
        "mem-1", user_id="bob", thread_id="thread-1"
    )
    assert [item["summary_id"] for item in result] == ["s-bob"]


# ---------------------------------------------------------------------------
# Finding 7: honest sandbox tool description
# ---------------------------------------------------------------------------


class _StubSandboxProvider:
    def __init__(self, name, config):
        self.provider_name = name
        self._config = config

    def get_provider_name(self):
        return self.provider_name

    def get_config(self):
        return dict(self._config)

    def execute_code(self, code, language="python", timeout=30, envs=None):
        raise AssertionError("not executed")

    def write_file(self, path, content):
        return True

    def read_file(self, path):
        return None

    def close(self):
        return None


def _execute_code_doc(provider) -> str:
    tools = SandboxManager(provider=provider).get_tools()
    execute_code = next(t for t in tools if t.__name__ == "execute_code")
    return inspect.getdoc(execute_code) or ""


@pytest.mark.unit
def test_sandbox_tool_description_is_honest_for_graalpy_subprocess():
    provider = _StubSandboxProvider(
        "graalpy",
        {
            "provider": "graalpy",
            "mode": "subprocess",
            "security_boundary": "bounded_execution_provider_not_strong_sandbox",
        },
    )
    doc = _execute_code_doc(provider)
    assert "trusted code only" in doc
    assert "no filesystem isolation" in doc
    assert "secure sandbox" not in doc
    assert "isolated environment" not in doc


@pytest.mark.unit
def test_sandbox_tool_description_for_untrusted_and_remote_sandboxes():
    graal = _StubSandboxProvider(
        "graalpy",
        {
            "provider": "graalpy",
            "mode": "java_wrapper",
            "security_boundary": "graalvm_untrusted_sandbox",
        },
    )
    doc = _execute_code_doc(graal)
    assert "trusted code only" not in doc
    assert "isolat" in doc

    e2b = _StubSandboxProvider("e2b", {"provider": "e2b"})
    doc = _execute_code_doc(e2b)
    assert "trusted code only" not in doc
    assert "isolat" in doc


@pytest.mark.unit
def test_sandbox_tool_description_honours_is_security_sandbox_attribute():
    provider = _StubSandboxProvider("custom", {"provider": "custom"})
    provider.is_security_sandbox = False
    doc = _execute_code_doc(provider)
    assert "trusted code only" in doc
    assert "no filesystem isolation" in doc

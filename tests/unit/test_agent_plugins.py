"""The MemoRizz plugins for Codex and Claude Code: files, hooks, installer."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from memorizz.cli import plugin_commands
from memorizz.cli.app import app
from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from tests.cli_text import plain

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_environment():
    """The plugin reads settings through load_layered_env, which copies a
    test's .env into os.environ; put the environment back afterwards."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugins" / "codex" / "memorizz"
CLAUDE_PLUGIN = REPO / "plugins" / "claude-code" / "memorizz"


def test_the_plugin_has_what_codex_loads():
    manifest = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "memorizz" and PLUGIN.name == manifest["name"]
    assert manifest["skills"] == "./skills/" and manifest["mcpServers"] == "./.mcp.json"
    for asset in (manifest["interface"]["logo"], manifest["interface"]["composerIcon"]):
        assert (PLUGIN / asset).is_file()
    assert len(manifest["interface"]["defaultPrompt"]) <= 3
    server = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]["memorizz"]
    assert server["args"][:2] == ["mcp", "serve"] and "--allow-writes" in server["args"]
    assert server["tools"]["memorizz_store_memory"]["approval_mode"] == "approve"
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    assert set(hooks) == {
        "SessionStart",
        "UserPromptSubmit",
        "Stop",
        "PreCompact",
        "SessionEnd",
    }
    for event, entries in hooks.items():
        command = entries[0]["hooks"][0]["command"]
        # Relative commands resolve against the project, not the plugin.
        assert command.startswith('"${PLUGIN_ROOT}/scripts/')
        script = PLUGIN / command.strip('"').replace("${PLUGIN_ROOT}/", "")
        assert script.is_file() and os.access(script, os.X_OK)
    # Stop returns at once (the save runs detached), so it is not async:
    # an exiting `claude -p` cancels async hooks.
    assert "async" not in hooks["Stop"][0]["hooks"][0]
    # Codex runs SessionEnd synchronously and allows at most 3 seconds.
    assert hooks["SessionEnd"][0]["hooks"][0]["timeout"] <= 3
    for skill in (
        "memorizz-memory",
        "memorizz-agents",
        "memory-curator",
        "memory-transfer",
    ):
        text = (PLUGIN / "skills" / skill / "SKILL.md").read_text()
        assert text.startswith(f"---\nname: {skill}\ndescription: ")
    marketplace = json.loads(
        (REPO / ".agents" / "plugins" / "marketplace.json").read_text()
    )
    entry = marketplace["plugins"][0]
    assert (
        marketplace["name"] == "memorizz"
        and (REPO / entry["source"]["path"]).resolve() == PLUGIN
    )


def _git_project(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / name
    (folder / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(folder)], check=True)
    return folder


def test_each_project_gets_its_own_memory_id(tmp_path: Path):
    first = _git_project(tmp_path / "a", "Shop API")
    second = _git_project(tmp_path / "b", "Shop API")
    found = plugin_commands.project_memory_id(first / "src")
    # From anywhere inside the repository, the same ID.
    assert found == plugin_commands.project_memory_id(first)
    assert (
        found.startswith("project-shop-api-")
        and len(found) == len("project-shop-api-") + 6
    )
    # Another checkout of the same name keeps its own memories.
    assert plugin_commands.project_memory_id(second) != found
    result = CliRunner().invoke(app, ["plugin", "memory-id", str(first)])
    assert result.exit_code == 0 and result.stdout.strip() == found


def _provider(tmp_path: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )


def test_the_session_hook_tells_codex_the_memory_id_and_recent_memories(tmp_path: Path):
    project = _git_project(tmp_path, "calc")
    memory_id = plugin_commands.project_memory_id(project)
    provider = _provider(tmp_path)
    try:
        for day, text in (
            ("2026-09-29", "Tests run with pytest -q."),
            ("2026-10-01", "add() must stay pure."),
        ):
            provider.store(
                {
                    "content": text,
                    "memory_id": memory_id,
                    "timestamp": f"{day}T10:00:00+00:00",
                },
                memory_store_type=MemoryType.KNOWLEDGE_BASE,
                memory_id=memory_id,
            )
        provider.store(
            {"content": "Another project's fact.", "memory_id": "project-other-000000"},
            memory_store_type=MemoryType.KNOWLEDGE_BASE,
            memory_id="project-other-000000",
        )
        context = plugin_commands.session_context(project, provider=provider)
    finally:
        provider.close()
    assert f"memory ID is `{memory_id}`" in context
    lines = context.splitlines()
    recent = lines[lines.index("Recent project memories (newest first):") + 1 :]
    assert recent == [
        "- 2026-10-01: add() must stay pure.",
        "- 2026-09-29: Tests run with pytest -q.",
    ]
    assert "Another project" not in context
    # Nothing stored yet: still the ID and how to use it.
    empty = plugin_commands.session_context(
        tmp_path / "elsewhere", provider=_FailingProvider()
    )
    assert "memory ID is `project-elsewhere-" in empty and "Recent" not in empty


class _FailingProvider:
    def list_all(self, *_a, **_k):
        raise RuntimeError("store unavailable")

    def retrieve_by_query(self, *_a, **_k):
        raise RuntimeError("store unavailable")


def test_hooks_never_fail_codex_and_prompt_recall_is_opt_in(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(plugin_commands, "_store", lambda: _FailingProvider())
    payload = json.dumps(
        {"cwd": str(tmp_path), "prompt": "How do we deploy this service?"}
    )
    started = CliRunner().invoke(
        app, ["plugin", "hook", "session-start"], input=payload
    )
    assert started.exit_code == 0 and "MemoRizz memory is connected" in started.stdout

    monkeypatch.setattr(plugin_commands, "_wants_prompt_recall", lambda: False)
    quiet = CliRunner().invoke(app, ["plugin", "hook", "prompt"], input=payload)
    assert quiet.exit_code == 0 and quiet.stdout == ""

    seen = {}

    class _Search:
        def retrieve_by_query(self, query, **kwargs):
            seen.update(query=query, **kwargs)
            return [
                {
                    "content": "Deploy with make release.",
                    "timestamp": "2026-10-01T00:00:00Z",
                }
            ]

        def close(self):
            pass

    monkeypatch.setattr(plugin_commands, "_wants_prompt_recall", lambda: True)
    monkeypatch.setattr(plugin_commands, "_store", lambda: _Search())
    recalled = CliRunner().invoke(app, ["plugin", "hook", "prompt"], input=payload)
    assert "- 2026-10-01: Deploy with make release." in recalled.stdout
    assert seen["memory_id"] == plugin_commands.project_memory_id(tmp_path)
    # A failing store, or a too-short prompt, says nothing.
    monkeypatch.setattr(plugin_commands, "_store", lambda: _FailingProvider())
    assert (
        CliRunner().invoke(app, ["plugin", "hook", "prompt"], input=payload).stdout
        == ""
    )
    assert plugin_commands.prompt_context(tmp_path, "hi", provider=_Search()) == ""


def test_the_prompt_hook_script_exits_at_once_when_recall_is_off(tmp_path: Path):
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "MEMORIZZ_BIN": "/nonexistent",
    }
    finished = subprocess.run(
        ["sh", str(PLUGIN / "scripts" / "hook-prompt.sh")],
        input='{"prompt": "How do we deploy?"}',
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert finished.returncode == 0 and finished.stdout == ""


def _fake_cli(tmp_path: Path, name: str, marketplaces: str = "") -> None:
    fake = tmp_path / "bin" / name
    fake.parent.mkdir(exist_ok=True)
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "{name} $@" >> {tmp_path}/calls.txt\n'
        f'if [ "$1 $2 $3" = "plugin marketplace list" ]; then echo "{marketplaces}"; fi\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)


@pytest.mark.parametrize(
    "agent, expected",
    [
        (
            "codex",
            [
                "codex plugin marketplace list",
                "codex plugin marketplace add {root}",
                "codex plugin add memorizz@memorizz",
            ],
        ),
        (
            "claude-code",
            [
                "claude plugin marketplace list",
                "claude plugin marketplace add {root}",
                "claude plugin install memorizz@memorizz",
            ],
        ),
    ],
)
def test_install_writes_one_marketplace_for_both_agents(
    tmp_path: Path, monkeypatch, agent, expected
):
    for name in ("codex", "claude"):
        _fake_cli(tmp_path, name)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    result = CliRunner().invoke(app, ["plugin", "install", agent, "--json"])
    assert result.exit_code == 0, result.output
    root = tmp_path / "home" / "plugin-marketplace"
    assert json.loads(result.stdout)["marketplace"] == str(root)
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls == [line.format(root=root) for line in expected]
    # Both agents' catalogs, each pointing at its own plugin.
    codex = json.loads((root / ".agents" / "plugins" / "marketplace.json").read_text())
    claude = json.loads((root / ".claude-plugin" / "marketplace.json").read_text())
    assert (
        root / codex["plugins"][0]["source"]["path"] / ".codex-plugin" / "plugin.json"
    ).is_file()
    assert (
        root / claude["plugins"][0]["source"] / ".claude-plugin" / "plugin.json"
    ).is_file()
    for folder in ("codex", "claude-code"):
        assert os.access(
            root / "plugins" / folder / "memorizz" / "scripts" / "memorizz.sh", os.X_OK
        )

    removed = CliRunner().invoke(app, ["plugin", "uninstall", agent, "--json"])
    assert removed.exit_code == 0
    tail = (tmp_path / "calls.txt").read_text().splitlines()[-2:]
    cli = "codex" if agent == "codex" else "claude"
    verb = "remove" if agent == "codex" else "uninstall"
    assert tail == [
        f"{cli} plugin {verb} memorizz@memorizz",
        f"{cli} plugin marketplace remove memorizz",
    ]
    assert CliRunner().invoke(app, ["plugin", "install", "word"]).exit_code != 0


def test_the_launcher_runs_the_recorded_memorizz(tmp_path: Path):
    plugins = plugin_commands.build_marketplace(tmp_path / "market", "/bin/echo")
    for plugin in plugins.values():
        ran = subprocess.run(
            [str(plugin / "scripts" / "memorizz.sh"), "mcp", "serve"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
            timeout=10,
        )
        assert ran.stdout.strip() == "mcp serve"


def test_the_claude_code_plugin_matches_the_codex_one():
    manifest = json.loads(
        (CLAUDE_PLUGIN / ".claude-plugin" / "plugin.json").read_text()
    )
    codex = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_text())
    assert manifest["name"] == codex["name"] == "memorizz"
    assert manifest["version"] == codex["version"]
    server = json.loads((CLAUDE_PLUGIN / ".mcp.json").read_text())["mcpServers"][
        "memorizz"
    ]
    assert server["command"] == "${CLAUDE_PLUGIN_ROOT}/scripts/memorizz.sh"
    assert (
        server["args"]
        == json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]["memorizz"][
            "args"
        ]
    )
    hooks = json.loads((CLAUDE_PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    for event in ("SessionStart", "UserPromptSubmit"):
        assert hooks[event][0]["hooks"][0]["command"].startswith(
            '"${CLAUDE_PLUGIN_ROOT}"/scripts/'
        )
    # The scripts and skills are the same files in both plugins.
    shared = sorted(
        path.relative_to(PLUGIN)
        for folder in ("scripts", "skills")
        for path in (PLUGIN / folder).rglob("*")
        if path.is_file()
    )
    assert shared
    for relative in shared:
        assert (CLAUDE_PLUGIN / relative).read_bytes() == (
            PLUGIN / relative
        ).read_bytes(), relative
    claude_shared = sorted(
        path.relative_to(CLAUDE_PLUGIN)
        for folder in ("scripts", "skills")
        for path in (CLAUDE_PLUGIN / folder).rglob("*")
        if path.is_file()
    )
    assert claude_shared == shared
    marketplace = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    assert (REPO / marketplace["plugins"][0]["source"]).resolve() == CLAUDE_PLUGIN


def test_claude_code_has_slash_commands_and_a_curator_subagent():
    for command in ("remember", "recall", "forget", "memory-status"):
        text = (CLAUDE_PLUGIN / "commands" / f"{command}.md").read_text()
        assert text.startswith("---\ndescription: ")
    curator = (CLAUDE_PLUGIN / "agents" / "memory-curator.md").read_text()
    assert "name: memory-curator" in curator
    assert "mcp__plugin_memorizz_memorizz__memorizz_update_memory" in curator


def _project_and_store(tmp_path: Path):
    project = _git_project(tmp_path, "shop")
    return project, _provider(tmp_path)


def _rows(provider, memory_type):
    return [
        row for row in provider.list_all(memory_type) or [] if isinstance(row, dict)
    ]


def test_a_turn_is_saved_with_its_prompt_and_secrets_removed(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PLUGIN_ROOT", "/codex/plugin")  # Codex runs the hook
    monkeypatch.delenv("MEMORIZZ_SESSION_CAPTURE", raising=False)
    monkeypatch.delenv(plugin_commands.PRINCIPAL_ENV, raising=False)
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding", lambda text: [0.1, 0.2, 0.3]
    )
    project, provider = _project_and_store(tmp_path)
    try:
        pending = plugin_commands._sessions_dir() / "s-1.prompt.json"
        pending.write_text(
            json.dumps(
                {
                    "session_id": "s-1",
                    "prompt": "Deploy with key sk-abcdefghijklmnopqrstuvwxyz123456",
                }
            )
        )
        thread = plugin_commands.capture_turn(
            {
                "session_id": "s-1",
                "cwd": str(project),
                "last_assistant_message": "Run make deploy.",
            },
            provider=provider,
        )
        assert thread == "codex-s-1" and not pending.exists()
        rows = [
            row
            for row in _rows(provider, MemoryType.CONVERSATION_MEMORY)
            if row.get("thread_id") == thread
        ]
        assert sorted(row["role"] for row in rows) == ["assistant", "user"]
        user = next(row for row in rows if row["role"] == "user")
        assert "sk-abcdef" not in user["content"] and "[REDACTED]" in user["content"]
        assert user["memory_id"] == plugin_commands.project_memory_id(project)
        assert user["agent_id"] == "codex"
        # Nothing pending (or capture off): nothing saved.
        assert (
            plugin_commands.capture_turn(
                {
                    "session_id": "s-2",
                    "cwd": str(project),
                    "last_assistant_message": "x",
                },
                provider=provider,
            )
            is None
        )
        monkeypatch.setenv("MEMORIZZ_SESSION_CAPTURE", "off")
        pending.write_text(json.dumps({"prompt": "hello"}))
        assert (
            plugin_commands.capture_turn(
                {
                    "session_id": "s-1",
                    "cwd": str(project),
                    "last_assistant_message": "hi",
                },
                provider=provider,
            )
            is None
        )
    finally:
        provider.close()


def test_a_session_is_summarized_and_the_next_session_starts_with_it(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PLUGIN_ROOT", raising=False)  # Claude Code runs the hook
    monkeypatch.delenv("MEMORIZZ_SESSION_CAPTURE", raising=False)
    monkeypatch.delenv("MEMORIZZ_SESSION_SUMMARY", raising=False)
    monkeypatch.delenv(plugin_commands.PRINCIPAL_ENV, raising=False)
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding", lambda text: [0.1, 0.2, 0.3]
    )
    project, provider = _project_and_store(tmp_path)

    class _Model:
        def generate(self, messages, **kwargs):
            return "We moved deploys to Thursdays and fixed the flaky login test."

        def get_last_usage(self):
            return {}

    monkeypatch.setattr("memorizz.episodic_capture.default_model", lambda: _Model())
    try:
        for prompt, answer in (
            ("Move deploys to Thursday", "Done."),
            ("Fix the flaky login test", "Fixed."),
        ):
            (plugin_commands._sessions_dir() / "s-9.prompt.json").write_text(
                json.dumps({"prompt": prompt})
            )
            plugin_commands.capture_turn(
                {
                    "session_id": "s-9",
                    "cwd": str(project),
                    "last_assistant_message": answer,
                },
                provider=provider,
            )
        made = plugin_commands.summarize_session(
            {"session_id": "s-9", "cwd": str(project)}, provider=provider
        )
        assert made
        summaries = _rows(provider, MemoryType.SUMMARIES)
        assert summaries and summaries[0]["agent_id"] == "claude-code"
        context = plugin_commands.session_context(project, provider=provider)
        assert "Last session summary" in context and "Thursdays" in context
        assert "saved to MemoRizz conversation memory" in context
        # Summaries off: nothing is made.
        monkeypatch.setenv("MEMORIZZ_SESSION_SUMMARY", "false")
        assert (
            plugin_commands.summarize_session(
                {"session_id": "s-9", "cwd": str(project)}, provider=provider
            )
            == []
        )
    finally:
        provider.close()


def test_a_shared_store_keeps_each_users_memories_apart(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    project, provider = _project_and_store(tmp_path)
    memory_id = plugin_commands.project_memory_id(project)
    try:
        for user, text in (("ada", "Ada's fact."), ("grace", "Grace's fact.")):
            provider.store(
                {
                    "content": text,
                    "memory_id": memory_id,
                    "user_id": user,
                    "timestamp": "2026-10-01T00:00:00Z",
                },
                memory_store_type=MemoryType.KNOWLEDGE_BASE,
                memory_id=memory_id,
            )
        monkeypatch.setenv(plugin_commands.PRINCIPAL_ENV, "ada")
        context = plugin_commands.session_context(project, provider=provider)
        assert "Ada's fact." in context and "Grace's fact." not in context
    finally:
        provider.close()


def test_the_hook_scripts_keep_the_prompt_and_save_the_turn(tmp_path: Path):
    project = _git_project(tmp_path, "shop")
    home = tmp_path / "home"
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "MEMORIZZ_HOME": str(home),
        "MEMORIZZ_MEMORY_ROOT": str(tmp_path / "memory"),
        "MEMORIZZ_BIN": str(Path(sys.executable).with_name("memorizz")),
        "CLAUDE_PLUGIN_ROOT": str(CLAUDE_PLUGIN),
    }
    scripts = CLAUDE_PLUGIN / "scripts"
    payload = {"session_id": "t-1", "cwd": str(project)}
    subprocess.run(
        [str(scripts / "hook-prompt.sh")],
        input=json.dumps({**payload, "prompt": "Where are the docs?"}),
        text=True,
        env=env,
        timeout=20,
        check=True,
    )
    assert (home / "plugin-sessions" / "t-1.prompt.json").is_file()
    started = time.monotonic()
    subprocess.run(
        [str(scripts / "hook-stop.sh")],
        input=json.dumps({**payload, "last_assistant_message": "In docs/."}),
        text=True,
        env=env,
        timeout=60,
        check=True,
    )
    assert time.monotonic() - started < 2  # the save runs detached
    # The prompt is set aside at once, so the next prompt can't take its place.
    assert not (home / "plugin-sessions" / "t-1.prompt.json").exists()
    while (
        any((home / "plugin-sessions").glob("t-1.saving.*"))
        and time.monotonic() - started < 60
    ):
        time.sleep(0.2)
    assert not any((home / "plugin-sessions").glob("t-1.*"))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    try:
        rows = [
            row
            for row in _rows(provider, MemoryType.CONVERSATION_MEMORY)
            if row.get("thread_id") == "claude-code-t-1"
        ]
        assert sorted(row["content"] for row in rows) == [
            "In docs/.",
            "Where are the docs?",
        ]
    finally:
        provider.close()


def test_install_saves_the_options_as_settings(tmp_path: Path, monkeypatch):
    for name in ("codex", "claude"):
        _fake_cli(tmp_path, name)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MEMORIZZ_MCP_SERVER_HARNESS_WORKSPACE_ROOTS", raising=False)
    root = tmp_path / "work"
    root.mkdir()
    refused = CliRunner().invoke(app, ["plugin", "install", "codex", "--allow-harness"])
    assert refused.exit_code != 0 and "--harness-root" in plain(refused.output)
    done = CliRunner().invoke(
        app,
        [
            "plugin",
            "install",
            "codex",
            "--json",
            "--user",
            "ada",
            "--no-capture",
            "--prompt-recall",
            "--allow-agents",
            "--allow-harness",
            "--harness-root",
            str(root),
            "--allow-traces",
        ],
    )
    assert done.exit_code == 0, done.output
    saved = (tmp_path / "home" / ".env").read_text()
    for line in (
        "MEMORIZZ_MCP_SERVER_LOCAL_PRINCIPAL=ada",
        "MEMORIZZ_SESSION_CAPTURE=off",
        "MEMORIZZ_PROMPT_RECALL=true",
        "MEMORIZZ_MCP_SERVER_ALLOW_AGENT_EXECUTION=true",
        "MEMORIZZ_MCP_SERVER_ALLOW_HARNESS_EXECUTION=true",
        f"MEMORIZZ_MCP_SERVER_HARNESS_WORKSPACE_ROOTS={root.resolve()}",
        "MEMORIZZ_MCP_SERVER_ALLOW_TRACE_QUERIES=true",
    ):
        assert line in saved.replace('"', ""), line
    assert "MEMORIZZ_SESSION_SUMMARY" not in saved  # left out: unchanged


def test_config_set_accepts_the_plugin_settings(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    ok = CliRunner().invoke(app, ["config", "set", "MEMORIZZ_PROMPT_RECALL", "true"])
    assert ok.exit_code == 0, ok.output
    bad = CliRunner().invoke(
        app, ["config", "set", "MEMORIZZ_SESSION_CAPTURE", "everything"]
    )
    assert bad.exit_code != 0
    assert "MEMORIZZ_PROMPT_RECALL=true" in (tmp_path / ".env").read_text().replace(
        '"', ""
    )


def test_install_remote_points_both_plugins_at_a_hosted_server(
    tmp_path: Path, monkeypatch
):
    for name in ("codex", "claude"):
        _fake_cli(tmp_path, name)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(plugin_commands.REMOTE_URL_ENV, raising=False)
    for bad in ("http://mcp.example.com/mcp", "ftp://127.0.0.1/mcp"):
        refused = CliRunner().invoke(
            app, ["plugin", "install", "codex", "--remote", bad]
        )
        assert refused.exit_code != 0 and "https://" in refused.output
    refused = CliRunner().invoke(
        app,
        [
            "plugin",
            "install",
            "codex",
            "--remote",
            "https://mcp.example.com/mcp",
            "--token-env",
            "bad name",
        ],
    )
    assert refused.exit_code != 0

    url = "https://mcp.example.com/mcp"
    done = CliRunner().invoke(
        app,
        [
            "plugin",
            "install",
            "codex",
            "--remote",
            url,
            "--token-env",
            "TEAM_TOKEN",
            "--json",
        ],
    )
    assert done.exit_code == 0, done.output
    plugins = tmp_path / "home" / "plugin-marketplace" / "plugins"
    codex = json.loads((plugins / "codex" / "memorizz" / ".mcp.json").read_text())[
        "mcpServers"
    ]["memorizz"]
    assert codex["type"] == "http" and codex["url"] == url
    assert codex["bearer_token_env_var"] == "TEAM_TOKEN" and "command" not in codex
    assert codex["tools"]["memorizz_store_memory"]["approval_mode"] == "approve"
    claude = json.loads(
        (plugins / "claude-code" / "memorizz" / ".mcp.json").read_text()
    )["mcpServers"]["memorizz"]
    assert claude == {
        "type": "http",
        "url": url,
        "headers": {"Authorization": "Bearer ${TEAM_TOKEN}"},
    }
    saved = (tmp_path / "home" / ".env").read_text().replace('"', "")
    assert (
        f"MEMORIZZ_PLUGIN_REMOTE_URL={url}" in saved
        and "MEMORIZZ_PLUGIN_REMOTE_TOKEN_ENV=TEAM_TOKEN" in saved
    )

    # Installing the other agent later keeps using the hosted server.
    assert CliRunner().invoke(app, ["plugin", "install", "claude-code"]).exit_code == 0
    claude = json.loads(
        (plugins / "claude-code" / "memorizz" / ".mcp.json").read_text()
    )["mcpServers"]["memorizz"]
    assert claude["headers"] == {"Authorization": "Bearer ${TEAM_TOKEN}"}

    back = CliRunner().invoke(app, ["plugin", "install", "codex", "--local"])
    assert back.exit_code == 0, back.output
    codex = json.loads((plugins / "codex" / "memorizz" / ".mcp.json").read_text())[
        "mcpServers"
    ]["memorizz"]
    assert codex["args"][:2] == ["mcp", "serve"] and "url" not in codex
    saved = (tmp_path / "home" / ".env").read_text().replace('"', "").splitlines()
    assert (
        "MEMORIZZ_PLUGIN_REMOTE_URL=" in saved
    )  # hooks (new processes) read it as unset


def test_hooks_use_the_hosted_server_in_remote_mode(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(plugin_commands.REMOTE_URL_ENV, "https://mcp.example.com/mcp")
    monkeypatch.setenv("PLUGIN_ROOT", "/codex")
    monkeypatch.delenv("MEMORIZZ_SESSION_CAPTURE", raising=False)
    monkeypatch.delenv("MEMORIZZ_SESSION_SUMMARY", raising=False)
    project = _git_project(tmp_path, "shop")
    calls = []

    def fake_remote(name, arguments, timeout=15.0):
        calls.append((name, arguments))
        if name == "memorizz_list_memories" and arguments["memory_type"] == "summaries":
            return {
                "memories": [
                    {
                        "content": "- Moved deploys to Thursdays.",
                        "created_at": "2026-09-30T10:00:00+00:00",
                        "agent_id": "codex",
                    }
                ]
            }
        if name == "memorizz_list_memories":
            return {
                "memories": [
                    {
                        "content": "Tests run with pytest -n 8.",
                        "timestamp": "2026-09-29T10:00:00+00:00",
                    }
                ]
            }
        if name == "memorizz_search_memories":
            return {
                "memories": [
                    {
                        "content": "Deploys need the VPN.",
                        "timestamp": "2026-09-28T10:00:00+00:00",
                    }
                ]
            }
        if name == "memorizz_summarize_session":
            return {"ok": True, "summary_ids": ["s-1"]}
        return {"ok": True}

    monkeypatch.setattr(plugin_commands, "remote_tool", fake_remote)
    monkeypatch.setattr(
        plugin_commands, "_store", lambda: pytest.fail("the local store was opened")
    )
    context = plugin_commands.session_context(project)
    assert "Moved deploys to Thursdays" in context and "pytest -n 8" in context
    monkeypatch.setenv("MEMORIZZ_PROMPT_RECALL", "true")
    assert "Deploys need the VPN." in plugin_commands.prompt_context(
        project, "How do we deploy the shop?"
    )
    (plugin_commands._sessions_dir() / "r-1.prompt.json").write_text(
        json.dumps({"prompt": "Ship it"})
    )
    assert (
        plugin_commands.capture_turn(
            {
                "session_id": "r-1",
                "cwd": str(project),
                "last_assistant_message": "Shipped.",
            }
        )
        == "codex-r-1"
    )
    assert plugin_commands.summarize_session(
        {"session_id": "r-1", "cwd": str(project)}
    ) == ["s-1"]
    recorded = dict(calls)["memorizz_record_turn"]
    assert (
        recorded["thread_id"] == "codex-r-1" and recorded["user_message"] == "Ship it"
    )
    assert recorded["memory_id"] == plugin_commands.project_memory_id(project)


def test_a_summary_waits_for_the_sessions_turn_saves(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    saving = plugin_commands._sessions_dir() / "w-1.saving.abc123"
    saving.write_text("{}")
    started = time.monotonic()
    plugin_commands._wait_for_saves("w-1", limit=0.6)  # still saving: waits it out
    assert time.monotonic() - started >= 0.5
    saving.unlink()
    started = time.monotonic()
    plugin_commands._wait_for_saves("w-1", limit=5)
    assert time.monotonic() - started < 0.5

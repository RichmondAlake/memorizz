"""One project, two coding agents, one memory: Codex works, Claude Code remembers.

    python examples/coding_agent_plugins/cross_agent_demo.py
    python examples/coding_agent_plugins/cross_agent_demo.py --workdir ~/memorizz-demo --ui

1. Codex (with the MemoRizz plugin) fixes a failing test in a small project and
   records why. The plugin saves its turn and, when the session ends, a summary.
2. Claude Code (with the MemoRizz plugin) opens the same project in a new
   session and is asked why the code is the way it is. It starts with Codex's
   session summary and can search what Codex saved.
3. The plugin records each session as a run, so the UI's Harnesses page
   shows both trajectories (and can compare them). --ui adds them to the UI
   you normally run.

Nothing touches your own setup: Codex gets a throwaway CODEX_HOME (your login is
linked in, nothing else), Claude Code loads the plugin for this run only
(--plugin-dir), and MemoRizz uses a fresh store under --workdir. Codex runs on
your Codex login; Claude Code on your Claude login (a few cents on Haiku).

Requires: `memorizz` with the MCP extra, the `codex` and `claude` CLIs, signed
in. Ollama (nomic-embed-text, qwen2.5:7b) is used for search and summaries
when it's running; without it search falls back to keywords and no summary
is written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
from pathlib import Path

CODEX_TASK = (
    "The test suite in this project has one failing test. Find it, fix the bug "
    "with the smallest change, and run the tests to confirm they all pass. "
    "Then save one fact to MemoRizz project memory: what you changed and why."
)
CLAUDE_QUESTION = (
    "I'm new to this project. Why do reward points use int() rather than "
    "round(), and what was done in the last session? Use MemoRizz memory."
)

FILES = {
    "pyproject.toml": '[project]\nname = "fernvale-checkout"\nversion = "0.3.0"\n\n[tool.pytest.ini_options]\npythonpath = ["src"]\n',
    "README.md": "# fernvale-checkout\n\nCheckout rules for the Fernvale Outfitters store. Run the tests with\n`python -m pytest -q -p no:cacheprovider`.\n",
    "src/checkout/__init__.py": '"""Checkout rules: shipping charges and reward points."""\n',
    "src/checkout/rewards.py": (
        '"""Fernvale Rewards points."""\n\n'
        'DOUBLE_POINT_CATEGORIES = {"boots"}\n\n\n'
        "def points_for(subtotal: float, category: str, member: bool) -> int:\n"
        '    """One point per whole dollar spent; members earn double points on boots."""\n'
        "    points = round(subtotal)\n"
        "    if member and category in DOUBLE_POINT_CATEGORIES:\n"
        "        points *= 2\n"
        "    return points\n"
    ),
    "tests/test_rewards.py": (
        "from checkout.rewards import points_for\n\n\n"
        "def test_points_are_whole_dollars_rounded_down():\n"
        '    assert points_for(49.99, "tents", member=False) == 49\n\n\n'
        "def test_members_earn_double_points_on_boots():\n"
        '    assert points_for(120.00, "boots", member=True) == 240\n'
    ),
}


def step(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}", flush=True)


def say(text: str, indent: str = "  ") -> None:
    for paragraph in str(text).strip().splitlines():
        print(
            textwrap.fill(
                paragraph, 96, initial_indent=indent, subsequent_indent=indent
            )
            if paragraph
            else ""
        )


def run(cmd, *, env, cwd=None, timeout=900, stdin=subprocess.DEVNULL):
    return subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        stdin=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ollama_ready() -> bool:
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/tags", timeout=2
        ) as response:
            return response.status == 200
    except Exception:
        return False


def find_memorizz() -> str:
    beside = Path(sys.executable).with_name("memorizz")
    found = str(beside) if beside.exists() else shutil.which("memorizz")
    if not found:
        sys.exit("memorizz isn't installed in this Python; pip install 'memorizz[mcp]'")
    return found


def store_rows(memory_root: Path, memory_id: str):
    """What the shared memory holds for the project, oldest first."""
    from memorizz.enums import MemoryType
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=memory_root, lazy_vector_indexes=True)
    )
    try:
        rows = {}
        for kind in (
            MemoryType.KNOWLEDGE_BASE,
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.SUMMARIES,
        ):
            rows[kind.value] = sorted(
                (
                    r
                    for r in provider.list_all(kind) or []
                    if isinstance(r, dict) and r.get("memory_id") == memory_id
                ),
                key=lambda r: str(r.get("timestamp") or r.get("created_at") or ""),
            )
        return rows
    finally:
        provider.close()


def session_runs(home: Path):
    """The runs the plugin recorded in the demo's MemoRizz home."""
    from memorizz.metaharness import SQLiteHarnessRunStore

    path = home / "harness-runs.sqlite3"
    if not path.exists():
        return []
    store = SQLiteHarnessRunStore(path)
    try:
        return [
            run
            for run in store.list(limit=20)
            if (run.task.get("metadata") or {}).get("source") == "plugin"
        ]
    finally:
        store.close()


def describe_run(item, home: Path) -> str:
    from collections import Counter

    from memorizz.metaharness import SQLiteHarnessRunStore

    store = SQLiteHarnessRunStore(home / "harness-runs.sqlite3")
    try:
        events = store.events(item.run_id, limit=5000)
    finally:
        store.close()
    kinds = Counter(event.type.value for event in events)
    memory_calls = [
        str(event.data.get("name", "")).split("__")[-1]
        for event in events
        if event.type.value == "tool_call" and "memorizz" in str(event.data.get("name"))
    ]
    result = item.result or {}
    cost = result.get("cost_usd")

    def count(n: int, word: str) -> str:
        return f"{n} {word}{'' if n == 1 else 's'}"

    return (
        f"{item.harness} run {item.run_id[:8]}: {count(kinds['command'], 'command')}, "
        f"{count(kinds['tool_call'], 'tool call')} ({len(memory_calls)} to MemoRizz: "
        f"{', '.join(memory_calls) or 'none'}), "
        f"{count(kinds['file_change'], 'file change')}, "
        f"{(result.get('latency_ms') or 0) / 1000:.0f}s"
        + (f", ${cost:.3f}" if cost is not None else "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Where the demo project and stores go (default: a temp folder).",
    )
    parser.add_argument("--claude-model", default="claude-haiku-4-5")
    parser.add_argument(
        "--ui",
        action="store_true",
        help=(
            "Also add both sessions to the Harnesses page of the MemoRizz UI you "
            "normally run (your usual MEMORIZZ_HOME). Memory stays in the demo store."
        ),
    )
    args = parser.parse_args()

    for tool in ("codex", "claude"):
        if not shutil.which(tool):
            sys.exit(f"`{tool}` isn't on PATH")
    auth = Path.home() / ".codex" / "auth.json"
    if not auth.exists():
        sys.exit(
            "Codex isn't signed in (no ~/.codex/auth.json); run `codex login` first"
        )
    memorizz = find_memorizz()

    workdir = (
        (args.workdir or Path(tempfile.mkdtemp(prefix="memorizz-cross-agent-")))
        .expanduser()
        .resolve()
    )
    if args.workdir and workdir.exists() and any(workdir.iterdir()):
        sys.exit(f"{workdir} isn't empty; pass a new folder so each run starts fresh")
    project = workdir / "fernvale-checkout"
    home = workdir / "memorizz-home"
    memory_root = workdir / "memorizz-memory"
    codex_home = workdir / "codex-home"
    for path in (project, home, codex_home):
        path.mkdir(parents=True, exist_ok=True)

    step("1. A small project with one failing test")
    for name, text in FILES.items():
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "add", "-A"], cwd=project, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=demo",
            "-c",
            "user.email=demo@example.com",
            "commit",
            "-qm",
            "Checkout rules",
        ],
        cwd=project,
        check=True,
    )
    say(f"Project: {project}")

    local_models = ollama_ready()
    settings = ["MEMORIZZ_BACKEND=filesystem", f"MEMORIZZ_MEMORY_ROOT={memory_root}"]
    if local_models:
        settings += [
            "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=ollama",
            "MEMORIZZ_DEFAULT_EMBEDDING_MODEL=nomic-embed-text",
            "MEMORIZZ_DEFAULT_LLM_PROVIDER=ollama",
            "MEMORIZZ_DEFAULT_LLM_MODEL=qwen2.5:7b",
        ]
    (home / ".env").write_text("\n".join(settings) + "\n")
    env = {
        **os.environ,
        "MEMORIZZ_HOME": str(home),
        "MEMORIZZ_MEMORY_ROOT": str(memory_root),
        "CODEX_HOME": str(codex_home),
    }
    for key in ("OPENAI_API_KEY",):  # Codex uses your ChatGPT login, not an API key
        env.pop(key, None)
    memory_id = run(
        [memorizz, "plugin", "memory-id", str(project)], env=env
    ).stdout.strip()
    say(f"Shared MemoRizz memory ID for this project: {memory_id}")
    say(
        "Search and summaries: "
        + (
            "local Ollama models"
            if local_models
            else "keyword search, no summaries (Ollama isn't running)"
        )
    )

    step("2. Install the MemoRizz plugin for Codex (in a throwaway CODEX_HOME)")
    link = codex_home / "auth.json"
    if not link.exists():
        link.symlink_to(auth)
    installed = run(
        [memorizz, "plugin", "install", "codex", "--json"], env=env, timeout=180
    )
    if installed.returncode != 0:
        print(installed.stdout, installed.stderr)
        return 1
    plugin_root = home / "plugin-marketplace" / "plugins"
    say(
        "Installed with `memorizz plugin install codex`. Claude Code will load the matching plugin"
    )
    say(f"for its run only: {plugin_root / 'claude-code' / 'memorizz'}")

    step("3. Session 1 — Codex fixes the bug and records why")
    say(f"Task: {CODEX_TASK}")
    last = workdir / "codex-last-message.txt"
    started = time.time()
    codex = run(
        [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--dangerously-bypass-hook-trust",
            "--skip-git-repo-check",
            "-C",
            str(project),
            "-o",
            str(last),
            CODEX_TASK,
        ],
        env=env,
        timeout=900,
    )
    if codex.returncode != 0:
        print(codex.stdout[-2000:], codex.stderr[-2000:])
        return 1
    print(f"\n  Codex ({time.time() - started:.0f}s):")
    say(last.read_text() if last.exists() else codex.stdout[-1500:], "    ")
    tests = run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        env=env,
        cwd=project,
        timeout=120,
    )
    say(
        f"Tests now: {tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else tests.returncode}"
    )
    say(f"Diff:\n{run(['git', 'diff'], env=env, cwd=project).stdout.strip()}")

    if local_models:
        say("Waiting for the plugin to save the turn and summarize the session …")
        for _ in range(60):
            if store_rows(memory_root, memory_id)["summaries"]:
                break
            time.sleep(3)

    rows = store_rows(memory_root, memory_id)
    step("4. What MemoRizz now holds for the project")
    for row in rows["knowledge_base"]:
        say(f"fact: {row.get('content')}")
    for row in rows["conversation_memory"]:
        say(
            f"turn [{row.get('thread_id')}] {row.get('role')}: {str(row.get('content'))[:140]}"
        )
    for row in rows["summaries"]:
        say(f"summary [{row.get('agent_id')}]: {row.get('content')}")

    step("5. Session 2 — Claude Code, a different agent, opens the same project")
    say(f"Question: {CLAUDE_QUESTION}")
    claude = run(
        [
            "claude",
            "-p",
            CLAUDE_QUESTION,
            "--plugin-dir",
            str(plugin_root / "claude-code" / "memorizz"),
            "--model",
            args.claude_model,
            "--allowedTools",
            "mcp__plugin_memorizz_memorizz",
            "--output-format",
            "json",
        ],
        env=env,
        cwd=project,
        timeout=600,
    )
    try:
        result = json.loads(claude.stdout)
    except ValueError:
        print(claude.stdout[-2000:], claude.stderr[-2000:])
        return 1
    print(f"\n  Claude Code (${result.get('total_cost_usd', 0):.3f}):")
    say(result.get("result", ""), "    ")

    # The plugin records each session as a run when the turn ends (detached).
    runs = []
    for _ in range(40):
        runs = session_runs(home)
        if {run.harness for run in runs} >= {"codex", "claude-code"}:
            break
        time.sleep(3)
    rows = store_rows(memory_root, memory_id)
    agents = sorted(
        {
            str(r.get("thread_id", "")).rsplit("-", 5)[0]
            for r in rows["conversation_memory"]
        }
    )
    step("6. One memory, two agents")
    say(f"Conversation threads in {memory_id}: {', '.join(agents)}")
    say(
        f"Facts: {len(rows['knowledge_base'])} · turns: {len(rows['conversation_memory'])} · session summaries: {len(rows['summaries'])}"
    )

    step("7. Each session as a run, with its trajectory")
    for item in sorted(runs, key=lambda run: run.created_at):
        say(describe_run(item, home))
    if args.ui and runs:
        logs = [str(item.task["metadata"]["session_log"]) for item in runs]
        # Your usual MEMORIZZ_HOME: the demo's own settings went only to its agents.
        copied = run(
            [memorizz, "plugin", "import-session", *logs], env=dict(os.environ)
        )
        say(copied.stdout.strip() or copied.stderr.strip())
        say(
            "Open your MemoRizz UI → Harnesses. Tick both runs and press Compare to see "
            "the two trajectories side by side."
        )
    else:
        say(
            "See them in the MemoRizz UI (Harnesses; tick both and press Compare):\n"
            f"  MEMORIZZ_HOME={home} memorizz ui --port 8806\n"
            "Or rerun with --ui to add them to the UI you normally use."
        )
    if not args.workdir:
        say(f"(Remove the demo folder when done: rm -rf {workdir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

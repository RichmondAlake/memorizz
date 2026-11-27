# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""A self-aware agent's own copy of the source: versions, its error log, and
the coding model that recommends and makes a fix there."""

import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from memorizz.memagent import MemAgent, self_repair
from memorizz.memagent.managers import SelfAwarenessManager
from memorizz.memagent.self_source import SelfSource, SelfSourceError


def _calls(*calls) -> Any:
    """An OpenAI-shaped response asking for tool calls."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=f"c{index}",
                            function=SimpleNamespace(
                                name=name, arguments=json.dumps(arguments)
                            ),
                        )
                        for index, (name, arguments) in enumerate(calls)
                    ],
                )
            )
        ]
    )


class _Model:
    """Scripted model: each ``generate`` pops a response, or raises."""

    model = "stub"

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.tools_seen: List[List[str]] = []

    def generate(self, messages, tools=None, **kwargs):
        self.tools_seen.append(
            [tool.get("function", {}).get("name") for tool in tools or []]
        )
        if self.error is not None:
            raise self.error
        if not self.responses:
            return "done"
        item = self.responses.pop(0)
        return item() if callable(item) else item

    def get_config(self):
        return {"provider": "openai", "model": "stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"
    for agent_id in list(self_repair._watched):
        self_repair.unwatch(agent_id)


def _agent(model, **config) -> MemAgent:
    return MemAgent(
        model=model,
        agent_id="agent-1",
        self_aware=True,
        self_aware_config={"own_source": True, **config},
        auto_register=False,
    )


@pytest.mark.unit
def test_the_copy_starts_as_the_install_and_keeps_each_change_as_a_version(home):
    source = SelfSource("agent-1")
    status = source.ensure()

    assert status["versions"] == 0 and status["changed_files"] == []
    assert status["base_version"] == status["installed_version"]
    assert str(home) in status["path"]

    source.edit_file(
        "memorizz/_env_io.py",
        'raw = os.environ.get("MEMORIZZ_HOME")',
        'raw = os.environ.get("MEMORIZZ_HOME") or None',
    )
    assert source.changed_files() == ["memorizz/_env_io.py"]
    assert "+    raw = os.environ.get" in source.diff()
    assert source.check()["ok"] is True

    kept = source.save_version("Treat an empty home as unset")
    assert kept["number"] == 1 and kept["files"] == 1
    assert source.changed_files() == []
    assert "or None" in source.diff(kept["id"])
    assert "or None" in source.patch()
    # Nothing to keep is not a version.
    assert source.save_version("again") is None

    # Going back is itself a version: nothing kept is lost.
    back = source.restore("installed")
    assert back["number"] == 2
    assert source.patch() == ""
    assert [v["number"] for v in source.versions()] == [2, 1, 0]
    assert source.versions()[-1]["installed"] is True


@pytest.mark.unit
def test_the_copy_is_never_the_installed_package(home):
    import memorizz

    source = SelfSource("agent-1")
    source.ensure()
    installed = open(memorizz.__file__, encoding="utf-8").read()
    source.edit_file(
        "memorizz/__init__.py", installed[:40], "# changed\n" + installed[:40]
    )

    assert open(memorizz.__file__, encoding="utf-8").read() == installed
    with pytest.raises(SelfSourceError):
        source.resolve("../../outside.py", must_exist=False)
    with pytest.raises(SelfSourceError):
        source.edit_file("memorizz/_env_io.py", "import", "IMPORT")  # many places


@pytest.mark.unit
def test_a_copy_that_does_not_compile_fails_the_check(home):
    source = SelfSource("agent-1")
    source.ensure()
    source.edit_file(
        "memorizz/_env_io.py",
        "def memorizz_home() -> Path:",
        "def memorizz_home( -> Path:",
    )

    check = source.check()

    assert check["ok"] is False
    assert "memorizz/_env_io.py" in check["steps"][0]["output"]


@pytest.mark.unit
def test_own_source_is_opt_in_and_a_coding_model_never_carries_a_key():
    plain = SelfAwarenessManager(config={}).get_config()
    assert plain["own_source"] is False and plain["coding_model"] is None

    config = SelfAwarenessManager(
        config={
            "own_source": True,
            "coding_model": {
                "provider": "Ollama",
                "model": "flint",
                "host": "http://10.0.0.5:11434",
                "api_key": "secret",
            },
        }
    ).get_config()
    assert config["coding_model"] == {
        "provider": "ollama",
        "model": "flint",
        "host": "http://10.0.0.5:11434",
    }
    assert SelfAwarenessManager(config={"coding_model": "flint"}).get_config()[
        "coding_model"
    ] == {"provider": "ollama", "model": "flint"}


@pytest.mark.unit
def test_turning_it_on_makes_the_copy_and_offers_the_tools(home):
    agent = _agent(_Model())

    assert agent.has_own_source() is True
    assert agent.self_source().exists()
    names = set(agent.tool_manager.list_tools())
    assert {"self_recent_errors", "self_recommend_fix", "self_apply_fix"} <= names

    agent.with_self_aware(True, {"own_source": False})
    assert "self_apply_fix" not in set(agent.tool_manager.list_tools())
    # The copy stays: its versions are the agent's work.
    assert agent.self_source().exists()


@pytest.mark.unit
def test_a_failed_turn_is_kept_with_the_file_and_line_it_came_from(home):
    agent = _agent(_Model(error=RuntimeError("model fell over")))

    agent.run("hello")

    errors = agent.self_errors()
    assert errors, "the failed model call should be in the agent's error log"
    failed = next(entry for entry in errors if "model fell over" in entry["error"])
    assert "RuntimeError" in failed["error"]
    assert any(
        frame["file"] == "memorizz/memagent/core.py" for frame in failed["frames"]
    )


@pytest.mark.unit
def test_another_agents_errors_are_not_this_agents(home):
    agent = _agent(_Model())
    self_repair.current_agent_id.set("someone-else")

    logging.getLogger("memorizz.anything").error("not mine")

    assert agent.self_errors() == []


@pytest.mark.unit
def test_recommending_reads_and_never_writes(home):
    model = _Model(
        [
            _calls(("recent_errors", {})),
            _calls(("read_source", {"path": "memorizz/_env_io.py", "end_line": 5})),
            "Problem: x\nCause: memorizz/_env_io.py:62\nFix: y\nRisk: none",
        ]
    )
    agent = _agent(model)
    steps: List[str] = []

    result = agent.recommend_self_fix(on_step=steps.append)

    assert result["recommendation"].startswith("Problem: x")
    assert "Reading its errors" in steps
    # The reviewer was never offered a way to change the source.
    offered = {name for names in model.tools_seen for name in names}
    assert "read_source" in offered and "edit_source" not in offered
    assert agent.self_source().changed_files() == []


@pytest.mark.unit
def test_a_fix_that_passes_the_check_becomes_a_version(home):
    edit: Dict[str, Any] = {
        "path": "memorizz/_env_io.py",
        "old_text": 'raw = os.environ.get("MEMORIZZ_HOME")',
        "new_text": 'raw = os.environ.get("MEMORIZZ_HOME") or None',
    }
    model = _Model(
        [
            _calls(("recent_errors", {})),
            _calls(("edit_source", edit)),
            _calls(("check_source", {})),
            "VERSION: Treat an empty MEMORIZZ_HOME as unset\nDone.",
        ]
    )
    agent = _agent(model)

    result = agent.fix_self("Treat empty as unset")

    assert result["outcome"] == "fixed"
    assert result["check"]["ok"] is True
    assert result["version"]["message"] == "Treat an empty MEMORIZZ_HOME as unset"
    assert "or None" in result["diff"]
    assert agent.self_source().status()["versions"] == 1


@pytest.mark.unit
def test_a_fix_that_breaks_the_source_is_dropped(home):
    broken = {
        "path": "memorizz/_env_io.py",
        "old_text": "def memorizz_home() -> Path:",
        "new_text": "def memorizz_home( -> Path:",
    }
    # It breaks the file, and has nothing better on the second try.
    model = _Model([_calls(("edit_source", broken)), "VERSION: break it", "sorry"])
    agent = _agent(model)

    result = agent.fix_self()

    assert result["outcome"] == "check_failed"
    assert result["version"] is None
    assert result["dropped"] == ["memorizz/_env_io.py"]
    source = agent.self_source()
    assert source.changed_files() == [] and source.status()["versions"] == 0


@pytest.mark.unit
def test_changing_nothing_is_an_answer_not_a_version(home):
    agent = _agent(
        _Model(["The model was unreachable; nothing in the source is wrong."])
    )

    result = agent.fix_self()

    assert result["outcome"] == "no_change"
    assert "unreachable" in result["answer"]


@pytest.mark.unit
def test_applying_a_fix_from_chat_needs_approval(home):
    agent = _agent(_Model())

    metadata = agent.tool_manager.get_tool_metadata("self_apply_fix")

    assert metadata is not None
    assert agent._effective_tool_policy("self_apply_fix", {})["requires_approval"]
    assert not agent._effective_tool_policy("self_recommend_fix", {}).get(
        "requires_approval"
    )


@pytest.mark.unit
def test_a_diff_is_read_into_files_hunks_and_line_numbers():
    from memorizz.memagent.self_source import parse_diff

    text = "\n".join(
        [
            "diff --git a/memorizz/a.py b/memorizz/a.py",
            "index 1..2 100644",
            "--- a/memorizz/a.py",
            "+++ b/memorizz/a.py",
            "@@ -7,3 +7,4 @@ def run():",
            " keep",
            "--- a line that starts with dashes",
            "+new one",
            "+new two",
            " keep too",
            "\\ No newline at end of file",
            "diff --git a/memorizz/old.py b/memorizz/old.py",
            "deleted file mode 100644",
            "--- a/memorizz/old.py",
            "+++ /dev/null",
            "@@ -1 +0,0 @@",
            "-gone",
        ]
    )

    first, second = parse_diff(text)

    assert (first["path"], first["status"]) == ("memorizz/a.py", "changed")
    assert (first["insertions"], first["deletions"]) == (2, 1)
    assert first["hunks"][0]["context"] == "def run():"
    assert [
        (line["kind"], line["old"], line["new"]) for line in first["hunks"][0]["lines"]
    ] == [
        ("same", 7, 7),
        ("del", 8, None),
        ("add", None, 8),
        ("add", None, 9),
        ("same", 9, 10),
    ]
    assert first["hunks"][0]["lines"][1]["text"] == "-- a line that starts with dashes"
    assert (second["status"], second["deletions"]) == ("deleted", 1)
    assert parse_diff("") == []


@pytest.mark.unit
def test_a_file_view_says_where_lines_were_taken_out(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    from memorizz.memagent.self_source import SelfSource

    source = SelfSource("viewer")
    source.ensure()
    source.edit_file(
        "memorizz/_env_io.py",
        "    base = raw if raw else DEFAULT_HOME\n    return",
        "    return",
    )

    view = source.file_view("memorizz/_env_io.py")

    assert view["changed"] == []
    (line,) = view["removed_before"]
    assert view["text"].splitlines()[line - 1].lstrip().startswith("return")
    old = source.file_view("memorizz/_env_io.py", old=True)
    assert len(old["changed"]) == 1
    assert "DEFAULT_HOME" in old["text"].splitlines()[old["changed"][0] - 1]


@pytest.mark.unit
def test_a_kept_fix_carries_the_account_of_how_it_came_about(home):
    from memorizz.memagent.self_repair import ErrorLog

    edit: Dict[str, Any] = {
        "path": "memorizz/_env_io.py",
        "old_text": 'raw = os.environ.get("MEMORIZZ_HOME")',
        "new_text": 'raw = os.environ.get("MEMORIZZ_HOME") or None',
    }
    model = _Model(
        [
            # The diagnosis: it reads the errors, then answers.
            _calls(("recent_errors", {})),
            "Problem: an empty home.\nCause: _env_io.py.\nFix: treat it as unset.",
            # The fix: it works from the diagnosis without rereading them.
            _calls(("read_source", {"path": "memorizz/_env_io.py"})),
            _calls(("edit_source", edit)),
            _calls(("check_source", {})),
            "VERSION: Treat an empty MEMORIZZ_HOME as unset\nDone.",
            # The explainer: why the change was made, as JSON in prose.
            'Here it is: {"summary": "An empty home was used as a path.", '
            '"root_cause": "memorizz/_env_io.py line 62 kept an empty string.", '
            '"changes": [{"file": "memorizz/_env_io.py", "line": "62", '
            '"what": "Adds or None.", "why": "Empty means unset."}], '
            '"risk": "None seen", "extra": 1}',
        ]
    )
    agent = _agent(model)
    ErrorLog(agent.agent_id).record(
        {
            "source": "exception",
            "error": "ValueError: empty home",
            "frames": [{"file": "memorizz/_env_io.py", "line": 62, "function": "x"}],
        }
    )

    diagnosis = agent.recommend_self_fix()
    result = agent.fix_self(diagnosis["recommendation"])

    source = agent.self_source()
    story = source.story(result["version"]["id"])
    assert story["errors_read_by_model"] is True
    assert [entry["error"] for entry in story["errors"]] == ["ValueError: empty home"]
    assert story["diagnosis"]["text"].startswith("Problem: an empty home.")
    assert story["diagnosis"]["prompt"]["task"].startswith("Find the cause")
    assert diagnosis["recommendation"] in story["prompt"]["task"]
    assert "smallest change" in story["prompt"]["instruction"]
    assert [step["text"] for step in story["steps"]] == [
        "Working on a fix",
        "Reading memorizz/_env_io.py",
        "Editing memorizz/_env_io.py",
        "Checking the source",
        "Checking the source",
        "Explaining the change",
    ]
    # Why it changed what it changed, kept with the version.
    assert story["explanation"] == {
        "summary": "An empty home was used as a path.",
        "root_cause": "memorizz/_env_io.py line 62 kept an empty string.",
        "risk": "None seen",
        "changes": [
            {
                "file": "memorizz/_env_io.py",
                "line": 62,
                "what": "Adds or None.",
                "why": "Empty means unset.",
            }
        ],
    }
    assert result["explanation"] == story["explanation"]
    # The coding model's work is traced under the agent, in a thread of its
    # own; the diagnosis was a separate run with its own.
    work = story["work_trace"]
    assert work["agent_id"] == agent.agent_id and work["thread_id"]
    assert work["memory_id"].startswith("self-repair-")
    assert story["diagnosis"]["work_trace"]["memory_id"] != work["memory_id"]
    assert story["check"]["ok"] is True and story["repaired"] is False
    assert story["answer"].startswith("VERSION:")
    # The install, and a version made by hand, have no account.
    assert source.story(source.versions()[-1]["id"]) is None
    source.create_file("memorizz/_by_hand.py", "X = 1\n")
    assert source.story(source.save_version("By hand")["id"]) is None
    assert source.story("not-a-version") is None


@pytest.mark.unit
def test_the_agents_own_traces_are_asked_for_by_its_id(home):
    from memorizz.memagent import self_repair

    asked: Dict[str, Any] = {}

    class _Provider:
        def query_trace_events(self, *, limit=250, agent_ids=None, **filters):
            asked.update(limit=limit, agent_ids=agent_ids, **filters)
            return {"items": [{"status": "error", "tool_name": "expand_summary"}]}

    class _Agent:
        agent_id = "agent-9"
        memory_provider = _Provider()

    events = self_repair._trace_events(_Agent(), 50, turn_id="t-1")

    assert asked == {"limit": 50, "agent_ids": ["agent-9"], "turn_id": "t-1"}
    assert events == [{"status": "error", "tool_name": "expand_summary"}]


_FAKE_DROID = '''#!/usr/bin/env python3
"""Stands in for Factory's CLI: `droid exec --cwd <dir> -o json [--auto low] -f <prompt>`."""
import json, os, pathlib, sys

args = sys.argv[1:]
here = pathlib.Path(__file__).resolve().parent
record = here / "droid.json"
prompt = pathlib.Path(args[args.index("-f") + 1]).read_text()
record.write_text(json.dumps({"args": args, "prompt": prompt, "env": sorted(os.environ)}))
if (here / "fail").exists():
    print(json.dumps({"type": "result", "is_error": True, "result": "Not signed in"}))
    sys.exit(1)
if "--auto" in args:
    target = pathlib.Path(args[args.index("--cwd") + 1]) / "memorizz" / "_env_io.py"
    text = target.read_text()
    target.write_text(text.replace('os.environ.get("MEMORIZZ_HOME")', 'os.environ.get("MEMORIZZ_HOME") or None', 1))
    answer = "VERSION: Treat an empty MEMORIZZ_HOME as unset\\nDone."
else:
    answer = "Problem: an empty home.\\nCause: memorizz/_env_io.py.\\nFix: treat it as unset.\\nRisk: none."
print("some log line")
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 3,
                  "result": answer, "session_id": "s-1", "usage": {"input_tokens": 4}}))
'''


@pytest.fixture()
def fake_droid(tmp_path, monkeypatch):
    folder = tmp_path / "bin"
    folder.mkdir()
    program = folder / "droid"
    program.write_text(_FAKE_DROID)
    program.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FACTORY_API_KEY", "fk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-droid")
    # Droid's process sees only a few variables, so the stand-in keeps its
    # notes beside itself.
    return folder / "droid.json"


@pytest.mark.unit
def test_droid_can_diagnose_and_fix_in_place_of_the_coding_model(home, fake_droid):
    import json as _json

    from memorizz.memagent.self_repair import ErrorLog

    model = _Model()  # never asked: Droid does the work
    agent = _agent(model)
    agent.with_self_aware(True, {"own_source": True, "coding_agent": "droid"})
    ErrorLog(agent.agent_id).record(
        {
            "source": "exception",
            "error": "ValueError: empty home",
            "frames": [{"file": "memorizz/_env_io.py", "line": 62, "function": "x"}],
        }
    )
    root = str(agent.self_source().root)

    diagnosis = agent.recommend_self_fix()
    seen = _json.loads(fake_droid.read_text())
    # Reading only: no --auto, in the agent's copy, asked for one JSON result.
    assert seen["args"][:5] == ["exec", "--cwd", root, "-o", "json"]
    assert "--auto" not in seen["args"]
    # Files only: Droid's command, web and messaging tools are taken away.
    assert seen["args"][seen["args"].index("--only-tools") + 1] == "Read,LS,Grep,Glob"
    assert "ValueError: empty home" in seen["prompt"]
    assert diagnosis["recommendation"].startswith("Problem: an empty home.")
    assert diagnosis["coding_model"] == "droid"

    result = agent.fix_self(diagnosis["recommendation"])
    seen = _json.loads(fake_droid.read_text())
    assert seen["args"][seen["args"].index("--auto") + 1] == "low"
    assert seen["args"][seen["args"].index("--only-tools") + 1] == (
        "Read,LS,Grep,Glob,Edit,ApplyPatch,Create"
    )
    # Droid gets its own key and nothing of anyone else's.
    assert "FACTORY_API_KEY" in seen["env"] and "OPENAI_API_KEY" not in seen["env"]
    assert result["outcome"] == "fixed" and result["check"]["ok"] is True
    assert result["version"]["message"] == "Treat an empty MEMORIZZ_HOME as unset"
    story = agent.self_source().story(result["version"]["id"])
    assert story["coding_model"] == "droid" and story["errors_read_by_model"] is True
    assert "Droid finished in 3 turns" in [step["text"] for step in story["steps"]]
    assert model.tools_seen == []


@pytest.mark.unit
def test_droid_says_plainly_when_it_cannot_run(home, fake_droid, monkeypatch):
    agent = _agent(_Model())
    agent.with_self_aware(
        True,
        {
            "own_source": True,
            "coding_agent": {"kind": "droid", "model": "custom:Flint-0"},
        },
    )
    assert agent.get_self_aware_config()["coding_agent"] == {
        "kind": "droid",
        "model": "custom:Flint-0",
    }

    (fake_droid.parent / "fail").write_text("1")
    with pytest.raises(RuntimeError, match="Droid stopped early: Not signed in"):
        agent.recommend_self_fix()

    monkeypatch.delenv("FACTORY_API_KEY")
    with pytest.raises(RuntimeError, match="FACTORY_API_KEY"):
        agent.recommend_self_fix()

    monkeypatch.setenv("FACTORY_API_KEY", "fk-test")
    agent.with_self_aware(
        True,
        {"own_source": True, "coding_agent": {"kind": "droid", "command": "no-droid"}},
    )
    with pytest.raises(RuntimeError, match="not installed"):
        agent.recommend_self_fix()


@pytest.mark.unit
def test_the_check_catches_a_console_page_or_script_that_cannot_be_drawn(home):
    source = SelfSource("agent-1")
    source.ensure()
    page = "memorizz/ui/templates/dashboard.html"
    text = (source.root / page).read_text(encoding="utf-8")

    (source.root / page).write_text(
        text + "\n{# a harmless note #}\n", encoding="utf-8"
    )
    passed = source.check()
    assert passed["ok"] is True
    assert "The console's changed pages compile" in [s["name"] for s in passed["steps"]]

    (source.root / page).write_text(text + "\n{% if broken %}\n", encoding="utf-8")
    failed = source.check()
    step = next(
        s
        for s in failed["steps"]
        if s["name"].startswith("The console's changed pages")
    )
    assert failed["ok"] is False and step["ok"] is False
    assert "dashboard.html" in step["output"]
    source.discard_changes()

    import shutil

    if shutil.which("node"):
        script = "memorizz/ui/static/js/app.js"
        original = (source.root / script).read_text(encoding="utf-8")
        (source.root / script).write_text(
            original + "\nfunction (((\n", encoding="utf-8"
        )
        broken = source.check()
        step = next(s for s in broken["steps"] if "scripts parse" in s["name"])
        assert broken["ok"] is False and step["ok"] is False


@pytest.mark.unit
def test_a_version_can_be_run_as_its_own_read_only_console(home, tmp_path):
    import urllib.request

    from memorizz.memagent import self_preview

    source = SelfSource("agent-1")
    source.ensure()
    # The frame every page is drawn in, so any page shows the change.
    page = "memorizz/ui/templates/base.html"
    text = (source.root / page).read_text(encoding="utf-8")
    assert "</body>" in text
    (source.root / page).write_text(
        text.replace("</body>", "<!-- changed by the agent --></body>", 1),
        encoding="utf-8",
    )
    kept = source.save_version("Mark every page")
    store = ["--backend", "filesystem", "--path", str(tmp_path / "preview-store")]

    try:
        preview = self_preview.start(source, kept["id"], store=store, label="test")

        assert preview["number"] == 1 and preview["read_only"] is True
        assert self_preview.status("agent-1")["port"] == preview["port"]
        with urllib.request.urlopen(preview["url"] + "/login", timeout=10) as reply:
            assert "changed by the agent" in reply.read().decode("utf-8")
        # The version's own folder, not the copy the agent goes on editing.
        assert (source.folder / "previews" / kept["commit"][:12] / "memorizz").is_dir()
        with pytest.raises(Exception):
            self_preview.start(source, "not-a-version", store=store)
    finally:
        assert self_preview.stop("agent-1") is True
    assert self_preview.status("agent-1") is None
    assert self_preview.stop("agent-1") is False


@pytest.mark.unit
def test_the_explainer_is_read_however_the_model_wraps_it():
    from memorizz.memagent.self_repair import _explanation, _is_repair_work

    assert _explanation("no json here") is None
    assert _explanation('```json\n{"changes": "x", "other": 1}\n```') is None
    told = _explanation(
        'Sure. {"summary": " Fixed. ", "changes": [{"file": "memorizz/a.py", '
        '"line": "x", "what": "w"}, {"line": 3}, "junk"]} Done.'
    )
    assert told == {
        "summary": "Fixed.",
        "changes": [{"file": "memorizz/a.py", "line": 0, "what": "w", "why": ""}],
    }
    # The coding model's own slips are not the agent's failures.
    assert _is_repair_work({"memory_id": "self-repair-abc"})
    assert _is_repair_work({"trace_tags": {"work": "self_repair"}})
    assert not _is_repair_work({"memory_id": "chat-1", "trace_tags": {"work": "x"}})

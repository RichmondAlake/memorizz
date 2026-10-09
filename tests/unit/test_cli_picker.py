"""The arrow-key picker and the commands that open it."""

from io import StringIO
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from memorizz.cli import commands, picker
from memorizz.cli.agent_factory import Session

pytestmark = pytest.mark.unit


def _run_picker(keys: str, items, **kwargs):
    from prompt_toolkit.application import create_app_session

    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        with create_app_session(input=pipe, output=DummyOutput()):
            return picker.pick(items, title="t", render=str, **kwargs)


def test_picker_moves_filters_and_selects():
    items = ["alpha agent", "beta agent", "gamma agent"]
    assert _run_picker("\r", items) == "alpha agent"
    assert _run_picker("\x1b[B\x1b[B\r", items) == "gamma agent"  # down, down, enter
    assert _run_picker("\x1b[A\r", items) == "gamma agent"  # up wraps to the end
    assert _run_picker("bet\r", items) == "beta agent"  # typing filters
    assert _run_picker("zzz\r\x1b", items) is None  # nothing matches, then escape
    assert _run_picker("\x1b", items) is None
    assert picker.pick([], title="t", render=str) is None


def test_picker_available_only_on_a_terminal(monkeypatch):
    tty = Console(force_terminal=True, file=StringIO())
    assert picker.picker_available(tty, {}) is True
    assert picker.picker_available(Console(file=StringIO()), {}) is False
    assert picker.picker_available(tty, {"MEMORIZZ_NO_PICKER": "1"}) is False
    assert picker.picker_available(tty, {"TERM": "dumb"}) is False
    assert picker.picker_available(tty, {"CI": "1"}) is False


@pytest.fixture
def tty_session(monkeypatch):
    output = StringIO()
    console = Console(file=output, force_terminal=True, color_system=None, width=140)
    agents = [
        SimpleNamespace(agent_id="a-1", name="assistant", application_mode="assistant"),
        SimpleNamespace(
            agent_id="b-2",
            name="Codex bug hunter",
            application_mode="assistant",
            is_favorite=True,
        ),
    ]
    provider = SimpleNamespace(list_memagents=lambda: agents)
    agent = SimpleNamespace(agent_id="a-1", meta_harness=None, llm_config={})
    sess = Session(
        agent=agent,
        provider=provider,
        llm_config={"provider": "ollama", "model": "m"},
        console=console,
    )
    for name in ("MEMORIZZ_NO_PICKER", "CI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    return sess, output, agents


def test_agents_opens_the_picker_and_loads_the_choice(tty_session, monkeypatch):
    sess, output, agents = tty_session
    seen = {}
    monkeypatch.setattr(
        picker, "pick", lambda items, **kw: (seen.update(kw), items[1])[1]
    )
    monkeypatch.setattr(
        commands, "cmd_agent", lambda session, args: seen.update({"loaded": args})
    )
    commands.dispatch("/agents", sess)
    assert seen["loaded"] == "b-2" and seen["title"] == "Switch agent"
    assert seen["is_current"](agents[0]) and not seen["is_current"](agents[1])
    assert "★ Codex bug hunter" in seen["render"](agents[1])
    # choosing the current agent or cancelling changes nothing
    monkeypatch.setattr(picker, "pick", lambda items, **kw: items[0])
    seen.pop("loaded")
    commands.dispatch("/agents", sess)
    assert "loaded" not in seen and "Already on a-1" in output.getvalue()
    monkeypatch.setattr(picker, "pick", lambda items, **kw: None)
    commands.dispatch("/agents", sess)
    assert "No change." in output.getvalue()
    # /agents list prints the plain list even on a terminal
    commands.dispatch("/agents list", sess)
    assert "Saved agents (2)" in output.getvalue()


def test_harnesses_picker_includes_delegate_and_off(tty_session, monkeypatch):
    sess, output, _ = tty_session
    monkeypatch.setattr(
        commands.harness_session
        if hasattr(commands, "harness_session")
        else __import__("memorizz.cli.harness_session", fromlist=["x"]),
        "list_harness_status",
        lambda session, **kw: [
            {"name": "codex", "ready": True, "version": "0.160.0", "reason": None}
        ],
    )
    captured = {}

    def fake_pick(items, **kw):
        captured["names"] = [i["name"] for i in items]
        return items[-1]  # "off"

    monkeypatch.setattr(picker, "pick", fake_pick)
    monkeypatch.setattr(
        commands, "cmd_harness", lambda session, args: captured.update({"chosen": args})
    )
    commands.dispatch("/harnesses", sess)
    assert (
        captured["names"] == ["codex", "delegate", "off"]
        and captured["chosen"] == "off"
    )


def test_sessions_and_menu_open_the_picker(tty_session, monkeypatch):
    sess, output, _ = tty_session
    from memorizz.cli import harness_session

    rows = [
        {
            "run_id": "6e07d725-full",
            "harness": "codex",
            "created_at": "2026-10-07T19:14:57",
            "status": "succeeded",
            "memory_id": "project-x",
            "workspace": "/w",
            "workspace_name": "w",
            "turns": "3",
            "session_id": "s",
            "title": "hello",
        }
    ]
    monkeypatch.setattr(harness_session, "plugin_sessions", lambda session, **kw: rows)
    monkeypatch.setattr(picker, "pick", lambda items, **kw: items[0])
    opened = {}
    monkeypatch.setattr(
        commands, "cmd_session", lambda session, args: opened.update({"run": args})
    )
    commands.dispatch("/sessions", sess)
    assert opened["run"] == "6e07d725"
    monkeypatch.setattr(
        picker, "pick", lambda items, **kw: next(a for a in items if a[0] == "/help")
    )
    commands.dispatch("/menu", sess)
    assert "Slash commands" in output.getvalue()
    commands.dispatch("/sessions list", sess)
    assert "Coding-agent sessions (1 newest)" in output.getvalue()


@pytest.mark.parametrize("focus", ["\x1b[I", "\x1b[O"])
def test_terminal_focus_reports_do_not_cancel_the_picker(focus):
    """Regression: clicking back into the terminal sent ESC [ I, which
    prompt_toolkit read as Esc and the picker closed with 'No change.'"""
    assert _run_picker(focus + "\r", ["alpha", "beta"]) == "alpha"
    assert _run_picker(focus + "\x1b[B" + focus + "\r", ["alpha", "beta"]) == "beta"


def test_focus_reports_do_not_type_into_the_prompt():
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import create_app_session

    from memorizz.cli import ui

    ui.ignore_focus_reports()
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[Ihello\x1b[O\r")
        with create_app_session(input=pipe, output=DummyOutput()):
            assert PromptSession().prompt("> ") == "hello"


def test_conversations_picker_survives_focus_reports(tmp_path):
    from prompt_toolkit.application import create_app_session

    from memorizz.cli import conversations

    summary = conversations.ConversationSummary(
        memory_id="m",
        thread_id="t",
        title="hello",
        preview="hello",
        created_at=None,
        updated_at=None,
        message_count=2,
    )
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[I\r")
        with create_app_session(input=pipe, output=DummyOutput()):
            assert conversations.pick_conversation([summary]) == summary

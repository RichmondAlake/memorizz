"""Look and feel of the chat: crest animation, status bar, prompt and arrow hotkeys."""

from io import StringIO

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import to_plain_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from memorizz.cli import commands, ui
from memorizz.cli.agent_factory import Session

pytestmark = pytest.mark.unit


@pytest.fixture
def session():
    output = StringIO()
    agent = type("Agent", (), {"agent_id": "agent-1", "meta_harness": None})()
    sess = Session(
        agent=agent,
        provider=object(),
        llm_config={"provider": "ollama", "model": "gemma4:latest"},
        memory_id="25bfd944-960b-48e8-89be-0b46aa",
        console=Console(file=output, color_system=None, width=160),
    )
    return sess, output


def test_crest_rows_share_one_width_and_draw_an_m():
    assert all(len(row) == ui.CREST_WIDTH for row in ui.CREST_ROWS)
    plain = ui.crest_text().plain
    assert plain.count("\n") == len(ui.CREST_ROWS) - 1
    assert "███╗       ███╗" in plain and plain.strip().endswith("V")
    partial = ui.crest_text(progress=3).plain.splitlines()
    assert partial[2].strip() and not partial[3].strip()
    assert ui.crest_text(None, pulse=7).plain == plain


def test_should_animate_only_on_a_quiet_colour_terminal():
    tty = Console(force_terminal=True, file=StringIO())
    assert ui.should_animate(tty, {}) is True
    assert ui.should_animate(Console(file=StringIO()), {}) is False
    assert ui.should_animate(tty, {"TERM": "dumb"}) is False
    assert ui.should_animate(tty, {"NO_COLOR": "1"}) is False
    assert ui.should_animate(tty, {"MEMORIZZ_NO_ANIMATION": "1"}) is False
    assert ui.should_animate(tty, {"CI": "true"}) is False


def test_animate_crest_is_a_no_op_off_terminal_and_runs_on_one(monkeypatch):
    assert ui.animate_crest(Console(file=StringIO())) is False
    out = StringIO()
    tty = Console(force_terminal=True, color_system="truecolor", file=out, width=120)
    monkeypatch.delenv("MEMORIZZ_NO_ANIMATION", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert ui.animate_crest(tty, frame_seconds=0) is True
    assert "█" in out.getvalue() and "╭" in out.getvalue()  # styled per glyph


def test_memory_label_and_toolbar(session):
    sess, _ = session
    assert ui.memory_label(sess) == "25bfd944"
    sess.memory_id = "project-memorizz-521ff6"
    assert ui.memory_label(sess) == "project-memorizz-521ff6"
    sess.memory_id = None
    assert ui.memory_label(sess) == "new"
    text = to_plain_text(ui.toolbar(sess))
    assert (
        "ollama/gemma4:latest" in text
        and "memory new" in text
        and "harness off" in text
    )
    assert (
        "← agents" in text
        and "→ harnesses" in text
        and "↓ menu" in text
        and "Tab commands" in text
    )
    sess.harness = "codex"
    assert "harness codex" in to_plain_text(ui.toolbar(sess))


def test_prompt_fragments_keep_the_plain_label(session):
    sess, _ = session
    assert to_plain_text(ui.prompt_fragments(sess)) == "memorizz> "
    sess.harness = "codex"
    frags = ui.prompt_fragments(sess)
    assert to_plain_text(frags) == "memorizz·codex> "
    assert ("class:prompt.scope", "codex") in frags
    sess.code_mode = True
    assert to_plain_text(ui.prompt_fragments(sess)) == "code·codex> "


def _prompt_with(keys: str, session) -> str:
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        ptk = PromptSession(
            input=pipe, output=DummyOutput(), key_bindings=ui.key_bindings(session)
        )
        return ptk.prompt("> ")


def test_arrow_keys_on_an_empty_line_run_the_actions(session):
    sess, _ = session
    assert _prompt_with("\x1b[D", sess) == "/agents"
    assert _prompt_with("\x1b[C", sess) == "/harnesses"
    assert _prompt_with("\x1b[B", sess) == "/menu"


def test_arrow_keys_keep_editing_when_the_line_has_text(session):
    sess, _ = session
    # left arrow moves the cursor; "x" lands before "o"; Enter submits the text
    assert _prompt_with("hello\x1b[Dx\r", sess) == "hellxo"
    assert _prompt_with("hi\x1b[C\x1b[B\r", sess) == "hi"


def test_menu_command_lists_hotkeys_and_common_commands(session):
    sess, output = session
    assert commands.dispatch("/menu", sess) is True
    text = output.getvalue()
    assert "Quick actions" in text
    for key, command, _ in ui.HOTKEYS:
        assert key in text and command in text
    assert "/sessions" in text and "/memory project" in text and "/compare" in text
    assert "now: ollama/gemma4:latest" in text and "memory 25bfd944" in text
    assert "/menu" in commands.command_completions()


def test_color_depth_follows_colorterm():
    from prompt_toolkit.output.color_depth import ColorDepth

    assert ui.color_depth({"COLORTERM": "truecolor"}) is ColorDepth.TRUE_COLOR
    assert ui.color_depth({"COLORTERM": "24bit"}) is ColorDepth.TRUE_COLOR
    assert ui.color_depth({}) is None
    assert ui.color_depth({"COLORTERM": "yes"}) is None

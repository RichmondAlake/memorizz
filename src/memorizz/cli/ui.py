# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Look and feel of the interactive chat: the crest animation, the frame
around the input, the status bar under it, the coloured prompt and the
arrow-key actions.

Everything here degrades to plain text: no animation on a non-terminal, a
dumb terminal, ``NO_COLOR`` or ``MEMORIZZ_NO_ANIMATION``; no frame on a dumb
terminal. The prompt's own line never changes (``memorizz> ``), so scripts
that drive the chat keep matching it.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, List, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, to_filter
from prompt_toolkit.formatted_text import HTML, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.text import Text

from . import harness_session

# The crest: a block "M" inside a shield. Each row is the same width so the
# reveal animation keeps its shape.
CREST_ROWS: Tuple[str, ...] = (
    "╭─────────────────╮",
    "│ ███╗       ███╗ │",
    "│ ████╗     ████║ │",
    "│ ██╔██╗   ██╔██║ │",
    "│ ██║╚██╗ ██╔╝██║ │",
    "│ ██║ ╚████╔╝ ██║ │",
    "╲ ██║  ╚██╔╝  ██║ ╱",
    " ╲╚═╝   ╚═╝   ╚═╝╱ ",
    "  ╲             ╱  ",
    "   ╲           ╱   ",
    "    ╲         ╱    ",
    "     ╲       ╱     ",
    "      ╲     ╱      ",
    "       ╲   ╱       ",
    "        ╲ ╱        ",
    "         V         ",
)
CREST_WIDTH = len(CREST_ROWS[0])
_SHIELD_CHARS = set("╭╮╰╯─│╲╱V ")
_PULSE_COLOURS = ("#ff3b3b", "#ff6b6b", "#ff3b3b", "#d7263d")
HOTKEYS: Tuple[Tuple[str, str, str], ...] = (
    ("←", "/agents", "switch agent"),
    ("→", "/harnesses", "run turns on a harness"),
    ("↓", "/menu", "quick actions"),
)

STYLE = Style.from_dict(
    {
        "prompt": "bold #7dd3fc",
        "prompt.scope": "bold #fbbf24",
        "prompt.arrow": "#9ca3af",
        "frame": "#4b5563",
        "frame.harness": "#3f8f63",
        "bottom-toolbar": "noreverse #c8d0e0",
        "bottom-toolbar.key": "bold #fbbf24",
        "bottom-toolbar.dim": "#8a93a6",
        "bottom-toolbar.model": "#7dd3fc",
        "bottom-toolbar.harness": "bold #86efac",
        "bottom-toolbar.update": "bold #fbbf24 bg:#3b2f0b",
    }
)


FOCUS_REPORTS = ("\x1b[I", "\x1b[O")


def ignore_focus_reports() -> None:
    """Treat terminal focus reports as no-ops.

    A terminal with focus reporting on sends ESC [ I / ESC [ O when the
    window gains or loses focus. prompt_toolkit does not know these
    sequences, so it reads them as the Esc key plus two characters: pickers
    cancel themselves when you click back into the window, and the prompt
    gains a stray "[I". Registering them as ignored keys fixes both.
    Idempotent; the CLI owns the terminal, so changing the shared table is safe.
    """
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.keys import Keys

    for sequence in FOCUS_REPORTS:
        ANSI_SEQUENCES.setdefault(sequence, Keys.Ignore)


@contextmanager
def quiet_keys(stream: Any = None):
    """Stop keys typed while the CLI works from echoing as ^[[B and the like.

    Between prompts the terminal is in line mode with echo on, so an arrow
    key pressed while a turn streams or a command loads is printed raw over
    the output. Echo is off inside the block; the keys stay queued for the
    next prompt or picker. A no-op when the stream is not a terminal.
    """
    try:
        import termios

        fd = (stream or sys.stdin).fileno()
        saved = termios.tcgetattr(fd)
        quiet = list(saved)
        quiet[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, quiet)
    except Exception:
        saved = None
    try:
        yield
    finally:
        if saved is not None:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, saved)
            except Exception:
                pass


@contextmanager
def busy(console: Console, message: str):
    """A spinner for work that takes a moment before anything prints, with
    keys kept from echoing over it. Plain (no spinner) off a terminal."""
    with quiet_keys():
        if console.is_terminal and not console.is_dumb_terminal:
            with console.status(f"[dim]{message}[/dim]", spinner="dots"):
                yield
        else:
            yield


def elapsed_suffix(seconds: float) -> str:
    """ " · 12s" for a turn still working, with the way out once it runs long."""
    whole = int(seconds)
    if whole < 1:
        return ""
    text = f"{whole}s" if whole < 60 else f"{whole // 60}m {whole % 60:02d}s"
    return f" · {text}" + (" · ctrl-c to stop" if whole >= 10 else "")


def should_animate(console: Console, environ: Any = None) -> bool:
    """Animate only on a real colour terminal and when nobody asked for quiet."""
    env = os.environ if environ is None else environ
    if not getattr(console, "is_terminal", False):
        return False
    if str(env.get("TERM", "")).lower() == "dumb":
        return False
    if env.get("NO_COLOR") or env.get("MEMORIZZ_NO_ANIMATION") or env.get("CI"):
        return False
    return True


def crest_text(progress: int | None = None, pulse: int = 0) -> Text:
    """The crest as Rich text: shield in gold, the M in red.

    ``progress`` shows only the first N rows (the reveal); ``pulse`` picks the
    M's colour for the glow frames.
    """
    shown = (
        len(CREST_ROWS) if progress is None else max(0, min(progress, len(CREST_ROWS)))
    )
    m_colour = _PULSE_COLOURS[pulse % len(_PULSE_COLOURS)]
    text = Text()
    for index, row in enumerate(CREST_ROWS):
        if index >= shown:
            text.append(" " * CREST_WIDTH)
        else:
            for char in row:
                style = "bold #fbbf24" if char in _SHIELD_CHARS else f"bold {m_colour}"
                text.append(char, style=style)
        if index < len(CREST_ROWS) - 1:
            text.append("\n")
    return text


def animate_crest(console: Console, *, frame_seconds: float = 0.035) -> bool:
    """Reveal the crest row by row, pulse it, and clear it before the banner.

    Returns True when it ran. About half a second on a terminal; a no-op
    otherwise.
    """
    if not should_animate(console):
        return False
    try:
        with Live(
            crest_text(0), console=console, transient=True, refresh_per_second=30
        ) as live:
            for progress in range(1, len(CREST_ROWS) + 1):
                live.update(crest_text(progress), refresh=True)
                time.sleep(frame_seconds)
            for pulse in range(1, len(_PULSE_COLOURS) + 1):
                live.update(crest_text(None, pulse), refresh=True)
                time.sleep(frame_seconds * 2)
    except Exception:
        return False
    return True


def color_depth(environ: Any = None):
    """Use 24-bit colour when the terminal advertises it (``COLORTERM``);
    prompt_toolkit only looks at ``TERM`` and would fall back to 256 colours."""
    env = os.environ if environ is None else environ
    if str(env.get("COLORTERM", "")).lower() in {"truecolor", "24bit"}:
        from prompt_toolkit.output.color_depth import ColorDepth

        return ColorDepth.TRUE_COLOR
    return None


# Lines the completion menu gets under the input while it is open.
MENU_SPACE = 8


class FramedPromptSession(PromptSession):
    """The chat's prompt: the input sits between two rules.

    prompt_toolkit lets the input grow to push the status bar to the bottom
    of the terminal, and keeps ``reserve_space_for_menu`` blank lines under
    it whenever completion is on; either would stretch the frame. The input
    keeps to its own lines, with room for the menu only while it is open.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for window in self.layout.find_all_windows():
            if getattr(window.content, "buffer", None) is self.default_buffer:
                window.dont_extend_height = to_filter(True)

    @property
    def reserve_space_for_menu(self) -> int:
        buffer = getattr(self, "default_buffer", None)
        return MENU_SPACE if buffer is not None and buffer.complete_state else 0

    @reserve_space_for_menu.setter
    def reserve_space_for_menu(self, value: int) -> None:
        pass  # decided per render above


def framed(environ: Any = None) -> bool:
    """Draw the frame anywhere but a dumb terminal, which shows no status bar
    either and would print the rule as text before every prompt."""
    env = os.environ if environ is None else environ
    return str(env.get("TERM", "")).lower() != "dumb"


def frame_rule(session: Any) -> Tuple[str, str]:
    """One rule across the terminal, green while turns run on a harness."""
    try:
        width = get_app().output.get_size().columns
    except Exception:
        width = 80
    harness = harness_session.active_harness(session)
    style = "class:frame.harness" if harness else "class:frame"
    return (style, "─" * max(1, width))


def prompt_message(session: Any) -> List[Tuple[str, str]]:
    """The prompt with the frame's top rule above it while typing.

    Once the line is submitted only the prompt stays, so the scrollback
    reads ``memorizz> question``.
    """
    fragments = prompt_fragments(session)
    if not framed() or get_app().is_done:
        return fragments
    return [frame_rule(session), ("", "\n"), *fragments]


def memory_label(session: Any) -> str:
    """How the status bar names the active memory: a project id in full,
    otherwise the first eight characters, or 'new' before the first turn."""
    memory_id = str(getattr(session, "memory_id", None) or "")
    if not memory_id:
        return "new"
    if memory_id.startswith("project-"):
        return memory_id
    return memory_id[:8]


def toolbar(session: Any) -> List[Tuple[str, str]]:
    """Under the prompt: the frame's bottom rule, then the status line
    (model, memory, harness and the hotkeys)."""
    harness = harness_session.active_harness(session) or "off"
    harness_model = harness_session.harness_model_label(session)
    if harness_model and harness not in {"off", harness_session.DELEGATE}:
        harness = f"{harness} · {harness_model}"
    harness_class = (
        "bottom-toolbar.harness" if harness != "off" else "bottom-toolbar.dim"
    )
    model = f"{session.provider_name}/{session.model_name}"
    keys = "  ".join(
        f"<bottom-toolbar.key>{key}</bottom-toolbar.key> {command.lstrip('/')}"
        for key, command, _ in HOTKEYS
    )
    status = HTML(
        f" <b>memorizz</b> <bottom-toolbar.dim>·</bottom-toolbar.dim> "
        f"<bottom-toolbar.model>{_escape(model)}</bottom-toolbar.model> "
        f"<bottom-toolbar.dim>· memory</bottom-toolbar.dim> {_escape(memory_label(session))} "
        f"<bottom-toolbar.dim>· harness</bottom-toolbar.dim> "
        f"<{harness_class}>{_escape(harness)}</{harness_class}>  "
        f"<bottom-toolbar.dim>│</bottom-toolbar.dim>  {keys}  "
        f"<bottom-toolbar.key>Tab</bottom-toolbar.key> commands"
        + update_fragment(session)
    )
    rule = [frame_rule(session), ("", "\n")] if framed() else []
    return rule + to_formatted_text(status)


def update_fragment(session: Any) -> str:
    """The status bar's update pill: '⬆ 0.17.0 available · /update'."""
    notice = getattr(session, "update_notice", None) or {}
    latest = notice.get("latest") if isinstance(notice, dict) else None
    if not latest:
        return ""
    return (
        "  <bottom-toolbar.update> ⬆ "
        f"{_escape(str(latest))} available · /update </bottom-toolbar.update>"
    )


def prompt_fragments(session: Any) -> List[Tuple[str, str]]:
    """The coloured prompt. Its plain text is exactly ``prompt_label(session)``."""
    label = harness_session.prompt_label(session)
    head, sep, _ = label.rpartition("> ")
    if not sep:
        return [("class:prompt", label)]
    if "·" in head:
        base, _, scope = head.partition("·")
        return [
            ("class:prompt", base),
            ("class:prompt.arrow", "·"),
            ("class:prompt.scope", scope),
            ("class:prompt.arrow", "> "),
        ]
    return [("class:prompt", head), ("class:prompt.arrow", "> ")]


def key_bindings(
    session: Any, run: Callable[[Any, str], None] | None = None
) -> KeyBindings:
    """Arrow keys on an empty line run the common actions.

    With text in the line the arrows keep their editing meaning; ``↑`` always
    walks history. The action is submitted as if typed, so the normal command
    dispatch (and history) sees it.
    """
    bindings = KeyBindings()
    submit = run or _submit

    @Condition
    def line_is_empty() -> bool:
        try:
            return not get_app().current_buffer.text
        except Exception:
            return False

    for key, command, _ in HOTKEYS:
        key_name = {"←": "left", "→": "right", "↓": "down"}[key]

        def _make(cmd: str):
            def handler(event: Any) -> None:
                submit(event, cmd)

            return handler

        bindings.add(key_name, filter=line_is_empty)(_make(command))
    return bindings


def _submit(event: Any, command: str) -> None:
    buffer = event.current_buffer
    buffer.text = command
    buffer.cursor_position = len(command)
    buffer.validate_and_handle()


def _escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


__all__ = [
    "CREST_ROWS",
    "FramedPromptSession",
    "HOTKEYS",
    "STYLE",
    "animate_crest",
    "framed",
    "frame_rule",
    "ignore_focus_reports",
    "color_depth",
    "crest_text",
    "key_bindings",
    "memory_label",
    "prompt_fragments",
    "prompt_message",
    "should_animate",
    "toolbar",
    "update_fragment",
]

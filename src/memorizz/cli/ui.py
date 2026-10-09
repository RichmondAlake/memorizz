# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Look and feel of the interactive chat: the crest animation, the status bar
under the prompt, the coloured prompt and the arrow-key actions.

Everything here degrades to plain text: no animation on a non-terminal, a
dumb terminal, ``NO_COLOR`` or ``MEMORIZZ_NO_ANIMATION``; the prompt text
itself never changes (``memorizz> ``), so scripts that drive the chat keep
matching it.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, List, Tuple

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
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
        "bottom-toolbar": "noreverse bg:#1f2430 #c8d0e0",
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


def memory_label(session: Any) -> str:
    """How the status bar names the active memory: a project id in full,
    otherwise the first eight characters, or 'new' before the first turn."""
    memory_id = str(getattr(session, "memory_id", None) or "")
    if not memory_id:
        return "new"
    if memory_id.startswith("project-"):
        return memory_id
    return memory_id[:8]


def toolbar(session: Any) -> HTML:
    """The status line under the prompt: model, memory, harness and the hotkeys."""
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
    return HTML(
        f" <b>memorizz</b> <bottom-toolbar.dim>·</bottom-toolbar.dim> "
        f"<bottom-toolbar.model>{_escape(model)}</bottom-toolbar.model> "
        f"<bottom-toolbar.dim>· memory</bottom-toolbar.dim> {_escape(memory_label(session))} "
        f"<bottom-toolbar.dim>· harness</bottom-toolbar.dim> "
        f"<{harness_class}>{_escape(harness)}</{harness_class}>  "
        f"<bottom-toolbar.dim>│</bottom-toolbar.dim>  {keys}  "
        f"<bottom-toolbar.key>Tab</bottom-toolbar.key> commands"
        + update_fragment(session)
    )


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
    "HOTKEYS",
    "STYLE",
    "animate_crest",
    "ignore_focus_reports",
    "color_depth",
    "crest_text",
    "key_bindings",
    "memory_label",
    "prompt_fragments",
    "should_animate",
    "toolbar",
    "update_fragment",
]

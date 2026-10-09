# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""A full-screen list picker for the chat: type to filter, ↑/↓ to move,
Enter to choose, Esc to cancel. The same widget behind ``/conversations``,
generalised for agents, harnesses, sessions, models and the quick menu.

It only opens on a real terminal; callers print a plain list otherwise (and
when ``MEMORIZZ_NO_PICKER`` is set, which scripted sessions use).
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence


def picker_available(console: Any, environ: Any = None) -> bool:
    env = os.environ if environ is None else environ
    if env.get("MEMORIZZ_NO_PICKER") or env.get("CI"):
        return False
    if str(env.get("TERM", "")).lower() == "dumb":
        return False
    return bool(getattr(console, "is_terminal", False))


def pick(
    items: Sequence[Any],
    *,
    title: str,
    render: Callable[[Any], str],
    detail: Optional[Callable[[Any], str]] = None,
    search_text: Optional[Callable[[Any], str]] = None,
    is_current: Optional[Callable[[Any], bool]] = None,
    verb: str = "select",
    initial_query: str = "",
) -> Optional[Any]:
    """Open the picker over ``items`` and return the chosen one, or None."""
    if not items:
        return None
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, ScrollOffsets, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import TextArea

    from .ui import ignore_focus_reports

    ignore_focus_reports()
    searchable = search_text or render
    selected = [0]
    search = TextArea(
        text=initial_query,
        height=1,
        prompt=FormattedText([("class:search-label", "Type to filter  ")]),
        multiline=False,
        wrap_lines=False,
        style="class:search",
    )

    def matches() -> List[Any]:
        needle = search.text.strip().lower()
        values = [
            item for item in items if not needle or needle in searchable(item).lower()
        ]
        selected[0] = max(0, min(selected[0], len(values) - 1))
        return values

    def render_header():
        return [
            ("class:title", f" {title}"),
            ("", "\n"),
            ("class:meta", f" {len(matches())} of {len(items)}"),
        ]

    def render_rows():
        values = matches()
        if not values:
            return [("class:empty", "\n  Nothing matches.")]
        try:
            width = get_app().output.get_size().columns
        except Exception:
            width = 100
        fragments = []
        for index, item in enumerate(values):
            chosen = index == selected[0]
            marker = "›" if chosen else " "
            current = "●" if is_current and is_current(item) else " "
            line = render(item).replace("\n", " ")[: max(10, width - 6)]
            fragments.append(
                (
                    "class:selected" if chosen else "class:row",
                    f"{marker} {current} {line}\n",
                )
            )
        return fragments

    def render_details():
        values = matches()
        if not values or detail is None:
            return []
        return [("class:details", " " + detail(values[selected[0]]).replace("\n", " "))]

    header = Window(
        FormattedTextControl(render_header), height=2, dont_extend_height=True
    )
    rows = Window(
        FormattedTextControl(
            render_rows, get_cursor_position=lambda: Point(x=0, y=selected[0])
        ),
        height=Dimension(min=4),
        scroll_offsets=ScrollOffsets(top=1, bottom=1),
        wrap_lines=False,
        always_hide_cursor=True,
    )
    details = Window(
        FormattedTextControl(render_details),
        height=1,
        dont_extend_height=True,
        wrap_lines=False,
    )
    footer = Window(
        FormattedTextControl(
            [
                ("class:key", " enter"),
                ("class:hint", f" {verb}   "),
                ("class:key", "↑/↓"),
                ("class:hint", " move   "),
                ("class:key", "esc"),
                ("class:hint", " cancel   "),
                ("class:key", "ctrl+u"),
                ("class:hint", " clear filter"),
            ]
        ),
        height=1,
        dont_extend_height=True,
    )

    bindings = KeyBindings()

    @bindings.add("up", eager=True)
    def _up(event):
        values = matches()
        if values:
            selected[0] = (selected[0] - 1) % len(values)
            event.app.invalidate()

    @bindings.add("down", eager=True)
    def _down(event):
        values = matches()
        if values:
            selected[0] = (selected[0] + 1) % len(values)
            event.app.invalidate()

    @bindings.add("pageup", eager=True)
    def _page_up(event):
        if matches():
            selected[0] = max(0, selected[0] - 10)
            event.app.invalidate()

    @bindings.add("pagedown", eager=True)
    def _page_down(event):
        values = matches()
        if values:
            selected[0] = min(len(values) - 1, selected[0] + 10)
            event.app.invalidate()

    @bindings.add("enter", eager=True)
    def _accept(event):
        values = matches()
        if values:
            event.app.exit(result=values[selected[0]])

    @bindings.add("escape", eager=True)
    @bindings.add("c-c", eager=True)
    def _cancel(event):
        event.app.exit(result=None)

    @bindings.add("c-u", eager=True)
    def _clear(event):
        search.text = ""
        search.buffer.cursor_position = 0
        selected[0] = 0
        event.app.invalidate()

    def _changed(_buffer):
        selected[0] = 0

    search.buffer.on_text_changed += _changed

    style = Style.from_dict(
        {
            "title": "bold #7dd3fc",
            "meta": "#8a93a6",
            "search": "bg:#1f2430 #e5e7eb",
            "search-label": "bold #fbbf24",
            "row": "",
            "selected": "reverse bold",
            "details": "#8a93a6",
            "empty": "#8a93a6 italic",
            "key": "bold #fbbf24",
            "hint": "#8a93a6",
        }
    )
    app = Application(
        layout=Layout(
            HSplit([header, search, rows, details, footer]), focused_element=search
        ),
        key_bindings=bindings,
        style=style,
        full_screen=True,
        mouse_support=False,
    )
    return app.run()


__all__ = ["pick", "picker_available"]

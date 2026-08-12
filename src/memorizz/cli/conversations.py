# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Conversation discovery and the interactive CLI resume picker."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..enums.memory_type import MemoryType


@dataclass(frozen=True)
class ConversationSummary:
    """One resumable conversation thread."""

    memory_id: str
    thread_id: str
    title: str
    preview: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    message_count: int

    @property
    def search_text(self) -> str:
        return " ".join(
            (self.title, self.preview, self.memory_id, self.thread_id)
        ).lower()


def _accepts_keyword(fn: Any, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in params
    )


def _text(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    return str(value or "").strip()


def _single_line(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip()


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return (
            parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        )
    except (TypeError, ValueError):
        return None


def conversation_row_parts(row: Any) -> Dict[str, Any]:
    """Normalize provider-specific conversation-row shapes."""
    if not isinstance(row, dict):
        return {}

    nested = row.get("content")
    payload = nested if isinstance(nested, dict) else {}
    role = _text(row.get("role") or payload.get("role")).lower()
    content = (
        payload.get("content")
        if isinstance(nested, dict)
        else row.get("content") or row.get("response") or row.get("query")
    )
    return {
        "memory_id": _text(row.get("memory_id") or payload.get("memory_id")),
        "thread_id": _text(
            row.get("thread_id")
            or row.get("conversation_id")
            or payload.get("thread_id")
            or payload.get("conversation_id")
        ),
        "role": role,
        "content": _single_line(content),
        "timestamp": _parse_timestamp(
            row.get("timestamp")
            or row.get("created_at")
            or payload.get("timestamp")
            or payload.get("created_at")
        ),
        "agent_id": _text(row.get("agent_id") or payload.get("agent_id")),
        "user_id": row.get("user_id", payload.get("user_id")),
        "has_user_id": "user_id" in row or "user_id" in payload,
    }


def _history_rows(provider: Any, memory_id: str, user_id: Optional[str]) -> List[Any]:
    retrieve = getattr(
        provider, "retrieve_conversation_history_ordered_by_timestamp", None
    )
    if not callable(retrieve):
        return []

    kwargs: Dict[str, Any] = {
        "memory_id": memory_id,
        "memory_type": MemoryType.CONVERSATION_MEMORY,
        "limit": None,
    }
    if _accepts_keyword(retrieve, "user_id"):
        kwargs["user_id"] = user_id
    try:
        return list(retrieve(**kwargs) or [])
    except Exception:
        return []


def discover_conversations(
    provider: Any,
    memory_ids: Iterable[str],
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> List[ConversationSummary]:
    """Group stored conversation rows into resumable memory/thread pairs."""
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    seen_memory_ids = set()

    for raw_memory_id in memory_ids:
        memory_id = _text(raw_memory_id)
        if not memory_id or memory_id in seen_memory_ids:
            continue
        seen_memory_ids.add(memory_id)

        for row in _history_rows(provider, memory_id, user_id):
            item = conversation_row_parts(row)
            thread_id = item.get("thread_id")
            if not thread_id:
                continue
            if item.get("role") not in ("user", "assistant"):
                continue
            if item.get("has_user_id") and item.get("user_id") != user_id:
                continue
            row_agent_id = item.get("agent_id")
            if agent_id and row_agent_id and row_agent_id != agent_id:
                continue
            grouped.setdefault((memory_id, thread_id), []).append(item)

    summaries: List[ConversationSummary] = []
    for (memory_id, thread_id), rows in grouped.items():
        rows.sort(
            key=lambda item: item.get("timestamp")
            or datetime.min.replace(tzinfo=timezone.utc)
        )
        user_rows = [row for row in rows if row.get("role") == "user"]
        title_source = (
            user_rows[0].get("content") if user_rows else rows[0].get("content")
        )
        preview_source = (
            user_rows[-1].get("content") if user_rows else rows[-1].get("content")
        )
        timestamps = [
            row["timestamp"] for row in rows if row.get("timestamp") is not None
        ]
        summaries.append(
            ConversationSummary(
                memory_id=memory_id,
                thread_id=thread_id,
                title=_single_line(title_source) or "Untitled conversation",
                preview=_single_line(preview_source),
                created_at=min(timestamps) if timestamps else None,
                updated_at=max(timestamps) if timestamps else None,
                message_count=len(rows),
            )
        )

    def _sort_key(item: ConversationSummary):
        value = item.updated_at
        if value is None:
            return float("-inf")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()

    summaries.sort(key=_sort_key, reverse=True)
    return summaries


def latest_thread_id(
    provider: Any,
    memory_id: str,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Optional[str]:
    """Return the most recently updated thread in one memory scope."""
    conversations = discover_conversations(
        provider,
        [memory_id],
        user_id=user_id,
        agent_id=agent_id,
    )
    return conversations[0].thread_id if conversations else None


def filter_conversations(
    conversations: Sequence[ConversationSummary], query: str
) -> List[ConversationSummary]:
    """Case-insensitive, all-terms search across titles, previews, and IDs."""
    terms = [term for term in _single_line(query).lower().split(" ") if term]
    if not terms:
        return list(conversations)
    return [
        conversation
        for conversation in conversations
        if all(term in conversation.search_text for term in terms)
    ]


def human_age(value: Optional[datetime], now: Optional[datetime] = None) -> str:
    """Compact relative time suitable for one-line picker rows."""
    if value is None:
        return "unknown"
    current = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    seconds = max(0, int((current - value).total_seconds()))
    if seconds < 60:
        return "now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    return f"{days // 365}y ago"


def pick_conversation(
    conversations: Sequence[ConversationSummary],
    *,
    current_memory_id: Optional[str] = None,
    current_thread_id: Optional[str] = None,
    initial_query: str = "",
) -> Optional[ConversationSummary]:
    """Open a searchable full-screen picker and return the selected thread."""
    if not conversations:
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

    selected_index = [0]
    search = TextArea(
        text=initial_query,
        height=1,
        prompt=FormattedText([("class:search-label", "Type to search  ")]),
        multiline=False,
        wrap_lines=False,
        style="class:search",
    )

    def matches() -> List[ConversationSummary]:
        values = filter_conversations(conversations, search.text)
        selected_index[0] = max(0, min(selected_index[0], len(values) - 1))
        return values

    def render_header():
        return [
            ("class:title", " Resume a previous conversation"),
            ("", "\n"),
            (
                "class:meta",
                f" {len(matches())} of {len(conversations)} conversations"
                "  ·  sorted by updated",
            ),
        ]

    def render_rows():
        values = matches()
        if not values:
            return [("class:empty", "\n  No conversations match your search.")]

        fragments = []
        try:
            terminal_width = get_app().output.get_size().columns
        except Exception:
            terminal_width = 100
        title_width = max(20, terminal_width - 29)
        for index, conversation in enumerate(values):
            selected = index == selected_index[0]
            current = (
                conversation.memory_id == current_memory_id
                and conversation.thread_id == current_thread_id
            )
            row_style = "class:selected" if selected else "class:row"
            marker = "›" if selected else " "
            current_marker = "●" if current else " "
            age = human_age(conversation.updated_at)
            title = conversation.title[:title_width]
            suffix = f"{conversation.message_count} msgs"
            fragments.extend(
                [
                    (
                        row_style,
                        f"{marker} {current_marker} {age:>8}  "
                        f"{title:<{title_width}} {suffix}\n",
                    )
                ]
            )
        return fragments

    def render_details():
        values = matches()
        if not values:
            return []
        conversation = values[selected_index[0]]
        return [
            (
                "class:details",
                " memory "
                + conversation.memory_id
                + "  ·  thread "
                + conversation.thread_id,
            )
        ]

    header = Window(
        FormattedTextControl(render_header),
        height=2,
        dont_extend_height=True,
    )
    rows = Window(
        FormattedTextControl(
            render_rows,
            get_cursor_position=lambda: Point(x=0, y=selected_index[0]),
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
                ("class:hint", " resume   "),
                ("class:key", "↑/↓"),
                ("class:hint", " browse   "),
                ("class:key", "esc"),
                ("class:hint", " cancel   "),
                ("class:key", "ctrl+u"),
                ("class:hint", " clear search"),
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
            selected_index[0] = (selected_index[0] - 1) % len(values)
            event.app.invalidate()

    @bindings.add("down", eager=True)
    def _down(event):
        values = matches()
        if values:
            selected_index[0] = (selected_index[0] + 1) % len(values)
            event.app.invalidate()

    @bindings.add("pageup", eager=True)
    def _page_up(event):
        values = matches()
        if values:
            selected_index[0] = max(0, selected_index[0] - 10)
            event.app.invalidate()

    @bindings.add("pagedown", eager=True)
    def _page_down(event):
        values = matches()
        if values:
            selected_index[0] = min(len(values) - 1, selected_index[0] + 10)
            event.app.invalidate()

    @bindings.add("enter", eager=True)
    def _accept(event):
        values = matches()
        if values:
            event.app.exit(result=values[selected_index[0]])

    @bindings.add("escape", eager=True)
    @bindings.add("c-c", eager=True)
    def _cancel(event):
        event.app.exit(result=None)

    @bindings.add("c-u", eager=True)
    def _clear(event):
        search.text = ""
        search.buffer.cursor_position = 0
        selected_index[0] = 0
        event.app.invalidate()

    def _search_changed(_buffer):
        selected_index[0] = 0
        try:
            get_app().invalidate()
        except Exception:
            pass

    search.buffer.on_text_changed += _search_changed

    style = Style.from_dict(
        {
            "title": "bold #00d7d7",
            "meta": "#888888",
            "search-label": "#aaaaaa",
            "search": "bg:#181818 #ffffff",
            "row": "#d0d0d0",
            "selected": "bg:#3a3400 #ffff5f bold",
            "empty": "#888888 italic",
            "details": "bg:#111111 #888888",
            "key": "bold #d0d0d0",
            "hint": "#777777",
        }
    )
    application = Application(
        layout=Layout(
            HSplit(
                [
                    header,
                    Window(height=1, char="─", style="class:meta"),
                    search,
                    Window(height=1, char=" "),
                    rows,
                    details,
                    footer,
                ]
            ),
            focused_element=search,
        ),
        key_bindings=bindings,
        style=style,
        full_screen=True,
        mouse_support=False,
    )
    try:
        return application.run()
    except (EOFError, KeyboardInterrupt):
        return None

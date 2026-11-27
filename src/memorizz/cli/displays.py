"""What a tool found, as terminal tables: search pages, a webset, a monitor.

A tool attaches a small ``display`` to its result (``ToolResult.display``);
the CLI prints it under the answer. A webset or monitor is read from Exa
again for its rows, with the key in the environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.table import Table
from rich.text import Text

MAX_ROWS = 25
_MET = {"yes": "[green]✓[/green]", "no": "[red]✕[/red]", "unclear": "[dim]?[/dim]"}


def _pages(pages: List[Dict[str, Any]]) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column(justify="right", style="dim", width=3)
    table.add_column(overflow="fold")
    for index, page in enumerate(pages, 1):
        body = Text()
        body.append(str(page.get("title") or page.get("url") or ""), style="bold")
        body.append("\n" + str(page.get("url") or ""), style="dim cyan")
        if page.get("snippet"):
            body.append("\n" + str(page["snippet"])[:220], style="dim")
        table.add_row(str(index), body)
    return table


def webset_renderable(table: Dict[str, Any]) -> RenderableType:
    """A webset (:func:`memorizz.internet_access.exa_tools.webset_table`)."""
    rows = table.get("rows") or []
    columns = [
        column
        for column in table.get("columns") or []
        if column["kind"] != "field"
        or any((row.get("cells") or {}).get(column["key"]) for row in rows)
    ]
    state = "done" if table.get("status") == "done" else "still building"
    grid = Table(
        title=f"Webset: {escape(str(table.get('title') or ''))}",
        caption=f"{table.get('found', len(rows))} of {table.get('wanted') or '?'} found, {state}"
        + (f"  {table['url']}" if table.get("url") else ""),
        title_justify="left",
        caption_justify="left",
        header_style="dim",
        expand=True,
    )
    grid.add_column("Name", overflow="fold", ratio=2)
    for column in columns:
        grid.add_column(
            escape(column["label"])[:28],
            overflow="fold",
            justify="center" if column["kind"] == "criterion" else "left",
        )
    for row in rows[:MAX_ROWS]:
        name = Text(str(row.get("name") or ""), style="bold")
        if row.get("url"):
            name.append("\n" + row["url"], style="dim cyan")
        cells = []
        for column in columns:
            value = str((row.get("cells") or {}).get(column["key"], ""))
            cells.append(
                _MET.get(value, "") if column["kind"] == "criterion" else escape(value)
            )
        grid.add_row(name, *cells)
    return grid


def monitor_renderable(view: Dict[str, Any]) -> RenderableType:
    """A monitor (:func:`memorizz.internet_access.exa_tools.monitor_view`)."""
    head = Text()
    head.append(f"Monitor: {view.get('name') or ''}", style="bold")
    facts = [
        f"every {view['every']}" if view.get("every") else "",
        str(view.get("status") or ""),
        f"next run {view['next_run_at']}" if view.get("next_run_at") else "",
    ]
    head.append("\n" + ", ".join(fact for fact in facts if fact), style="dim")
    latest = next(
        (run for run in view.get("runs") or [] if run.get("status") == "completed"),
        None,
    )
    if not latest:
        return Group(head, Text("No run has finished yet.", style="dim"))
    parts: List[RenderableType] = [head]
    if latest.get("summary"):
        parts.append(Text(str(latest["summary"])))
    if latest.get("results"):
        parts.append(_pages(latest["results"][:8]))
    return Group(*parts)


def renderable(display: Dict[str, Any], exa: Any = None) -> Optional[RenderableType]:
    """What to print for one display, or None when there is nothing to show.

    ``exa`` is an ``ExaProvider``; without one it is made from the
    environment when a webset or monitor has to be read.
    """
    kind = display.get("kind")
    if kind == "web_search":
        pages = display.get("results") or []
        if not pages:
            return None
        return Group(
            Text(
                f"Searched with {display.get('provider') or 'the web'}: "
                f"{display.get('query') or ''}",
                style="dim",
            ),
            _pages(pages[:5]),
        )
    if kind not in {"exa_webset", "exa_monitor"} or not display.get("id"):
        return None
    from ..internet_access.exa_tools import monitor_view, webset_table

    try:
        if exa is None:
            from ..internet_access.providers.exa import ExaProvider

            exa = ExaProvider()
        if kind == "exa_webset":
            return webset_renderable(
                webset_table(
                    exa.get_webset(display["id"]),
                    exa.list_webset_items(display["id"]),
                )
            )
        return monitor_renderable(
            monitor_view(
                exa.get_monitor(display["id"]), exa.list_monitor_runs(display["id"])
            )
        )
    except Exception as exc:  # the answer above still stands
        return Text(f"({kind.replace('_', ' ')} {display['id']}: {exc})", style="dim")


def print_displays(
    console: Console, displays: List[Dict[str, Any]], exa: Any = None
) -> None:
    """Print each display once; a later one about the same thing wins."""
    latest: Dict[str, Dict[str, Any]] = {}
    for display in displays:
        if isinstance(display, dict) and display.get("kind"):
            key = f"{display['kind']}:{display.get('id') or display.get('query')}"
            latest[key] = display
    for display in latest.values():
        shown = renderable(display, exa)
        if shown is not None:
            console.print(shown)

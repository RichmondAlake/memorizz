# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""``memorizz desktop serve``: this Mac's apps as MCP tools.

Reading tools carry ``readOnlyHint`` so MemoRizz runs them freely; opening an
app, clicking, typing, pressing keys and scrolling are marked as changes and go
through the host's approval policy.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .macos import DesktopError, MacDesktop

INSTRUCTIONS = (
    "Operate apps on the user's Mac. read_window(app) shows any running app's "
    "window as numbered elements without changing anything, so use it first; "
    "open_app is only for launching an app that list_apps does not show. "
    "open_url opens a link or deep link in an app, e.g. a Notion page link with "
    "app='Notion' shows that page in the Notion app. Act by number "
    "with click, type_text, press_keys and scroll; each action returns the "
    "window again with new numbers, so always use the latest ones. Ask the user "
    "before sending, deleting, buying or posting anything."
)


def create_desktop_mcp_server(
    desktop: Optional[MacDesktop] = None,
    *,
    state_path: Optional[str] = None,
    screenshots: Optional[str] = None,
):
    """Build the server; the desktop backend starts on the first tool call.

    ``state_path`` keeps the last read's numbering between server processes;
    ``screenshots`` is where each step's window screenshot is saved.
    """
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    from .. import __version__

    server = MCPServer(
        "memorizz-desktop",
        title="MemoRizz Desktop",
        description="Open, read and operate the apps on this Mac.",
        instructions=INSTRUCTIONS,
        version=__version__,
    )
    backend = {"desktop": desktop}
    look = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    act = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )

    def run(action: Callable[[MacDesktop], Any]) -> Any:
        try:
            if backend["desktop"] is None:
                backend["desktop"] = MacDesktop(
                    state_path=state_path, screenshots=screenshots
                )
            return action(backend["desktop"])
        except DesktopError as exc:
            raise ToolError(str(exc)) from exc

    # The tools are async so they run on the main thread, where AppKit's list
    # of running apps gets refreshed (sync tools would run in worker threads).

    @server.tool(annotations=look)
    async def list_apps() -> str:
        """List the apps running on this Mac and which one is in front."""
        return run(lambda mac: mac.list_apps())

    @server.tool(annotations=look)
    async def read_window(
        app: Optional[str] = None, include_menus: bool = False, limit: int = 150
    ) -> str:
        """Read what a running app shows: its window as numbered buttons, fields and text.

        Changes nothing and needs no approval, so read before acting. app
        defaults to the app used last. include_menus adds the menu bar (and the
        items of an open menu).
        """
        return run(lambda mac: mac.read_window(app, menus=include_menus, limit=limit))

    @server.tool(annotations=act)
    async def open_app(name: str) -> str:
        """Launch an app that is not running (or switch to it), then read its window.

        To look at an app that is already running, use read_window instead.
        """
        return run(lambda mac: mac.open_app(name))

    @server.tool(annotations=act)
    async def open_url(url: str, app: Optional[str] = None) -> str:
        """Open a link in an app on this Mac, then read its window.

        app picks the app (e.g. "Notion" opens a notion.so or app.notion.com
        page in the Notion app); without it the link opens where macOS
        usually opens it. Deep links such as notion://, slack:// or mailto:
        work too. The app comes to the front.
        """
        return run(lambda mac: mac.open_url(url, app=app))

    @server.tool(annotations=act)
    async def click(element: int, double: bool = False) -> str:
        """Click a numbered element from the latest read, then read the window again."""
        return run(lambda mac: mac.click(element, double=double))

    @server.tool(annotations=act)
    async def type_text(
        text: str,
        element: Optional[int] = None,
        submit: bool = False,
        in_front: bool = False,
    ) -> str:
        """Type text into a numbered field (or wherever the cursor is); submit presses Return.

        Typing reaches the app behind other windows. Set in_front only when
        typing in the background had no effect.
        """
        return run(
            lambda mac: mac.type_text(
                text, element=element, submit=submit, in_front=in_front
            )
        )

    @server.tool(annotations=act)
    async def press_keys(
        keys: str, app: Optional[str] = None, in_front: bool = False
    ) -> str:
        """Press keys or shortcuts, e.g. "cmd+n", "down down return", "escape".

        Keys reach the app behind other windows. Set in_front only when they
        had no effect there.
        """
        return run(lambda mac: mac.press_keys(keys, app=app, in_front=in_front))

    @server.tool(annotations=act)
    async def scroll(
        direction: str = "down", amount: int = 5, element: Optional[int] = None
    ) -> str:
        """Scroll the window (or a numbered element) up or down by a few lines."""
        return run(lambda mac: mac.scroll(direction, amount=amount, element=element))

    return server


def run_desktop_mcp_server(
    state_path: Optional[str] = None, screenshots: Optional[str] = None
) -> None:
    create_desktop_mcp_server(state_path=state_path, screenshots=screenshots).run(
        transport="stdio"
    )

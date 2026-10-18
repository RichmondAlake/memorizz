"""Desktop control: outlines, acting by number, and the MCP server's policy.

A fake accessibility tree stands in for macOS, so nothing here touches the
screen and the tests run on any platform.
"""

import asyncio
from pathlib import Path

import pytest

from memorizz.desktop_control import DesktopError, MacDesktop, parse_keys
from memorizz.desktop_control.macos import App

CMD, SHIFT = 1 << 20, 1 << 17


def node(role, *children, **attributes):
    values = {"AXRole": role, "AXChildren": list(children)}
    values.update(attributes)
    return values


class FakeAX:
    def __init__(self, roots, *, trusted=True, front=None, can_raise=True):
        self.roots = roots
        self.is_trusted = trusted
        self.front = front
        self.can_raise = can_raise
        self.performed = []
        self.focused = []

    def trusted(self, prompt=False):
        return self.is_trusted

    def application(self, pid):
        return self.roots[pid]

    def attribute(self, element, name):
        return element.get(name)

    def attributes(self, element, names):
        return {name: element[name] for name in names if name in element}

    def actions(self, element):
        return element.get("actions", [])

    def perform(self, element, action):
        self.performed.append((element.get("AXTitle") or element.get("AXRole"), action))
        return action in element.get("actions", [])

    def set(self, element, name, value):
        if name == "AXFrontmost" and self.can_raise:
            self.front = element["pid"]
            return True
        if name == "AXFocused" and element.get("focusable"):
            self.focused.append(
                element.get("AXPlaceholderValue") or element.get("AXTitle")
            )
            return True
        return False


class FakeInput:
    def __init__(self):
        self.events = []

    def click(self, x, y, count=1):
        self.events.append(("click", x, y, count))

    def key(self, code, flags, pid):
        self.events.append(("key", code, flags, pid))

    def text(self, character, pid):
        self.events.append(("text", character, pid))

    def scroll(self, x, y, lines, pid):
        self.events.append(("scroll", x, y, lines, pid))


class FakeApps:
    def __init__(self, apps, installed=None, front=None):
        self.apps = list(apps)
        self.front = front
        self.installed = installed or {}
        self.launched = []

    def running(self):
        return list(self.apps)

    def front_pid(self):
        return self.front()

    def by_bundle(self, bundle_id):
        return [app for app in self.apps if app.bundle_id == bundle_id]

    def locate(self, name):
        found = self.installed.get(name)
        return (found[0], found[1]) if found else None

    def launch(self, path):
        self.launched.append(path)
        for app_path, bundle_id, app in self.installed.values():
            if app_path == path:
                self.apps.append(app)

    def raise_app(self, pid):
        pass


def calculator_window():
    return node(
        "AXWindow",
        node("AXButton", AXTitle="Clear", actions=["AXPress"]),
        node("AXButton", node("AXStaticText", AXValue="7"), actions=["AXPress"]),
        node(
            "AXTextField",
            AXPlaceholderValue="Search",
            AXValue="hello",
            AXFocused=True,
            focusable=True,
        ),
        node(
            "AXTextField",
            AXSubrole="AXSecureTextField",
            AXTitle="Password",
            AXValue="hunter2",
        ),
        node("AXStaticText", AXValue="\u200e84"),
        node(
            "AXScrollArea",
            node(
                "AXStaticText",
                AXValue="below the fold",
                AXPosition=(0.0, 2000.0),
                AXSize=(100.0, 20.0),
            ),
            AXPosition=(0.0, 0.0),
            AXSize=(400.0, 600.0),
        ),
        node("AXCheckBox", AXTitle="Scientific", AXValue=1),
        node("AXGroup", AXDescription="Card", actions=["AXPress"]),
        node(
            "AXRow",
            node("AXStaticText", AXValue="Groceries"),
            AXPosition=(10.0, 100.0),
            AXSize=(200.0, 20.0),
        ),
        # A toolbar outside the window's reported frame (Chrome on a second display).
        node(
            "AXButton",
            AXTitle="Back",
            AXPosition=(0.0, -40.0),
            AXSize=(20.0, 20.0),
            actions=["AXPress"],
        ),
        node(
            "AXPopUpButton",
            node("AXMenu", node("AXMenuItem", AXTitle="Hidden item")),
            AXTitle="Mode",
            AXValue="Basic",
        ),
        node("AXButton", AXSubrole="AXCloseButton", actions=["AXPress"]),
        AXTitle="Calculator",
        AXPosition=(0.0, 0.0),
        AXSize=(400.0, 600.0),
    )


def make_desktop(*, trusted=True, can_raise=True, front=1):
    window = calculator_window()
    calculator = node("AXApplication", AXFocusedWindow=window, pid=7)
    terminal = node("AXApplication", pid=1)
    ax = FakeAX(
        {7: calculator, 1: terminal}, trusted=trusted, front=front, can_raise=can_raise
    )
    apps = FakeApps(
        [App("cmux", 1, "com.cmux"), App("Calculator", 7, "com.apple.calculator")],
        installed={
            "Notes": (
                "/System/Applications/Notes.app",
                "com.apple.Notes",
                App("Notes", 9, "com.apple.Notes"),
            )
        },
        front=lambda: ax.front,
    )
    ax.roots[9] = node(
        "AXApplication", AXFocusedWindow=node("AXWindow", AXTitle="Notes"), pid=9
    )
    desktop = MacDesktop(accessibility=ax, keyboard=FakeInput(), apps=apps)
    desktop.settle = 0
    desktop.front_timeout = 0.01
    return desktop


def test_parse_keys_reads_chords_and_sequences():
    assert parse_keys("cmd+shift+t") == [(17, CMD | SHIFT)]
    assert parse_keys("down, down return") == [(125, 0), (125, 0), (36, 0)]
    with pytest.raises(DesktopError, match="Unknown key"):
        parse_keys("cmd+banana")
    with pytest.raises(DesktopError, match="only modifiers"):
        parse_keys("cmd+shift")


def test_read_window_numbers_what_a_person_can_see_and_use():
    outline = make_desktop().read_window("Calculator")
    lines = outline.splitlines()
    assert lines[0] == 'Calculator — window "Calculator"'
    assert lines[1:12] == [
        '[1] button "Clear"',
        '[2] button "7"',
        '[3] text field "Search" = "hello" (focused)',
        '[4] password field "Password"',
        '[5] text "84"',
        '[6] checkbox "Scientific" = on',
        '[7] clickable group "Card"',
        '[8] row "Groceries"',
        '[9] button "Back"',
        '[10] pop-up button "Mode" = "Basic"',
        "[11] close window button",
    ]
    # Scrolled-out text, a closed menu and a password stay out of the outline.
    assert "below the fold" not in outline
    assert "Hidden item" not in outline
    assert "hunter2" not in outline


def test_read_window_explains_missing_accessibility():
    with pytest.raises(DesktopError, match="Accessibility"):
        make_desktop(trusted=False).read_window("Calculator")


def test_click_presses_through_accessibility_without_moving_the_pointer():
    desktop = make_desktop()
    desktop.read_window("Calculator")
    result = desktop.click(1)
    assert result.startswith('Pressed [1] button "Clear".')
    assert ("Clear", "AXPress") in desktop.ax.performed
    assert desktop.input.events == []


def test_click_falls_back_to_the_pointer_at_the_element_center():
    desktop = make_desktop()
    desktop.read_window("Calculator")
    result = desktop.click(8)
    assert result.startswith('Clicked [8] row "Groceries".')
    assert desktop.input.events == [("click", 110.0, 110.0, 1)]
    assert desktop.ax.front == 7


def test_the_pointer_refuses_when_the_app_cannot_come_forward():
    desktop = make_desktop(can_raise=False)
    desktop.read_window("Calculator")
    with pytest.raises(DesktopError, match="Could not bring Calculator"):
        desktop.click(8)
    with pytest.raises(DesktopError, match="Could not bring Calculator"):
        desktop.type_text("hello", in_front=True)
    assert desktop.input.events == []


def test_keys_and_typing_go_to_the_app_without_taking_focus():
    desktop = make_desktop()
    desktop.read_window("Calculator")
    desktop.type_text("7")
    desktop.press_keys("return")
    desktop.scroll("down", amount=3)
    assert desktop.input.events == [
        ("text", "7", 7),
        ("key", 36, 0, 7),
        ("scroll", 200.0, 300.0, -3, 7),
    ]
    assert desktop.ax.front == 1  # the user's app stays in front


def test_typing_needs_a_target_app_and_focuses_the_field():
    desktop = make_desktop()
    with pytest.raises(DesktopError, match="Open or read an app first"):
        desktop.type_text("hello")
    desktop.read_window("Calculator")
    result = desktop.type_text("a\nb", element=3, submit=True)
    assert desktop.ax.focused == ["Search"]
    assert desktop.input.events == [
        ("text", "a", 7),
        ("key", 36, 0, 7),
        ("text", "b", 7),
        ("key", 36, 0, 7),
    ]
    assert result.startswith("Typed 3 characters into Calculator and pressed Return.")


def test_press_keys_goes_to_the_app_last_read():
    desktop = make_desktop()
    desktop.read_window("Calculator")
    desktop.press_keys("cmd+n")
    assert desktop.input.events == [("key", 45, CMD, 7)]
    assert desktop.ax.front == 1


def test_numbers_survive_a_new_server_process(tmp_path):
    state = tmp_path / "last-read.json"
    first = make_desktop()
    first.state_path = state
    first.read_window("Calculator")

    # MCP hosts may start a fresh server for the next call.
    second = MacDesktop(
        accessibility=first.ax, keyboard=FakeInput(), apps=first.apps, state_path=state
    )
    second.settle = 0
    assert second.click(1).startswith('Pressed [1] button "Clear".')
    assert ("Clear", "AXPress") in first.ax.performed
    second.type_text("5")  # still knows which app was read last
    assert second.input.events == [("text", "5", 7)]


def test_a_changed_window_asks_for_a_fresh_read_instead_of_guessing(tmp_path):
    state = tmp_path / "last-read.json"
    first = make_desktop()
    first.state_path = state
    first.read_window("Calculator")
    window = first.ax.roots[7]["AXFocusedWindow"]
    window["AXChildren"][0]["AXTitle"] = "Delete everything"

    second = MacDesktop(
        accessibility=first.ax, keyboard=FakeInput(), apps=first.apps, state_path=state
    )
    with pytest.raises(DesktopError, match="window changed"):
        second.click(1)
    assert first.ax.performed == []


class FakeScreen:
    def __init__(self, allowed=True):
        self.is_allowed = allowed

    def allowed(self, prompt=False):
        return self.is_allowed

    def capture(self, pid, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpg")
        return True


def test_each_step_leaves_a_screenshot_path_for_workflow_memory(tmp_path):
    desktop = make_desktop()
    desktop.screenshots = tmp_path / "screenshots"
    desktop._screen = FakeScreen()
    result = desktop.read_window("Calculator")
    line = result.splitlines()[-1]
    assert line.startswith("Screenshot: ") and line.endswith("-calculator-read.jpg")
    assert Path(line.split(": ", 1)[1]).read_bytes() == b"jpg"
    assert desktop.click(1).splitlines()[-1].endswith("-calculator-click.jpg")


def test_no_screenshot_without_screen_recording(tmp_path):
    desktop = make_desktop()
    desktop.screenshots = tmp_path / "screenshots"
    desktop._screen = FakeScreen(allowed=False)
    assert "Screenshot:" not in desktop.read_window("Calculator")
    assert not (tmp_path / "screenshots").exists()


def test_stale_numbers_ask_for_a_fresh_read():
    desktop = make_desktop()
    with pytest.raises(DesktopError, match="no element \\[3\\]"):
        desktop.click(3)


def test_open_app_launches_an_installed_app_and_reads_it():
    desktop = make_desktop()
    result = desktop.open_app("Notes")
    assert desktop.apps.launched == ["/System/Applications/Notes.app"]
    assert result.startswith('Opened Notes.\n\nNotes — window "Notes"')
    with pytest.raises(DesktopError, match="No app called 'Nope'"):
        desktop.open_app("Nope")
    with pytest.raises(DesktopError, match="Give the app's name"):
        desktop.open_app("-n")


def test_open_app_without_accessibility_still_opens():
    desktop = make_desktop(trusted=False)
    assert desktop.open_app("Notes").startswith("Opened Notes. macOS has not given")


def test_server_marks_reads_free_and_actions_for_approval():
    pytest.importorskip("mcp")
    from memorizz.desktop_control.server import create_desktop_mcp_server
    from memorizz.mcp.security import tool_is_mutating

    server = create_desktop_mcp_server(make_desktop())
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert set(tools) == {
        "list_apps",
        "read_window",
        "open_app",
        "click",
        "type_text",
        "press_keys",
        "scroll",
    }
    mutating = {
        name
        for name, tool in tools.items()
        if tool_is_mutating(
            name,
            {
                "annotations": tool.annotations.model_dump(
                    by_alias=True, exclude_none=True
                )
            },
        )
    }
    assert mutating == {"open_app", "click", "type_text", "press_keys", "scroll"}


def test_server_returns_outlines_and_reports_desktop_errors():
    pytest.importorskip("mcp")
    from memorizz.desktop_control.server import create_desktop_mcp_server

    server = create_desktop_mcp_server(make_desktop())
    result = asyncio.run(server.call_tool("read_window", {"app": "Calculator"}))
    assert '[1] button "Clear"' in result.content[0].text
    with pytest.raises(Exception, match="no element \\[99\\]"):
        asyncio.run(server.call_tool("click", {"element": 99}))

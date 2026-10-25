# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Operate macOS apps through the Accessibility API.

``read_window`` turns an app's front window into a numbered outline of its
buttons, fields and text instead of a screenshot, so even a small model can act
by number (``click(7)``). Numbers come from the latest read; every read and
every action replaces them.

Keystrokes and pointer clicks go to whichever app is in front, so each one first
brings the target app forward and refuses to act when it cannot: nothing is
typed into the terminal running MemoRizz by mistake.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

ACCESSIBILITY_HINT = (
    "macOS has not given the app running MemoRizz (your terminal) Accessibility "
    "access, so apps can be opened but not read or operated. Turn it on in System "
    "Settings > Privacy & Security > Accessibility (memorizz desktop doctor names "
    "the app), then try again."
)
INSTALL_HINT = (
    "Desktop control needs pyobjc: pip install 'memorizz[desktop]' "
    "(or uv tool install 'memorizz[desktop]')."
)


class DesktopError(RuntimeError):
    """A desktop action could not run; the message says what to do instead."""


# Roles a person acts on, with the words the outline uses for them.
ACTION_ROLES = {
    "AXButton": "button",
    "AXMenuButton": "menu button",
    "AXPopUpButton": "pop-up button",
    "AXCheckBox": "checkbox",
    "AXRadioButton": "radio button",
    "AXTextField": "text field",
    "AXTextArea": "text area",
    "AXComboBox": "combo box",
    "AXLink": "link",
    "AXMenuItem": "menu item",
    "AXMenuBarItem": "menu",
    "AXSlider": "slider",
    "AXIncrementor": "stepper",
    "AXDisclosureTriangle": "disclosure",
    "AXRow": "row",
    "AXCell": "cell",
    "AXTab": "tab",
}
TEXT_ROLES = {"AXStaticText": "text", "AXHeading": "heading"}
_SUBROLES = {
    "AXSearchField": "search field",
    "AXSecureTextField": "password field",
    "AXTabButton": "tab",
    "AXOutlineRow": "row",
    "AXCloseButton": "close window button",
    "AXMinimizeButton": "minimize button",
    "AXZoomButton": "zoom button",
    "AXFullScreenButton": "full screen button",
}
# Fields take focus rather than a press, and show their current value.
FIELD_ROLES = {"AXTextField", "AXTextArea", "AXComboBox"}
_VALUE_ROLES = FIELD_ROLES | {"AXSlider", "AXIncrementor", "AXPopUpButton"}
_TOGGLE_ROLES = {"AXCheckBox", "AXRadioButton"}
# Their children are a closed menu or list, not something on screen.
_OPAQUE_ROLES = {"AXPopUpButton", "AXMenuButton", "AXComboBox", "AXMenuBarItem"}
# Labelled containers that web views and Electron apps sometimes make pressable.
_PRESSABLE_CONTAINERS = {"AXGroup", "AXImage"}
_PRESS_ACTIONS = ("AXPress", "AXConfirm", "AXPick")

_ATTRIBUTES = (
    "AXRole",
    "AXSubrole",
    "AXTitle",
    "AXDescription",
    "AXValue",
    "AXPlaceholderValue",
    "AXEnabled",
    "AXFocused",
    "AXSelected",
    "AXPosition",
    "AXSize",
    "AXChildren",
)

_KEY_CODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25,
    "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33,
    "i": 34, "p": 35, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42,
    ",": 43, "/": 44, "n": 45, "m": 46, ".": 47, "`": 50,
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51,
    "backspace": 51, "escape": 53, "esc": 53, "forwarddelete": 117,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121, "left": 123,
    "right": 124, "down": 125, "up": 126, "f1": 122, "f2": 120, "f3": 99,
    "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109,
    "f11": 103, "f12": 111,
}  # fmt: skip
_MODIFIERS = {
    "cmd": 1 << 20,
    "command": 1 << 20,
    "shift": 1 << 17,
    "ctrl": 1 << 18,
    "control": 1 << 18,
    "alt": 1 << 19,
    "option": 1 << 19,
    "opt": 1 << 19,
    "fn": 1 << 23,
}
_RETURN, _TAB = _KEY_CODES["return"], _KEY_CODES["tab"]


def parse_keys(keys: str) -> List[Tuple[int, int]]:
    """``"cmd+shift+t down return"`` -> ``[(key code, modifier flags), ...]``."""
    chords: List[Tuple[int, int]] = []
    for chord in str(keys or "").replace(",", " ").split():
        flags, code = 0, None
        for part in (piece.strip().lower() for piece in chord.split("+")):
            if part in _MODIFIERS:
                flags |= _MODIFIERS[part]
            elif part in _KEY_CODES and code is None:
                code = _KEY_CODES[part]
            else:
                raise DesktopError(
                    f"Unknown key {part!r} in {chord!r}. Use letters, digits, "
                    "return, tab, space, escape, delete, up/down/left/right, "
                    "home/end, pageup/pagedown or f1-f12, joined to cmd, shift, "
                    "ctrl or option with '+'."
                )
        if code is None:
            raise DesktopError(f"{chord!r} has no key to press, only modifiers.")
        chords.append((code, flags))
    if not chords:
        raise DesktopError("Say which keys to press, e.g. 'cmd+n' or 'return'.")
    return chords


@dataclass
class App:
    name: str
    pid: int
    bundle_id: str = ""
    active: bool = False


@dataclass
class _Element:
    pid: int
    ref: Any  # None once the number has been reloaded in a new process
    role: str
    kind: str
    label: str
    frame: Optional[Tuple[float, float, float, float]]
    # Where to find it again: child indices from the window, the menu bar or
    # the app, with the role, own label and window title it must still have.
    base: str = "window"
    path: Tuple[int, ...] = ()
    own_label: str = ""
    window: str = ""

    def describe(self) -> str:
        return f'{self.kind} "{self.label}"' if self.label else self.kind


@dataclass
class _Walk:
    pid: int
    limit: int
    bounds: Optional[Tuple[float, float, float, float]] = None
    window: str = ""
    visited: int = 0
    truncated: bool = False
    lines: List[str] = field(default_factory=list)
    elements: Dict[int, _Element] = field(default_factory=dict)


def _clean(value: Any, limit: int = 160) -> str:
    if not isinstance(value, str):
        return ""
    # Drop invisible format marks (Calculator's display starts with U+200E).
    visible = "".join(ch for ch in value if unicodedata.category(ch) != "Cf")
    text = " ".join(visible.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _frame(attrs: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    position, size = attrs.get("AXPosition"), attrs.get("AXSize")
    if not (isinstance(position, tuple) and isinstance(size, tuple)):
        return None
    return (position[0], position[1], size[0], size[1])


def _overlaps(inner, outer) -> bool:
    # Zero-sized containers still hold visible children in some apps.
    if inner[2] <= 0 or inner[3] <= 0:
        return True
    return (
        inner[0] < outer[0] + outer[2]
        and inner[0] + inner[2] > outer[0]
        and inner[1] < outer[1] + outer[3]
        and inner[1] + inner[3] > outer[1]
    )


def _center(frame) -> Tuple[float, float]:
    return (frame[0] + frame[2] / 2, frame[1] + frame[3] / 2)


def _value(role: str, subrole: str, attrs: Dict[str, Any]) -> str:
    value = attrs.get("AXValue")
    if subrole == "AXSecureTextField" or value is None:
        return ""
    if role in _TOGGLE_ROLES:
        return "on" if value in (1, True) else "off"
    if role in _VALUE_ROLES:
        text = _clean(value if isinstance(value, str) else str(value), 120)
        return f'"{text}"' if text else ""
    return ""


class _Accessibility:
    """The pyobjc Accessibility calls MacDesktop makes; tests swap in a fake."""

    def __init__(self) -> None:
        try:
            import ApplicationServices as ax
        except ImportError as exc:
            raise DesktopError(INSTALL_HINT) from exc
        self._ax = ax

    def trusted(self, prompt: bool = False) -> bool:
        ax = self._ax
        if prompt:
            options = {ax.kAXTrustedCheckOptionPrompt: True}
            return bool(ax.AXIsProcessTrustedWithOptions(options))
        return bool(ax.AXIsProcessTrusted())

    def application(self, pid: int) -> Any:
        element = self._ax.AXUIElementCreateApplication(pid)
        self._ax.AXUIElementSetMessagingTimeout(element, 3.0)
        return element

    def _check(self, error: int) -> None:
        if error == self._ax.kAXErrorAPIDisabled:
            raise DesktopError(ACCESSIBILITY_HINT)

    def attribute(self, element: Any, name: str) -> Any:
        error, value = self._ax.AXUIElementCopyAttributeValue(element, name, None)
        self._check(error)
        return self._plain(value) if error == 0 else None

    def attributes(self, element: Any, names: Sequence[str]) -> Dict[str, Any]:
        error, values = self._ax.AXUIElementCopyMultipleAttributeValues(
            element, list(names), 0, None
        )
        self._check(error)
        if error != 0 or values is None:
            return {}
        result = {}
        for name, value in zip(names, values):
            plain = self._plain(value)
            if plain is not None:
                result[name] = plain
        return result

    def _plain(self, value: Any) -> Any:
        """Points and sizes as tuples, strings as str, AX errors as None."""
        ax = self._ax
        if value is None:
            return None
        if type(value).__name__ == "AXValueRef":
            kind = ax.AXValueGetType(value)
            if kind == ax.kAXValueCGPointType:
                ok, point = ax.AXValueGetValue(value, kind, None)
                return (float(point.x), float(point.y)) if ok else None
            if kind == ax.kAXValueCGSizeType:
                ok, size = ax.AXValueGetValue(value, kind, None)
                return (float(size.width), float(size.height)) if ok else None
            return None  # AX errors and ranges
        if isinstance(value, str):
            return str(value)
        if isinstance(value, (list, tuple)) or type(value).__name__.endswith("Array"):
            return list(value)
        return value

    def actions(self, element: Any) -> List[str]:
        error, names = self._ax.AXUIElementCopyActionNames(element, None)
        self._check(error)
        return [str(name) for name in names or []] if error == 0 else []

    def perform(self, element: Any, action: str) -> bool:
        return self._ax.AXUIElementPerformAction(element, action) == 0

    def set(self, element: Any, name: str, value: Any) -> bool:
        return self._ax.AXUIElementSetAttributeValue(element, name, value) == 0


class _Input:
    """Synthetic input. Keys, text and scrolling go straight to one app's
    process, so they work behind other windows and never mix with what the
    user is typing elsewhere; only a pointer click needs the app in front."""

    def __init__(self) -> None:
        try:
            import Quartz as quartz
        except ImportError as exc:
            raise DesktopError(INSTALL_HINT) from exc
        self._q = quartz

    def _post(self, event: Any, pid: Optional[int] = None) -> None:
        if pid is None:
            self._q.CGEventPost(self._q.kCGHIDEventTap, event)
        else:
            self._q.CGEventPostToPid(pid, event)

    def _move(self, x: float, y: float) -> None:
        q = self._q
        self._post(
            q.CGEventCreateMouseEvent(
                None, q.kCGEventMouseMoved, (x, y), q.kCGMouseButtonLeft
            )
        )

    def click(self, x: float, y: float, count: int = 1) -> None:
        q = self._q
        self._move(x, y)
        time.sleep(0.05)
        for clicks in range(1, count + 1):
            for kind in (q.kCGEventLeftMouseDown, q.kCGEventLeftMouseUp):
                event = q.CGEventCreateMouseEvent(
                    None, kind, (x, y), q.kCGMouseButtonLeft
                )
                q.CGEventSetIntegerValueField(event, q.kCGMouseEventClickState, clicks)
                self._post(event)
            time.sleep(0.05)

    def key(self, code: int, flags: int, pid: int) -> None:
        q = self._q
        for down in (True, False):
            event = q.CGEventCreateKeyboardEvent(None, code, down)
            q.CGEventSetFlags(event, flags)
            self._post(event, pid)
        time.sleep(0.03)

    def text(self, character: str, pid: int) -> None:
        # One character per event: some apps read only the first of a batch.
        q = self._q
        units = len(character.encode("utf-16-le")) // 2
        for down in (True, False):
            event = q.CGEventCreateKeyboardEvent(None, 0, down)
            q.CGEventKeyboardSetUnicodeString(event, units, character)
            self._post(event, pid)
        time.sleep(0.008)

    def scroll(self, x: float, y: float, lines: int, pid: int) -> None:
        q = self._q
        event = q.CGEventCreateScrollWheelEvent(
            None, q.kCGScrollEventUnitLine, 1, lines
        )
        q.CGEventSetLocation(event, (x, y))
        self._post(event, pid)


class _Screen:
    """Window screenshots saved as files (needs Screen Recording permission)."""

    def __init__(self) -> None:
        try:
            import Quartz as quartz
        except ImportError as exc:
            raise DesktopError(INSTALL_HINT) from exc
        self._q = quartz

    def allowed(self, prompt: bool = False) -> bool:
        if prompt:
            return bool(self._q.CGRequestScreenCaptureAccess())
        return bool(self._q.CGPreflightScreenCaptureAccess())

    def _window_id(self, pid: int) -> Optional[int]:
        q = self._q
        options = (
            q.kCGWindowListOptionOnScreenOnly | q.kCGWindowListExcludeDesktopElements
        )
        for info in q.CGWindowListCopyWindowInfo(options, q.kCGNullWindowID) or []:
            # Front to back, so the first normal window is the app's front one.
            if (
                int(info.get(q.kCGWindowOwnerPID, -1)) == pid
                and int(info.get(q.kCGWindowLayer, 1)) == 0
            ):
                return int(info[q.kCGWindowNumber])
        return None

    def capture(self, pid: int, path: Path) -> bool:
        window = self._window_id(pid)
        if window is None:
            return False  # minimised or on another Space
        path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["screencapture", "-x", "-o", "-t", "jpg", "-l", str(window), str(path)],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0 and path.is_file()


class _Apps:
    """Running apps from AppKit; launching and raising through ``open``."""

    def __init__(self) -> None:
        try:
            import AppKit as appkit
        except ImportError as exc:
            raise DesktopError(INSTALL_HINT) from exc
        self._appkit = appkit

    @staticmethod
    def _app(app: Any) -> App:
        return App(
            name=str(app.localizedName() or ""),
            pid=int(app.processIdentifier()),
            bundle_id=str(app.bundleIdentifier() or ""),
            active=bool(app.isActive()),
        )

    def _refresh(self) -> None:
        # NSWorkspace updates its running and frontmost apps from run-loop
        # notifications, so a process without an app run loop pumps it.
        appkit = self._appkit
        if threading.current_thread() is threading.main_thread():
            appkit.NSRunLoop.currentRunLoop().runUntilDate_(
                appkit.NSDate.dateWithTimeIntervalSinceNow_(0.05)
            )

    def front_pid(self) -> Optional[int]:
        self._refresh()
        app = self._appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app is not None else None

    def running(self) -> List[App]:
        appkit = self._appkit
        self._refresh()
        return [
            self._app(app)
            for app in appkit.NSWorkspace.sharedWorkspace().runningApplications()
            if app.activationPolicy() == appkit.NSApplicationActivationPolicyRegular
            and not app.isTerminated()
        ]

    def by_bundle(self, bundle_id: str) -> List[App]:
        found = (
            self._appkit.NSRunningApplication.runningApplicationsWithBundleIdentifier_(
                bundle_id
            )
        )
        return [self._app(app) for app in found or [] if not app.isTerminated()]

    def locate(self, name: str) -> Optional[Tuple[str, str]]:
        """The installed app's path and bundle id, if macOS knows the name."""
        appkit = self._appkit
        path = appkit.NSWorkspace.sharedWorkspace().fullPathForApplication_(name)
        if not path:
            return None
        bundle = appkit.NSBundle.bundleWithPath_(path)
        return str(path), str((bundle and bundle.bundleIdentifier()) or "")

    def launch(self, path: str) -> None:
        result = subprocess.run(
            ["open", "-a", path], capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise DesktopError(f"macOS could not open {path}: {detail}")

    def open_url(self, url: str, path: Optional[str] = None) -> None:
        command = ["open"] + (["-a", path] if path else []) + [url]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise DesktopError(f"macOS could not open {url}: {detail}")

    def handler(self, url: str) -> Optional[str]:
        """The name of the app macOS opens ``url`` with, if any."""
        appkit = self._appkit
        target = appkit.NSURL.URLWithString_(url)
        found = (
            appkit.NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(target)
            if target is not None
            else None
        )
        if found is None:
            return None
        bundle = appkit.NSBundle.bundleWithURL_(found)
        info = (bundle and bundle.infoDictionary()) or {}
        name = info.get("CFBundleDisplayName") or info.get("CFBundleName")
        return str(name or Path(str(found.path())).stem)

    def raise_app(self, pid: int) -> None:
        app = (
            self._appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(
                pid
            )
        )
        bundle = app.bundleURL() if app is not None else None
        if bundle is not None:
            subprocess.run(
                ["open", "-a", str(bundle.path())], capture_output=True, timeout=15
            )


class MacDesktop:
    """Open, read and operate the apps on this Mac."""

    settle = 0.6  # seconds an app gets to react before the window is read again
    front_timeout = 2.0
    launch_timeout = 20.0
    max_nodes = 6000
    max_depth = 60

    state_ttl = 3600.0  # seconds a saved read stays usable

    def __init__(
        self,
        *,
        accessibility=None,
        keyboard=None,
        apps=None,
        state_path=None,
        screenshots=None,
        screen=None,
    ) -> None:
        if accessibility is None and sys.platform != "darwin":
            raise DesktopError(f"Desktop control needs macOS; this is {sys.platform}.")
        self.ax = accessibility or _Accessibility()
        self.input = keyboard or _Input()
        self.apps = apps or _Apps()
        self._elements: Dict[int, _Element] = {}
        self._bounds: Dict[int, Tuple[float, float, float, float]] = {}
        self._pid: Optional[int] = None  # the app the last read or action targeted
        self._woken: set = set()
        # MCP hosts may start a new server process for every call, so the last
        # read is kept on disk for the next one.
        self.state_path = Path(state_path) if state_path else None
        self._load()
        # Each read and action leaves a screenshot of the app's window here,
        # and its path in the result, so a stored workflow shows every step.
        self.screenshots = Path(screenshots) if screenshots else None
        self._screen = screen

    def _snapshot(self, app: App, action: str) -> str:
        if self.screenshots is None:
            return ""
        try:
            if self._screen is None:
                self._screen = _Screen()
            if not self._screen.allowed():
                return ""
            now = time.time()
            stamp = (
                time.strftime("%H%M%S", time.localtime(now))
                + f"{int(now * 1000) % 1000:03d}"
            )
            name = "".join(
                ch if ch.isalnum() else "-" for ch in app.name.lower()
            ).strip("-")
            path = (
                self.screenshots
                / time.strftime("%Y-%m-%d", time.localtime(now))
                / f"{stamp}-{name or 'app'}-{action}.jpg"
            )
            if self._screen.capture(app.pid, path):
                return f"\n\nScreenshot: {path}"
        except (DesktopError, OSError, subprocess.SubprocessError):
            pass
        return ""

    def _save(self) -> None:
        if self.state_path is None:
            return
        data = {
            "saved_at": time.time(),
            "pid": self._pid,
            "bounds": {str(pid): list(frame) for pid, frame in self._bounds.items()},
            "elements": {
                str(number): {
                    "pid": item.pid,
                    "role": item.role,
                    "kind": item.kind,
                    "label": item.label,
                    "frame": list(item.frame) if item.frame else None,
                    "base": item.base,
                    "path": list(item.path),
                    "own_label": item.own_label,
                    "window": item.window,
                }
                for number, item in self._elements.items()
            },
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(".tmp")
            descriptor = os.open(
                str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            os.replace(temporary, self.state_path)
        except OSError:
            pass  # numbers then hold only within this process

    def _load(self) -> None:
        if self.state_path is None:
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if time.time() - float(data.get("saved_at") or 0) > self.state_ttl:
                return
            self._pid = data.get("pid")
            self._bounds = {
                int(pid): tuple(frame) for pid, frame in data["bounds"].items()
            }
            self._elements = {
                int(number): _Element(
                    pid=int(item["pid"]),
                    ref=None,
                    role=item["role"],
                    kind=item["kind"],
                    label=item["label"],
                    frame=tuple(item["frame"]) if item.get("frame") else None,
                    base=item.get("base", "window"),
                    path=tuple(item.get("path") or ()),
                    own_label=item.get("own_label", ""),
                    window=item.get("window", ""),
                )
                for number, item in data["elements"].items()
            }
        except (OSError, ValueError, KeyError, TypeError):
            self._elements, self._bounds, self._pid = {}, {}, None

    def _resolve(self, item: _Element) -> Any:
        """The live element behind a number, also in a later server process."""
        if item.ref is not None:
            return item.ref
        stale = DesktopError(
            "The window changed since the last read; call read_window again and "
            "use the new numbers."
        )
        root = self.ax.application(item.pid)
        if item.base == "window":
            node = self._window(root, 0)
            title = (
                _clean(self.ax.attributes(node, ("AXTitle",)).get("AXTitle"), 80)
                if node is not None
                else None
            )
            if node is None or title != item.window:
                raise stale
        elif item.base == "menubar":
            node = self.ax.attribute(root, "AXMenuBar")
        else:
            node = root
        for index in item.path:
            children = (
                (self.ax.attribute(node, "AXChildren") or [])
                if node is not None
                else []
            )
            if index >= len(children):
                raise stale
            node = children[index]
        attrs = self.ax.attributes(
            node, ("AXRole", "AXTitle", "AXDescription", "AXPlaceholderValue")
        )
        if attrs.get("AXRole") != item.role or self._label(attrs) != item.own_label:
            raise stale
        item.ref = node
        return node

    # -- state -------------------------------------------------------------

    def permissions(self, *, prompt: bool = False) -> Dict[str, Any]:
        if self._screen is None:
            self._screen = _Screen()
        return {
            "platform": sys.platform,
            "accessibility": self.ax.trusted(prompt=prompt),
            "screen_recording": self._screen.allowed(prompt=prompt),
        }

    def _require_access(self) -> None:
        if not self.ax.trusted():
            raise DesktopError(ACCESSIBILITY_HINT)

    def _front_pid(self) -> Optional[int]:
        return self.apps.front_pid()

    def _find(self, name: str) -> Optional[App]:
        wanted = str(name or "").strip().lower()
        if not wanted:
            return None
        apps = self.apps.running()
        exact = [
            app for app in apps if wanted in (app.name.lower(), app.bundle_id.lower())
        ]
        if exact:
            return exact[0]
        prefix = [app for app in apps if app.name.lower().startswith(wanted)]
        return prefix[0] if len(prefix) == 1 else None

    def _app(self, pid: int) -> App:
        found = next((app for app in self.apps.running() if app.pid == pid), None)
        return found or App(name=f"app {pid}", pid=pid)

    def _current_app(self) -> App:
        if self._pid is None:
            raise DesktopError(
                "Open or read an app first (open_app or read_window) so keys and "
                "text go to the right place."
            )
        return self._app(self._pid)

    def _element(self, number: int) -> _Element:
        try:
            return self._elements[int(number)]
        except (KeyError, TypeError, ValueError):
            raise DesktopError(
                f"There is no element [{number}] in the last read. Call read_window "
                "again and use a number from it."
            ) from None

    def _wait_front(self, pid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if self._front_pid() == pid:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def _bring_forward(self, app: App) -> None:
        """Put ``app`` in front, or raise: keys and pointer clicks go to the front app."""
        if self._front_pid() == app.pid:
            return
        self.ax.set(self.ax.application(app.pid), "AXFrontmost", True)
        if self._wait_front(app.pid, self.front_timeout / 2):
            return
        self.apps.raise_app(app.pid)
        if self._wait_front(app.pid, self.front_timeout):
            return
        raise DesktopError(
            f"Could not bring {app.name} to the front, so nothing was sent to it. "
            "Try open_app first."
        )

    # -- reading -----------------------------------------------------------

    def list_apps(self) -> str:
        apps = sorted(self.apps.running(), key=lambda app: app.name.lower())
        front = self._front_pid()
        lines = [
            f"- {app.name}" + (" (in front)" if app.pid == front else "")
            for app in apps
        ]
        return (
            "Running apps:\n"
            + "\n".join(lines)
            + "\n\nopen_app(name) brings one forward or launches another; "
            "read_window(app) shows what is in its window."
        )

    def read_window(
        self, app: Optional[str] = None, *, menus: bool = False, limit: int = 150
    ) -> str:
        self._require_access()
        if app:
            target = self._find(app)
            if target is None:
                raise DesktopError(
                    f"{app!r} is not running; open_app({app!r}) starts it."
                )
        elif self._pid is not None:
            target = self._app(self._pid)
        else:
            front = self._front_pid()
            if front is None:
                raise DesktopError(
                    "No app is in front; name one, e.g. read_window('Notes')."
                )
            target = self._app(front)
        return self._outline(target, menus=menus, limit=limit) + self._snapshot(
            target, "read"
        )

    def _window(self, root: Any, wait: float) -> Any:
        deadline = time.monotonic() + wait
        while True:
            for name in ("AXFocusedWindow", "AXMainWindow"):
                window = self.ax.attribute(root, name)
                if window is not None:
                    return window
            windows = self.ax.attribute(root, "AXWindows") or []
            if windows:
                return windows[0]
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.25)

    def _outline(
        self, app: App, *, menus: bool = False, limit: int = 150, wait: float = 0.0
    ) -> str:
        root = self.ax.application(app.pid)
        if app.pid not in self._woken:
            self._woken.add(app.pid)
            # Electron and Chromium apps build their tree only when asked.
            if self.ax.set(root, "AXManualAccessibility", True):
                time.sleep(0.3)
        state = _Walk(pid=app.pid, limit=max(10, min(int(limit or 150), 400)))
        window = self._window(root, wait)
        if window is None:
            header = f"{app.name} is open but shows no window."
        else:
            attrs = self.ax.attributes(window, ("AXTitle", "AXPosition", "AXSize"))
            state.bounds = _frame(attrs)
            if state.bounds:
                self._bounds[app.pid] = state.bounds
            title = _clean(attrs.get("AXTitle"), 80)
            state.window = title
            header = (
                f'{app.name} — window "{title}"'
                if title
                else f"{app.name} — front window"
            )
            self._walk(window, state, depth=0, inside=False)
        # Menus that are open right now: pop-ups, context menus.
        for index, child in enumerate(self.ax.attribute(root, "AXChildren") or []):
            if self.ax.attributes(child, ("AXRole",)).get("AXRole") == "AXMenu":
                self._menu_items(child, state, "app", (index,))
        if menus:
            menubar = self.ax.attribute(root, "AXMenuBar")
            if menubar is not None:
                self._menu_bar(menubar, state)
        self._elements = state.elements
        self._pid = app.pid
        self._save()
        footer = (
            "Act by number: click(n), type_text(text, element=n), "
            'press_keys("cmd+f"), scroll("down"). Numbers change after every '
            "read or action."
        )
        if state.truncated:
            footer = (
                f"Showing the first {state.limit} elements; scroll, or read with a "
                f"higher limit, for more. {footer}"
            )
        body = "\n".join(state.lines) or "(nothing readable in this window)"
        return f"{header}\n{body}\n\n{footer}"

    def _walk(
        self,
        element: Any,
        state: _Walk,
        *,
        depth: int,
        inside: bool,
        clip=None,
        path: Tuple[int, ...] = (),
    ) -> None:
        if state.truncated or state.visited >= self.max_nodes or depth > self.max_depth:
            return
        state.visited += 1
        attrs = self.ax.attributes(element, _ATTRIBUTES)
        frame = _frame(attrs)
        # Clip to the nearest scroll area, not the window: some windows (Chrome
        # on a second display) report a frame that leaves out their toolbar.
        if frame and clip and not _overlaps(frame, clip):
            return  # scrolled out of view
        role = str(attrs.get("AXRole") or "")
        subrole = str(attrs.get("AXSubrole") or "")
        children = list(attrs.get("AXChildren") or [])
        listed = False
        if role in ACTION_ROLES:
            label = self._label(attrs) or self._text_inside(children)
            kind = _SUBROLES.get(subrole) or ACTION_ROLES[role]
            self._add(
                state, element, role, subrole, kind, label, attrs, frame, path=path
            )
            listed = True
        elif role in TEXT_ROLES:
            if not inside:
                text = _clean(attrs.get("AXValue")) or self._label(attrs)
                if text:
                    self._add(
                        state,
                        element,
                        role,
                        subrole,
                        TEXT_ROLES[role],
                        text,
                        attrs,
                        frame,
                        path=path,
                    )
        elif role in _PRESSABLE_CONTAINERS and self._label(attrs):
            if any(action in _PRESS_ACTIONS for action in self.ax.actions(element)):
                kind = f"clickable {role[2:].lower()}"
                self._add(
                    state,
                    element,
                    role,
                    subrole,
                    kind,
                    self._label(attrs),
                    attrs,
                    frame,
                    path=path,
                )
                listed = True
        if role in _OPAQUE_ROLES:
            return
        if role == "AXScrollArea" and frame:
            clip = frame
        for index, child in enumerate(children):
            self._walk(
                child,
                state,
                depth=depth + 1,
                inside=inside or listed,
                clip=clip,
                path=path + (index,),
            )

    @staticmethod
    def _label(attrs: Dict[str, Any]) -> str:
        for name in ("AXTitle", "AXDescription", "AXPlaceholderValue"):
            text = _clean(attrs.get(name), 100)
            if text:
                return text
        return ""

    def _text_inside(self, children: List[Any], depth: int = 0) -> str:
        """Text shown inside an unlabelled button, link or row."""
        parts: List[str] = []
        for child in children:
            if len(parts) >= 4 or depth > 3:
                break
            attrs = self.ax.attributes(
                child, ("AXRole", "AXValue", "AXTitle", "AXDescription", "AXChildren")
            )
            if attrs.get("AXRole") in TEXT_ROLES:
                text = _clean(attrs.get("AXValue")) or self._label(attrs)
            else:
                text = self._text_inside(list(attrs.get("AXChildren") or []), depth + 1)
            if text:
                parts.append(text)
        return _clean(" ".join(parts), 100)

    def _add(
        self,
        state,
        element,
        role,
        subrole,
        kind,
        label,
        attrs,
        frame,
        *,
        path,
        base="window",
    ) -> None:
        number = len(state.elements) + 1
        state.elements[number] = _Element(
            state.pid,
            element,
            role,
            kind,
            label,
            frame,
            base=base,
            path=tuple(path),
            own_label=self._label(attrs),
            window=state.window,
        )
        line = f"[{number}] {kind}" + (f' "{label}"' if label else "")
        value = _value(role, subrole, attrs)
        if value and value != f'"{label}"':
            line += f" = {value}"
        flags = [
            name
            for name, attribute, wanted in (
                ("focused", "AXFocused", True),
                ("selected", "AXSelected", True),
                ("disabled", "AXEnabled", False),
            )
            if attrs.get(attribute) is wanted
        ]
        if flags:
            line += f" ({', '.join(flags)})"
        state.lines.append(line)
        if len(state.elements) >= state.limit:
            state.truncated = True

    def _menu_items(
        self, menu: Any, state: _Walk, base: str, path: Tuple[int, ...]
    ) -> None:
        for index, item in enumerate(self.ax.attribute(menu, "AXChildren") or []):
            attrs = self.ax.attributes(
                item, ("AXRole", "AXTitle", "AXEnabled", "AXSelected")
            )
            title = _clean(attrs.get("AXTitle"), 100)
            if title and attrs.get("AXRole") == "AXMenuItem":
                self._add(
                    state,
                    item,
                    "AXMenuItem",
                    "",
                    "menu item",
                    title,
                    attrs,
                    None,
                    path=path + (index,),
                    base=base,
                )

    def _menu_bar(self, menubar: Any, state: _Walk) -> None:
        for index, item in enumerate(self.ax.attribute(menubar, "AXChildren") or []):
            attrs = self.ax.attributes(
                item, ("AXRole", "AXTitle", "AXSelected", "AXChildren")
            )
            title = _clean(attrs.get("AXTitle"), 60)
            if not title or title == "Apple":
                continue
            self._add(
                state,
                item,
                "AXMenuBarItem",
                "",
                "menu",
                title,
                {"AXTitle": attrs.get("AXTitle")},
                None,
                path=(index,),
                base="menubar",
            )
            if attrs.get("AXSelected") is True:  # this menu is open
                for menu_index, menu in enumerate(attrs.get("AXChildren") or []):
                    self._menu_items(menu, state, "menubar", (index, menu_index))

    # -- acting --------------------------------------------------------------

    def open_app(self, name: str) -> str:
        wanted = str(name or "").strip()
        if not wanted or wanted.startswith("-"):
            raise DesktopError("Give the app's name, e.g. open_app('Notes').")
        app = self._find(wanted)
        if app is None:
            located = self.apps.locate(wanted)
            if located is None:
                raise DesktopError(
                    f"No app called {wanted!r} is installed. list_apps shows the "
                    "running ones."
                )
            path, bundle_id = located
            self.apps.launch(path)
            app = self._wait_for_launch(wanted, bundle_id)
        elif self.ax.trusted():
            self._bring_forward(app)
        else:
            self.apps.raise_app(app.pid)
        self._pid = app.pid
        if not self.ax.trusted():
            return f"Opened {app.name}. {ACCESSIBILITY_HINT}"
        return (
            f"Opened {app.name}.\n\n"
            + self._outline(app, wait=5.0)
            + self._snapshot(app, "open")
        )

    def open_url(self, url: str, app: Optional[str] = None) -> str:
        link = str(url or "").strip()
        scheme = urlparse(link).scheme.lower()
        if not scheme or link.startswith("-") or any(ch.isspace() for ch in link):
            raise DesktopError(
                "Give the whole link, e.g. open_url('https://www.notion.so/…') "
                "or a deep link such as 'notion://…' or 'slack://…'."
            )
        if scheme == "file":
            # A file can be a script macOS would run; apps go through open_app.
            raise DesktopError(
                "open_url opens links, not files. Use open_app to open an app."
            )
        wanted = str(app or "").strip()
        path = None
        if wanted:
            located = None if wanted.startswith("-") else self.apps.locate(wanted)
            if located is None:
                raise DesktopError(
                    f"No app called {wanted!r} is installed. Leave app out to "
                    "open the link with its usual app."
                )
            path = located[0]
        self.apps.open_url(link, path)
        name = wanted or self.apps.handler(link)
        found = None
        deadline = time.monotonic() + self.launch_timeout
        while name:
            found = self._find(name)
            if found is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.25)
        if found is None:
            return f"Opened {link}."
        self._pid = found.pid
        if not self.ax.trusted():
            return f"Opened {link} in {found.name}. {ACCESSIBILITY_HINT}"
        time.sleep(self.settle)  # the app loads the link after it comes forward
        return (
            f"Opened {link} in {found.name}.\n\n"
            + self._outline(found, wait=5.0)
            + self._snapshot(found, "open")
        )

    def _wait_for_launch(self, name: str, bundle_id: str) -> App:
        deadline = time.monotonic() + self.launch_timeout
        while True:
            found = self.apps.by_bundle(bundle_id) if bundle_id else []
            app = found[0] if found else self._find(name)
            if app is not None:
                return app
            if time.monotonic() >= deadline:
                raise DesktopError(
                    f"{name} did not start within {self.launch_timeout:.0f}s."
                )
            time.sleep(0.25)

    def click(self, element: int, *, double: bool = False) -> str:
        self._require_access()
        item = self._element(element)
        app = self._app(item.pid)
        ref = self._resolve(item)
        how = ""
        if not double:
            # A press reaches a window behind others, so the user's focus (and
            # their typing) stays where it is.
            if item.role in FIELD_ROLES and self.ax.set(ref, "AXFocused", True):
                how = "Focused"
            else:
                actions = self.ax.actions(ref)
                for action in _PRESS_ACTIONS:
                    if action in actions and self.ax.perform(ref, action):
                        how = "Pressed"
                        break
            if not how and item.role in {"AXRow", "AXCell"}:
                if self.ax.set(ref, "AXSelected", True):
                    how = "Selected"
        if not how:
            if not item.frame:
                raise DesktopError(
                    f"[{element}] cannot be pressed and has no place on screen; "
                    "try another element or press_keys."
                )
            # The last resort moves the real pointer, so the app comes forward.
            self._bring_forward(app)
            self.input.click(*_center(item.frame), count=2 if double else 1)
            how = "Double-clicked" if double else "Clicked"
        time.sleep(self.settle)
        return (
            f"{how} [{element}] {item.describe()}.\n\n"
            + self._outline(app)
            + self._snapshot(app, "click")
        )

    def type_text(
        self,
        text: str,
        *,
        element: Optional[int] = None,
        submit: bool = False,
        in_front: bool = False,
    ) -> str:
        self._require_access()
        if element is not None:
            item = self._element(element)
            app = self._app(item.pid)
            if not self.ax.set(self._resolve(item), "AXFocused", True):
                if not item.frame:
                    raise DesktopError(f"[{element}] cannot take typing; pick a field.")
                self._bring_forward(app)
                self.input.click(*_center(item.frame))
            time.sleep(0.15)
        else:
            app = self._current_app()
        if in_front:
            self._bring_forward(app)
        for index, line in enumerate(str(text).split("\n")):
            if index:
                self.input.key(_RETURN, 0, app.pid)
            for character in line:
                if character == "\t":
                    self.input.key(_TAB, 0, app.pid)
                else:
                    self.input.text(character, app.pid)
        if submit:
            self.input.key(_RETURN, 0, app.pid)
        time.sleep(self.settle)
        done = f"Typed {len(text)} characters into {app.name}"
        done += " and pressed Return." if submit else "."
        return f"{done}\n\n" + self._outline(app) + self._snapshot(app, "type")

    def press_keys(
        self, keys: str, *, app: Optional[str] = None, in_front: bool = False
    ) -> str:
        self._require_access()
        chords = parse_keys(keys)
        if app:
            target = self._find(app)
            if target is None:
                raise DesktopError(
                    f"{app!r} is not running; open_app({app!r}) starts it."
                )
        else:
            target = self._current_app()
        if in_front:
            self._bring_forward(target)
        for code, flags in chords:
            self.input.key(code, flags, target.pid)
        time.sleep(self.settle)
        return (
            f"Pressed {keys} in {target.name}.\n\n"
            + self._outline(target)
            + self._snapshot(target, "keys")
        )

    def scroll(
        self, direction: str = "down", *, amount: int = 5, element: Optional[int] = None
    ) -> str:
        self._require_access()
        way = str(direction or "down").strip().lower()
        if way not in {"up", "down"}:
            raise DesktopError("Scroll 'up' or 'down'.")
        lines = max(1, min(int(amount or 5), 50))
        if element is not None:
            item = self._element(element)
            app, frame = self._app(item.pid), item.frame
        else:
            app = self._current_app()
            frame = self._bounds.get(app.pid)
        if not frame:
            raise DesktopError("Read the window first so I know where to scroll.")
        self.input.scroll(*_center(frame), lines if way == "up" else -lines, app.pid)
        time.sleep(self.settle)
        return (
            f"Scrolled {way} in {app.name}.\n\n"
            + self._outline(app)
            + self._snapshot(app, "scroll")
        )

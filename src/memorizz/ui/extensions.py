"""Optional pages that are not part of the published package.

An extension is a module that exposes ``ui_extension() -> UIExtension``. The
UI looks for the modules named in :data:`LOCAL_EXTENSIONS` (present only in a
source checkout that has them), in ``MEMORIZZ_UI_EXTENSIONS`` (a comma
separated list of importable modules) and in the ``memorizz.ui_extensions``
entry point group. When none is found nothing changes: no route, no sidebar
item, no extra work at start-up.

An extension that fails to load is logged and skipped; it never stops the UI.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

#: Extension modules that live beside the package in some checkouts.
LOCAL_EXTENSIONS = ("memorizz.simuground",)

ENTRY_POINT_GROUP = "memorizz.ui_extensions"


@dataclass
class UIExtension:
    """What one optional page adds to the UI."""

    key: str  # the page's ``active_page`` value, e.g. "simuground"
    label: str  # sidebar and breadcrumb text
    href: str  # where the sidebar item goes
    router: Any  # a fastapi.APIRouter
    #: Sidebar group the item is listed under. Only "Build" is rendered today
    #: (base.html calls ``extension_nav('Build')``); add a call to use another.
    group: str = "Build"
    #: Inner SVG of a 24x24 stroke icon, in the sidebar's icon style.
    icon: str = '<circle cx="12" cy="12" r="8"/>'
    templates_dir: Optional[Path] = None
    #: Served at ``/ext/<key>``.
    static_dir: Optional[Path] = None
    #: Called once when the UI stops (close what the page keeps open).
    on_shutdown: Optional[Callable[[], None]] = None
    #: POST paths (regular expression) that are not worth an activity line.
    quiet_posts: Optional[str] = None
    #: Shown as a small tag beside the label, e.g. "Preview".
    badge: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def nav_item(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "href": self.href,
            "group": self.group,
            "icon": self.icon,
            "badge": self.badge,
        }


def _module_names() -> List[str]:
    names = [name for name in LOCAL_EXTENSIONS]
    configured = os.environ.get("MEMORIZZ_UI_EXTENSIONS", "")
    names.extend(part.strip() for part in configured.split(",") if part.strip())
    seen: set = set()
    return [name for name in names if not (name in seen or seen.add(name))]


def _from_module(name: str) -> Optional[UIExtension]:
    try:
        if importlib.util.find_spec(name) is None:
            return None
    except (ImportError, ValueError):
        return None
    module = importlib.import_module(name)
    factory = getattr(module, "ui_extension", None)
    if factory is None:
        return None
    return factory()


def discover() -> List[UIExtension]:
    """Every extension that can be loaded, in a stable order, one per key."""
    if os.environ.get("MEMORIZZ_UI_EXTENSIONS", "").strip().lower() in ("0", "off"):
        return []
    found: List[UIExtension] = []
    for name in _module_names():
        try:
            extension = _from_module(name)
        except Exception:
            logger.exception("UI extension %s could not be loaded", name)
            continue
        if extension is not None:
            found.append(extension)
    try:
        from importlib.metadata import entry_points

        for entry in entry_points(group=ENTRY_POINT_GROUP):
            try:
                extension = entry.load()()
            except Exception:
                logger.exception("UI extension %s could not be loaded", entry.name)
                continue
            if extension is not None:
                found.append(extension)
    except Exception:
        logger.debug("No UI extension entry points", exc_info=True)
    unique: dict = {}
    for extension in found:
        unique.setdefault(extension.key, extension)
    return list(unique.values())


def install(app: Any, templates: Any) -> List[UIExtension]:
    """Add every discovered extension to the app; return the ones installed."""
    installed: List[UIExtension] = []
    for extension in discover():
        try:
            _install_one(app, templates, extension)
        except Exception:
            logger.exception("UI extension %s could not be installed", extension.key)
            continue
        installed.append(extension)
    templates.env.globals["ui_extensions"] = [e.nav_item() for e in installed]
    app.state.ui_extensions = installed
    if installed:
        from . import activity

        activity.add_quiet_posts([e.quiet_posts for e in installed if e.quiet_posts])
    return installed


def _install_one(app: Any, templates: Any, extension: UIExtension) -> None:
    if extension.templates_dir is not None:
        from jinja2 import ChoiceLoader, FileSystemLoader

        loader = templates.env.loader
        added = FileSystemLoader(str(extension.templates_dir))
        if isinstance(loader, ChoiceLoader):
            if not any(
                str(extension.templates_dir) in getattr(item, "searchpath", [])
                for item in loader.loaders
            ):
                loader.loaders.append(added)
        else:
            # The UI's own templates stay first: an extension cannot replace one.
            templates.env.loader = ChoiceLoader([loader, added])
    if extension.static_dir is not None:
        from fastapi.staticfiles import StaticFiles

        app.mount(
            f"/ext/{extension.key}",
            StaticFiles(directory=str(extension.static_dir)),
            name=f"ext-{extension.key}",
        )
    app.include_router(extension.router)


def shutdown(app: Any) -> None:
    """Run every installed extension's shutdown hook."""
    for extension in getattr(app.state, "ui_extensions", None) or []:
        if extension.on_shutdown is None:
            continue
        try:
            extension.on_shutdown()
        except Exception:
            logger.exception("UI extension %s did not shut down cleanly", extension.key)

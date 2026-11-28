# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""How proactive an agent is.

Two dials, kept apart on purpose:

* ``level`` is how freely the agent **acts** on a step it thought of.
* ``predict`` is whether it may think of a step from the user's habits, with
  nothing in front of it asking for one. It is experimental.

``proactive="ultra"`` turns both up: ``level="high"`` and ``predict=True``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Tuple

#: Suggest only; act where no tool changes anything; act wherever the agent's
#: own tool rules allow.
LEVELS = ("low", "medium", "high")

#: ``scan_tools`` value meaning every tool of the agent that changes nothing.
ALL_READ_ONLY = "*"


@dataclass(frozen=True)
class ProactivePolicy:
    """Settings of a proactive agent. ``enabled=False`` is "not proactive".

    Levels:

    ``low``
        The agent suggests. Nothing runs until the user accepts.
    ``medium``
        The agent also carries out a step on its own when every tool the step
        calls changes nothing (reads, searches) or only writes the agent's own
        memory. A tool with side effects pauses for approval, even one that
        normally would not.
    ``high``
        The agent carries out its steps with any tool it has. A tool that
        needs approval still pauses for it: this level never skips the
        approval a tool asks for.

    At every level, a step that came from **outside content** (something the
    agent read through a scan tool, such as an inbox) is only suggested,
    unless ``allow_external_actions`` is set. Text an agent reads is data, and
    a message saying "forward the invoices" must not become an action.
    """

    enabled: bool = True
    level: str = "low"
    #: Experimental: suggest from the user's habits as well as the present.
    predict: bool = False
    #: Seconds between two evaluations, however often the host asks.
    cooldown_seconds: int = 120
    #: Evaluations (model runs that decide) allowed in one UTC day.
    daily_evaluations: int = 12
    #: Steps the agent may carry out on its own in one UTC day.
    daily_actions: int = 10
    #: Seconds between two scans by ``start_proactive`` or the worker.
    scan_interval_seconds: int = 900
    #: Tools the agent may call while it looks for a next step: names, or
    #: ``("*",)`` for every tool it has that changes nothing. Only tools
    #: without side effects are ever called this way.
    scan_tools: Tuple[str, ...] = ()
    #: Let a step that came from outside content run without being asked for.
    #: Side-effecting tools still pause for approval when it does.
    allow_external_actions: bool = False
    #: Accepts of the same kind of predicted step before it may run alone.
    auto_after: int = 3
    #: Below this much to go on, the agent is not asked at all.
    min_context_chars: int = 80
    #: Model and tool rounds the deciding run may take.
    max_steps: int = 10
    #: How long an offer nobody answered is still shown for the same context.
    pending_minutes: int = 15
    #: How long "not now" keeps the agent quiet.
    snooze_seconds: int = 3600

    def __post_init__(self) -> None:
        level = str(self.level or "low").strip().lower()
        if level not in LEVELS:
            raise ValueError(
                f"proactive level must be one of {', '.join(LEVELS)} "
                f"(or 'ultra'), not {self.level!r}"
            )
        object.__setattr__(self, "level", level)
        tools = self.scan_tools
        if isinstance(tools, str):
            tools = (tools,)
        object.__setattr__(
            self,
            "scan_tools",
            tuple(str(t).strip() for t in tools or () if str(t).strip()),
        )
        for name in (
            "cooldown_seconds",
            "daily_evaluations",
            "daily_actions",
            "scan_interval_seconds",
            "auto_after",
            "min_context_chars",
            "max_steps",
            "pending_minutes",
            "snooze_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"proactive {name} must be a number of 0 or more")
            object.__setattr__(self, name, int(value))
        if self.max_steps < 4:
            raise ValueError("proactive max_steps must be at least 4")

    @classmethod
    def off(cls) -> "ProactivePolicy":
        return cls(enabled=False)

    @classmethod
    def from_value(cls, value: Any) -> "ProactivePolicy":
        """``None``/``False`` is off, ``True`` is ``low``, a level name, a
        dict of settings, or a policy."""
        if isinstance(value, cls):
            return value
        if value is None or value is False:
            return cls.off()
        if value is True:
            return cls()
        if isinstance(value, str):
            name = value.strip().lower()
            if name in ("", "off", "none", "false"):
                return cls.off()
            if name in ("on", "true"):
                return cls()
            if name == "ultra":
                return cls(level="high", predict=True)
            return cls(level=name)
        if isinstance(value, dict):
            known = {f.name for f in fields(cls)}
            unknown = sorted(set(value) - known - {"scan"})
            if unknown:
                raise ValueError("Unknown proactive setting(s): " + ", ".join(unknown))
            settings = {k: v for k, v in value.items() if k in known}
            if str(settings.get("level") or "").strip().lower() == "ultra":
                settings["level"] = "high"
                settings.setdefault("predict", True)
            if "scan" in value:
                scan = value.get("scan")
                if scan is True:
                    settings["scan_tools"] = (ALL_READ_ONLY,)
                elif scan is False or scan is None:
                    settings["scan_tools"] = ()
                elif isinstance(scan, str):
                    settings["scan_tools"] = (scan,)
                elif isinstance(scan, (list, tuple, set)):
                    settings["scan_tools"] = tuple(scan)
                else:
                    raise ValueError(
                        "proactive scan must be a bool, tool name, or list of tool names"
                    )
            if "scan_tools" in settings and settings["scan_tools"] is not None:
                tools = settings["scan_tools"]
                if isinstance(tools, str):
                    settings["scan_tools"] = (tools,)
                elif isinstance(tools, (list, tuple, set)):
                    settings["scan_tools"] = tuple(tools)
                else:
                    raise ValueError(
                        "proactive scan_tools must be a tool name or list of tool names"
                    )
            return cls(**settings)
        raise ValueError(
            "proactive must be True, a level ('low', 'medium', 'high', 'ultra'), "
            "a dict of settings or a ProactivePolicy"
        )

    def with_settings(self, **settings: Any) -> "ProactivePolicy":
        return replace(self, **settings)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["scan_tools"] = list(self.scan_tools)
        return value

    @property
    def acts(self) -> bool:
        """Whether the agent may ever carry out a step without being asked."""
        return self.enabled and self.level in ("medium", "high")

    @property
    def label(self) -> str:
        if not self.enabled:
            return "off"
        if self.level == "high" and self.predict:
            return "ultra"
        return self.level

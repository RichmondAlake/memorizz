"""Where an agent's traces go: the settings and the contract every
observability provider meets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Providers an agent can use, the default first.
OBSERVABILITY_PROVIDERS = ("memorizz", "sentry")
DEFAULT_OBSERVABILITY_PROVIDER = "memorizz"

_FIELDS = (
    "provider",
    "org",
    "project",
    "url",
    "environment",
    "include_content",
    "sample_rate",
)
_ALIASES = {
    "name": "provider",
    "organization": "org",
    "organisation": "org",
    "org_slug": "org",
    "project_id": "project",
    "project_slug": "project",
    "dashboard_url": "url",
    "env": "environment",
    "traces_sample_rate": "sample_rate",
    "send_content": "include_content",
    "builtin": "memorizz",
    "default": "memorizz",
    "local": "memorizz",
}
# Credentials come from the environment (Settings), never from a saved agent.
_SECRET_WORDS = ("dsn", "api_key", "apikey", "secret", "token", "password")


@dataclass(frozen=True)
class ObservabilityConfig:
    """Where one agent's traces go.

    ``provider`` picks the service. The other fields are optional:

    * ``org`` and ``project``: the provider's organisation slug and project,
      used to build the link to its dashboard.
    * ``url``: the dashboard link itself, when it cannot be built (a
      self-hosted install, a saved search).
    * ``environment``: the environment name the provider files traces under.
    * ``include_content``: also send what the traces contain (tool arguments
      and results, reasoning text, the user id). Off unless asked for: only
      names, timings, token counts and statuses leave the process.
    * ``sample_rate``: the share of turns sent, 0 to 1 (1 when unset).
    * ``options``: settings only one provider understands.
    """

    provider: str = DEFAULT_OBSERVABILITY_PROVIDER
    org: Optional[str] = None
    project: Optional[str] = None
    url: Optional[str] = None
    environment: Optional[str] = None
    include_content: bool = False
    sample_rate: Optional[float] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "ObservabilityConfig":
        """Build a config from None/True (the built-in provider), a provider
        name, a dict, another ``ObservabilityConfig`` or a provider instance.
        Raises ValueError on anything else, on an unknown provider and on a
        credential in the dict."""
        if value is None or value is True or value is False:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, ObservabilityProvider):
            return value.config
        if isinstance(value, str):
            return cls(provider=provider_name(value))
        if not isinstance(value, dict):
            raise ValueError(
                "observability_provider must be a provider name, a dict, an "
                "ObservabilityConfig or an ObservabilityProvider"
            )
        known: Dict[str, Any] = {}
        options: Dict[str, Any] = dict(value.get("options") or {})
        for key, item in value.items():
            if key == "options":
                continue
            name = _ALIASES.get(str(key), str(key))
            if name in _FIELDS:
                known[name] = item
            else:
                options[str(key)] = item
        for key in options:
            if any(word in key.lower() for word in _SECRET_WORDS):
                raise ValueError(
                    f"observability_provider must not hold credentials ({key}); "
                    "set the provider's key in the environment or in Settings"
                )
        rate = known.get("sample_rate")
        if rate in ("", None):
            rate = None
        else:
            try:
                rate = float(rate)
            except (TypeError, ValueError):
                raise ValueError("observability sample_rate must be a number") from None
            if not 0.0 <= rate <= 1.0:
                raise ValueError("observability sample_rate must be between 0 and 1")
            if rate == 1.0:
                rate = None  # every turn is the default
        url = _text(known.get("url"))
        if url and not url.lower().startswith(("https://", "http://")):
            raise ValueError("observability url must start with https:// or http://")
        return cls(
            provider=provider_name(
                known.get("provider") or DEFAULT_OBSERVABILITY_PROVIDER
            ),
            org=_text(known.get("org")),
            project=_text(known.get("project")),
            url=url,
            environment=_text(known.get("environment")),
            include_content=_flag(known.get("include_content")),
            sample_rate=rate,
            options=options,
        )

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict holding only what was set (plus the provider)."""
        data: Dict[str, Any] = {"provider": self.provider}
        for name in ("org", "project", "url", "environment", "sample_rate"):
            item = getattr(self, name)
            if item is not None:
                data[name] = item
        if self.include_content:
            data["include_content"] = True
        if self.options:
            data["options"] = dict(self.options)
        return data

    @property
    def is_builtin(self) -> bool:
        return self.provider == DEFAULT_OBSERVABILITY_PROVIDER


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def provider_name(value: Any) -> str:
    """A provider name as the registry spells it. Raises ValueError when it
    names no provider."""
    from . import registered_providers

    name = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    name = _ALIASES.get(name, name)
    known = registered_providers()
    if name not in known:
        raise ValueError(
            f"Unknown observability provider {value!r}; choose one of "
            + ", ".join(known)
        )
    return name


class ObservabilityProvider:
    """One place an agent's traces are sent.

    A provider receives each finished turn as the trace events MemoRizz
    recorded for it and forwards them. Sending must never fail a turn: the
    agent calls :meth:`export_turn` inside a guard, and a provider that is
    not ready (no credentials, package missing) drops the turn and says why
    through :meth:`status`.

    To add one, subclass this, set ``name`` and ``label``, implement
    :meth:`export_turn`, and either pass an instance as
    ``MemAgent(observability_provider=...)`` or register the class with
    :func:`memorizz.observability.register_observability_provider` so saved
    agents can name it.
    """

    #: Registry name, as saved on an agent.
    name = ""
    #: What people call it.
    label = ""
    #: False for a provider that only describes MemoRizz's own trace store.
    external = True

    def __init__(self, config: Any = None) -> None:
        if isinstance(config, ObservabilityConfig):
            self.config = config
            return
        # The settings are checked as the built-in provider's and then given
        # this class's own name, so a provider nobody registered still builds.
        data = dict(config) if isinstance(config, dict) else {}
        data.pop("name", None)
        data["provider"] = DEFAULT_OBSERVABILITY_PROVIDER
        self.config = replace(
            ObservabilityConfig.from_value(data),
            provider=self.name or DEFAULT_OBSERVABILITY_PROVIDER,
        )

    # -- what a provider answers -------------------------------------------

    def missing(self) -> List[str]:
        """What stops this provider from sending (environment variables,
        packages). Empty when it is ready."""
        return []

    def dashboard_url(
        self, *, agent_id: Optional[str] = None, agent_name: Optional[str] = None
    ) -> Optional[str]:
        """Where a person opens this agent's traces at the provider."""
        return self.config.url

    def trace_url(self, root_trace_id: Any, *, started_at: Any = None) -> Optional[str]:
        """Where a person opens one turn's trace at the provider, given the
        turn's MemoRizz ``root_trace_id`` (and, when known, when it started).
        None when the provider has no page for a single trace."""
        return None

    def trace_reference(self, trace_context: Dict[str, Any]) -> Optional[str]:
        """The id the provider files a turn's trace under, when it differs
        from MemoRizz's ``root_trace_id``."""
        return None

    def status(self) -> Dict[str, Any]:
        """Whether the provider is ready, and what is missing when not."""
        missing = self.missing()
        return {
            "provider": self.name,
            "label": self.label or self.name,
            "external": bool(self.external),
            "ready": not missing,
            "missing": missing,
            "dashboard_url": self.dashboard_url(),
            "include_content": bool(self.config.include_content),
        }

    # -- what a provider does ----------------------------------------------

    def export_turn(
        self,
        trace_context: Dict[str, Any],
        events: List[Dict[str, Any]],
        *,
        agent_name: Optional[str] = None,
    ) -> Optional[str]:
        """Send one finished turn. ``events`` are the turn's trace events in
        the order they happened; ``trace_context`` names the agent, thread and
        ``root_trace_id``. Returns the provider's id for the trace, or None
        when nothing was sent."""
        raise NotImplementedError

    def export_mark(self) -> str:
        """What MemoRizz writes on its own copy of a turn this provider took,
        naming where the turn went. The provider's name, unless one provider
        can send to places that are opened differently."""
        return self.name

    def flush(self, timeout: float = 2.0) -> None:
        """Wait up to ``timeout`` seconds for queued traces to be sent."""

    def close(self) -> None:
        """Flush and release what the provider holds."""
        self.flush()


class MemorizzObservability(ObservabilityProvider):
    """The built-in provider: traces stay in the agent's memory provider and
    are read in the MemoRizz UI (Traces, Failures, Usage). It sends nothing
    anywhere; the agent's own trace store does the recording."""

    name = "memorizz"
    label = "MemoRizz"
    external = False

    def dashboard_url(
        self, *, agent_id: Optional[str] = None, agent_name: Optional[str] = None
    ) -> Optional[str]:
        if self.config.url:
            return self.config.url
        if agent_id:
            from urllib.parse import quote

            return f"/traces?agent_id={quote(str(agent_id), safe='')}"
        return "/traces"

    def export_turn(
        self,
        trace_context: Dict[str, Any],
        events: List[Dict[str, Any]],
        *,
        agent_name: Optional[str] = None,
    ) -> Optional[str]:
        return None


def observability_settings(
    observability: Any = None, observability_provider: Any = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[ObservabilityProvider]]:
    """An agent's ``(observability, observability_config, provider instance)``
    from its two arguments.

    ``observability`` is the switch: ``None`` or ``True`` records traces,
    ``False`` records none. ``observability_provider`` says where they go: a
    name (``"sentry"``), a dict (``{"provider": "sentry", "org": "acme"}``),
    an ``ObservabilityConfig`` or a provider instance; without one the traces
    go to MemoRizz's own store. A provider may also be given as
    ``observability`` itself (``observability="sentry"``).

    The config comes back as None for the built-in provider with nothing set,
    and the instance only when one was passed in. Raises ValueError for an
    unknown provider, a credential in the settings, or two providers.
    """
    provider = observability_provider
    if observability is not None and not isinstance(observability, bool):
        if provider is not None:
            raise ValueError(
                "Name the observability provider once: as observability or as "
                "observability_provider, not both"
            )
        provider, observability = observability, True
    enabled = True if observability is None else bool(observability)
    if provider is None or provider is True or provider is False:
        return enabled, None, None
    instance = provider if isinstance(provider, ObservabilityProvider) else None
    if instance is not None:
        from . import register_observability_provider, registered_providers

        # A provider passed as an object can be named by this process's saved
        # agents from now on (a refresh must not swap it for the built-in one).
        if instance.name and instance.name not in registered_providers():
            register_observability_provider(type(instance))
    config = ObservabilityConfig.from_value(provider)
    data = config.to_dict()
    if instance is None and data == {"provider": DEFAULT_OBSERVABILITY_PROVIDER}:
        return enabled, None, None
    return enabled, data, instance


def observability_change(
    current_enabled: Any = True,
    current_config: Any = None,
    *,
    enabled: Optional[bool] = None,
    provider: Optional[str] = None,
    org: Optional[str] = None,
    include_content: Optional[bool] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """An agent's ``(observability, observability_config)`` after a change.

    Each argument left as None keeps what the agent has. Naming a provider
    without ``enabled`` turns tracing on (asking for a provider means wanting
    its traces). Changing provider drops the settings that belonged to the
    old one, and ``"memorizz"`` goes back to MemoRizz's own store alone.
    ``org=""`` clears the organisation. Raises ValueError for an unknown
    provider, and for ``org`` or ``include_content`` when the traces stay in
    MemoRizz, which has neither.
    """
    config: Dict[str, Any] = (
        dict(current_config) if isinstance(current_config, dict) else {}
    )
    if provider is not None and str(provider).strip():
        chosen = provider_name(provider)
        if config.get("provider", DEFAULT_OBSERVABILITY_PROVIDER) != chosen:
            config = {}
        config["provider"] = chosen
    external = config.get("provider", DEFAULT_OBSERVABILITY_PROVIDER) != (
        DEFAULT_OBSERVABILITY_PROVIDER
    )
    if (org is not None and str(org).strip()) or include_content:
        if not external:
            raise ValueError(
                "An organisation and sending content belong to a provider such "
                "as sentry; name one with the provider option"
            )
    if org is not None and external:
        if str(org).strip():
            config["org"] = str(org).strip()
        else:
            config.pop("org", None)
    if include_content is not None and external:
        config["include_content"] = bool(include_content)
    if enabled is None:
        named = provider is not None and bool(str(provider).strip())
        is_on = True if named else current_enabled is not False
    else:
        is_on = bool(enabled)
    if not config:
        return is_on, None
    data = ObservabilityConfig.from_value(config).to_dict()
    if data == {"provider": DEFAULT_OBSERVABILITY_PROVIDER}:
        return is_on, None
    return is_on, data

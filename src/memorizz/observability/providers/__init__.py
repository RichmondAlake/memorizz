"""Observability providers: where an agent's traces go.

``MemAgent(observability=True)`` keeps traces in MemoRizz's own store, read in
the MemoRizz UI. ``observability_provider`` sends each turn somewhere else as
well::

    MemAgent(observability=True, observability_provider="sentry")
    MemAgent(observability_provider={"provider": "sentry", "org": "acme"})
    MemAgent(observability=False)   # record no traces at all

Credentials are read from the environment (or set in the UI's Settings); they
are never part of an agent's saved configuration.

========  ==============================================================
Provider  Needs
========  ==============================================================
memorizz  Nothing: the agent's memory provider holds the traces
sentry    SENTRY_DSN (or MEMORIZZ_SENTRY_DSN) and the ``sentry`` extra
========  ==============================================================
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Type

from .base import (
    DEFAULT_OBSERVABILITY_PROVIDER,
    OBSERVABILITY_PROVIDERS,
    MemorizzObservability,
    ObservabilityConfig,
    ObservabilityProvider,
    observability_change,
    observability_settings,
)


def _sentry() -> Type[ObservabilityProvider]:
    from .sentry import SentryObservability

    return SentryObservability


_REGISTRY: Dict[str, Callable[[], Type[ObservabilityProvider]]] = {
    "memorizz": lambda: MemorizzObservability,
    "sentry": _sentry,
}


def registered_providers() -> Tuple[str, ...]:
    """Every provider name an agent can use, the built-in one first."""
    return tuple(_REGISTRY)


def register_observability_provider(
    provider_class: Type[ObservabilityProvider],
) -> Type[ObservabilityProvider]:
    """Make a provider class available by its ``name``, so a saved agent can
    name it. Usable as a class decorator."""
    name = str(getattr(provider_class, "name", "") or "").strip().lower()
    if not name:
        raise ValueError("An observability provider class needs a name")
    _REGISTRY[name] = lambda: provider_class
    return provider_class


def create_observability_provider(config: Any = None) -> ObservabilityProvider:
    """The provider a configuration names (the built-in one when none is)."""
    if isinstance(config, ObservabilityProvider):
        return config
    resolved = ObservabilityConfig.from_value(config)
    return _REGISTRY[resolved.provider]()(resolved)


def observability_status() -> List[Dict[str, Any]]:
    """Every provider with whether it is ready to use."""
    return [create_observability_provider(name).status() for name in _REGISTRY]


__all__ = [
    "DEFAULT_OBSERVABILITY_PROVIDER",
    "OBSERVABILITY_PROVIDERS",
    "MemorizzObservability",
    "ObservabilityConfig",
    "ObservabilityProvider",
    "create_observability_provider",
    "observability_change",
    "observability_settings",
    "observability_status",
    "register_observability_provider",
    "registered_providers",
]

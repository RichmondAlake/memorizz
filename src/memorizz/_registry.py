# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Name-keyed provider registry shared by the peripheral packages.

Sandbox, internet-access and browser-control providers are all registered by
name and built the same way: the config is passed as keyword arguments first,
then as a single ``config=`` argument for providers with that constructor.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Type,
    TypeVar,
)

T = TypeVar("T")


def _lowercase(name: str) -> str:
    return name.lower()


class ProviderRegistry(Generic[T]):
    """Register provider classes by name and build configured instances.

    ``unknown_message`` is logged (``%s`` = name) when ``create`` meets an
    unregistered name; ``init_error_message`` (``%s`` = name, ``%s`` = config
    keys) is logged when neither constructor shape accepts the config, before
    the ``TypeError`` propagates. ``normalize`` maps a name to its registry
    key, ``drop_config_keys`` are stripped from the config before
    construction, and ``validate`` runs ``validate_configuration()`` on the
    new provider and raises ``ValueError`` with its message.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger,
        unknown_message: str,
        init_error_message: Optional[str] = None,
        normalize: Callable[[str], str] = _lowercase,
        drop_config_keys: Iterable[str] = (),
        validate: bool = True,
    ) -> None:
        self.providers: Dict[str, Type[T]] = {}
        self._logger = logger
        self._unknown_message = unknown_message
        self._init_error_message = init_error_message
        self._normalize = normalize
        self._drop_config_keys = frozenset(drop_config_keys)
        self._validate = validate

    def register(self, name: str, provider_cls: Type[T]) -> None:
        self.providers[self._normalize(name)] = provider_cls

    def get(self, name: str) -> Optional[Type[T]]:
        if not name:
            return None
        return self.providers.get(self._normalize(name))

    def create(
        self, name: str, config: Optional[Mapping[str, Any]] = None
    ) -> Optional[T]:
        provider_cls = self.get(name)
        if provider_cls is None:
            self._logger.warning(self._unknown_message, name)
            return None
        values = {
            key: value
            for key, value in dict(config or {}).items()
            if key not in self._drop_config_keys
        }
        try:
            provider = provider_cls(**values)
        except TypeError:
            try:
                provider = provider_cls(config=values)  # type: ignore[call-arg]
            except TypeError:
                if self._init_error_message:
                    self._logger.error(
                        self._init_error_message, name, list(values.keys())
                    )
                raise
        if self._validate:
            issue = provider.validate_configuration()  # type: ignore[attr-defined]
            if issue:
                raise ValueError(issue)
        return provider


__all__ = ["ProviderRegistry"]

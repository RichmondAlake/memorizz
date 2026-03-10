# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Base classes for sandbox code execution providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import ExecutionResult

logger = logging.getLogger(__name__)


class SandboxProvider(ABC):
    """
    Interface for providers that offer sandboxed code execution.

    Each provider wraps a different execution backend (cloud or local) but
    exposes a uniform API so MemAgent can swap them without code changes.
    """

    provider_name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return getattr(self, "provider_name", self.__class__.__name__).lower()

    def get_config(self) -> Dict[str, Any]:
        """Return serializable config information (never includes secrets)."""
        return {"provider": self.get_provider_name()}

    def validate_configuration(self) -> Optional[str]:
        """
        Validate runtime prerequisites for this sandbox provider.

        Returns:
            ``None`` when configuration is usable, otherwise an actionable
            human-readable error string.
        """
        return None

    # --- Core execution ---

    @abstractmethod
    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        envs: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute code in the sandbox and return the result.

        Each call is stateless — a fresh sandbox context is created and torn
        down automatically.

        Args:
            code: Source code to execute.
            language: Programming language (default ``"python"``).
            timeout: Maximum execution time in seconds.
            envs: Optional environment variables for the sandbox.

        Returns:
            ExecutionResult with stdout, stderr, and error information.
        """

    # --- File operations ---

    @abstractmethod
    def write_file(self, path: str, content: str) -> bool:
        """
        Write a text file inside the sandbox.

        Args:
            path: Absolute or relative path inside the sandbox filesystem.
            content: Text content to write.

        Returns:
            True on success, False on failure.
        """

    @abstractmethod
    def read_file(self, path: str) -> Optional[str]:
        """
        Read a text file from the sandbox.

        Args:
            path: Absolute or relative path inside the sandbox filesystem.

        Returns:
            File contents as a string, or None if the file does not exist.
        """

    # --- Lifecycle ---

    def close(self) -> None:
        """Clean up resources (override when necessary)."""
        return None


# --- Provider registry ---

_PROVIDER_REGISTRY: Dict[str, type[SandboxProvider]] = {}


def register_provider(name: str, provider_cls: type[SandboxProvider]) -> None:
    """Register a sandbox provider by name."""
    _PROVIDER_REGISTRY[name.lower()] = provider_cls


def get_provider_class(name: str) -> Optional[type[SandboxProvider]]:
    """Return the provider class for a given name."""
    if not name:
        return None
    return _PROVIDER_REGISTRY.get(name.lower())


def create_sandbox_provider(
    name: str, config: Optional[Dict[str, Any]] = None
) -> Optional[SandboxProvider]:
    """
    Instantiate a sandbox provider from the registry.

    Args:
        name: Provider name (``"e2b"``, ``"daytona"``, ``"graalpy"``).
        config: Provider-specific configuration dict.

    Returns:
        A SandboxProvider instance, or None if the name is unknown.
    """
    provider_cls = get_provider_class(name)
    if not provider_cls:
        logger.warning("Unknown sandbox provider: %s", name)
        return None

    config = dict(config or {})
    try:
        provider = provider_cls(**config)
    except TypeError:
        try:
            provider = provider_cls(config=config)  # type: ignore[arg-type]
        except TypeError as exc:
            logger.error(
                "Failed to initialize sandbox provider '%s' with config keys: %s",
                name,
                list(config.keys()),
            )
            raise exc

    validation_error = provider.validate_configuration()
    if validation_error:
        raise ValueError(validation_error)

    return provider

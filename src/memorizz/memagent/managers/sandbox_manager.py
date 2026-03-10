# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Manager responsible for routing sandbox code execution to providers."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from ...sandbox.base import SandboxProvider, create_sandbox_provider
from ...sandbox.models import ExecutionResult

logger = logging.getLogger(__name__)


class SandboxManager:
    """Wrapper over SandboxProvider implementations.

    Follows the same manager pattern as InternetAccessManager — thin routing
    layer that delegates to whichever provider is configured.
    """

    def __init__(self, provider: Optional[SandboxProvider] = None):
        self.provider = provider

    @classmethod
    def from_config(
        cls, config: Union[str, Dict[str, Any], SandboxProvider]
    ) -> "SandboxManager":
        """Create a SandboxManager from flexible config input.

        Args:
            config: One of:
                - A string provider name (``"e2b"``, ``"daytona"``, ``"graalpy"``).
                - A dict with ``"provider"`` key and optional provider kwargs.
                - An already-instantiated SandboxProvider instance.

        Returns:
            Configured SandboxManager.
        """
        if isinstance(config, SandboxProvider):
            return cls(provider=config)

        if isinstance(config, str):
            provider = create_sandbox_provider(config)
            if provider is None:
                raise ValueError(
                    f"Unknown sandbox provider '{config}'. "
                    "Supported providers: e2b, daytona, graalpy."
                )
            return cls(provider=provider)

        if isinstance(config, dict):
            config_data = dict(config)
            provider_name = config_data.pop("provider", "e2b")
            provider = create_sandbox_provider(provider_name, config_data)
            if provider is None:
                raise ValueError(
                    f"Unknown sandbox provider '{provider_name}'. "
                    "Supported providers: e2b, daytona, graalpy."
                )
            return cls(provider=provider)

        raise ValueError(
            f"Invalid sandbox config type: {type(config)}. "
            "Expected str, dict, or SandboxProvider instance."
        )

    def set_provider(
        self, provider: Optional[SandboxProvider]
    ) -> Optional[SandboxProvider]:
        """Attach or detach a sandbox provider."""
        previous = self.provider
        if previous and previous is not provider:
            try:
                previous.close()
            except Exception as exc:
                logger.debug("Failed to close previous sandbox provider: %s", exc)
        self.provider = provider
        return previous

    def is_enabled(self) -> bool:
        """Return True if a provider is available."""
        return self.provider is not None

    def get_provider_name(self) -> Optional[str]:
        if not self.provider:
            return None
        return self.provider.get_provider_name()

    def get_provider_config(self) -> Optional[Dict[str, Any]]:
        if not self.provider:
            return None
        return self.provider.get_config()

    # --- Execution ---

    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        envs: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute code using the configured provider."""
        if not self.provider:
            raise ValueError("Sandbox provider is not configured")
        return self.provider.execute_code(
            code=code, language=language, timeout=timeout, envs=envs
        )

    # --- File operations ---

    def write_file(self, path: str, content: str) -> bool:
        """Write a file inside the sandbox."""
        if not self.provider:
            raise ValueError("Sandbox provider is not configured")
        return self.provider.write_file(path=path, content=content)

    def read_file(self, path: str) -> Optional[str]:
        """Read a file from the sandbox."""
        if not self.provider:
            raise ValueError("Sandbox provider is not configured")
        return self.provider.read_file(path=path)

    # --- Tool generation ---

    def get_tools(self) -> List[Callable]:
        """Return tool functions suitable for MemAgent tool registration.

        Returns three tools:
        - ``execute_code``: Execute code in the sandbox.
        - ``sandbox_write_file``: Write a file in the sandbox.
        - ``sandbox_read_file``: Read a file from the sandbox.
        """
        manager = self  # capture for closures

        def execute_code(code: str, language: str = "python") -> str:
            """Execute code in a secure sandbox environment and return the output.

            Use this tool to run Python (or other language) code safely in an isolated
            environment. The code runs in a fresh sandbox — variables do not persist
            between calls.

            Args:
                code: The source code to execute.
                language: Programming language (default: python).

            Returns:
                JSON string with stdout, stderr, error, and execution results.
            """
            result = manager.execute_code(code=code, language=language)
            return result.to_json()

        def sandbox_write_file(path: str, content: str) -> str:
            """Write a file inside the sandbox environment.

            Args:
                path: File path inside the sandbox.
                content: Text content to write.

            Returns:
                JSON string indicating success or failure.
            """
            success = manager.write_file(path=path, content=content)
            return json.dumps({"success": success, "path": path})

        def sandbox_read_file(path: str) -> str:
            """Read a file from the sandbox environment.

            Args:
                path: File path inside the sandbox.

            Returns:
                JSON string with the file content or an error message.
            """
            content = manager.read_file(path=path)
            if content is not None:
                return json.dumps({"content": content, "path": path})
            return json.dumps({"error": f"File not found: {path}", "path": path})

        return [execute_code, sandbox_write_file, sandbox_read_file]

    # --- Lifecycle ---

    def close(self) -> None:
        """Clean up provider resources."""
        if self.provider:
            try:
                self.provider.close()
            except Exception as exc:
                logger.debug("Failed to close sandbox provider: %s", exc)

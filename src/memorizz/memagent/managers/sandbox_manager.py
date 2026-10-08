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
from ._provider_slot import ProviderSlot

logger = logging.getLogger(__name__)


class SandboxManager(ProviderSlot):
    """Wrapper over SandboxProvider implementations.

    Follows the same manager pattern as InternetAccessManager — thin routing
    layer that delegates to whichever provider is configured.
    """

    provider_kind = "sandbox"

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

    def _provider_security_posture(self) -> Dict[str, Any]:
        """Report whether the configured provider is a real security boundary.

        Order of precedence: an explicit ``is_security_sandbox`` attribute on
        the provider (or key in ``get_config()``), then the provider's
        ``security_boundary`` config entry, then the provider name and mode.
        """
        provider = self.provider
        name = ""
        config: Dict[str, Any] = {}
        if provider is not None:
            try:
                name = str(provider.get_provider_name() or "")
            except Exception:
                name = str(getattr(provider, "provider_name", "") or "")
            try:
                config = dict(provider.get_config() or {})
            except Exception:
                config = {}
        name = name.strip().lower()
        mode = str(config.get("mode") or "").strip().lower()

        flag = getattr(provider, "is_security_sandbox", None)
        if not isinstance(flag, bool):
            flag = config.get("is_security_sandbox")
        if isinstance(flag, bool):
            is_sandbox: Optional[bool] = flag
        else:
            boundary = str(config.get("security_boundary") or "").strip().lower()
            if boundary:
                is_sandbox = "not_strong_sandbox" not in boundary and (
                    "not_a_sandbox" not in boundary
                )
            elif name == "graalpy":
                is_sandbox = mode == "java_wrapper"
            elif name in {"e2b", "daytona"}:
                is_sandbox = True
            else:
                is_sandbox = None

        return {
            "name": name,
            "mode": mode,
            "is_security_sandbox": is_sandbox,
            "allow_network": config.get("allow_network"),
        }

    def _execute_code_description(self) -> str:
        """Build an honest ``execute_code`` tool description for the provider."""
        posture = self._provider_security_posture()
        label = posture["name"] or "configured"
        if posture["mode"]:
            label = f"{label} ({posture['mode']} mode)"

        if posture["is_security_sandbox"] is True:
            summary = (
                f"Execute code in the isolated {label} sandbox and return the "
                "output. The provider enforces a security boundary between the "
                "code and this host, so it is suitable for untrusted code."
            )
        else:
            if posture["is_security_sandbox"] is False:
                reason = (
                    "This is NOT a security sandbox: trusted code only, no "
                    "filesystem isolation. Code runs with this process's host "
                    "identity and can read and write the host filesystem; only "
                    "environment variables are filtered"
                )
            else:
                reason = (
                    "This provider declares no security boundary: treat it as "
                    "trusted code only, no filesystem isolation"
                )
            if posture["allow_network"] is False:
                reason += " and network access is denied by policy."
            elif posture["allow_network"] is True:
                reason += " and network access is allowed."
            else:
                reason += "."
            summary = (
                f"Execute code with the {label} provider and return the output. "
                f"{reason} Never run code from untrusted sources with this tool."
            )

        return (
            f"{summary}\n\n"
            "Stateful providers keep one bounded session, so files written with "
            "sandbox_write_file remain available until close().\n\n"
            "Args:\n"
            "    code: The source code to execute.\n"
            "    language: Programming language (default: python).\n\n"
            "Returns:\n"
            "    JSON string with stdout, stderr, error, and execution results."
        )

    def get_tools(self) -> List[Callable]:
        """Return tool functions suitable for MemAgent tool registration.

        Returns three tools:
        - ``execute_code``: Execute code in the sandbox.
        - ``sandbox_write_file``: Write a file in the sandbox.
        - ``sandbox_read_file``: Read a file from the sandbox.

        The ``execute_code`` description states honestly whether the provider
        is a security boundary (E2B/Daytona, GraalPy ``java_wrapper``) or a
        bounded host subprocess for trusted code only (GraalPy ``subprocess``).
        """
        manager = self  # capture for closures

        def execute_code(code: str, language: str = "python") -> str:
            """Execute code with the configured sandbox provider and return the output.

            Args:
                code: The source code to execute.
                language: Programming language (default: python).

            Returns:
                JSON string with stdout, stderr, error, and execution results.
            """
            result = manager.execute_code(code=code, language=language)
            return result.to_json()

        execute_code.__doc__ = manager._execute_code_description()

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

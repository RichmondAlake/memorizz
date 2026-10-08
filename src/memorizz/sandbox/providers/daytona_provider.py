# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Daytona sandbox provider — cloud-based dev environments with unlimited runtime.

Daytona offers full development environments with:
- Git operations built-in
- process.code_run for multi-language support

Requires: ``pip install daytona``
Environment: ``DAYTONA_API_KEY``, ``DAYTONA_API_URL`` (optional), ``DAYTONA_TARGET`` (optional)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

from ..base import SandboxProvider, register_provider
from ..models import ExecutionResult

logger = logging.getLogger(__name__)


class DaytonaSandboxProvider(SandboxProvider):
    """Cloud sandbox provider powered by Daytona (https://daytona.io).

    One sandbox per provider instance is shared by ``write_file`` →
    ``execute_code`` → ``read_file``, so files written in one call are visible
    to the next (as ``E2BSandboxProvider`` does with its session). ``close()``
    removes the sandbox; the next call starts a fresh one.
    """

    provider_name = "daytona"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        target: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.api_key = api_key or os.environ.get("DAYTONA_API_KEY", "")
        self.api_url = api_url or os.environ.get(
            "DAYTONA_API_URL", "https://app.daytona.io/api"
        )
        self.target = target or os.environ.get("DAYTONA_TARGET", "us")
        if not self.api_key:
            logger.warning(
                "Daytona API key not provided. Set DAYTONA_API_KEY environment variable "
                "or pass api_key to the constructor."
            )
        self._client: Any = None
        self._sandbox: Any = None
        self._lock = threading.Lock()

    def get_config(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "api_url": self.api_url,
            "target": self.target,
            "api_key_set": bool(self.api_key),
        }

    # --- Session ---

    def _sandbox_session(self, envs: Optional[Dict[str, str]] = None) -> Any:
        """The provider's sandbox, created on first use.

        ``envs`` apply when the sandbox is created; a later call with
        different values keeps the running sandbox and logs that.
        """
        with self._lock:
            if self._sandbox is not None:
                if envs:
                    logger.debug(
                        "Daytona sandbox already running; env vars apply to the "
                        "next session"
                    )
                return self._sandbox
            try:
                from daytona import Daytona, DaytonaConfig
            except ImportError as exc:
                raise RuntimeError(
                    "daytona SDK is not installed. "
                    "Install it with: pip install daytona"
                ) from exc

            client = Daytona(
                DaytonaConfig(
                    api_key=self.api_key,
                    api_url=self.api_url,
                    target=self.target,
                )
            )
            create_kwargs: Dict[str, Any] = {}
            if envs:
                create_kwargs["env"] = envs
            self._sandbox = client.create(**create_kwargs)
            self._client = client
            return self._sandbox

    # --- Core execution ---

    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        envs: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute code in the provider's sandbox, bounded by ``timeout``."""
        try:
            sandbox = self._sandbox_session(envs)
        except RuntimeError as exc:
            return ExecutionResult(error=str(exc), exit_code=1)
        except Exception as exc:
            logger.error("Daytona sandbox creation failed: %s", exc)
            return ExecutionResult(
                error=f"Daytona sandbox error: {exc}",
                exit_code=1,
            )

        try:
            # Use process.code_run for language-agnostic execution
            response = sandbox.process.code_run(
                code, language=language, timeout=timeout
            )

            stdout = response.result.split("\n") if response.result else []
            # Daytona returns exit_code on the response
            exit_code = getattr(response, "exit_code", 0)

            return ExecutionResult(
                stdout=stdout,
                stderr=[],
                error=None if exit_code == 0 else response.result,
                exit_code=exit_code,
                results=[response.result] if response.result else [],
                metadata={"provider": "daytona", "target": self.target},
            )

        except Exception as exc:
            logger.error("Daytona execution failed: %s", exc)
            return ExecutionResult(
                error=f"Daytona sandbox error: {exc}",
                exit_code=1,
            )

    # --- File operations ---

    def write_file(self, path: str, content: str) -> bool:
        """Write a file inside the provider's sandbox."""
        try:
            sandbox = self._sandbox_session()
            sandbox.fs.upload_file(path, content.encode("utf-8"))
            return True
        except Exception as exc:
            logger.error("Daytona write_file failed: %s", exc)
            return False

    def read_file(self, path: str) -> Optional[str]:
        """Read a file from the provider's sandbox."""
        try:
            sandbox = self._sandbox_session()
            data = sandbox.fs.download_file(path)
            return data.decode("utf-8") if isinstance(data, bytes) else str(data)
        except Exception as exc:
            logger.error("Daytona read_file failed: %s", exc)
            return None

    # --- Lifecycle ---

    def close(self) -> None:
        """Remove the sandbox; the next call starts a fresh one."""
        with self._lock:
            sandbox, client = self._sandbox, self._client
            self._sandbox = None
            self._client = None
        if sandbox is not None and client is not None:
            try:
                client.remove(sandbox)
            except Exception as exc:
                logger.debug("Daytona sandbox removal failed: %s", exc)


# Auto-register
register_provider("daytona", DaytonaSandboxProvider)

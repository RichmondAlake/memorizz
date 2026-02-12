"""Daytona sandbox provider — cloud-based dev environments with unlimited runtime.

Daytona offers full development environments with:
- ~90ms cold start
- Unlimited session duration
- GPU support
- Git operations built-in
- process.code_run for multi-language support

Requires: ``pip install daytona``
Environment: ``DAYTONA_API_KEY``, ``DAYTONA_API_URL`` (optional), ``DAYTONA_TARGET`` (optional)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from ..base import SandboxProvider, register_provider
from ..models import ExecutionResult

logger = logging.getLogger(__name__)


class DaytonaSandboxProvider(SandboxProvider):
    """Cloud sandbox provider powered by Daytona (https://daytona.io).

    Each ``execute_code`` call creates a fresh sandbox, executes the code,
    and removes it — keeping every invocation stateless.
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

    def get_config(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "api_url": self.api_url,
            "target": self.target,
            "api_key_set": bool(self.api_key),
        }

    # --- Core execution ---

    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        envs: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute code in a fresh Daytona sandbox (stateless)."""
        try:
            from daytona import Daytona, DaytonaConfig
        except ImportError:
            return ExecutionResult(
                error=(
                    "daytona SDK is not installed. "
                    "Install it with: pip install daytona"
                ),
                exit_code=1,
            )

        sandbox = None
        daytona = None
        try:
            daytona_config = DaytonaConfig(
                api_key=self.api_key,
                api_url=self.api_url,
                target=self.target,
            )
            daytona = Daytona(daytona_config)

            # Create sandbox with env vars if provided
            create_kwargs = {}
            if envs:
                create_kwargs["env"] = envs
            sandbox = daytona.create(**create_kwargs)

            # Use process.code_run for language-agnostic execution
            response = sandbox.process.code_run(code, language=language)

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
        finally:
            if sandbox and daytona:
                try:
                    daytona.remove(sandbox)
                except Exception:
                    pass

    # --- File operations ---

    def write_file(self, path: str, content: str) -> bool:
        """Write a file inside a temporary Daytona sandbox."""
        try:
            from daytona import Daytona, DaytonaConfig
        except ImportError:
            logger.error("daytona SDK is not installed.")
            return False

        sandbox = None
        daytona = None
        try:
            daytona_config = DaytonaConfig(
                api_key=self.api_key,
                api_url=self.api_url,
                target=self.target,
            )
            daytona = Daytona(daytona_config)
            sandbox = daytona.create()
            sandbox.fs.upload_file(path, content.encode("utf-8"))
            return True
        except Exception as exc:
            logger.error("Daytona write_file failed: %s", exc)
            return False
        finally:
            if sandbox and daytona:
                try:
                    daytona.remove(sandbox)
                except Exception:
                    pass

    def read_file(self, path: str) -> Optional[str]:
        """Read a file from a temporary Daytona sandbox."""
        try:
            from daytona import Daytona, DaytonaConfig
        except ImportError:
            logger.error("daytona SDK is not installed.")
            return None

        sandbox = None
        daytona = None
        try:
            daytona_config = DaytonaConfig(
                api_key=self.api_key,
                api_url=self.api_url,
                target=self.target,
            )
            daytona = Daytona(daytona_config)
            sandbox = daytona.create()
            data = sandbox.fs.download_file(path)
            return data.decode("utf-8") if isinstance(data, bytes) else str(data)
        except Exception as exc:
            logger.error("Daytona read_file failed: %s", exc)
            return None
        finally:
            if sandbox and daytona:
                try:
                    daytona.remove(sandbox)
                except Exception:
                    pass


# Auto-register
register_provider("daytona", DaytonaSandboxProvider)

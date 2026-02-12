"""E2B sandbox provider — cloud-based code execution using Firecracker microVMs.

E2B is the default sandbox provider for MemAgent. It offers:
- Firecracker microVM isolation (~150ms cold start)
- Stateful Jupyter-style code execution
- Full filesystem access within the sandbox
- Internet access from inside the sandbox

Requires: ``pip install e2b-code-interpreter``
Environment: ``E2B_API_KEY``
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from ..base import SandboxProvider, register_provider
from ..models import ExecutionResult

logger = logging.getLogger(__name__)


class E2BSandboxProvider(SandboxProvider):
    """Cloud sandbox provider powered by E2B (https://e2b.dev).

    Each ``execute_code`` call spins up a fresh sandbox, executes the code,
    and tears it down — keeping every invocation stateless.
    """

    provider_name = "e2b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        template: str = "code-interpreter",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.api_key = api_key or os.environ.get("E2B_API_KEY", "")
        self.template = template
        if not self.api_key:
            logger.warning(
                "E2B API key not provided. Set E2B_API_KEY environment variable "
                "or pass api_key to the constructor."
            )

    def get_config(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "template": self.template,
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
        """Execute code in a fresh E2B sandbox (stateless)."""
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            return ExecutionResult(
                error=(
                    "e2b-code-interpreter is not installed. "
                    "Install it with: pip install e2b-code-interpreter"
                ),
                exit_code=1,
            )

        sandbox = None
        try:
            sandbox = Sandbox(api_key=self.api_key, timeout=timeout)
            execution = sandbox.run_code(
                code,
                language=language if language != "python" else None,
                envs=envs,
                timeout=timeout,
            )

            stdout = (
                [msg.line for msg in execution.logs.stdout]
                if execution.logs.stdout
                else []
            )
            stderr = (
                [msg.line for msg in execution.logs.stderr]
                if execution.logs.stderr
                else []
            )
            error = None
            if execution.error:
                error = (
                    f"{execution.error.name}: {execution.error.value}\n"
                    f"{execution.error.traceback}"
                )

            results = [str(r) for r in execution.results] if execution.results else []

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                error=error,
                exit_code=1 if execution.error else 0,
                results=results,
                metadata={"provider": "e2b", "template": self.template},
            )

        except Exception as exc:
            logger.error("E2B execution failed: %s", exc)
            return ExecutionResult(
                error=f"E2B sandbox error: {exc}",
                exit_code=1,
            )
        finally:
            if sandbox:
                try:
                    sandbox.kill()
                except Exception:
                    pass

    # --- File operations ---

    def write_file(self, path: str, content: str) -> bool:
        """Write a file inside a temporary E2B sandbox."""
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            logger.error("e2b-code-interpreter is not installed.")
            return False

        sandbox = None
        try:
            sandbox = Sandbox(api_key=self.api_key)
            sandbox.files.write(path, content)
            return True
        except Exception as exc:
            logger.error("E2B write_file failed: %s", exc)
            return False
        finally:
            if sandbox:
                try:
                    sandbox.kill()
                except Exception:
                    pass

    def read_file(self, path: str) -> Optional[str]:
        """Read a file from a temporary E2B sandbox."""
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            logger.error("e2b-code-interpreter is not installed.")
            return None

        sandbox = None
        try:
            sandbox = Sandbox(api_key=self.api_key)
            content = sandbox.files.read(path)
            return content
        except Exception as exc:
            logger.error("E2B read_file failed: %s", exc)
            return None
        finally:
            if sandbox:
                try:
                    sandbox.kill()
                except Exception:
                    pass


# Auto-register
register_provider("e2b", E2BSandboxProvider)

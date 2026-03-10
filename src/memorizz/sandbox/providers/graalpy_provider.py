# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""GraalPy sandbox provider — local code execution using GraalVM's Python runtime.

GraalPy is a local sandbox option that requires no cloud services. It offers:
- No API keys or network dependency
- No per-hour billing
- JVM-based isolation via GraalVM's Polyglot API
- Two security modes: subprocess (simple) and Java wrapper (full sandboxing)

Two modes are available:
- **subprocess** (default): Runs code via ``graalpy -c "<code>"`` subprocess.
  Simple to set up, but isolation is limited to OS-level process boundaries.
- **java_wrapper**: Uses a Java bridge with GraalVM's ``SandboxPolicy.UNTRUSTED``
  for full restriction of filesystem, network, and CPU access.

Requires: GraalPy installed on the system (``graalpy`` on PATH or explicit path).
For java_wrapper mode: JVM + GraalVM polyglot dependencies.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional

from ..base import SandboxProvider, register_provider
from ..models import ExecutionResult

logger = logging.getLogger(__name__)


class GraalPySandboxProvider(SandboxProvider):
    """Local sandbox provider powered by GraalPy (https://graalvm.org/python/).

    Each ``execute_code`` call runs in an isolated subprocess.
    No cloud dependencies, no API keys, no billing.
    """

    provider_name = "graalpy"

    def __init__(
        self,
        graalpy_path: Optional[str] = None,
        mode: str = "subprocess",
        java_wrapper_jar: Optional[str] = None,
        sandbox_policy: str = "UNTRUSTED",
        working_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GraalPy sandbox provider.

        Args:
            graalpy_path: Path to the ``graalpy`` binary. If None, looks for
                ``graalpy`` on PATH.
            mode: Execution mode — ``"subprocess"`` (simple) or
                ``"java_wrapper"`` (full GraalVM sandboxing).
            java_wrapper_jar: Path to the Java wrapper JAR for
                ``java_wrapper`` mode.
            sandbox_policy: GraalVM sandbox policy for ``java_wrapper`` mode.
                One of ``"TRUSTED"``, ``"CONSTRAINED"``, ``"UNTRUSTED"``.
            working_dir: Working directory for sandbox execution.
                Defaults to a temporary directory.
        """
        super().__init__(config)
        env_graalpy_path = (os.environ.get("GRAALPY_PATH", "") or "").strip()
        self.graalpy_path = (
            graalpy_path or env_graalpy_path or shutil.which("graalpy") or "graalpy"
        )
        self.mode = mode
        self.java_wrapper_jar = java_wrapper_jar
        self.sandbox_policy = sandbox_policy
        self.working_dir = working_dir

        if mode == "java_wrapper" and not java_wrapper_jar:
            logger.warning(
                "GraalPy java_wrapper mode requires java_wrapper_jar path. "
                "Falling back to subprocess mode."
            )
            self.mode = "subprocess"

    def _resolve_executable(self) -> Optional[str]:
        """Return the resolved graalpy executable path when available."""
        configured = (self.graalpy_path or "").strip() or "graalpy"
        if os.path.sep in configured or configured.startswith("."):
            if os.path.isfile(configured) and os.access(configured, os.X_OK):
                return configured
            return None
        return shutil.which(configured)

    def validate_configuration(self) -> Optional[str]:
        """Validate that graalpy executable (and optional wrapper) are available."""
        resolved_executable = self._resolve_executable()
        if not resolved_executable:
            return (
                "GraalPy sandbox requires a working `graalpy` executable, but it was "
                f"not found for graalpy_path='{self.graalpy_path}'. "
                "Install GraalPy and ensure `graalpy` is on PATH, or configure "
                "`graalpy_path` with an absolute executable path. "
                "Install docs: https://www.graalvm.org/python/"
            )

        if self.mode == "java_wrapper":
            wrapper_path = (self.java_wrapper_jar or "").strip()
            if not wrapper_path:
                return (
                    "GraalPy java_wrapper mode requires `java_wrapper_jar`. "
                    "Provide a valid JAR path or use subprocess mode."
                )
            if not os.path.isfile(wrapper_path):
                return (
                    "GraalPy java_wrapper_jar does not exist at "
                    f"'{wrapper_path}'. Provide a valid file path."
                )

        return None

    def get_config(self) -> Dict[str, Any]:
        resolved_executable = self._resolve_executable()
        return {
            "provider": self.provider_name,
            "mode": self.mode,
            "graalpy_path": self.graalpy_path,
            "sandbox_policy": self.sandbox_policy,
            "graalpy_available": resolved_executable is not None,
            "graalpy_resolved_path": resolved_executable or "",
        }

    # --- Core execution ---

    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        envs: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute code using GraalPy (stateless subprocess)."""
        if language != "python":
            return ExecutionResult(
                error=f"GraalPy only supports Python execution, got: {language}",
                exit_code=1,
            )

        if self.mode == "java_wrapper":
            return self._execute_java_wrapper(code, timeout, envs)
        return self._execute_subprocess(code, timeout, envs)

    def _execute_subprocess(
        self,
        code: str,
        timeout: int,
        envs: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute code via ``graalpy -c`` subprocess."""
        env = os.environ.copy()
        if envs:
            env.update(envs)

        try:
            result = subprocess.run(
                [self.graalpy_path, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
                env=env,
            )

            stdout = result.stdout.strip().split("\n") if result.stdout.strip() else []
            stderr = result.stderr.strip().split("\n") if result.stderr.strip() else []

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                error=result.stderr.strip() if result.returncode != 0 else None,
                exit_code=result.returncode,
                results=stdout,
                metadata={"provider": "graalpy", "mode": "subprocess"},
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                error=f"GraalPy execution timed out after {timeout}s",
                exit_code=124,
                metadata={"provider": "graalpy", "mode": "subprocess"},
            )
        except FileNotFoundError:
            return ExecutionResult(
                error=(
                    f"GraalPy not found at '{self.graalpy_path}'. "
                    "Install GraalPy or set graalpy_path to the correct location. "
                    "See: https://www.graalvm.org/python/"
                ),
                exit_code=127,
            )
        except Exception as exc:
            logger.error("GraalPy subprocess execution failed: %s", exc)
            return ExecutionResult(
                error=f"GraalPy subprocess error: {exc}",
                exit_code=1,
            )

    def _execute_java_wrapper(
        self,
        code: str,
        timeout: int,
        envs: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute code via the Java Polyglot wrapper with SandboxPolicy."""
        java_path = shutil.which("java")
        if not java_path:
            return ExecutionResult(
                error=(
                    "Java runtime not found. The java_wrapper mode requires "
                    "a JVM with GraalVM polyglot support installed."
                ),
                exit_code=127,
            )

        env = os.environ.copy()
        if envs:
            env.update(envs)

        # Write code to a temp file to avoid shell escaping issues
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp:
                tmp.write(code)
                code_file = tmp.name

            result = subprocess.run(
                [
                    java_path,
                    "-jar",
                    self.java_wrapper_jar,
                    "--sandbox-policy",
                    self.sandbox_policy,
                    "--code-file",
                    code_file,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
                env=env,
            )

            stdout = result.stdout.strip().split("\n") if result.stdout.strip() else []
            stderr = result.stderr.strip().split("\n") if result.stderr.strip() else []

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                error=result.stderr.strip() if result.returncode != 0 else None,
                exit_code=result.returncode,
                results=stdout,
                metadata={
                    "provider": "graalpy",
                    "mode": "java_wrapper",
                    "sandbox_policy": self.sandbox_policy,
                },
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                error=f"GraalPy Java wrapper timed out after {timeout}s",
                exit_code=124,
                metadata={"provider": "graalpy", "mode": "java_wrapper"},
            )
        except Exception as exc:
            logger.error("GraalPy Java wrapper execution failed: %s", exc)
            return ExecutionResult(
                error=f"GraalPy Java wrapper error: {exc}",
                exit_code=1,
            )
        finally:
            try:
                os.unlink(code_file)
            except (OSError, UnboundLocalError):
                pass

    # --- File operations ---

    def write_file(self, path: str, content: str) -> bool:
        """Write a file to the local working directory."""
        target_dir = self.working_dir or tempfile.gettempdir()
        full_path = os.path.join(target_dir, path) if not os.path.isabs(path) else path
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as exc:
            logger.error("GraalPy write_file failed: %s", exc)
            return False

    def read_file(self, path: str) -> Optional[str]:
        """Read a file from the local working directory."""
        target_dir = self.working_dir or tempfile.gettempdir()
        full_path = os.path.join(target_dir, path) if not os.path.isabs(path) else path
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:
            logger.error("GraalPy read_file failed: %s", exc)
            return None


# Auto-register
register_provider("graalpy", GraalPySandboxProvider)

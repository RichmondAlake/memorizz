# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""GraalPy execution provider — local code execution using GraalVM Python.

GraalPy is a local sandbox option that requires no cloud services. It offers:
- No API keys or network dependency
- No per-hour billing
- bounded subprocess execution for trusted code
- an explicit Java/GraalVM ``UNTRUSTED`` wrapper for untrusted code

Two modes are available:
- **subprocess** (default): Runs code via ``graalpy -c "<code>"`` with a
  private working directory, an environment allowlist, path confinement, and
  POSIX resource limits. It is an execution provider, not a security sandbox.
- **java_wrapper**: Uses a Java bridge with GraalVM's ``SandboxPolicy.UNTRUSTED``
  as the supported untrusted-code boundary, in addition to host limits.

Requires: GraalPy installed on the system (``graalpy`` on PATH or explicit path).
For java_wrapper mode: JVM + GraalVM polyglot dependencies.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..base import SandboxProvider, register_provider
from ..models import ExecutionResult

logger = logging.getLogger(__name__)


class GraalPySandboxProvider(SandboxProvider):
    """Local GraalPy execution provider (https://graalvm.org/python/).

    Subprocess mode is bounded but shares the host OS security identity. Never
    treat it as an isolation boundary for adversarial input. ``java_wrapper``
    mode refuses any policy other than ``UNTRUSTED``.
    """

    provider_name = "graalpy"

    def __init__(
        self,
        graalpy_path: Optional[str] = None,
        mode: str = "subprocess",
        java_wrapper_jar: Optional[str] = None,
        sandbox_policy: str = "UNTRUSTED",
        working_dir: Optional[str] = None,
        allow_network: bool = False,
        env_allowlist: Optional[Iterable[str]] = None,
        max_memory_mb: int = 512,
        max_cpu_seconds: int = 30,
        max_processes: int = 32,
        max_file_bytes: int = 10_000_000,
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
        self.mode = str(mode or "subprocess").strip().lower()
        self.java_wrapper_jar = java_wrapper_jar
        self.sandbox_policy = str(sandbox_policy or "UNTRUSTED").upper()
        self.allow_network = bool(allow_network)
        self.env_allowlist = {
            str(item)
            for item in (
                env_allowlist or ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ")
            )
        }
        self.max_memory_mb = max(128, int(max_memory_mb))
        self.max_cpu_seconds = max(1, int(max_cpu_seconds))
        self.max_processes = max(1, int(max_processes))
        self.max_file_bytes = max(1_024, int(max_file_bytes))
        parent = Path(working_dir).expanduser().resolve() if working_dir else None
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        self._private_dir = Path(
            tempfile.mkdtemp(
                prefix="memorizz-graalpy-",
                dir=str(parent) if parent is not None else None,
            )
        ).resolve()
        try:
            os.chmod(self._private_dir, 0o700)
        except OSError:
            pass
        self.working_dir = str(self._private_dir)

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
        if self.mode == "subprocess":
            resolved_executable = self._resolve_executable()
            if not resolved_executable:
                return (
                    "GraalPy execution requires a working `graalpy` executable, "
                    f"but it was not found for graalpy_path='{self.graalpy_path}'. "
                    "Install GraalPy and ensure `graalpy` is on PATH, or configure "
                    "`graalpy_path` with an absolute executable path. "
                    "Install docs: https://www.graalvm.org/python/"
                )
        elif self.mode == "java_wrapper":
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
            if not shutil.which("java"):
                return "GraalPy java_wrapper mode requires a Java runtime on PATH."
            if not zipfile.is_zipfile(wrapper_path):
                return "GraalPy java_wrapper_jar must be an executable JAR/ZIP file."
            try:
                with zipfile.ZipFile(wrapper_path) as archive:
                    names = set(archive.namelist())
                    class_name = "memorizz/sandbox/MemorizzGraalSandbox.class"
                    if class_name not in names:
                        return (
                            "GraalPy java_wrapper_jar does not contain the "
                            "package-shipped MemorizzGraalSandbox class."
                        )
                    manifest = archive.read("META-INF/MANIFEST.MF").decode(
                        "utf-8", errors="replace"
                    )
                    normalized_manifest = manifest.replace("\r\n", "\n")
                    if (
                        "Main-Class: memorizz.sandbox.MemorizzGraalSandbox"
                        not in normalized_manifest
                    ):
                        return (
                            "GraalPy java_wrapper_jar manifest must declare "
                            "memorizz.sandbox.MemorizzGraalSandbox as Main-Class."
                        )
            except (KeyError, OSError, zipfile.BadZipFile) as exc:
                return f"GraalPy java_wrapper_jar is invalid: {exc}"
            if self.sandbox_policy != "UNTRUSTED":
                return (
                    "GraalPy java_wrapper mode is an untrusted-code boundary only "
                    "when sandbox_policy='UNTRUSTED'."
                )
            if os.name != "posix":
                return (
                    "GraalPy java_wrapper UNTRUSTED mode requires POSIX host "
                    "resource limits in this release."
                )
        else:
            return "GraalPy mode must be 'subprocess' or 'java_wrapper'."

        return None

    def get_config(self) -> Dict[str, Any]:
        resolved_executable = self._resolve_executable()
        return {
            "provider": self.provider_name,
            "mode": self.mode,
            "graalpy_path": self.graalpy_path,
            "sandbox_policy": self.sandbox_policy,
            "security_boundary": (
                "graalvm_untrusted_sandbox"
                if self.mode == "java_wrapper"
                else "bounded_execution_provider_not_strong_sandbox"
            ),
            "allow_network": self.allow_network,
            "network_policy": self._network_policy(),
            "environment_allowlist": sorted(self.env_allowlist),
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_processes": self.max_processes,
            "max_file_bytes": self.max_file_bytes,
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
        validation_error = self.validate_configuration()
        if validation_error:
            return ExecutionResult(
                error=validation_error,
                exit_code=2,
                metadata=self.get_config(),
            )
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
        env = self._safe_environment(envs)

        try:
            result = subprocess.run(
                self._subprocess_command(code),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
                env=env,
                preexec_fn=self._resource_limiter(),
            )

            stdout = result.stdout.strip().split("\n") if result.stdout.strip() else []
            stderr = result.stderr.strip().split("\n") if result.stderr.strip() else []

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                error=result.stderr.strip() if result.returncode != 0 else None,
                exit_code=result.returncode,
                results=stdout,
                metadata=self.get_config(),
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                error=f"GraalPy execution timed out after {timeout}s",
                exit_code=124,
                metadata=self.get_config(),
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

        env = self._safe_environment(envs)

        # Write code to a temp file to avoid shell escaping issues
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir=self.working_dir
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
                    "--max-cpu-seconds",
                    str(self.max_cpu_seconds),
                    "--max-memory-mb",
                    str(self.max_memory_mb),
                    "--max-threads",
                    str(self.max_processes),
                    "--max-output-bytes",
                    str(self.max_file_bytes),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
                env=env,
                preexec_fn=self._resource_limiter(),
            )

            stdout = result.stdout.strip().split("\n") if result.stdout.strip() else []
            stderr = result.stderr.strip().split("\n") if result.stderr.strip() else []

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                error=result.stderr.strip() if result.returncode != 0 else None,
                exit_code=result.returncode,
                results=stdout,
                metadata=self.get_config(),
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                error=f"GraalPy Java wrapper timed out after {timeout}s",
                exit_code=124,
                metadata=self.get_config(),
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
        """Write a relative path inside the private provider directory."""
        try:
            full_path = self._resolve_private_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with full_path.open("w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as exc:
            logger.error("GraalPy write_file failed: %s", exc)
            return False

    def read_file(self, path: str) -> Optional[str]:
        """Read a relative path inside the private provider directory."""
        try:
            full_path = self._resolve_private_path(path)
            with full_path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:
            logger.error("GraalPy read_file failed: %s", exc)
            return None

    def _resolve_private_path(self, path: str) -> Path:
        raw = Path(str(path or ""))
        if not str(path or "").strip() or raw.is_absolute() or ".." in raw.parts:
            raise ValueError("GraalPy file paths must be relative without '..'")
        resolved = (self._private_dir / raw).resolve()
        resolved.relative_to(self._private_dir)
        return resolved

    def _safe_environment(self, values: Optional[Dict[str, str]]) -> Dict[str, str]:
        environment = {
            key: os.environ[key] for key in self.env_allowlist if key in os.environ
        }
        for key, value in (values or {}).items():
            if str(key) not in self.env_allowlist:
                raise ValueError(
                    f"Environment variable '{key}' is not in env_allowlist"
                )
            environment[str(key)] = str(value)
        environment["HOME"] = self.working_dir
        environment["TMPDIR"] = self.working_dir
        return environment

    def _subprocess_command(self, code: str) -> list[str]:
        command = [self.graalpy_path, "-c", code]
        if self.allow_network:
            return command
        if platform.system() == "Darwin" and os.path.isfile("/usr/bin/sandbox-exec"):
            return [
                "/usr/bin/sandbox-exec",
                "-p",
                "(version 1)(allow default)(deny network*)",
                *command,
            ]
        if platform.system() == "Linux":
            unshare = shutil.which("unshare")
            if unshare:
                # If the host forbids unprivileged network namespaces,
                # ``unshare`` fails before guest code starts: fail closed.
                return [unshare, "--net", "--", *command]
        raise RuntimeError(
            "Network-denied GraalPy subprocess execution is unavailable on "
            "this host. Use java_wrapper UNTRUSTED mode, or explicitly set "
            "allow_network=True only for trusted code."
        )

    def _network_policy(self) -> str:
        if self.mode == "java_wrapper":
            return "blocked_by_graalvm_io_none"
        if self.allow_network:
            return "explicitly_allowed_for_trusted_code"
        if platform.system() == "Darwin" and os.path.isfile("/usr/bin/sandbox-exec"):
            return "blocked_by_sandbox_exec"
        if platform.system() == "Linux" and shutil.which("unshare"):
            return "blocked_by_network_namespace_fail_closed"
        return "blocked_fail_closed_no_supported_enforcer"

    def _resource_limiter(self):
        if os.name != "posix":
            return None

        def limit() -> None:
            import resource

            cpu = min(self.max_cpu_seconds, 86_400)
            memory = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            if hasattr(resource, "RLIMIT_NPROC"):
                resource.setrlimit(
                    resource.RLIMIT_NPROC,
                    (self.max_processes, self.max_processes),
                )
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (self.max_file_bytes, self.max_file_bytes),
            )

        return limit

    def close(self) -> None:
        shutil.rmtree(self._private_dir, ignore_errors=True)

    def __enter__(self) -> "GraalPySandboxProvider":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


# Auto-register
register_provider("graalpy", GraalPySandboxProvider)

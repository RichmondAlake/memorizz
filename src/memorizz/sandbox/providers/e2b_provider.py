# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Stateful E2B Code Interpreter provider using the current SDK factory."""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
from collections.abc import Iterable as IterableABC
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Iterable, Optional

from ..base import SandboxProvider, register_provider
from ..models import ExecutionResult

logger = logging.getLogger(__name__)

_E2B_MIN = (2, 26, 0)
_E2B_MAX = (2, 38, 0)
_CODE_INTERPRETER_MIN = (2, 9, 0)
_CODE_INTERPRETER_MAX = (2, 10, 0)


def _version_tuple(value: str) -> tuple[int, int, int]:
    """Return a conservative release tuple without another runtime dependency."""
    numbers = [int(item) for item in str(value).split(".")[:3] if item.isdigit()]
    padded = numbers + [0, 0, 0]
    return padded[0], padded[1], padded[2]


def _installed_sdk_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for distribution, key in (
        ("e2b", "e2b"),
        ("e2b-code-interpreter", "e2b_code_interpreter"),
    ):
        try:
            versions[key] = version(distribution)
        except PackageNotFoundError:
            continue
    return versions


def _sdk_compatibility_error(versions: Dict[str, str]) -> Optional[str]:
    e2b_version = versions.get("e2b")
    interpreter_version = versions.get("e2b_code_interpreter")
    if e2b_version and not _E2B_MIN <= _version_tuple(e2b_version) < _E2B_MAX:
        return (
            f"Unsupported e2b SDK {e2b_version}; MemoRizz 0.5 supports "
            "e2b>=2.26.0,<2.38.0 with e2b-code-interpreter 2.9.x."
        )
    if interpreter_version and not (
        _CODE_INTERPRETER_MIN
        <= _version_tuple(interpreter_version)
        < _CODE_INTERPRETER_MAX
    ):
        return (
            "Unsupported e2b-code-interpreter SDK "
            f"{interpreter_version}; MemoRizz 0.5 supports 2.9.x."
        )
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("line", "text", "value", "message"):
            if value.get(key) is not None:
                return str(value[key])
        return json.dumps(value, ensure_ascii=False, default=str)
    for attribute in ("line", "text", "value", "message"):
        item = getattr(value, attribute, None)
        if item is not None:
            return str(item)
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"), ensure_ascii=False)
    return str(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read one field from old dictionary or current SDK object shapes."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _lines(values: Optional[Iterable[Any]]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, dict)) or not isinstance(values, IterableABC):
        values = [values]
    return [line for item in values if (line := _text(item))]


class E2BSandboxProvider(SandboxProvider):
    """One bounded E2B session shared by write → execute → read operations."""

    provider_name = "e2b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        template: Optional[str] = None,
        session_timeout: int = 300,
        max_execution_timeout: int = 120,
        allow_internet_access: bool = False,
        cpu_count: Optional[int] = None,
        memory_mb: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.api_key = str(api_key or os.environ.get("E2B_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError(
                "E2B_API_KEY is required for the E2B sandbox provider. Set the "
                "environment variable or pass api_key explicitly."
            )
        self.template = str(template).strip() if template else None
        self.session_timeout = max(30, min(int(session_timeout), 3_600))
        self.max_execution_timeout = max(
            1, min(int(max_execution_timeout), self.session_timeout)
        )
        self.allow_internet_access = bool(allow_internet_access)
        self.cpu_count = max(1, int(cpu_count)) if cpu_count is not None else None
        self.memory_mb = max(128, int(memory_mb)) if memory_mb is not None else None
        self._sandbox: Any = None
        self._lock = threading.RLock()
        self._factory_mode: Optional[str] = None
        self._sdk_versions = _installed_sdk_versions()

    def validate_configuration(self) -> Optional[str]:
        if not self.api_key:
            return (
                "E2B_API_KEY is required for the E2B sandbox provider. Set the "
                "environment variable or pass api_key explicitly."
            )
        if (
            self.cpu_count is not None or self.memory_mb is not None
        ) and not self.template:
            return (
                "E2B CPU/memory policies require an explicit E2B template that "
                "enforces those resource limits."
            )
        return _sdk_compatibility_error(self._sdk_versions)

    def get_config(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "template": self.template,
            "api_key_set": bool(self.api_key),
            "session_timeout": self.session_timeout,
            "max_execution_timeout": self.max_execution_timeout,
            "allow_internet_access": self.allow_internet_access,
            "cpu_count": self.cpu_count,
            "memory_mb": self.memory_mb,
            "resource_policy_enforcement": (
                "e2b_template" if self.template else "provider_default"
            ),
            "egress_policy_enforcement": "e2b_create",
            "sdk_versions": dict(self._sdk_versions),
        }

    @staticmethod
    def _accepted_kwargs(callable_value: Any, values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            signature = inspect.signature(callable_value)
        except (TypeError, ValueError):
            return {key: value for key, value in values.items() if value is not None}
        accepts_any = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        return {
            key: value
            for key, value in values.items()
            if value is not None and (accepts_any or key in signature.parameters)
        }

    def _create_session(self) -> Any:
        validation_error = self.validate_configuration()
        if validation_error:
            raise ValueError(validation_error)
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as exc:
            raise RuntimeError(
                "e2b-code-interpreter is not installed. Install the "
                "'memorizz[sandbox-e2b]' extra."
            ) from exc

        compatibility_error = _sdk_compatibility_error(self._sdk_versions)
        if compatibility_error:
            raise RuntimeError(compatibility_error)

        values = {
            "api_key": self.api_key,
            "template": self.template,
            "timeout": self.session_timeout,
            "allow_internet_access": self.allow_internet_access,
            "metadata": {
                "managed_by": "memorizz",
                "egress": "allowed" if self.allow_internet_access else "blocked",
                "resource_policy": (
                    f"template:{self.cpu_count or 'default'}cpu:"
                    f"{self.memory_mb or 'default'}mb"
                ),
            },
        }
        factory = getattr(Sandbox, "create", None)
        if callable(factory):
            self._factory_mode = "Sandbox.create"
            return factory(**self._accepted_kwargs(factory, values))
        # Compatibility for pre-v2 SDKs. New installations always use create().
        self._factory_mode = "Sandbox constructor (legacy compatibility)"
        return Sandbox(**self._accepted_kwargs(Sandbox, values))

    def _session(self) -> Any:
        with self._lock:
            if self._sandbox is None:
                self._sandbox = self._create_session()
            return self._sandbox

    def _metadata(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "template": self.template or "e2b-default-code-interpreter",
            "factory": self._factory_mode,
            "session_timeout": self.session_timeout,
            "execution_timeout": timeout,
            "egress_allowed": self.allow_internet_access,
            "cpu_count": self.cpu_count,
            "memory_mb": self.memory_mb,
            "resource_policy_enforcement": (
                "e2b_template" if self.template else "provider_default"
            ),
            "egress_policy_enforcement": "e2b_create",
            "stateful_session": True,
            "sdk_versions": dict(self._sdk_versions),
        }

    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        envs: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        bounded_timeout = max(1, min(int(timeout), self.max_execution_timeout))
        try:
            sandbox = self._session()
            kwargs = {
                "language": None if language == "python" else language,
                "envs": dict(envs or {}),
                "timeout": bounded_timeout,
            }
            execution = sandbox.run_code(
                code,
                **self._accepted_kwargs(sandbox.run_code, kwargs),
            )
            logs = _field(execution, "logs")
            stdout = _lines(
                _field(logs, "stdout")
                if logs is not None
                else _field(execution, "stdout")
            )
            stderr = _lines(
                _field(logs, "stderr")
                if logs is not None
                else _field(execution, "stderr")
            )
            execution_error = _field(execution, "error")
            error = None
            if execution_error:
                name = _text(_field(execution_error, "name"))
                value = _text(_field(execution_error, "value"))
                traceback = _text(_field(execution_error, "traceback"))
                error = ": ".join(item for item in (name, value) if item)
                if traceback:
                    error = f"{error}\n{traceback}" if error else traceback
                if not error:
                    error = _text(execution_error)
            results = _lines(_field(execution, "results"))
            if not results:
                results = _lines(_field(execution, "text"))
            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                error=error,
                exit_code=1 if execution_error else 0,
                results=results,
                metadata=self._metadata(bounded_timeout),
            )
        except Exception as exc:
            logger.error("E2B execution failed: %s", exc)
            return ExecutionResult(
                error=f"E2B sandbox error: {exc}",
                exit_code=1,
                metadata=self._metadata(bounded_timeout),
            )

    def write_file(self, path: str, content: str) -> bool:
        try:
            with self._lock:
                self._session().files.write(str(path), content)
            return True
        except Exception as exc:
            logger.error("E2B write_file failed: %s", exc)
            return False

    def read_file(self, path: str) -> Optional[str]:
        try:
            with self._lock:
                return _text(self._session().files.read(str(path)))
        except Exception as exc:
            logger.error("E2B read_file failed: %s", exc)
            return None

    def close(self) -> None:
        with self._lock:
            sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception as exc:
                logger.debug("E2B sandbox termination failed: %s", exc)

    def __enter__(self) -> "E2BSandboxProvider":
        self._session()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


register_provider("e2b", E2BSandboxProvider)

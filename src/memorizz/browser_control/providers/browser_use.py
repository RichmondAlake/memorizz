# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Browser Use provider executed with an isolated tool interpreter."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

from ..base import BrowserControlProvider, register_provider
from ..models import BrowserControlResult

_RESULT_MARKER = "__MEMORIZZ_BROWSER_CONTROL_RESULT__="
_DEFAULT_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "BROWSER_USE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "BU_CDP_URL",
    "BU_CDP_WS",
    "BU_NAME",
    "ANONYMIZED_TELEMETRY",
)


def _string_list(values: Optional[Iterable[Any]]) -> list[str]:
    result: list[str] = []
    for value in values or []:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


class BrowserUseProvider(BrowserControlProvider):
    """Run Browser Use in a separately installed tool environment.

    The isolation is intentional: current Browser Use releases and MemoRizz's
    MCP client require incompatible major SDK lines. MemoRizz resolves the
    Python interpreter behind the ``browser-use`` entry point, writes a fixed
    wrapper to a private temporary directory, and executes it directly in that
    environment. Model-authored Python is never executed by this provider.
    """

    provider_name = "browseruse"

    def __init__(
        self,
        command: Union[str, Sequence[str]] = "browser-use",
        python_command: Optional[Union[str, Sequence[str]]] = None,
        llm_provider: str = "openai",
        model: Optional[str] = None,
        headless: bool = True,
        use_vision: bool = True,
        allowed_domains: Optional[Sequence[str]] = None,
        prohibited_domains: Optional[Sequence[str]] = None,
        block_ip_addresses: bool = True,
        use_cloud: bool = False,
        cdp_url_env: Optional[str] = "BU_CDP_URL",
        max_steps: int = 25,
        task_timeout: int = 600,
        env_allowlist: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        normalized_command = self._normalize_command(command)
        if not normalized_command or not normalized_command[0]:
            raise ValueError("Browser Use command must not be empty")
        self.command = normalized_command
        self.python_command = (
            self._normalize_command(python_command)
            if python_command is not None
            else None
        )
        if self.python_command is not None and not self.python_command[0]:
            raise ValueError("Browser Use python_command must not be empty")
        self.llm_provider = (
            str(llm_provider or "openai").strip().lower().replace("-", "")
        )
        self.model = str(model).strip() if model else None
        self.headless = bool(headless)
        self.use_vision = bool(use_vision)
        self.allowed_domains = _string_list(allowed_domains)
        self.prohibited_domains = _string_list(prohibited_domains)
        self.block_ip_addresses = bool(block_ip_addresses)
        self.use_cloud = bool(use_cloud)
        self.cdp_url_env = str(cdp_url_env or "").strip() or None
        self.max_steps = max(1, min(int(max_steps), 100))
        self.task_timeout = max(10, min(int(task_timeout), 3_600))
        self.env_allowlist = tuple(
            _string_list(env_allowlist or _DEFAULT_ENV_ALLOWLIST)
        )

    @staticmethod
    def _normalize_command(
        command: Union[str, Sequence[str]],
    ) -> list[str]:
        if isinstance(command, str):
            return [command.strip()]
        return [str(item).strip() for item in command]

    @staticmethod
    def _resolve_command(command: Sequence[str]) -> Optional[list[str]]:
        first = command[0]
        if os.path.isabs(first):
            path = Path(first)
            if not path.is_file() or not os.access(path, os.X_OK):
                return None
            resolved = str(path)
        else:
            resolved = shutil.which(first) or ""
            if not resolved:
                return None
        return [resolved, *command[1:]]

    def _resolved_command(self) -> Optional[list[str]]:
        return self._resolve_command(self.command)

    def _resolved_python_command(self) -> Optional[list[str]]:
        """Resolve the Python executable owned by the isolated Browser Use tool.

        Browser Use's CLI surface has changed across releases, while the Python
        quickstart API remains the supported agent interface. ``uv tool`` and
        ``pipx`` entry points contain an absolute interpreter shebang, so the
        default path discovers that interpreter without importing Browser Use
        into MemoRizz's MCP environment. Deployments with another launcher can
        provide ``python_command`` explicitly.
        """
        if self.python_command is not None:
            return self._resolve_command(self.python_command)
        resolved_cli = self._resolved_command()
        if resolved_cli is None:
            return None
        entry_point = Path(resolved_cli[0]).resolve()
        try:
            with entry_point.open("r", encoding="utf-8") as entry_file:
                first_line = entry_file.readline().strip()
        except (OSError, UnicodeError):
            first_line = ""
        if first_line.startswith("#!"):
            try:
                shebang = shlex.split(first_line[2:].strip())
            except ValueError:
                shebang = []
            if shebang:
                interpreter = shebang[0]
                if os.path.isabs(interpreter) and os.access(interpreter, os.X_OK):
                    return shebang

        # Windows/pipx-style launchers commonly sit beside python.exe; this is
        # also a safe fallback for entry points whose shebang cannot be read.
        for candidate_name in ("python", "python3", "python.exe"):
            candidate = entry_point.parent / candidate_name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return [str(candidate)]
        return None

    def validate_configuration(self) -> Optional[str]:
        if self.llm_provider not in {"browseruse", "openai", "anthropic", "google"}:
            return (
                "Browser Use llm_provider must be one of: browseruse, openai, "
                "anthropic, google"
            )
        if self.allowed_domains and "*" in self.allowed_domains:
            return "Use an empty allowed_domains list for unrestricted navigation; '*' is rejected"
        if self._resolved_command() is None:
            return (
                f"Browser Use CLI '{self.command[0]}' was not found. Install it "
                "in an isolated Python 3.11+ tool environment with `uv tool "
                "install --python 3.12 browser-use`, then run `browser-use "
                "install` and `browser-use --doctor`."
            )
        if self._resolved_python_command() is None:
            return (
                "MemoRizz could not resolve the isolated Python interpreter "
                "behind the Browser Use entry point. Configure "
                "MEMORIZZ_BROWSER_USE_PYTHON_COMMAND with that environment's "
                "python executable."
            )
        required_key = {
            "browseruse": "BROWSER_USE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }[self.llm_provider]
        if not os.getenv(required_key):
            return f"{required_key} is required by Browser Use llm_provider={self.llm_provider}"
        return None

    def get_config(self) -> Dict[str, Any]:
        """Persist policy and env-variable names, never credential values."""
        return {
            "provider": self.provider_name,
            "command": list(self.command),
            "python_command": (
                list(self.python_command) if self.python_command is not None else None
            ),
            "llm_provider": self.llm_provider,
            "model": self.model,
            "headless": self.headless,
            "use_vision": self.use_vision,
            "allowed_domains": list(self.allowed_domains),
            "prohibited_domains": list(self.prohibited_domains),
            "block_ip_addresses": self.block_ip_addresses,
            "use_cloud": self.use_cloud,
            "cdp_url_env": self.cdp_url_env,
            "max_steps": self.max_steps,
            "task_timeout": self.task_timeout,
            "env_allowlist": list(self.env_allowlist),
            "execution_boundary": "isolated_browser_use_python",
            "requires_durable_approval": True,
        }

    def _worker_config(self, task: str, max_steps: int) -> Dict[str, Any]:
        return {
            "task": task,
            "llm_provider": self.llm_provider,
            "model": self.model,
            "headless": self.headless,
            "use_vision": self.use_vision,
            "allowed_domains": list(self.allowed_domains),
            "prohibited_domains": list(self.prohibited_domains),
            "block_ip_addresses": self.block_ip_addresses,
            "use_cloud": self.use_cloud,
            "cdp_url_env": self.cdp_url_env,
            "max_steps": max_steps,
        }

    @staticmethod
    def _worker_source(config: Dict[str, Any]) -> str:
        encoded = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
        # This is a fixed program. The untrusted task is decoded from a quoted
        # JSON string instead of being concatenated as executable Python.
        return f"""import asyncio
import importlib.metadata
import inspect
import json
import os

from browser_use import Agent, Browser, ChatAnthropic, ChatBrowserUse, ChatGoogle, ChatOpenAI
try:
    from browser_use import BrowserProfile
except ImportError:  # Browser Use releases before BrowserProfile was public
    BrowserProfile = None

CONFIG = json.loads({encoded!r})
MARKER = {_RESULT_MARKER!r}

def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except Exception:
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(mode="json")
            except Exception:
                pass
        return str(value)

def _accepts(target, name):
    parameters = inspect.signature(target).parameters
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

def _supported_kwargs(target, values, required):
    unsupported = sorted(name for name in required if not _accepts(target, name))
    if unsupported:
        raise RuntimeError(
            "Installed Browser Use cannot enforce required browser policy fields: "
            + ", ".join(unsupported)
        )
    return {{name: value for name, value in values.items() if _accepts(target, name)}}

async def _main():
    provider = CONFIG["llm_provider"]
    model = CONFIG.get("model")
    if provider == "browseruse":
        llm = ChatBrowserUse(**({{"model": model}} if model else {{}}))
    elif provider == "anthropic":
        llm = ChatAnthropic(**({{"model": model}} if model else {{}}))
    elif provider == "google":
        llm = ChatGoogle(**({{"model": model}} if model else {{}}))
    else:
        llm = ChatOpenAI(**({{"model": model}} if model else {{}}))

    policy_kwargs = {{
        "headless": bool(CONFIG["headless"]),
        "block_ip_addresses": bool(CONFIG["block_ip_addresses"]),
        "keep_alive": False,
    }}
    required_policy_fields = {{"headless"}}
    if CONFIG.get("block_ip_addresses"):
        required_policy_fields.add("block_ip_addresses")
    if CONFIG.get("allowed_domains"):
        policy_kwargs["allowed_domains"] = CONFIG["allowed_domains"]
        required_policy_fields.add("allowed_domains")
    if CONFIG.get("prohibited_domains"):
        policy_kwargs["prohibited_domains"] = CONFIG["prohibited_domains"]
        required_policy_fields.add("prohibited_domains")
    if CONFIG.get("use_cloud"):
        policy_kwargs["use_cloud"] = True
        required_policy_fields.add("use_cloud")
    cdp_env = CONFIG.get("cdp_url_env")
    if cdp_env and os.environ.get(cdp_env):
        policy_kwargs["cdp_url"] = os.environ[cdp_env]
        required_policy_fields.add("cdp_url")

    browser = None
    policy_adapter = None
    try:
        if BrowserProfile is not None and _accepts(Browser, "browser_profile"):
            browser_profile = BrowserProfile(
                **_supported_kwargs(
                    BrowserProfile, policy_kwargs, required_policy_fields
                )
            )
            browser = Browser(browser_profile=browser_profile)
            policy_adapter = "browser_profile"
        else:
            browser = Browser(
                **_supported_kwargs(Browser, policy_kwargs, required_policy_fields)
            )
            policy_adapter = "browser_constructor"
        agent = Agent(
            task=CONFIG["task"],
            llm=llm,
            browser=browser,
            use_vision=bool(CONFIG["use_vision"]),
        )
        history = await agent.run(max_steps=int(CONFIG["max_steps"]))
        successful = history.is_successful()
        if successful is None:
            successful = bool(history.is_done() and not history.has_errors())
        payload = {{
            "success": bool(successful),
            "output": _json_safe(history.final_result()),
            "urls": [_json_safe(item) for item in history.urls()],
            "actions": [_json_safe(item) for item in history.action_names()],
            "errors": [str(item) for item in history.errors() if item],
            "steps": int(history.number_of_steps()),
            "duration_seconds": _json_safe(history.total_duration_seconds()),
            "browser_use_version": importlib.metadata.version("browser-use"),
            "browser_policy_adapter": policy_adapter,
        }}
    except Exception as exc:
        payload = {{
            "success": False,
            "output": None,
            "urls": [],
            "actions": [],
            "errors": [f"{{type(exc).__name__}}: {{exc}}"],
            "steps": 0,
            "duration_seconds": None,
        }}
    finally:
        if browser is not None:
            try:
                await browser.stop()
            except Exception:
                pass
    print(MARKER + json.dumps(payload, ensure_ascii=False, default=str))

asyncio.run(_main())
"""

    def _environment(self) -> Dict[str, str]:
        return {
            name: value
            for name in self.env_allowlist
            if (value := os.environ.get(name)) is not None
        }

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - exercised on Windows CI only
                process.terminate()
            process.wait(timeout=3)
        except Exception:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover
                    process.kill()
            except Exception:
                pass

    def run_task(
        self,
        task: str,
        *,
        max_steps: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> BrowserControlResult:
        normalized_task = str(task or "").strip()
        if not normalized_task:
            raise ValueError("browser_control task must not be empty")
        if len(normalized_task) > 20_000:
            raise ValueError("browser_control task exceeds the 20,000-character limit")
        issue = self.validate_configuration()
        if issue:
            raise ValueError(issue)

        bounded_steps = max(1, min(int(max_steps or self.max_steps), self.max_steps))
        bounded_timeout = max(
            10, min(int(timeout or self.task_timeout), self.task_timeout)
        )
        python_command = self._resolved_python_command()
        assert python_command is not None
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="memorizz-browser-") as temp_dir:
            worker_path = Path(temp_dir) / "worker.py"
            worker_path.write_text(
                self._worker_source(
                    self._worker_config(normalized_task, bounded_steps)
                ),
                encoding="utf-8",
            )
            try:
                worker_path.chmod(0o600)
            except OSError:  # pragma: no cover - Windows permissions differ
                pass

            environment = self._environment()
            popen_kwargs: Dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "env": environment,
                "cwd": temp_dir,
            }
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(
                [*python_command, str(worker_path)],
                **popen_kwargs,
            )
            try:
                stdout, stderr = process.communicate(timeout=bounded_timeout)
            except subprocess.TimeoutExpired:
                self._terminate(process)
                return BrowserControlResult(
                    success=False,
                    task=normalized_task,
                    errors=[f"Browser task timed out after {bounded_timeout} seconds"],
                    duration_seconds=round(time.monotonic() - started, 3),
                    metadata=self._result_metadata(bounded_steps, bounded_timeout),
                )

        payload: Optional[Dict[str, Any]] = None
        for line in reversed((stdout or "").splitlines()):
            if line.startswith(_RESULT_MARKER):
                try:
                    candidate = json.loads(line[len(_RESULT_MARKER) :])
                    if isinstance(candidate, dict):
                        payload = candidate
                except json.JSONDecodeError:
                    pass
                break
        if payload is None:
            detail = (
                stderr or stdout or "Browser Use returned no structured result"
            ).strip()
            return BrowserControlResult(
                success=False,
                task=normalized_task,
                errors=[detail[-4_000:]],
                duration_seconds=round(time.monotonic() - started, 3),
                metadata={
                    **self._result_metadata(bounded_steps, bounded_timeout),
                    "return_code": process.returncode,
                },
            )

        return BrowserControlResult(
            success=bool(payload.get("success")),
            task=normalized_task,
            output=payload.get("output"),
            urls=[str(item) for item in payload.get("urls") or []],
            actions=[str(item) for item in payload.get("actions") or []],
            errors=[str(item) for item in payload.get("errors") or []],
            steps=int(payload.get("steps") or 0),
            duration_seconds=(
                float(payload["duration_seconds"])
                if isinstance(payload.get("duration_seconds"), (int, float))
                else round(time.monotonic() - started, 3)
            ),
            metadata={
                **self._result_metadata(bounded_steps, bounded_timeout),
                "browser_use_version": payload.get("browser_use_version"),
                "browser_policy_adapter": payload.get("browser_policy_adapter"),
                "return_code": process.returncode,
            },
        )

    def _result_metadata(self, max_steps: int, timeout: int) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "execution_boundary": "isolated_browser_use_python",
            "llm_provider": self.llm_provider,
            "model": self.model,
            "max_steps": max_steps,
            "timeout": timeout,
            "headless": self.headless,
            "use_vision": self.use_vision,
            "allowed_domains": list(self.allowed_domains),
            "prohibited_domains": list(self.prohibited_domains),
            "block_ip_addresses": self.block_ip_addresses,
            "use_cloud": self.use_cloud,
            "private_worker_process": True,
        }


register_provider("browseruse", BrowserUseProvider)
register_provider("browser-use", BrowserUseProvider)

"""First-party adapters for MemAgent, Codex, Claude Code, and OpenHands."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AdapterOutcome, AgentHarness, EventSink, SubprocessHarness
from .models import (
    HarnessCapabilities,
    HarnessContextPack,
    HarnessEvent,
    HarnessEventType,
    HarnessTask,
)
from .security import build_child_environment


def _text_blocks(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = message
    if isinstance(content, str):
        return content
    values: List[str] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            values.append(str(block.get("text") or ""))
    return "\n".join(item for item in values if item)


class NativeMemAgentHarness(AgentHarness):
    name = "memagent"

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def probe(self) -> HarnessCapabilities:
        available = self.agent is not None
        usage_reporting = callable(
            getattr(getattr(self.agent, "model", None), "get_last_usage", None)
        )
        return HarnessCapabilities(
            name=self.name,
            available=available,
            version=None,
            command="in-process",
            error_code=None if available else "agent_not_configured",
            error=None if available else "No in-process MemAgent is configured",
            remediation=(
                None
                if available
                else "Attach a MemAgent to MetaHarness or select a saved MemAgent by agent_id."
            ),
            structured_events=True,
            resume=False,
            mcp=True,
            per_action_approvals=True,
            requires_external_isolation=False,
            usage_reporting=usage_reporting,
            metadata={
                "cost_reporting": False,
                "token_reporting": usage_reporting,
                "network_modes": ["none"],
                "task_tool_policy": False,
                "agent_id": getattr(self.agent, "agent_id", None),
                "cancellation": "pre_start_only",
                "wall_time_enforcement": "provider_cooperative",
                "configured_max_steps": getattr(self.agent, "max_steps", None),
            },
        )

    def run(
        self,
        task: HarnessTask,
        *,
        workspace: Path,
        context_pack: HarnessContextPack,
        emit: EventSink,
        cancel_event: Any,
    ) -> AdapterOutcome:
        if cancel_event.is_set():
            return AdapterOutcome(
                error_code="canceled", error="Harness run was canceled"
            )
        emit(HarnessEvent(task.run_id, HarnessEventType.STATUS, {"status": "running"}))
        context = dict(task.context)
        if context_pack.rendered:
            context["harness_memory_context"] = context_pack.rendered
            context["harness_memory_source_ids"] = context_pack.source_ids
        response = self.agent.run(
            task.task,
            memory_id=task.memory_id,
            thread_id=task.thread_id,
            user_id=task.user_id,
            context=context,
            tool_context={
                "_memorizz_harness_native": True,
                "workspace": str(workspace),
                "harness_run_id": task.run_id,
            },
            observability_context={
                "run_id": task.run_id,
                "memory_id": task.memory_id,
                "thread_id": task.thread_id,
                "user_id": task.user_id,
            },
        )
        emit(
            HarnessEvent(
                task.run_id,
                HarnessEventType.MESSAGE,
                {"role": "assistant", "text": str(response)},
            )
        )
        usage: Dict[str, Any] = {}
        usage_getter = getattr(
            getattr(self.agent, "model", None), "get_last_usage", None
        )
        if callable(usage_getter):
            try:
                candidate = usage_getter() or {}
            except Exception:
                candidate = {}
            if isinstance(candidate, dict):
                usage = dict(candidate)
        return AdapterOutcome(
            final_response=str(response),
            usage=usage,
            exit_code=0,
        )


class PersistedMemAgentHarness(AgentHarness):
    """Load one explicitly selected saved MemAgent for a standalone run."""

    name = "memagent"

    def __init__(self, memory_provider: Any) -> None:
        self.memory_provider = memory_provider

    def probe(self) -> HarnessCapabilities:
        available = callable(getattr(self.memory_provider, "retrieve_memagent", None))
        saved_agent_count: Optional[int] = None
        error = None
        if available and callable(
            getattr(self.memory_provider, "list_memagents", None)
        ):
            try:
                saved_agent_count = len(self.memory_provider.list_memagents() or [])
            except Exception as exc:
                error = f"Saved MemAgents could not be listed: {type(exc).__name__}"
        if available and saved_agent_count == 0:
            error = "No saved MemAgent is available; create one before native execution"
        return HarnessCapabilities(
            name=self.name,
            available=available,
            command="in-process",
            error=(
                error if available else "The memory provider cannot load saved agents"
            ),
            error_code=(
                "saved_agent_unavailable"
                if available
                else "memory_provider_unsupported"
            ),
            remediation=(
                "Create and save a MemAgent, then select it with agent_id."
                if available
                else "Configure a memory provider that supports saved MemAgents."
            ),
            structured_events=True,
            resume=False,
            mcp=True,
            per_action_approvals=True,
            requires_external_isolation=False,
            usage_reporting=False,
            metadata={
                "cost_reporting": False,
                "token_reporting": False,
                "network_modes": ["none"],
                "task_tool_policy": False,
                "requires_agent_id": True,
                "explicit_only": True,
                "saved_agent_count": saved_agent_count,
                "cancellation": "pre_start_only",
                "wall_time_enforcement": "provider_cooperative",
            },
        )

    def run(
        self,
        task: HarnessTask,
        *,
        workspace: Path,
        context_pack: HarnessContextPack,
        emit: EventSink,
        cancel_event: Any,
    ) -> AdapterOutcome:
        if not task.agent_id:
            return AdapterOutcome(
                error_code="agent_id_required",
                error="A persisted MemAgent harness run requires agent_id",
            )
        agent = None
        try:
            from ..memagent import MemAgent

            agent = MemAgent.load(
                task.agent_id,
                memory_provider=self.memory_provider,
                meta_harness=False,
                meta_harness_mode=None,
            )
            return NativeMemAgentHarness(agent).run(
                task,
                workspace=workspace,
                context_pack=context_pack,
                emit=emit,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            return AdapterOutcome(
                error_code="memagent_load_or_run_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if agent is not None:
                try:
                    agent.close(close_memory_provider=False)
                except Exception:
                    pass


class CodexHarness(SubprocessHarness):
    name = "codex"
    executable = "codex"
    supports_output_schema = True
    auth_environment = ("CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_HOME")
    authentication_remediation = (
        "Set OPENAI_API_KEY or CODEX_API_KEY, or authenticate the CLI with "
        "`codex login`, then rerun `memorizz harness doctor codex`."
    )

    def probe(self) -> HarnessCapabilities:
        capability = super().probe()
        if not capability.available or capability.error or not capability.command:
            return capability
        authentication = capability.metadata.get("authentication_configured")
        if authentication is not None or capability.metadata.get(
            "authentication_checked"
        ):
            return capability

        metadata = dict(capability.metadata)
        metadata["authentication_checked"] = True
        try:
            result = subprocess.run(
                [str(capability.command), "login", "status"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env=build_child_environment(allowed_names=self.auth_environment),
            )
        except (OSError, subprocess.TimeoutExpired):
            metadata.update(
                authentication_status="unknown",
                authentication_note=(
                    "Codex login status could not be verified; runtime authentication "
                    "failures will still be normalized."
                ),
            )
            updated = replace(capability, metadata=metadata)
        else:
            diagnostic = f"{result.stdout}\n{result.stderr}".lower()
            if result.returncode == 0:
                metadata.update(
                    authentication_configured=True,
                    authentication_status="authenticated",
                    authentication_mode="cli_managed",
                    authentication_note=None,
                )
                updated = replace(capability, metadata=metadata)
            elif "not logged in" in diagnostic:
                metadata.update(
                    authentication_configured=False,
                    authentication_status="missing",
                    authentication_mode="cli_managed_or_environment",
                )
                updated = replace(
                    capability,
                    error_code="authentication_required",
                    error="Codex CLI is installed but is not authenticated.",
                    remediation=self.authentication_remediation,
                    metadata=metadata,
                )
            else:
                metadata.update(
                    authentication_status="unknown",
                    authentication_note=(
                        "Codex login status could not be verified; runtime authentication "
                        "failures will still be normalized."
                    ),
                )
                updated = replace(capability, metadata=metadata)
        with self._process_lock:
            self._probe_cache = updated
        return updated

    def _capabilities(self, **values: Any) -> HarnessCapabilities:
        probe_error = values.get("error")
        environment_auth = any(
            os.environ.get(name) or self.extra_env.get(name)
            for name in ("CODEX_API_KEY", "OPENAI_API_KEY")
        )
        return HarnessCapabilities(
            name=self.name,
            available=values["available"],
            command=values.get("command"),
            version=values.get("version"),
            error_code=(
                "harness_unavailable"
                if probe_error and not values["available"]
                else None
            ),
            error=probe_error,
            remediation=(
                f"Install {self.command!r}, ensure it is executable, then rerun "
                "`memorizz harness doctor codex`."
                if probe_error and not values["available"]
                else None
            ),
            structured_events=True,
            resume=False,
            mcp=True,
            per_action_approvals=True,
            requires_external_isolation=False,
            usage_reporting=True,
            metadata={
                "cost_reporting": False,
                "token_reporting": True,
                "network_policy": "codex_sandbox",
                "network_modes": ["none"],
                "task_tool_policy": False,
                "output_schema": True,
                # Absence of an environment key is not a failure: Codex may use
                # its authenticated CODEX_HOME credential store.
                "authentication_configured": True if environment_auth else None,
                "authentication_mode": (
                    "environment" if environment_auth else "cli_managed_or_environment"
                ),
                "authentication_note": (
                    None
                    if environment_auth
                    else "No API key was detected in the process environment; an authenticated Codex CLI session may still be used."
                ),
            },
        )

    def build_command(
        self, task: HarnessTask, *, workspace: Path, prompt: str
    ) -> List[str]:
        sandbox = "workspace-write" if task.writes_workspace else "read-only"
        command = [
            self.command,
            "exec",
            "--json",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            sandbox,
            "--cd",
            str(workspace),
        ]
        if not task.metadata.get("persist_session", False):
            command.insert(3, "--ephemeral")
        if task.model or self.default_model:
            command.extend(["--model", str(task.model or self.default_model)])
        output_schema_path = task.metadata.get("_memorizz_output_schema_path")
        if output_schema_path:
            command.extend(["--output-schema", str(output_schema_path)])
        # Trusted SDK callers may supply non-policy Codex tuning, but the
        # host-owned safety layer is appended afterwards so it cannot be
        # weakened. Project-scoped Codex config can otherwise enable hooks,
        # web search, extra MCP servers, egress, or additional writable roots.
        for value in task.metadata.get("codex_config", []) or []:
            command.extend(["--config", str(value)])
        policy_overrides = [
            'approval_policy="never"',
            'web_search="disabled"',
            "tools.web_search=false",
            "features.skill_mcp_dependency_install=false",
            "hooks={}",
            "mcp_servers={}",
            "sandbox_workspace_write.network_access=false",
            "sandbox_workspace_write.writable_roots=[]",
            "sandbox_workspace_write.exclude_slash_tmp=true",
            "sandbox_workspace_write.exclude_tmpdir_env_var=true",
        ]
        for value in policy_overrides:
            command.extend(["--config", value])
        # Only the generated first-party MemoRizz server may be restored after
        # clearing project/user MCP configuration.
        for value in task.metadata.get("_memorizz_codex_mcp_config", []) or []:
            command.extend(["--config", str(value)])
        # The generated MemoRizz prompt begins with a Markdown delimiter. An
        # explicit end-of-options marker prevents the CLI from interpreting
        # that prompt (or any user task text within it) as another flag.
        command.extend(["--", prompt])
        return command

    def parse_event(
        self, run_id: str, payload: Dict[str, Any]
    ) -> tuple[List[HarnessEvent], Dict[str, Any]]:
        event_type = str(payload.get("type") or "")
        events: List[HarnessEvent] = []
        updates: Dict[str, Any] = {}
        if event_type == "thread.started":
            checkpoint = {"thread_id": payload.get("thread_id")}
            updates["checkpoint"] = checkpoint
            events.append(HarnessEvent(run_id, HarnessEventType.STATUS, checkpoint))
        elif event_type.startswith("item."):
            item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
            kind = str(item.get("type") or "")
            if kind == "agent_message":
                text = str(item.get("text") or _text_blocks(item))
                if text:
                    updates["final_response"] = text
                    events.append(
                        HarnessEvent(
                            run_id,
                            HarnessEventType.MESSAGE,
                            {"role": "assistant", "text": text},
                        )
                    )
            elif kind == "command_execution":
                events.append(HarnessEvent(run_id, HarnessEventType.COMMAND, item))
            elif kind in {"file_change", "file_changes"}:
                events.append(HarnessEvent(run_id, HarnessEventType.FILE_CHANGE, item))
            elif kind in {"mcp_tool_call", "tool_call", "web_search"}:
                events.append(HarnessEvent(run_id, HarnessEventType.TOOL_CALL, item))
            else:
                events.append(HarnessEvent(run_id, HarnessEventType.LOG, payload))
        elif event_type == "turn.completed":
            usage = dict(payload.get("usage") or {})
            updates["usage"] = usage
            events.append(HarnessEvent(run_id, HarnessEventType.USAGE, usage))
        elif event_type in {"turn.failed", "error"}:
            error = str(
                payload.get("message") or payload.get("error") or "Codex run failed"
            )
            updates.update(error_code="codex_failed", error=error)
            events.append(
                HarnessEvent(run_id, HarnessEventType.ERROR, {"error": error})
            )
        else:
            events.append(HarnessEvent(run_id, HarnessEventType.LOG, payload))
        return events, updates


class ClaudeCodeHarness(SubprocessHarness):
    name = "claude-code"
    executable = "claude"
    supports_output_schema = True
    auth_environment = (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    )
    authentication_remediation = (
        "Set ANTHROPIC_API_KEY or enable CLAUDE_CODE_USE_BEDROCK, "
        "CLAUDE_CODE_USE_VERTEX, or CLAUDE_CODE_USE_FOUNDRY, then rerun "
        "`memorizz harness doctor claude-code`."
    )

    def _authentication_configured(self) -> bool:
        api_key = self.extra_env.get("ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if str(api_key or "").strip():
            return True
        for name in self.auth_environment[1:]:
            value = self.extra_env.get(name)
            if value is None:
                value = os.environ.get(name)
            if str(value or "").strip().lower() in {"1", "true", "yes", "on"}:
                return True
        return False

    def _capabilities(self, **values: Any) -> HarnessCapabilities:
        probe_error = values.get("error")
        authenticated = self._authentication_configured()
        authentication_error = bool(
            values["available"] and not probe_error and not authenticated
        )
        return HarnessCapabilities(
            name=self.name,
            available=values["available"],
            command=values.get("command"),
            version=values.get("version"),
            error_code=(
                "harness_unavailable"
                if probe_error and not values["available"]
                else "authentication_required"
                if authentication_error
                else None
            ),
            error=(
                probe_error
                or (
                    "ANTHROPIC_API_KEY or a supported cloud-provider mode is required "
                    "because --bare disables OAuth/keychain authentication"
                    if authentication_error
                    else None
                )
            ),
            remediation=(
                f"Install {self.command!r}, ensure it is executable, then rerun "
                "`memorizz harness doctor claude-code`."
                if probe_error and not values["available"]
                else self.authentication_remediation
                if authentication_error
                else None
            ),
            structured_events=True,
            resume=False,
            mcp=True,
            per_action_approvals=True,
            requires_external_isolation=False,
            usage_reporting=True,
            metadata={
                "cost_reporting": True,
                "token_reporting": True,
                "network_policy": "tool_allowlist",
                "network_modes": ["none", "full"],
                "forbidden_tools": ["bash"],
                "task_tool_policy": True,
                "output_schema": True,
                "authentication_configured": authenticated,
                "authentication_mode": "environment_or_cloud_provider",
                "required_environment_any": list(self.auth_environment),
            },
        )

    def build_command(
        self, task: HarnessTask, *, workspace: Path, prompt: str
    ) -> List[str]:
        command = [
            self.command,
            "--bare",
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "acceptEdits" if task.writes_workspace else "dontAsk",
        ]
        # Ephemeral runs are the safe default. A caller may explicitly retain a
        # session when it intends to use the adapter checkpoint for resumption.
        if not task.metadata.get("persist_session", False):
            command.append("--no-session-persistence")
        allowed = list(task.permissions.allowed_tools) or ["Read", "Glob", "Grep"]
        if task.writes_workspace:
            for tool in ("Edit", "Write"):
                if tool not in allowed:
                    allowed.append(tool)
        if task.permissions.network == "full":
            for tool in ("WebFetch", "WebSearch"):
                if tool not in allowed:
                    allowed.append(tool)
        if task.permissions.mcp_access != "none":
            allowed.append("mcp__memorizz__*")
        command.extend(["--allowedTools", ",".join(allowed)])
        denied = ["Bash", *task.permissions.denied_tools]
        if task.permissions.network != "full":
            denied.extend(["WebFetch", "WebSearch"])
        denied = list(dict.fromkeys(str(item) for item in denied if str(item).strip()))
        command.extend(["--disallowedTools", ",".join(denied)])
        if task.model or self.default_model:
            command.extend(["--model", str(task.model or self.default_model)])
        effort = task.metadata.get("claude_effort")
        if effort is not None:
            normalized_effort = str(effort).strip().lower()
            if normalized_effort not in {"low", "medium", "high", "xhigh", "max"}:
                raise ValueError(
                    "claude_effort must be low, medium, high, xhigh, or max"
                )
            command.extend(["--effort", normalized_effort])
        if task.output_schema:
            command.extend(
                [
                    "--json-schema",
                    json.dumps(
                        task.output_schema,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        if task.budget.max_cost_usd is not None:
            command.extend(["--max-budget-usd", str(task.budget.max_cost_usd)])
        command.extend(["--max-turns", str(task.budget.max_steps)])
        mcp_config = task.metadata.get("mcp_config_path")
        if mcp_config:
            command.extend(["--mcp-config", str(mcp_config), "--strict-mcp-config"])
        # Keep the prompt after every option and terminate option parsing. In
        # particular, MemoRizz prompts start with ``---`` and must remain an
        # opaque positional value to Claude Code.
        command.extend(["--", prompt])
        return command

    def parse_event(
        self, run_id: str, payload: Dict[str, Any]
    ) -> tuple[List[HarnessEvent], Dict[str, Any]]:
        kind = str(payload.get("type") or "")
        subtype = str(payload.get("subtype") or "")
        events: List[HarnessEvent] = []
        updates: Dict[str, Any] = {}
        if kind == "system" and subtype == "init":
            checkpoint = {"session_id": payload.get("session_id")}
            updates["checkpoint"] = checkpoint
            events.append(
                HarnessEvent(
                    run_id,
                    HarnessEventType.STATUS,
                    {
                        **checkpoint,
                        "model": payload.get("model"),
                        "capabilities": payload.get("capabilities") or [],
                        "mcp_servers": payload.get("mcp_servers") or [],
                        "mcp_server_errors": payload.get("mcp_server_errors") or [],
                    },
                )
            )
        elif kind == "assistant":
            message = payload.get("message") or payload
            text = _text_blocks(message)
            if text:
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.MESSAGE,
                        {"role": "assistant", "text": text},
                    )
                )
            content = message.get("content") if isinstance(message, dict) else []
            for block in content or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    events.append(
                        HarnessEvent(run_id, HarnessEventType.TOOL_CALL, block)
                    )
        elif kind == "user":
            events.append(HarnessEvent(run_id, HarnessEventType.TOOL_RESULT, payload))
        elif kind == "result":
            response = str(payload.get("result") or "")
            updates["final_response"] = response
            updates["cost_usd"] = payload.get("total_cost_usd")
            updates["usage"] = dict(payload.get("usage") or {})
            updates["checkpoint"] = {"session_id": payload.get("session_id")}
            if payload.get("is_error"):
                updates.update(
                    error_code="claude_failed",
                    error=response or "Claude Code run failed",
                )
            events.append(HarnessEvent(run_id, HarnessEventType.COMPLETE, payload))
        elif kind == "system" and subtype == "api_retry":
            events.append(HarnessEvent(run_id, HarnessEventType.STATUS, payload))
        else:
            events.append(HarnessEvent(run_id, HarnessEventType.LOG, payload))
        return events, updates


class OpenHandsHarness(SubprocessHarness):
    name = "openhands"
    executable = "openhands"
    auth_environment = (
        "LLM_API_KEY",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    )
    authentication_remediation = (
        "Configure the OpenHands LLM provider with LLM_API_KEY and LLM_MODEL, "
        "a supported provider key, or its authenticated CLI configuration, then "
        "rerun `memorizz harness doctor openhands`."
    )

    def __init__(self, *, external_isolation: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.external_isolation = bool(external_isolation)

    def _capabilities(self, **values: Any) -> HarnessCapabilities:
        probe_error = values.get("error")
        environment_auth = any(
            os.environ.get(name) or self.extra_env.get(name)
            for name in (
                "LLM_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
            )
        )
        isolation_error = bool(
            values["available"] and not probe_error and not self.external_isolation
        )
        return HarnessCapabilities(
            name=self.name,
            available=values["available"],
            command=values.get("command"),
            version=values.get("version"),
            error_code=(
                "harness_unavailable"
                if probe_error and not values["available"]
                else "external_isolation_required"
                if isolation_error
                else None
            ),
            error=(
                probe_error
                or (
                    "Headless OpenHands is always-approve; configure an isolated "
                    "command wrapper and set external_isolation=true"
                    if isolation_error
                    else None
                )
            ),
            remediation=(
                f"Install {self.command!r}, ensure it is executable, then rerun "
                "`memorizz harness doctor openhands`."
                if probe_error and not values["available"]
                else "Configure an isolated Docker, remote, or sandbox command wrapper and set MEMORIZZ_OPENHANDS_EXTERNAL_ISOLATION=true."
                if isolation_error
                else None
            ),
            structured_events=True,
            resume=False,
            mcp=True,
            per_action_approvals=False,
            requires_external_isolation=True,
            usage_reporting=False,
            metadata={
                "cost_reporting": False,
                "token_reporting": False,
                "network_modes": ["none", "restricted", "full"],
                "external_isolation_configured": self.external_isolation,
                "headless_always_approve": True,
                "task_tool_policy": False,
                # OpenHands can resolve credentials from its provider config,
                # so a missing environment key is diagnostic rather than fatal.
                "authentication_configured": True if environment_auth else None,
                "authentication_mode": (
                    "environment" if environment_auth else "provider_or_cli_managed"
                ),
                "authentication_note": (
                    None
                    if environment_auth
                    else "No API key was detected in the process environment; OpenHands provider or CLI configuration may still supply authentication."
                ),
            },
        )

    def build_command(
        self, task: HarnessTask, *, workspace: Path, prompt: str
    ) -> List[str]:
        return [self.command, "--headless", "--json", "--task", prompt]

    def parse_event(
        self, run_id: str, payload: Dict[str, Any]
    ) -> tuple[List[HarnessEvent], Dict[str, Any]]:
        kind = str(payload.get("type") or "")
        events: List[HarnessEvent] = []
        updates: Dict[str, Any] = {}
        if kind == "action":
            action = str(payload.get("action") or "")
            event_type = (
                HarnessEventType.COMMAND
                if action in {"run", "execute", "command"}
                else HarnessEventType.FILE_CHANGE
                if action in {"write", "edit", "patch"}
                else HarnessEventType.TOOL_CALL
            )
            events.append(HarnessEvent(run_id, event_type, payload))
        elif kind == "observation":
            events.append(HarnessEvent(run_id, HarnessEventType.TOOL_RESULT, payload))
        elif kind in {"message", "assistant", "final"}:
            text = str(
                payload.get("message")
                or payload.get("content")
                or payload.get("text")
                or ""
            )
            if text:
                updates["final_response"] = text
            events.append(
                HarnessEvent(
                    run_id,
                    HarnessEventType.MESSAGE,
                    {"role": "assistant", "text": text},
                )
            )
        elif kind in {"error", "failure"}:
            error = str(
                payload.get("error") or payload.get("message") or "OpenHands run failed"
            )
            updates.update(error_code="openhands_failed", error=error)
            events.append(
                HarnessEvent(run_id, HarnessEventType.ERROR, {"error": error})
            )
        else:
            events.append(HarnessEvent(run_id, HarnessEventType.LOG, payload))
        return events, updates


__all__ = [
    "ClaudeCodeHarness",
    "CodexHarness",
    "NativeMemAgentHarness",
    "OpenHandsHarness",
    "PersistedMemAgentHarness",
]

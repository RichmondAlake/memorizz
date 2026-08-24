"""Memory-first control plane for native and external agent harnesses."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

from ..approval import (
    ApprovalStateError,
    ApprovalStatus,
    ApprovalStore,
    default_approval_store,
)
from ..observability import ObservabilityStore
from .adapters import ClaudeCodeHarness, CodexHarness, OpenHandsHarness
from .base import AgentHarness
from .context import HarnessContextBuilder
from .models import (
    HarnessEvent,
    HarnessEventType,
    HarnessPlan,
    HarnessResult,
    HarnessRun,
    HarnessStatus,
    HarnessTask,
    canonical_hash,
    count_harness_steps,
    utcnow_iso,
)
from .router import HarnessReadinessError, HarnessRouter
from .security import (
    HarnessSecurityError,
    build_child_environment,
    redact,
    resolve_workspace,
    workspace_diff,
    workspace_snapshot,
)
from .store import HarnessRunStore, SQLiteHarnessRunStore


class MetaHarness:
    """Route, govern, execute, remember, and evaluate agent harness runs."""

    def __init__(
        self,
        *,
        memory_provider: Any = None,
        adapters: Optional[Iterable[AgentHarness]] = None,
        run_store: Optional[HarnessRunStore] = None,
        approval_store: Optional[ApprovalStore] = None,
        router: Optional[HarnessRouter] = None,
        context_max_chars: int = 24_000,
        allowed_workspace_roots: Optional[Iterable[str]] = None,
        recover_interrupted: bool = False,
        agent: Any = None,
        learning_control_plane: Any = None,
        context_cache_entries: int = 128,
    ) -> None:
        self.memory_provider = memory_provider
        self.run_store = run_store or SQLiteHarnessRunStore()
        self.approval_store = approval_store or default_approval_store()
        self.router = router or HarnessRouter()
        self.context_builder = HarnessContextBuilder(
            memory_provider, max_chars=context_max_chars
        )
        self.allowed_workspace_roots = [
            str(Path(item).expanduser().resolve())
            for item in (allowed_workspace_roots or [])
        ]
        self.agent = agent
        self.learning_control_plane = learning_control_plane
        self._owned_learning_control_planes: Dict[str, Any] = {}
        self._context_cache_entries = max(1, min(int(context_cache_entries), 1_024))
        self._context_pack_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._context_cache_lock = threading.RLock()
        self.adapters: Dict[str, AgentHarness] = {}
        for adapter in adapters or []:
            self.register(adapter)
        self._cancel_events: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._closed = False
        self.recovered_runs = (
            self.run_store.recover_interrupted() if recover_interrupted else 0
        )

    def recover_interrupted_runs(self) -> int:
        """Mark orphaned queued/running rows interrupted when a worker starts."""
        self.recovered_runs = self.run_store.recover_interrupted()
        return self.recovered_runs

    @classmethod
    def from_env(
        cls,
        *,
        memory_provider: Any = None,
        agent: Any = None,
        run_store: Optional[HarnessRunStore] = None,
        approval_store: Optional[ApprovalStore] = None,
        allowed_workspace_roots: Optional[Iterable[str]] = None,
    ) -> "MetaHarness":
        from .config import load_harness_config

        configured = load_harness_config()
        adapter_config = dict(configured.get("adapters") or {})
        adapters: List[AgentHarness] = []
        codex = dict(adapter_config.get("codex") or {})
        if codex.get("enabled", True):
            adapters.append(
                CodexHarness(
                    command=os.getenv(
                        "MEMORIZZ_CODEX_COMMAND", codex.get("command") or "codex"
                    ),
                    default_model=os.getenv("MEMORIZZ_CODEX_MODEL")
                    or codex.get("model")
                    or None,
                )
            )
        claude = dict(adapter_config.get("claude-code") or {})
        if claude.get("enabled", True):
            adapters.append(
                ClaudeCodeHarness(
                    command=os.getenv(
                        "MEMORIZZ_CLAUDE_CODE_COMMAND",
                        claude.get("command") or "claude",
                    ),
                    default_model=os.getenv("MEMORIZZ_CLAUDE_CODE_MODEL")
                    or claude.get("model")
                    or None,
                )
            )
        openhands = dict(adapter_config.get("openhands") or {})
        if openhands.get("enabled", True):
            adapters.append(
                OpenHandsHarness(
                    command=os.getenv(
                        "MEMORIZZ_OPENHANDS_COMMAND",
                        openhands.get("command") or "openhands",
                    ),
                    default_model=os.getenv("MEMORIZZ_OPENHANDS_MODEL")
                    or openhands.get("model")
                    or None,
                    external_isolation=(
                        os.getenv("MEMORIZZ_OPENHANDS_EXTERNAL_ISOLATION", "")
                        .strip()
                        .lower()
                        in {"1", "true", "yes", "on"}
                        or bool(openhands.get("external_isolation", False))
                    ),
                )
            )
        if agent is not None:
            from .adapters import NativeMemAgentHarness

            adapters.insert(0, NativeMemAgentHarness(agent))
        elif callable(getattr(memory_provider, "retrieve_memagent", None)):
            from .adapters import PersistedMemAgentHarness

            adapters.insert(0, PersistedMemAgentHarness(memory_provider))
        allowlist = os.getenv("MEMORIZZ_HARNESS_ALLOWLIST")
        router = HarnessRouter(
            preference=configured.get("preference")
            or ("memagent", "codex", "claude-code", "openhands"),
            allowlist=(
                [item.strip() for item in allowlist.split(",") if item.strip()]
                if allowlist
                else configured.get("allowlist")
            ),
        )
        return cls(
            memory_provider=memory_provider,
            adapters=adapters,
            run_store=run_store,
            approval_store=approval_store,
            router=router,
            context_max_chars=int(
                os.getenv(
                    "MEMORIZZ_HARNESS_CONTEXT_MAX_CHARS",
                    str(configured.get("context_max_chars") or 24_000),
                )
            ),
            allowed_workspace_roots=(
                list(allowed_workspace_roots)
                if allowed_workspace_roots is not None
                else configured.get("allowed_workspace_roots") or [os.getcwd()]
            ),
            agent=agent,
        )

    def register(self, adapter: AgentHarness) -> "MetaHarness":
        name = str(adapter.name or "").strip().lower().replace("_", "-")
        if not name:
            raise ValueError("Harness adapter name is required")
        self.adapters[name] = adapter
        return self

    def list_harnesses(self, *, probe: bool = True) -> List[Dict[str, Any]]:
        rows = []
        for name, adapter in sorted(self.adapters.items()):
            if probe:
                rows.append(adapter.probe().to_dict())
            else:
                rows.append({"name": name})
        return rows

    def probe(self, name: str) -> Dict[str, Any]:
        normalized = str(name).strip().lower().replace("_", "-")
        normalized = {
            "claude": "claude-code",
            "claudecode": "claude-code",
            "open-hands": "openhands",
            "native": "memagent",
            "memorizz": "memagent",
        }.get(normalized, normalized)
        adapter = self.adapters.get(normalized)
        if adapter is None:
            raise KeyError(f"Unknown harness: {name}")
        return adapter.probe().to_dict()

    def _routing_metrics(self) -> Dict[str, Dict[str, float]]:
        groups: Dict[str, List[HarnessRun]] = {}
        for run in self.run_store.list(limit=500):
            if run.harness and run.status.terminal:
                groups.setdefault(run.harness, []).append(run)
        result: Dict[str, Dict[str, float]] = {}
        for harness, runs in groups.items():
            verified = 0
            costs: List[float] = []
            for run in runs:
                payload = dict(run.result or {})
                if payload.get("verified") and payload.get("ok"):
                    verified += 1
                if payload.get("cost_usd") is not None:
                    costs.append(float(payload["cost_usd"]))
            result[harness] = {
                "verified_success_rate": verified / len(runs),
                "average_cost_usd": sum(costs) / len(costs) if costs else 0.0,
            }
        return result

    @staticmethod
    def _bounded_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: snapshot.get(key)
            for key in ("workspace", "git_repository", "head", "dirty", "fingerprint")
        }

    def _prepare(
        self, task: HarnessTask
    ) -> tuple[HarnessTask, Path, Dict[str, Any], Dict[str, Any]]:
        aliases = {
            "claude": "claude-code",
            "claudecode": "claude-code",
            "open-hands": "openhands",
            "native": "memagent",
            "memorizz": "memagent",
        }
        if task.harness in aliases:
            payload = task.to_dict()
            payload["harness"] = aliases[task.harness]
            task = HarnessTask.from_dict(payload)
        if not task.permissions.allowed_roots and self.allowed_workspace_roots:
            payload = task.to_dict()
            payload["permissions"]["allowed_roots"] = list(self.allowed_workspace_roots)
            task = HarnessTask.from_dict(payload)
        if self.allowed_workspace_roots:
            # A task can narrow the operator's root policy, never widen it.
            resolve_workspace(task.workspace, self.allowed_workspace_roots)
        workspace = resolve_workspace(task.workspace, task.permissions.allowed_roots)
        before = workspace_snapshot(
            workspace, include_untracked_contents=task.writes_workspace
        )
        if (
            task.writes_workspace
            and before.get("dirty")
            and not task.permissions.allow_dirty_workspace
        ):
            raise HarnessSecurityError(
                "Write-capable harness runs reject dirty Git worktrees by default; "
                "set allow_dirty_workspace=True only when the existing changes are expected"
            )
        adapter, decision = self.router.select(
            task, self.adapters, metrics=self._routing_metrics()
        )
        resolved_payload = task.to_dict()
        resolved_payload["harness"] = adapter.name
        resolved_task = HarnessTask.from_dict(resolved_payload)
        return resolved_task, workspace, before, decision.to_dict()

    def _create_run(
        self,
        task: HarnessTask,
        *,
        status: HarnessStatus,
        routing: Optional[Dict[str, Any]] = None,
        approval_proposal_id: Optional[str] = None,
    ) -> HarnessRun:
        run = HarnessRun(
            run_id=task.run_id,
            task=redact(task.to_dict()),
            status=status,
            harness=task.harness if task.harness != "auto" else None,
            routing=dict(routing or {}),
            approval_proposal_id=approval_proposal_id,
        )
        self.run_store.create(run)
        self._emit(task.run_id, HarnessEventType.STATUS, {"status": status.value})
        return run

    def _proposal(
        self,
        task: HarnessTask,
        *,
        before: Dict[str, Any],
        routing: Dict[str, Any],
    ):
        arguments = task.approval_arguments(
            workspace_fingerprint=str(before.get("fingerprint") or "")
        )
        proposal = self.approval_store.propose(
            owner_id=str(
                task.metadata.get("approval_owner_id") or task.agent_id or "metaharness"
            ),
            tool_name="metaharness.run",
            arguments=arguments,
            policy_reason=(
                "The external harness can modify a workspace or use network, secrets, "
                "governed MCP capabilities, or a model-supplied host verification "
                "command. Approve the exact bounded run envelope."
            ),
            checkpoint={
                "version": 1,
                "operation": "harness_run",
                "task": task.to_dict(),
                "workspace_snapshot": self._bounded_snapshot(before),
                "routing": routing,
                "thread_id": task.thread_id,
            },
            ttl_seconds=int(task.metadata.get("approval_ttl_seconds") or 900),
        )
        return proposal

    @staticmethod
    def _approval_required(task: HarnessTask) -> bool:
        return bool(
            task.writes_workspace
            or task.permissions.require_approval
            or (
                task.verification.command
                and task.metadata.get("model_initiated", False)
            )
        )

    @staticmethod
    def _preparation_failure(
        exc: Exception,
    ) -> tuple[str, str, Optional[str], Dict[str, Any]]:
        if isinstance(exc, HarnessReadinessError):
            details = exc.to_dict()
            return exc.code, str(exc), exc.remediation, {"rejection": details}
        return type(exc).__name__, str(exc), None, {}

    def run(self, task: HarnessTask | Dict[str, Any]) -> HarnessResult:
        resolved_input = (
            task if isinstance(task, HarnessTask) else HarnessTask.from_dict(task)
        )
        try:
            task_value, workspace, before, routing = self._prepare(resolved_input)
        except Exception as exc:
            error_code, error, remediation, routing = self._preparation_failure(exc)
            failed = HarnessResult(
                run_id=resolved_input.run_id,
                harness=resolved_input.harness,
                status=HarnessStatus.FAILED,
                routing=routing,
                error_code=error_code,
                error=error,
                remediation=remediation,
            )
            self._create_run(
                resolved_input, status=HarnessStatus.FAILED, routing=routing
            )
            self.run_store.update(
                resolved_input.run_id,
                finished_at=utcnow_iso(),
                result=failed.to_dict(),
            )
            return failed

        if self._approval_required(task_value):
            proposal = self._proposal(task_value, before=before, routing=routing)
            self._create_run(
                task_value,
                status=HarnessStatus.PENDING_APPROVAL,
                routing=routing,
                approval_proposal_id=proposal.proposal_id,
            )
            self._emit(
                task_value.run_id,
                HarnessEventType.APPROVAL,
                {"proposal": proposal.to_dict(include_arguments=True)},
            )
            return HarnessResult(
                run_id=task_value.run_id,
                harness=task_value.harness,
                status=HarnessStatus.PENDING_APPROVAL,
                checkpoint={"proposal_id": proposal.proposal_id},
                routing=routing,
                error_code="approval_required",
                error="Host approval is required before this harness run can start.",
            )

        self._create_run(task_value, status=HarnessStatus.QUEUED, routing=routing)
        return self._execute(
            task_value, workspace=workspace, before=before, routing=routing
        )

    def start(self, task: HarnessTask | Dict[str, Any]) -> HarnessRun:
        resolved_input = (
            task if isinstance(task, HarnessTask) else HarnessTask.from_dict(task)
        )
        try:
            task_value, workspace, before, routing = self._prepare(resolved_input)
        except Exception as exc:
            error_code, error, remediation, routing = self._preparation_failure(exc)
            failed = HarnessResult(
                run_id=resolved_input.run_id,
                harness=resolved_input.harness,
                status=HarnessStatus.FAILED,
                routing=routing,
                error_code=error_code,
                error=error,
                remediation=remediation,
            )
            run = self._create_run(
                resolved_input, status=HarnessStatus.FAILED, routing=routing
            )
            return self.run_store.update(
                run.run_id, finished_at=utcnow_iso(), result=failed.to_dict()
            )
        if self._approval_required(task_value):
            proposal = self._proposal(task_value, before=before, routing=routing)
            run = self._create_run(
                task_value,
                status=HarnessStatus.PENDING_APPROVAL,
                routing=routing,
                approval_proposal_id=proposal.proposal_id,
            )
            self._emit(
                task_value.run_id,
                HarnessEventType.APPROVAL,
                {"proposal": proposal.to_dict(include_arguments=True)},
            )
            return run
        run = self._create_run(task_value, status=HarnessStatus.QUEUED, routing=routing)
        thread = threading.Thread(
            target=self._execute,
            kwargs={
                "task": task_value,
                "workspace": workspace,
                "before": before,
                "routing": routing,
            },
            name=f"memorizz-harness-{task_value.run_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._threads[task_value.run_id] = thread
        thread.start()
        return run

    async def arun(self, task: HarnessTask | Dict[str, Any]) -> HarnessResult:
        """Asynchronously execute a task without blocking the event loop."""
        return await asyncio.to_thread(self.run, task)

    async def astart(self, task: HarnessTask | Dict[str, Any]) -> HarnessRun:
        return await asyncio.to_thread(self.start, task)

    def approve(
        self, proposal_id: str, *, approver_id: str, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        proposal = self.approval_store.approve(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        )
        return proposal.to_dict(include_arguments=True)

    def list_approvals(
        self,
        *,
        status: Optional[str] = None,
        owner_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List only meta-harness execution proposals from the shared store."""
        return [
            proposal.to_dict(include_arguments=True)
            for proposal in self.approval_store.list(
                owner_id=owner_id, status=status, limit=limit
            )
            if proposal.tool_name == "metaharness.run"
        ]

    def reject(
        self, proposal_id: str, *, approver_id: str, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        proposal = self.approval_store.reject(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        )
        checkpoint = dict(proposal.checkpoint or {})
        task = checkpoint.get("task") or {}
        run_id = task.get("run_id")
        if run_id and self.run_store.get(run_id):
            result = HarnessResult(
                run_id=run_id,
                harness=str(task.get("harness") or "auto"),
                status=HarnessStatus.CANCELED,
                error_code="approval_rejected",
                error="The host rejected this harness run.",
            )
            self.run_store.update(
                run_id,
                status=HarnessStatus.CANCELED.value,
                finished_at=utcnow_iso(),
                result=result.to_dict(),
            )
            self._emit(
                run_id,
                HarnessEventType.COMPLETE,
                {"status": result.status.value, "error_code": result.error_code},
            )
        return proposal.to_dict(include_arguments=True)

    def _consume_approved_checkpoint(
        self, proposal_id: str
    ) -> tuple[
        Optional[HarnessTask],
        Optional[Path],
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        Optional[HarnessResult],
    ]:
        proposal = self.approval_store.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown approval proposal: {proposal_id}")
        if proposal.status != ApprovalStatus.APPROVED:
            raise ApprovalStateError(
                "Harness proposal must be approved and unconsumed before it can resume"
            )
        checkpoint = dict(proposal.checkpoint or {})
        task = HarnessTask.from_dict(checkpoint.get("task") or {})
        if self.allowed_workspace_roots:
            resolve_workspace(task.workspace, self.allowed_workspace_roots)
        workspace = resolve_workspace(task.workspace, task.permissions.allowed_roots)
        current = workspace_snapshot(
            workspace, include_untracked_contents=task.writes_workspace
        )
        expected = str(
            (checkpoint.get("workspace_snapshot") or {}).get("fingerprint") or ""
        )
        if expected and current.get("fingerprint") != expected:
            result = HarnessResult(
                run_id=task.run_id,
                harness=task.harness,
                status=HarnessStatus.FAILED,
                error_code="workspace_changed_after_approval",
                error=(
                    "The workspace changed after approval. Create a new proposal so "
                    "the approver can review the current execution envelope."
                ),
            )
            self.run_store.update(
                task.run_id,
                status=result.status.value,
                finished_at=utcnow_iso(),
                result=result.to_dict(),
            )
            self._emit(
                task.run_id,
                HarnessEventType.COMPLETE,
                {"status": result.status.value, "error_code": result.error_code},
            )
            return None, None, None, None, result
        expected_arguments = task.approval_arguments(workspace_fingerprint=expected)
        self.approval_store.consume(
            proposal_id,
            expected_tool_name="metaharness.run",
            expected_arguments=expected_arguments,
        )
        persisted = self.run_store.get(task.run_id)
        if persisted is not None and (
            persisted.cancel_requested or persisted.status == HarnessStatus.CANCELED
        ):
            result = HarnessResult(
                run_id=task.run_id,
                harness=task.harness,
                status=HarnessStatus.CANCELED,
                error_code="canceled_before_resume",
                error="The approved harness run was canceled before execution.",
            )
            self.run_store.update(
                task.run_id,
                status=result.status.value,
                finished_at=utcnow_iso(),
                result=result.to_dict(),
            )
            self._emit(
                task.run_id,
                HarnessEventType.COMPLETE,
                {"status": result.status.value, "error_code": result.error_code},
            )
            return None, None, None, None, result
        self.run_store.update(task.run_id, status=HarnessStatus.QUEUED.value)
        return (
            task,
            workspace,
            current,
            dict(checkpoint.get("routing") or {}),
            None,
        )

    def resume_approval(self, proposal_id: str) -> HarnessResult:
        """Consume and synchronously execute one approved exact checkpoint."""
        task, workspace, current, routing, failure = self._consume_approved_checkpoint(
            proposal_id
        )
        if failure is not None:
            return failure
        assert task is not None and workspace is not None and current is not None
        return self._execute(
            task,
            workspace=workspace,
            before=current,
            routing=dict(routing or {}),
        )

    def resume_approval_start(self, proposal_id: str) -> HarnessRun:
        """Consume an approval and resume its checkpoint in a background worker."""
        task, workspace, current, routing, failure = self._consume_approved_checkpoint(
            proposal_id
        )
        if failure is not None:
            run = self.run_store.get(failure.run_id)
            if run is None:  # pragma: no cover - store invariant
                raise RuntimeError("Failed harness approval has no durable run")
            return run
        assert task is not None and workspace is not None and current is not None
        thread = threading.Thread(
            target=self._execute,
            kwargs={
                "task": task,
                "workspace": workspace,
                "before": current,
                "routing": dict(routing or {}),
            },
            name=f"memorizz-harness-{task.run_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._threads[task.run_id] = thread
        thread.start()
        run = self.run_store.get(task.run_id)
        if run is None:  # pragma: no cover - store invariant
            raise RuntimeError("Approved harness run disappeared before execution")
        return run

    def _emit(
        self,
        run_id: str,
        event_type: HarnessEventType | str,
        data: Dict[str, Any],
    ) -> HarnessEvent:
        event = HarnessEvent(run_id, event_type, redact(data))
        return self.run_store.append_event(event)

    def _configure_mcp(self, task: HarnessTask, temporary: Path) -> None:
        if task.permissions.mcp_access == "none":
            return
        writes = task.permissions.mcp_access == "governed_write"
        server_env = {
            "MEMORIZZ_MCP_SERVER_TRANSPORT": "stdio",
            "MEMORIZZ_MCP_SERVER_ALLOW_WRITES": "true" if writes else "false",
            "MEMORIZZ_MCP_SERVER_ALLOW_AGENT_EXECUTION": "false",
            "MEMORIZZ_MCP_SERVER_LOCAL_PRINCIPAL": str(task.user_id or ""),
        }
        if task.agent_id:
            server_env["MEMORIZZ_MCP_SERVER_AGENT_IDS"] = task.agent_id
        config = {
            "mcpServers": {
                "memorizz": {
                    "command": sys.executable,
                    "args": ["-m", "memorizz.mcp_server"],
                    "env": server_env,
                }
            }
        }
        target = temporary / "mcp.json"
        target.write_text(json.dumps(config, indent=2), encoding="utf-8")
        task.metadata["mcp_config_path"] = str(target)
        env_toml = ",".join(
            f"{json.dumps(key)}={json.dumps(value)}"
            for key, value in server_env.items()
        )
        task.metadata["_memorizz_codex_mcp_config"] = [
            f"mcp_servers.memorizz.command={json.dumps(sys.executable)}",
            "mcp_servers.memorizz.args=" + json.dumps(["-m", "memorizz.mcp_server"]),
            "mcp_servers.memorizz.env={" + env_toml + "}",
        ]
        if task.harness == "openhands":
            self._emit(
                task.run_id,
                HarnessEventType.STATUS,
                {
                    "mcp": "context_pack_fallback",
                    "reason": (
                        "OpenHands reads MCP configuration from its user config; the "
                        "run remains isolated and receives the same bounded memory pack."
                    ),
                },
            )

    @staticmethod
    def _configure_output_schema(task: HarnessTask, temporary: Path) -> None:
        """Materialize a validated schema for adapters that require a file."""
        if not task.output_schema or task.harness != "codex":
            return
        target = temporary / "output-schema.json"
        target.write_text(
            json.dumps(task.output_schema, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        task.metadata["_memorizz_output_schema_path"] = str(target)

    def _execute(
        self,
        task: HarnessTask,
        *,
        workspace: Path,
        before: Dict[str, Any],
        routing: Dict[str, Any],
    ) -> HarnessResult:
        adapter = self.adapters[task.harness]
        cancel_event = threading.Event()
        with self._lock:
            self._cancel_events[task.run_id] = cancel_event
        leased = False
        if task.writes_workspace:
            leased = self.run_store.acquire_workspace(str(workspace), task.run_id)
            if not leased:
                result = HarnessResult(
                    run_id=task.run_id,
                    harness=task.harness,
                    status=HarnessStatus.FAILED,
                    routing=routing,
                    error_code="workspace_busy",
                    error="Another write-capable harness run owns this workspace.",
                )
                self.run_store.update(
                    task.run_id,
                    status=result.status.value,
                    finished_at=utcnow_iso(),
                    result=result.to_dict(),
                )
                with self._lock:
                    self._cancel_events.pop(task.run_id, None)
                    self._threads.pop(task.run_id, None)
                return result
        started = time.monotonic()
        started_at = utcnow_iso()
        self.run_store.update(
            task.run_id,
            status=HarnessStatus.RUNNING.value,
            started_at=started_at,
            heartbeat_at=started_at,
        )
        self._emit(task.run_id, HarnessEventType.STATUS, {"status": "running"})
        monitor_stop = threading.Event()

        def monitor_control_state() -> None:
            """Bridge durable cancel requests into the live adapter process."""
            last_heartbeat = 0.0
            while not monitor_stop.wait(0.25):
                try:
                    persisted = self.run_store.get(task.run_id)
                    if persisted is None:
                        return
                    if persisted.cancel_requested:
                        cancel_event.set()
                        return
                    now = time.monotonic()
                    if now - last_heartbeat >= 2.0:
                        self.run_store.update(task.run_id, heartbeat_at=utcnow_iso())
                        last_heartbeat = now
                except Exception:
                    # The adapter's own wall-time limit remains the final safety
                    # boundary if the durable store becomes temporarily unavailable.
                    return

        persisted = self.run_store.get(task.run_id)
        if persisted is not None and persisted.cancel_requested:
            cancel_event.set()
        monitor = threading.Thread(
            target=monitor_control_state,
            name=f"memorizz-harness-control-{task.run_id[:8]}",
            daemon=True,
        )
        monitor.start()
        context_pack = None
        phase_timings_ms: Dict[str, int] = {}
        try:
            phase_started = time.monotonic()
            context_pack = self._context_for_task(task)
            phase_timings_ms["context"] = round(
                (time.monotonic() - phase_started) * 1_000
            )
            self._emit(
                task.run_id,
                HarnessEventType.STATUS,
                {
                    "memory_context": {
                        "source_ids": context_pack.source_ids,
                        "token_estimate": context_pack.token_estimate,
                        "truncated": context_pack.truncated,
                        "fingerprint": context_pack.fingerprint,
                        "content_fingerprint": context_pack.content_fingerprint,
                        "metadata": context_pack.metadata,
                    }
                },
            )
            with tempfile.TemporaryDirectory(prefix="memorizz-harness-") as temp_dir:
                phase_started = time.monotonic()
                temporary = Path(temp_dir)
                self._configure_mcp(task, temporary)
                phase_timings_ms["mcp_setup"] = round(
                    (time.monotonic() - phase_started) * 1_000
                )
                phase_started = time.monotonic()
                self._configure_output_schema(task, temporary)
                phase_timings_ms["output_schema_setup"] = round(
                    (time.monotonic() - phase_started) * 1_000
                )
                phase_started = time.monotonic()
                outcome = adapter.run(
                    task,
                    workspace=workspace,
                    context_pack=context_pack,
                    emit=lambda event: self._emit(event.run_id, event.type, event.data),
                    cancel_event=cancel_event,
                )
                phase_timings_ms["adapter"] = round(
                    (time.monotonic() - phase_started) * 1_000
                )
            self._apply_reported_budget(task, outcome)
            phase_started = time.monotonic()
            if outcome.error or cancel_event.is_set():
                verification = {
                    "required": bool(task.verification.required),
                    "verified": False,
                    "skipped": True,
                    "reason": (
                        "run_canceled" if cancel_event.is_set() else "adapter_failed"
                    ),
                }
                if task.verification.required:
                    self._emit(
                        task.run_id,
                        HarnessEventType.VERIFICATION,
                        verification,
                    )
            else:
                verification = self._verify(task, workspace, cancel_event=cancel_event)
            phase_timings_ms["verification"] = round(
                (time.monotonic() - phase_started) * 1_000
            )
            phase_started = time.monotonic()
            after = workspace_snapshot(
                workspace, include_untracked_contents=task.writes_workspace
            )
            diff = workspace_diff(workspace) if task.writes_workspace else None
            phase_timings_ms["workspace_finalize"] = round(
                (time.monotonic() - phase_started) * 1_000
            )
            if outcome.error_code == "canceled" or cancel_event.is_set():
                status = HarnessStatus.CANCELED
            elif outcome.error_code and outcome.error_code.endswith("budget_exceeded"):
                status = HarnessStatus.BUDGET_EXCEEDED
            elif outcome.error:
                status = HarnessStatus.FAILED
            elif verification.get("required") and not verification.get("verified"):
                status = HarnessStatus.VERIFICATION_FAILED
            else:
                status = HarnessStatus.SUCCEEDED
            total_latency_ms = round((time.monotonic() - started) * 1_000)
            phase_timings_ms["total"] = total_latency_ms
            result = HarnessResult(
                run_id=task.run_id,
                harness=task.harness,
                status=status,
                final_response=outcome.final_response,
                verified=bool(verification.get("verified")),
                verification=verification,
                usage=outcome.usage,
                cost_usd=outcome.cost_usd,
                latency_ms=total_latency_ms,
                phase_timings_ms=phase_timings_ms,
                workspace_diff=diff,
                workspace_fingerprint_before=before.get("fingerprint"),
                workspace_fingerprint_after=after.get("fingerprint"),
                checkpoint=outcome.checkpoint,
                context_pack=context_pack,
                routing=routing,
                error_code=outcome.error_code,
                error=outcome.error,
                remediation=outcome.remediation,
            )
        except Exception as exc:
            total_latency_ms = round((time.monotonic() - started) * 1_000)
            phase_timings_ms["total"] = total_latency_ms
            result = HarnessResult(
                run_id=task.run_id,
                harness=task.harness,
                status=HarnessStatus.FAILED,
                latency_ms=total_latency_ms,
                phase_timings_ms=phase_timings_ms,
                context_pack=context_pack,
                routing=routing,
                error_code=type(exc).__name__,
                error=str(exc),
            )
        finally:
            monitor_stop.set()
            monitor.join(timeout=1.0)
            if leased:
                self.run_store.release_workspace(str(workspace), task.run_id)
            with self._lock:
                self._cancel_events.pop(task.run_id, None)
                self._threads.pop(task.run_id, None)

        self.run_store.update(
            task.run_id,
            status=result.status.value,
            finished_at=utcnow_iso(),
            heartbeat_at=utcnow_iso(),
            result=redact(result.to_dict()),
        )
        self._emit(
            task.run_id,
            HarnessEventType.COMPLETE,
            {
                "status": result.status.value,
                "ok": result.ok,
                "verified": result.verified,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "error_code": result.error_code,
            },
        )
        self._record_memory_evidence(task, result)
        return result

    def _context_for_task(self, task: HarnessTask):
        """Build or reuse one scoped context snapshot for a workflow.

        A panel often gives several harnesses the same memory scope while their
        role-specific task wording differs. ``memory_query`` keeps retrieval
        aligned to the original request, and ``shared_context_key`` lets those
        stages reuse the exact same bounded snapshot without repeating provider
        reads or observing different memory mid-workflow.
        """
        memory_query = str(task.context.get("memory_query") or task.task)
        strategy = (
            str(task.context.get("memory_context_strategy") or "retrieval")
            .strip()
            .lower()
        )
        if strategy not in {"retrieval", "evidence_pack"}:
            raise ValueError(
                "memory_context_strategy must be retrieval or evidence_pack"
            )
        shared_key = str(task.context.get("shared_context_key") or "").strip()
        context_agent_id = self._context_agent_id(task)
        cache_key = None
        if shared_key:
            cache_key = canonical_hash(
                {
                    "shared_context_key": shared_key,
                    "query": memory_query,
                    "strategy": strategy,
                    "memory_id": task.memory_id,
                    "user_id": task.user_id,
                    "thread_id": task.thread_id,
                    "context_agent_id": context_agent_id,
                }
            )
            with self._context_cache_lock:
                cached = self._context_pack_cache.get(cache_key)
                if cached is not None:
                    self._context_pack_cache.move_to_end(cache_key)
                    return replace(
                        cached,
                        metadata={
                            **dict(cached.metadata or {}),
                            "shared_context_reused": True,
                            "context_cache_hit": True,
                            "context_access_latency_ms": 0,
                        },
                    )
                # Build while holding the dedicated cache lock. This prevents
                # parallel delegates from issuing duplicate provider queries
                # without blocking run cancellation or worker bookkeeping.
                context_started = time.monotonic()
                pack = self._build_context_pack(task, memory_query, strategy=strategy)
                context_latency_ms = round((time.monotonic() - context_started) * 1_000)
                pack = replace(
                    pack,
                    metadata={
                        **dict(pack.metadata or {}),
                        "context_cache_hit": False,
                        "context_access_latency_ms": context_latency_ms,
                        "initial_context_build_latency_ms": context_latency_ms,
                    },
                )
                self._context_pack_cache[cache_key] = pack
                while len(self._context_pack_cache) > self._context_cache_entries:
                    self._context_pack_cache.popitem(last=False)
                return pack

        context_started = time.monotonic()
        pack = self._build_context_pack(task, memory_query, strategy=strategy)
        context_latency_ms = round((time.monotonic() - context_started) * 1_000)
        return replace(
            pack,
            metadata={
                **dict(pack.metadata or {}),
                "context_cache_hit": False,
                "context_access_latency_ms": context_latency_ms,
                "initial_context_build_latency_ms": context_latency_ms,
            },
        )

    @staticmethod
    def _context_agent_id(task: HarnessTask) -> str:
        return str(
            task.context.get("memory_context_agent_id")
            or task.agent_id
            or "metaharness"
        )

    def _build_context_pack(
        self, task: HarnessTask, memory_query: str, *, strategy: str
    ):
        if strategy == "evidence_pack" and self.memory_provider is not None:
            plane = self._learning_plane(task)
            if plane is not None and getattr(
                getattr(plane, "config", None), "retrieval_enabled", True
            ):
                evidence = plane.retrieve_evidence(
                    memory_query,
                    memory_id=task.memory_id,
                    user_id=task.user_id,
                    thread_id=task.thread_id,
                    run_id=task.run_id,
                    trace_id=str(task.context.get("trace_id") or "") or None,
                )
                from .models import HarnessContextPack

                records = [item.to_dict() for item in evidence.items]
                retrieval_modes = sorted(
                    {
                        str(item.metadata.get("retrieval_mode"))
                        for item in evidence.items
                        if item.metadata.get("retrieval_mode")
                    }
                )
                retrieval_reasons = sorted(
                    {
                        str(item.metadata.get("retrieval_reason"))
                        for item in evidence.items
                        if item.metadata.get("retrieval_reason")
                    }
                )
                retrieval_degraded = any(
                    bool(item.metadata.get("retrieval_degraded"))
                    for item in evidence.items
                )
                return HarnessContextPack(
                    query=memory_query,
                    rendered=evidence.render(),
                    records=records,
                    source_ids=[item.source_id for item in evidence.items],
                    token_estimate=evidence.tokens_used,
                    truncated=bool(evidence.rejected_count),
                    metadata={
                        "strategy": "evidence_pack",
                        "pack_id": evidence.pack_id,
                        "candidate_count": evidence.candidate_count,
                        "selected_count": len(evidence.items),
                        "rejected_count": evidence.rejected_count,
                        "candidate_tokens": evidence.candidate_tokens,
                        "tokens_saved": max(
                            0, evidence.candidate_tokens - evidence.tokens_used
                        ),
                        "retrieval_modes": retrieval_modes,
                        "retrieval_reasons": retrieval_reasons,
                        "retrieval_degraded": retrieval_degraded,
                        "context_agent_id": self._context_agent_id(task),
                        "shared_context_reused": False,
                    },
                )
        pack = self.context_builder.build(
            memory_query,
            memory_id=task.memory_id,
            user_id=task.user_id,
            thread_id=task.thread_id,
        )
        pack.metadata.update(
            {
                "strategy": "retrieval",
                "context_agent_id": self._context_agent_id(task),
                "shared_context_reused": False,
            }
        )
        return pack

    @staticmethod
    def _stop_verification_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass

    def _verify(
        self,
        task: HarnessTask,
        workspace: Path,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        spec = task.verification
        if not spec.command:
            return {"required": bool(spec.required), "verified": False}
        cwd = workspace
        if spec.cwd:
            candidate = (workspace / spec.cwd).resolve()
            if candidate != workspace and workspace not in candidate.parents:
                raise HarnessSecurityError("Verification cwd escapes the workspace")
            cwd = candidate
        started = time.monotonic()
        process: Optional[subprocess.Popen[str]] = None
        try:
            command = (
                ["cmd.exe", "/d", "/s", "/c", spec.command]
                if os.name == "nt"
                else ["/bin/sh", "-lc", spec.command]
            )
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=build_child_environment(allowed_names=task.permissions.allowed_env),
                start_new_session=os.name != "nt",
            )
            deadline = time.monotonic() + spec.timeout_seconds
            canceled = False
            timed_out = False
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if cancel_event is not None and cancel_event.is_set():
                        canceled = True
                    elif time.monotonic() >= deadline:
                        timed_out = True
                    else:
                        continue
                    self._stop_verification_process(process)
                    try:
                        stdout, stderr = process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        stdout, stderr = "", ""
                    break
            return_code = process.returncode
            if canceled:
                return_code = 130
            elif timed_out:
                return_code = 124
            evidence = {
                "required": True,
                "verified": return_code == 0,
                "command": spec.command,
                "cwd": str(cwd),
                "return_code": return_code,
                "stdout": redact((stdout or "")[-20_000:]),
                "stderr": redact((stderr or "")[-20_000:]),
                "duration_ms": round((time.monotonic() - started) * 1000),
            }
            if canceled:
                evidence["error"] = "verification_canceled"
            elif timed_out:
                evidence["error"] = "verification_timeout"
        except OSError as exc:
            evidence = {
                "required": True,
                "verified": False,
                "command": spec.command,
                "cwd": str(cwd),
                "return_code": 127,
                "stdout": "",
                "stderr": redact(str(exc)[-20_000:]),
                "duration_ms": round((time.monotonic() - started) * 1000),
                "error": "verification_start_failed",
            }
        self._emit(task.run_id, HarnessEventType.VERIFICATION, evidence)
        return evidence

    @staticmethod
    def _apply_reported_budget(task: HarnessTask, outcome: Any) -> None:
        """Apply provider-reported limits to every adapter, including in-process ones."""
        usage = dict(outcome.usage or {})
        checks = (
            (
                "input_token_budget_exceeded",
                usage.get("input_tokens"),
                task.budget.max_input_tokens,
            ),
            (
                "output_token_budget_exceeded",
                usage.get("output_tokens"),
                task.budget.max_output_tokens,
            ),
        )
        for code, actual, limit in checks:
            if limit is not None and actual is not None and int(actual) > int(limit):
                outcome.error_code = code
                outcome.error = f"Harness {code.replace('_', ' ')} ({actual} > {limit})"
                return
        if (
            task.budget.max_cost_usd is not None
            and outcome.cost_usd is not None
            and float(outcome.cost_usd) > float(task.budget.max_cost_usd)
        ):
            outcome.error_code = "cost_budget_exceeded"
            outcome.error = (
                "Harness cost budget exceeded "
                f"({outcome.cost_usd} > {task.budget.max_cost_usd})"
            )

    def _record_memory_evidence(self, task: HarnessTask, result: HarnessResult) -> None:
        if self.memory_provider is None:
            return
        events = [
            event.to_dict()
            for event in self.run_store.events(task.run_id, limit=10_000)
        ]
        context = {
            "root_trace_id": task.run_id,
            "run_id": task.run_id,
            "agent_id": task.agent_id or "metaharness",
            "memory_id": task.memory_id,
            "thread_id": task.thread_id,
            "user_id": task.user_id,
        }
        metrics = {
            "harness": result.harness,
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
            "status": result.status.value,
            "context_tokens": (
                result.context_pack.token_estimate if result.context_pack else 0
            ),
            "input_tokens": result.usage.get("input_tokens"),
            "output_tokens": result.usage.get("output_tokens"),
        }
        try:
            store = ObservabilityStore(self.memory_provider)
            if events:
                store.record_trace_bundle(trace_context=context, events=events)

            plane = self._learning_plane(task)
            if plane is not None:
                from ..learning import OutcomeEvidence, OutcomeStatus

                if result.ok:
                    outcome_status = OutcomeStatus.SUCCESS
                elif result.final_response or result.workspace_diff:
                    outcome_status = OutcomeStatus.PARTIAL
                elif result.status == HarnessStatus.CANCELED:
                    outcome_status = OutcomeStatus.UNKNOWN
                else:
                    outcome_status = OutcomeStatus.FAILURE
                plane.begin_run(task.task, scope=context)
                plane.complete_run(
                    result.final_response,
                    status=result.status.value,
                    tool_call_count=sum(
                        1 for event in events if event.get("type") == "tool_call"
                    ),
                    metrics=metrics,
                    scope=context,
                )
                plane.record_workflow(
                    workflow_id=task.run_id,
                    outcome=outcome_status.value,
                    canonical_hash=canonical_hash(
                        {
                            "task": task.task,
                            "harness": task.harness,
                            "mode": task.mode,
                        }
                    ),
                    step_count=count_harness_steps(events),
                    skills_activated=task.context.get("skills_activated") or [],
                    scope=context,
                )
                plane.record_outcome(
                    OutcomeEvidence.from_value(
                        outcome_status,
                        verified=result.verified,
                        source="metaharness_verification",
                        score=1.0 if result.ok else 0.0,
                        metrics=metrics,
                        evidence_refs=[task.run_id],
                        recorded_by="metaharness",
                    ),
                    scope=context,
                    external_id=task.run_id,
                )
            else:
                store.record_outcome(
                    trace_context=context,
                    status="success" if result.ok else "failure",
                    verified=result.verified,
                    source="metaharness_verification",
                    score=1.0 if result.ok else 0.0,
                    metrics=metrics,
                    external_id=task.run_id,
                )
        except Exception:
            # A completed external task must not be changed to failed merely
            # because optional observability persistence is unavailable.
            return

    def _learning_plane(self, task: HarnessTask) -> Any:
        if self.learning_control_plane is not None:
            return self.learning_control_plane
        agent_plane = getattr(self.agent, "learning_control_plane", None)
        agent_id = self._context_agent_id(task)
        if (
            agent_plane is not None
            and str(getattr(agent_plane, "agent_id", "")) == agent_id
        ):
            return agent_plane
        if self.memory_provider is None:
            return None
        with self._lock:
            existing = self._owned_learning_control_planes.get(agent_id)
            if existing is not None:
                return existing
            from ..learning import LearningControlPlane

            plane = LearningControlPlane(
                self.memory_provider,
                agent_id=agent_id,
                config={"enabled": True, "compile_async": False},
            )
            self._owned_learning_control_planes[agent_id] = plane
            return plane

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self.run_store.get(run_id)
        return run.to_dict() if run else None

    def list_runs(
        self, *, limit: int = 100, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [
            run.to_dict() for run in self.run_store.list(limit=limit, status=status)
        ]

    def events(
        self, run_id: str, *, after: int = 0, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        return [
            event.to_dict()
            for event in self.run_store.events(run_id, after=after, limit=limit)
        ]

    def stream(
        self, run_id: str, *, after: int = 0, poll_seconds: float = 0.1
    ) -> Generator[Dict[str, Any], None, None]:
        cursor = max(0, int(after))
        while True:
            events = self.run_store.events(run_id, after=cursor, limit=1000)
            for event in events:
                cursor = int(event.sequence or cursor)
                yield event.to_dict()
            run = self.run_store.get(run_id)
            if run is None or (run.status.terminal and not events):
                return
            time.sleep(max(0.02, min(float(poll_seconds), 2.0)))

    def cancel(self, run_id: str) -> Dict[str, Any]:
        run = self.run_store.get(run_id)
        if run is None:
            raise KeyError(f"Unknown harness run: {run_id}")
        if run.status.terminal:
            return {
                "ok": False,
                "run_id": run_id,
                "status": run.status.value,
                "reason": "already_terminal",
            }
        # Persist first so a request from another CLI/UI/MCP process is visible
        # to the worker's control monitor.
        run = self.run_store.update(run_id, cancel_requested=True)
        if run.status == HarnessStatus.PENDING_APPROVAL:
            if run.approval_proposal_id:
                proposal = self.approval_store.get(run.approval_proposal_id)
                if proposal is not None and proposal.status == ApprovalStatus.PENDING:
                    self.approval_store.reject(
                        proposal.proposal_id,
                        approver_id="metaharness:cancel",
                        decision_reason="Harness run canceled by host",
                    )
            result = HarnessResult(
                run_id=run_id,
                harness=run.harness or "auto",
                status=HarnessStatus.CANCELED,
                error_code="canceled_before_approval",
                error="Harness run was canceled before approval.",
            )
            self.run_store.update(
                run_id,
                status=result.status.value,
                finished_at=utcnow_iso(),
                result=result.to_dict(),
            )
            self._emit(
                run_id,
                HarnessEventType.COMPLETE,
                {"status": result.status.value, "error_code": result.error_code},
            )
            return {
                "ok": True,
                "run_id": run_id,
                "status": result.status.value,
                "process_signaled": False,
            }
        with self._lock:
            event = self._cancel_events.get(run_id)
        if event is not None:
            event.set()
        adapter = self.adapters.get(run.harness or "")
        stopped = bool(adapter and adapter.cancel(run_id))
        self._emit(run_id, HarnessEventType.STATUS, {"status": "canceling"})
        return {
            "ok": True,
            "run_id": run_id,
            "status": "canceling",
            "process_signaled": stopped,
        }

    def retry(self, run_id: str) -> HarnessResult:
        run = self.run_store.get(run_id)
        if run is None:
            raise KeyError(f"Unknown harness run: {run_id}")
        if not run.status.terminal:
            raise ValueError("Only terminal harness runs may be retried")
        payload = dict(run.task)
        payload.pop("run_id", None)
        task = HarnessTask.from_dict(payload)
        return self.run(task)

    def run_plan(
        self, task: HarnessTask | Dict[str, Any], plan: HarnessPlan
    ) -> List[HarnessResult]:
        """Run a bounded serial planner/implementer/reviewer composition."""
        base = task if isinstance(task, HarnessTask) else HarnessTask.from_dict(task)
        results: List[HarnessResult] = []
        prior = ""
        for stage in plan.stages:
            payload = base.to_dict()
            payload.pop("run_id", None)
            payload["harness"] = stage.harness
            payload["mode"] = "stage"
            payload["task"] = stage.instruction or base.task
            payload["permissions"] = {
                **base.permissions.to_dict(),
                "workspace_mode": stage.workspace_mode,
            }
            if stage.verification is not None:
                payload["verification"] = stage.verification.to_dict()
            payload["context"] = {
                **base.context,
                "harness_stage": stage.name,
                "prior_stage_results": prior[-20_000:],
            }
            result = self.run(HarnessTask.from_dict(payload))
            results.append(result)
            prior += f"\n[{stage.name}:{result.status.value}]\n{result.final_response}"
            if result.status != HarnessStatus.SUCCEEDED:
                break
        return results

    def compare(
        self,
        task: HarnessTask | Dict[str, Any],
        harnesses: Iterable[str],
    ) -> Dict[str, Any]:
        """Run an explicit comparison; callers must provide isolated workspaces for writes."""
        base = task if isinstance(task, HarnessTask) else HarnessTask.from_dict(task)
        if base.writes_workspace and not base.metadata.get("comparison_isolated"):
            raise ValueError(
                "Write-capable comparisons require comparison_isolated=True and a "
                "caller-managed disposable workspace for each run"
            )
        results = []
        for harness in harnesses:
            payload = base.to_dict()
            payload.pop("run_id", None)
            payload["harness"] = harness
            results.append(self.run(HarnessTask.from_dict(payload)).to_dict())
        successful = [row for row in results if row.get("ok")]
        return {
            "task_fingerprint": canonical_hash(base.task),
            "results": results,
            "count": len(results),
            "verified_successes": len(successful),
        }

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            active = list(self._cancel_events.items())
            threads = list(self._threads.values())
        for run_id, event in active:
            event.set()
            run = self.run_store.get(run_id)
            adapter = self.adapters.get(run.harness or "") if run else None
            if adapter:
                adapter.cancel(run_id)
        for thread in threads:
            thread.join(timeout=5.0)
        for plane in self._owned_learning_control_planes.values():
            try:
                plane.close()
            except Exception:
                pass
        self._owned_learning_control_planes.clear()
        with self._context_cache_lock:
            self._context_pack_cache.clear()
        self.run_store.close()
        self._closed = True

    def __enter__(self) -> "MetaHarness":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["MetaHarness"]

"""Memory-first control plane for native and external agent harnesses."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Mapping, Optional

from .._env_io import env_bool, memorizz_home
from ..approval import (
    ApprovalStateError,
    ApprovalStatus,
    ApprovalStore,
    default_approval_store,
)
from ..observability import ObservabilityStore
from .adapters import (
    ClaudeCodeHarness,
    CodexHarness,
    DeepSeekHarness,
    HermesHarness,
    OpenHandsHarness,
    PiHarness,
)
from .base import AgentHarness
from .config import HARNESS_NAMES
from .context import HarnessContextBuilder
from .handoff import fit_handoffs, handoff_memory_record, stage_handoff
from .models import (
    HarnessEvent,
    HarnessEventType,
    HarnessOrchestration,
    HarnessPermissions,
    HarnessPlan,
    HarnessResult,
    HarnessRun,
    HarnessStage,
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

logger = logging.getLogger(__name__)

HARNESS_ALIASES = {
    "claude": "claude-code",
    "claudecode": "claude-code",
    "open-hands": "openhands",
    "native": "memagent",
    "memorizz": "memagent",
}
# Staged plans and comparisons: a bounded number of runs per workflow, a poll
# interval for the driver thread, and how long a workflow may go without a
# driver heartbeat before readers report it interrupted.
MAX_ORCHESTRATION_STEPS = 8
ORCHESTRATION_POLL_SECONDS = 0.25
ORCHESTRATION_HEARTBEAT_SECONDS = 2.0
ORCHESTRATION_STALE_SECONDS = 60.0
# Delegates that edit one workspace take turns (one lock per workspace).
_EDIT_TURNS: Dict[str, threading.Lock] = {}
_EDIT_TURNS_GUARD = threading.Lock()


def normalize_harness_name(name: Any) -> str:
    normalized = str(name or "").strip().lower().replace("_", "-")
    return HARNESS_ALIASES.get(normalized, normalized)


class _OrchestrationStopped(Exception):
    """The host is shutting down; the workflow driver must exit."""


def _parse_iso(value: Any) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


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
        scratch_root: Optional[str | Path] = None,
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
        # Empty folders for tasks with no project (see scratch_workspace()).
        self.scratch_root = Path(scratch_root).expanduser() if scratch_root else None
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
        # Workflow drivers: orchestration id -> (thread, stop event).
        self._orchestrations: Dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._lock = threading.RLock()
        self._closing = False
        self._closed = False
        self._judge = None
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
                        env_bool("MEMORIZZ_OPENHANDS_EXTERNAL_ISOLATION")
                        or bool(openhands.get("external_isolation", False))
                    ),
                )
            )
        deepseek = dict(adapter_config.get("deepseek") or {})
        if deepseek.get("enabled", True):
            adapters.append(
                DeepSeekHarness(
                    command=os.getenv(
                        "MEMORIZZ_DEEPSEEK_COMMAND",
                        deepseek.get("command") or "claude",
                    ),
                    default_model=os.getenv("MEMORIZZ_DEEPSEEK_MODEL")
                    or deepseek.get("model")
                    or None,
                )
            )
        pi = dict(adapter_config.get("pi") or {})
        if pi.get("enabled", True):
            adapters.append(
                PiHarness(
                    command=os.getenv("MEMORIZZ_PI_COMMAND", pi.get("command") or "pi"),
                    default_model=os.getenv("MEMORIZZ_PI_MODEL")
                    or pi.get("model")
                    or None,
                    provider=os.getenv("MEMORIZZ_PI_PROVIDER")
                    or pi.get("provider")
                    or None,
                    external_isolation=(
                        env_bool("MEMORIZZ_PI_EXTERNAL_ISOLATION")
                        or bool(pi.get("external_isolation", False))
                    ),
                )
            )
        hermes = dict(adapter_config.get("hermes") or {})
        if hermes.get("enabled", True):
            adapters.append(
                HermesHarness(
                    command=os.getenv(
                        "MEMORIZZ_HERMES_COMMAND", hermes.get("command") or "hermes"
                    ),
                    default_model=os.getenv("MEMORIZZ_HERMES_MODEL")
                    or hermes.get("model")
                    or None,
                    provider=os.getenv("MEMORIZZ_HERMES_PROVIDER")
                    or hermes.get("provider")
                    or None,
                    base_url=os.getenv("MEMORIZZ_HERMES_BASE_URL")
                    or hermes.get("base_url")
                    or None,
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
            preference=configured.get("preference") or tuple(HARNESS_NAMES),
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

    def _scratch_root(self) -> Path:
        return self.scratch_root or memorizz_home() / "harness-workspaces"

    def _workspace_roots(self) -> List[str]:
        """The operator's roots plus the managed scratch folder, once it exists."""
        roots = list(self.allowed_workspace_roots)
        scratch = self._scratch_root()
        if roots and scratch.is_dir():
            resolved = str(scratch.resolve())
            if resolved not in roots:
                roots.append(resolved)
        return roots

    def scratch_workspace(self) -> str:
        """Create an empty folder for a task that needs no project, such as a
        question answered from the web. It stays for inspection afterwards."""
        root = self._scratch_root()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = root / f"{stamp}-{uuid.uuid4().hex[:6]}"
        path.mkdir(mode=0o700)
        return str(path.resolve())

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
        normalized = normalize_harness_name(name)
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
        if task.metadata.get("judge") is not None:
            from .judging import judge_config

            task.metadata["judge"] = judge_config(task.metadata["judge"])
        if task.harness in HARNESS_ALIASES:
            payload = task.to_dict()
            payload["harness"] = HARNESS_ALIASES[task.harness]
            task = HarnessTask.from_dict(payload)
        roots = self._workspace_roots()
        if not task.permissions.allowed_roots and roots:
            payload = task.to_dict()
            payload["permissions"]["allowed_roots"] = roots
            task = HarnessTask.from_dict(payload)
        if roots:
            # A task can narrow the operator's root policy, never widen it.
            resolve_workspace(task.workspace, roots)
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

    def _approval_covered(self, task: HarnessTask, workspace: Path) -> bool:
        """A delegate's run is covered by an approval already given: the
        MemAgent harness run it is part of (still running, approved for at
        least this access in the same workspace) or a playground grant."""
        if task.permissions.require_approval or (
            task.verification.command and task.metadata.get("model_initiated")
        ):
            return False
        parent_id = task.metadata.get("parent_run_id")
        grant_id = task.metadata.get("access_grant_id")
        if parent_id:
            parent = self.run_store.get(str(parent_id))
            if parent is None or parent.status != HarnessStatus.RUNNING:
                return False
            granted = HarnessPermissions.from_value(
                dict(parent.task or {}).get("permissions") or {}
            )
            folder = dict(parent.task or {}).get("workspace")
        elif grant_id:
            grant = self.delegate_access(str(grant_id), agent_id=None)
            if grant is None:
                return False
            granted = HarnessPermissions.from_value(grant["permissions"])
            folder = grant["workspace"]
        else:
            return False
        try:
            same_place = Path(str(folder)).resolve() == Path(workspace).resolve()
        except (OSError, TypeError):
            return False
        wanted = task.permissions
        network_rank = {"none": 0, "restricted": 1, "full": 2}
        return bool(
            same_place
            and (not task.writes_workspace or granted.workspace_mode == "direct")
            and network_rank.get(wanted.network, 2)
            <= network_rank.get(granted.network, 0)
            and (
                wanted.mcp_access != "governed_write"
                or granted.mcp_access == "governed_write"
            )
            and not wanted.allowed_env
        )

    def grant_delegate_access(
        self,
        *,
        agent_id: str,
        approver_id: str,
        workspace: str = "",
        network: str = "none",
        write: bool = False,
        mcp_access: str = "read_only",
        allow_subagents: bool = False,
        ttl_seconds: int = 8 * 3600,
    ) -> Dict[str, Any]:
        """Let an agent's harness delegates use this access outside a harness
        run (in the playground, say) without asking for each delegate run.

        Recorded as an approved approval (``metaharness.delegate_access``)
        that expires; a blank workspace gets a fresh scratch folder.
        """
        folder = str(workspace or "").strip() or self.scratch_workspace()
        roots = self._workspace_roots()
        resolved = (
            resolve_workspace(folder, roots) if roots else resolve_workspace(folder)
        )
        permissions = HarnessPermissions(
            workspace_mode="direct" if write else "read_only",
            network=network,
            mcp_access=mcp_access,
            allow_subagents=bool(allow_subagents),
            require_approval=False,
        )
        ttl = max(60, min(int(ttl_seconds), 86_400))
        proposal = self.approval_store.propose(
            owner_id=str(agent_id),
            tool_name="metaharness.delegate_access",
            arguments={
                "agent_id": str(agent_id),
                "workspace": str(resolved),
                "permissions": permissions.to_dict(),
            },
            policy_reason=(
                "This agent's harness delegates may use this workspace and access "
                "for their runs until the grant expires."
            ),
            ttl_seconds=ttl,
        )
        self.approval_store.approve(
            proposal.proposal_id,
            approver_id=approver_id,
            decision_reason="Granted for the agent's harness delegates",
        )
        return self.delegate_access(proposal.proposal_id, agent_id=str(agent_id)) or {}

    def delegate_access(
        self, grant_id: str, *, agent_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """A grant from ``grant_delegate_access`` while it holds (approved and
        not expired, for this agent), as ``run_on_harness`` takes it."""
        proposal = self.approval_store.get(str(grant_id or ""))
        if (
            proposal is None
            or proposal.tool_name != "metaharness.delegate_access"
            or proposal.status != ApprovalStatus.APPROVED
            or proposal.expired
        ):
            return None
        arguments = dict(proposal.arguments or {})
        if agent_id is not None and arguments.get("agent_id") != str(agent_id):
            return None
        return {
            "grant_id": proposal.proposal_id,
            "agent_id": arguments.get("agent_id"),
            "workspace": arguments.get("workspace"),
            "permissions": dict(arguments.get("permissions") or {}),
            "approver_id": proposal.approver_id,
            "expires_at": proposal.expires_at.isoformat(),
        }

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

        if self._approval_required(task_value) and not self._approval_covered(
            task_value, workspace
        ):
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
        if self._approval_required(task_value) and not self._approval_covered(
            task_value, workspace
        ):
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
        self._register_worker(task_value.run_id, thread)
        thread.start()
        return run

    def _register_worker(self, run_id: str, thread: threading.Thread) -> None:
        # Register the cancel signal before the thread runs, so close() can
        # stop a run that has been started but has not reached the adapter.
        with self._lock:
            self._threads[run_id] = thread
            self._cancel_events.setdefault(run_id, threading.Event())

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
        roots = self._workspace_roots()
        if roots:
            resolve_workspace(task.workspace, roots)
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
        self._register_worker(task.run_id, thread)
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
        # Some hosts start MCP servers with a bare environment. These settings
        # are paths and names, never secrets, and keep the same memory store,
        # credentials and cached tool lists in reach. Connection strings stay
        # out of the config file.
        for key in ("MEMORIZZ_HOME", "MEMORIZZ_MEMORY_ROOT", "MEMORIZZ_BACKEND"):
            if os.environ.get(key):
                server_env[key] = os.environ[key]
        # A local UI connects its store at runtime, not through the
        # environment; name the same folder so the server sees its agents.
        root = getattr(getattr(self.memory_provider, "config", None), "root_path", None)
        if root and type(self.memory_provider).__name__ == "FileSystemProvider":
            server_env["MEMORIZZ_BACKEND"] = "filesystem"
            server_env["MEMORIZZ_MEMORY_ROOT"] = str(root)
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
        adapter = self.adapters.get(task.harness)
        if adapter is not None and not adapter.probe().mcp:
            self._emit(
                task.run_id,
                HarnessEventType.STATUS,
                {
                    "mcp": "context_pack_fallback",
                    "reason": (
                        f"{task.harness} has no MCP client; it reads MemoRizz memory "
                        "from the bounded memory pack in its prompt."
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

    def _lease_holder(self, workspace: Any) -> Optional[str]:
        holder = getattr(self.run_store, "workspace_holder", None)
        return holder(str(workspace)) if callable(holder) else None

    def _edit_turn(
        self, task: HarnessTask, workspace: Any, cancel_event: threading.Event
    ) -> Optional[threading.Lock]:
        """Delegates that edit (under a parent run or a playground grant) take
        turns in a workspace instead of failing as busy. Returns the held
        turn, or None for other runs or when canceled while waiting."""
        if not (
            task.metadata.get("parent_run_id") or task.metadata.get("access_grant_id")
        ):
            return None
        with _EDIT_TURNS_GUARD:
            turn = _EDIT_TURNS.setdefault(str(workspace), threading.Lock())
        waiting = False
        while not turn.acquire(timeout=0.25):
            if cancel_event.is_set():
                return None
            if not waiting:
                waiting = True
                self._emit(
                    task.run_id,
                    HarnessEventType.STATUS,
                    {"status": "waiting", "reason": "another delegate is editing"},
                )
        return turn

    def _execute(
        self,
        task: HarnessTask,
        *,
        workspace: Path,
        before: Dict[str, Any],
        routing: Dict[str, Any],
    ) -> HarnessResult:
        adapter = self.adapters[task.harness]
        with self._lock:
            cancel_event = self._cancel_events.setdefault(
                task.run_id, threading.Event()
            )
        if self._closing:
            cancel_event.set()
        leased = False
        edit_turn = None
        if task.writes_workspace:
            edit_turn = self._edit_turn(task, workspace, cancel_event)
            leased = self.run_store.acquire_workspace(str(workspace), task.run_id)
            # A delegate of a MemAgent run approved for edits works under
            # that run's lease.
            shared = not leased and self._lease_holder(workspace) == str(
                task.metadata.get("parent_run_id") or "-"
            )
            if not leased and not shared:
                if edit_turn is not None:
                    edit_turn.release()
                canceled = cancel_event.is_set()
                result = HarnessResult(
                    run_id=task.run_id,
                    harness=task.harness,
                    status=HarnessStatus.CANCELED if canceled else HarnessStatus.FAILED,
                    routing=routing,
                    error_code="canceled" if canceled else "workspace_busy",
                    error=(
                        "Harness run was canceled while waiting for the workspace."
                        if canceled
                        else "Another write-capable harness run owns this workspace."
                    ),
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
                # Private per-run scratch for adapters (removed after the run).
                task.metadata["_memorizz_run_dir"] = str(temporary)
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
            if edit_turn is not None:
                edit_turn.release()
            with self._lock:
                self._cancel_events.pop(task.run_id, None)

        try:
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
            self._remember_single_run(task)
            if task.metadata.get("judge") and result.ok and result.final_response:
                try:
                    self.judge_runs([task.run_id], task.metadata["judge"])
                except ValueError as exc:
                    if not self._closing:
                        try:
                            run = self.run_store.get(task.run_id)
                            if run is not None:
                                self._judge_service().record_failure(
                                    run.to_dict(), task.metadata["judge"], str(exc)
                                )
                        except Exception:
                            logger.exception(
                                "Could not record the optional judge error"
                            )
                except Exception:
                    logger.exception("Could not queue the optional answer judge")
        finally:
            # Stay visible to close() until the final writes are done, so it
            # never closes the store under a run that is still recording.
            with self._lock:
                self._threads.pop(task.run_id, None)
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
        return self._judged_runs([run.to_dict()])[0] if run else None

    def list_runs(
        self, *, limit: int = 100, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self._judged_runs(
            [run.to_dict() for run in self.run_store.list(limit=limit, status=status)]
        )

    def _judge_service(self):
        from .judging import HarnessJudge

        with self._lock:
            if self._closing:
                raise ValueError("The harness service is shutting down")
            if self._judge is None:
                if not callable(getattr(self.run_store, "create_judgment", None)):
                    raise ValueError("This run store does not support saved judgments")
                self._judge = HarnessJudge(self.run_store)
            return self._judge

    def _judged_runs(self, runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._closing and callable(
            getattr(self.run_store, "latest_judgments", None)
        ):
            return self._judge_service().annotate(runs)
        return runs

    def judge_settings(self, value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._judge_service().settings(value)

    def judge_runs(
        self, run_ids: List[str], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if (
            not isinstance(run_ids, list)
            or not 1 <= len(run_ids) <= 8
            or any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
        ):
            raise ValueError("Select between one and eight run IDs to judge")
        runs = []
        for run_id in dict.fromkeys(run_ids):
            run = self.run_store.get(run_id)
            if run is None:
                raise KeyError(f"Unknown harness run: {run_id}")
            runs.append(run.to_dict())
        return self._judge_service().start(runs, config)

    def get_judgment(self, judgment_id: str) -> Optional[Dict[str, Any]]:
        return self._judge_service().get(judgment_id)

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

    def cancel(self, run_id: str, *, before_start: bool = False) -> Dict[str, Any]:
        """Cancel a run. With ``before_start``, a run this process has not
        created yet (a delegate's, being prepared) starts already canceled
        instead of raising ``KeyError``."""
        run = self.run_store.get(run_id)
        if run is None and before_start:
            with self._lock:
                self._cancel_events.setdefault(run_id, threading.Event()).set()
            return {
                "ok": True,
                "run_id": run_id,
                "status": "canceling",
                "process_signaled": False,
            }
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

    def _retry_task(self, run_id: str) -> HarnessTask:
        run = self.run_store.get(run_id)
        if run is None:
            raise KeyError(f"Unknown harness run: {run_id}")
        if not run.status.terminal:
            raise ValueError("Only terminal harness runs may be retried")
        payload = dict(run.task)
        payload.pop("run_id", None)
        # Adapter scratch paths belong to the finished run.
        payload["metadata"] = {
            key: value
            for key, value in dict(payload.get("metadata") or {}).items()
            if not key.startswith("_memorizz_") and key != "mcp_config_path"
        }
        return HarnessTask.from_dict(payload)

    def retry(self, run_id: str) -> HarnessResult:
        """Run a finished run's exact task again and wait for the result."""
        return self.run(self._retry_task(run_id))

    def retry_start(self, run_id: str) -> HarnessRun:
        """Start a finished run's exact task again in the background; risky
        envelopes pause for approval like any new run."""
        return self.start(self._retry_task(run_id))

    @staticmethod
    def conversation_id_for(run: Mapping[str, Any]) -> str:
        """The conversation a run belongs to; a run on its own starts one."""
        metadata = dict((run.get("task") or {}).get("metadata") or {})
        return str(metadata.get("conversation_id") or f"hxc-{run.get('run_id')}")

    def conversation(
        self, conversation_id: str, *, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """A harness conversation's runs, oldest first. Its first run is the
        one it was started from (``hxc-<run id>``)."""
        root = conversation_id.removeprefix("hxc-")
        runs = [
            run
            for run in self.list_runs(limit=limit)
            if run.get("run_id") == root
            or dict((run.get("task") or {}).get("metadata") or {}).get(
                "conversation_id"
            )
            == conversation_id
        ]
        if root and not any(run.get("run_id") == root for run in runs):
            first = self.get_run(root)
            if first is not None:
                runs.append(first)
        return sorted(runs, key=lambda run: str(run.get("created_at") or ""))

    def continue_conversation(
        self, conversation_id: str, task: HarnessTask | Dict[str, Any]
    ) -> HarnessRun:
        """Start the next turn of a conversation in the background.

        Works with every harness: earlier turns (each request, answer, changed
        files and commands) reach the new run as bounded context, and the turn
        is recorded in the same conversation and thread.
        """
        base = task if isinstance(task, HarnessTask) else HarnessTask.from_dict(task)
        prior: List[Dict[str, Any]] = []
        for run in self.conversation(conversation_id):
            if not HarnessStatus(run.get("status")).terminal:
                continue
            events = self.events(run["run_id"], limit=10_000)
            handoff = stage_handoff(run, events)
            metadata = dict((run.get("task") or {}).get("metadata") or {})
            request = str(
                metadata.get("conversation_message")
                or (run.get("task") or {}).get("task")
                or ""
            )
            handoff["request"] = request[:2_000]
            prior.append(handoff)
        limit = int(base.context.get("model_context_max_chars") or 12_000)
        payload = base.to_dict()
        payload.pop("run_id", None)
        payload["thread_id"] = base.thread_id or conversation_id
        payload["metadata"] = {
            **base.metadata,
            "conversation_id": conversation_id,
            "conversation_turn": len(prior) + 1,
            "conversation_message": base.task,
        }
        payload["context"] = {
            **base.context,
            "conversation": fit_handoffs(prior, budget=max(1_000, min(limit, 50_000))),
        }
        return self.start(HarnessTask.from_dict(payload))

    def run_plan(
        self, task: HarnessTask | Dict[str, Any], plan: HarnessPlan
    ) -> List[HarnessResult]:
        """Run a bounded serial planner/implementer/reviewer composition."""
        base = task if isinstance(task, HarnessTask) else HarnessTask.from_dict(task)
        plan = HarnessPlan.from_value(plan)
        results: List[HarnessResult] = []
        handoffs: List[Dict[str, Any]] = []
        thread_id = base.thread_id or f"harness-plan-{uuid.uuid4()}"
        for index, stage in enumerate(plan.stages):
            stage_task = self._stage_task(
                base, stage, prior=self._fit_prior(base, handoffs)
            )
            result = self.run(stage_task)
            results.append(result)
            handoff = self._stage_handoff(result.run_id, stage.name)
            if handoff is not None:
                handoffs.append(handoff)
                self._remember_handoff(
                    stage_task,
                    handoff,
                    workflow={"kind": "plan", "step": index, "steps": len(plan.stages)},
                    thread_id=thread_id,
                )
            if result.status != HarnessStatus.SUCCEEDED:
                break
        return results

    def _stage_handoff(self, run_id: str, name: str) -> Optional[Dict[str, Any]]:
        run = self.run_store.get(run_id)
        if run is None:
            return None
        events = [
            event.to_dict() for event in self.run_store.events(run_id, limit=10_000)
        ]
        return stage_handoff(run.to_dict(), events, name=name)

    @staticmethod
    def _fit_prior(
        base: HarnessTask, handoffs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Leave room for the prompt's stage-context wrapper and other keys.
        limit = int(base.context.get("model_context_max_chars") or 12_000)
        return fit_handoffs(handoffs, budget=max(1_000, min(limit, 50_000) - 1_000))

    def _remember_single_run(self, task: HarnessTask) -> None:
        """Save a standalone run's answer to conversation memory, like a
        workflow step. Workflow steps save their own handoffs, and a MemAgent
        records the answer in its own conversation."""
        if not task.memory_id or task.metadata.get("orchestration"):
            return
        if task.metadata.get("origin_agent_id"):
            return
        try:
            handoff = self._stage_handoff(task.run_id, "")
        except Exception:
            logger.warning("Could not build a harness handoff", exc_info=True)
            return
        if handoff is not None:
            self._remember_handoff(
                task,
                handoff,
                workflow={"kind": "run"},
                thread_id=task.thread_id or f"harness-run-{task.run_id}",
            )

    def _remember_handoff(
        self,
        task: HarnessTask,
        handoff: Dict[str, Any],
        *,
        workflow: Dict[str, Any],
        thread_id: str,
    ) -> Optional[str]:
        """Write one handoff to conversation memory in the task's memory scope,
        so retrieval, compaction and summaries can use it like any turn."""
        if self.memory_provider is None or not task.memory_id:
            return None
        record = handoff_memory_record(
            handoff,
            workflow=workflow,
            memory_id=task.memory_id,
            thread_id=thread_id,
            agent_id=task.agent_id or "metaharness",
            user_id=task.user_id,
        )
        try:
            from ..enums.memory_type import MemoryType

            stored = self.memory_provider.store(
                record,
                memory_store_type=MemoryType.CONVERSATION_MEMORY,
                memory_id=task.memory_id,
            )
        except Exception:
            # Memory is evidence, not a precondition: a read-only or unavailable
            # provider must not fail a workflow whose harness work succeeded.
            logger.warning("Could not save harness handoff to memory", exc_info=True)
            return None
        return str(stored) if stored else None

    @staticmethod
    def _stage_permissions(
        base: HarnessPermissions, workspace_mode: str
    ) -> Dict[str, Any]:
        value = {**base.to_dict(), "workspace_mode": workspace_mode}
        derived = bool(
            base.workspace_mode == "direct"
            or base.network != "none"
            or base.allowed_tools
            or base.allowed_env
            or base.mcp_access == "governed_write"
        )
        if base.require_approval == derived:
            # Not set explicitly: derive it again for this stage's mode, so a
            # read-only stage does not inherit an edit stage's approval.
            value["require_approval"] = None
        return value

    def _stage_task(
        self,
        base: HarnessTask,
        stage: HarnessStage,
        *,
        prior: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HarnessTask:
        """One plan stage as a task: the stage's harness, mode and instruction,
        with earlier stages' results as bounded evidence."""
        payload = base.to_dict()
        payload.pop("run_id", None)
        payload["harness"] = stage.harness
        payload["mode"] = "stage"
        payload["task"] = stage.instruction or base.task
        payload["permissions"] = self._stage_permissions(
            base.permissions, stage.workspace_mode
        )
        if stage.verification is not None:
            payload["verification"] = stage.verification.to_dict()
        if stage.model:
            payload["model"] = stage.model
        if stage.agent_id:
            payload["agent_id"] = stage.agent_id
        payload["context"] = {
            **base.context,
            **(context or {}),
            "harness_stage": stage.name,
            "prior_stage_results": list(prior),
        }
        if metadata:
            payload["metadata"] = {**base.metadata, **metadata}
        return HarnessTask.from_dict(payload)

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

    # ------------------------------------------------------------ workflows

    def start_plan(
        self,
        task: HarnessTask | Dict[str, Any],
        plan: HarnessPlan | Dict[str, Any] | List[Any],
    ) -> Dict[str, Any]:
        """Start a staged plan in the background and return its durable record.

        Stages run in order, each on its own harness, with earlier stages'
        results as bounded evidence. Unlike ``run_plan``, a stage that needs
        approval waits for the host decision (approve and resume it through the
        usual approval API) instead of ending the plan. The plan stops at the
        first stage that does not succeed.
        """
        base = task if isinstance(task, HarnessTask) else HarnessTask.from_dict(task)
        plan = HarnessPlan.from_value(plan)
        if len(plan.stages) > MAX_ORCHESTRATION_STEPS:
            raise ValueError(
                f"A plan may have at most {MAX_ORCHESTRATION_STEPS} stages"
            )
        steps = []
        for stage in plan.stages:
            stage.harness = self._registered_harness(stage.harness, allow_auto=True)
            steps.append({**stage.to_dict(), "run_id": None})
        return self._start_orchestration("plan", base, steps)

    def _registered_harness(self, name: Any, *, allow_auto: bool) -> str:
        normalized = normalize_harness_name(name)
        if normalized == "auto" and allow_auto:
            return normalized
        if not normalized or normalized == "auto":
            raise ValueError("Name each harness to compare; auto is not allowed")
        if normalized not in self.adapters:
            raise ValueError(f"Unknown harness: {name}")
        return normalized

    def start_compare(
        self,
        task: HarnessTask | Dict[str, Any],
        harnesses: Iterable[str],
        models: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run one read-only task on several harnesses side by side.

        Every harness gets the same task, limits and memory snapshot, so the
        results differ only by harness. Runs start together; the comparison
        finishes when all of them have.
        """
        base = task if isinstance(task, HarnessTask) else HarnessTask.from_dict(task)
        if base.writes_workspace:
            raise ValueError(
                "Comparisons are read-only because every harness shares the "
                "workspace. Use a plan with one edit stage, or compare() with "
                "caller-managed isolated workspaces."
            )
        names: List[str] = []
        for name in harnesses or []:
            normalized = self._registered_harness(name, allow_auto=False)
            if normalized not in names:
                names.append(normalized)
        if len(names) < 2:
            raise ValueError("A comparison needs at least two different harnesses")
        if len(names) > MAX_ORCHESTRATION_STEPS:
            raise ValueError(
                f"A comparison may use at most {MAX_ORCHESTRATION_STEPS} harnesses"
            )
        # A model per harness: one model name rarely runs on every harness.
        chosen = {
            normalize_harness_name(name): str(model).strip()
            for name, model in dict(models or {}).items()
            if str(model or "").strip()
        }
        steps = [
            {"name": name, "harness": name, "run_id": None, "model": chosen.get(name)}
            for name in names
        ]
        return self._start_orchestration("compare", base, steps)

    def is_scratch_workspace(self, workspace: Any) -> bool:
        """Whether a workspace is one of the managed scratch folders."""
        if not str(workspace or "").strip():
            return False
        try:
            Path(str(workspace)).resolve().relative_to(self._scratch_root().resolve())
        except (OSError, ValueError):
            return False
        return True

    def rerun_orchestration(
        self, orchestration_id: str, *, agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start a plan or comparison again with the settings it ran with.

        A workflow that ran in a scratch folder gets a fresh one, as it did the
        first time. ``agent_id`` fills in the saved MemAgent when the original
        named none. The new record notes which workflow it repeats.
        """
        record = self._orchestration_store().get_orchestration(orchestration_id)
        if record is None:
            raise KeyError(f"Harness workflow not found: {orchestration_id}")
        value = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        kind = str(value.get("kind") or "")
        if kind not in {"plan", "compare"}:
            raise ValueError(f"A {kind or 'workflow'} cannot be run again")
        task = dict(value.get("task") or {})
        steps = [dict(step) for step in value.get("steps") or []]
        # Stored settings have secrets masked; running them would not repeat
        # the original, so ask for a fresh launch instead.
        if "[REDACTED]" in json.dumps([task, steps], default=str):
            raise ValueError(
                "This workflow's saved settings had secrets removed, so it can't "
                "be repeated exactly. Use Edit and run to launch it again."
            )
        task.pop("run_id", None)
        if agent_id and not task.get("agent_id"):
            task["agent_id"] = agent_id
        if self.is_scratch_workspace(task.get("workspace")):
            task["workspace"] = self.scratch_workspace()
        task["metadata"] = {
            **dict(task.get("metadata") or {}),
            "rerun_of": orchestration_id,
        }
        if kind == "plan":
            stages = [HarnessStage.from_value(step).to_dict() for step in steps]
            return self.start_plan(task, stages)
        return self.start_compare(
            task,
            [step.get("harness") for step in steps],
            models={step.get("harness"): step.get("model") for step in steps},
        )

    def _orchestration_store(self) -> Any:
        if not callable(getattr(self.run_store, "create_orchestration", None)):
            raise TypeError(
                "This harness run store does not support plans or comparisons"
            )
        return self.run_store

    def _start_orchestration(
        self, kind: str, base: HarnessTask, steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        store = self._orchestration_store()
        if self._closing:
            raise RuntimeError("This MetaHarness is closed")
        record = HarnessOrchestration(
            orchestration_id=str(uuid.uuid4()),
            kind=kind,
            status=HarnessStatus.QUEUED,
            task=redact(base.to_dict()),
            steps=steps,
            heartbeat_at=utcnow_iso(),
        )
        store.create_orchestration(record)
        stop = threading.Event()
        thread = threading.Thread(
            target=self._drive_orchestration,
            args=(record.orchestration_id, kind, base, [dict(s) for s in steps], stop),
            name=f"memorizz-harness-{kind}-{record.orchestration_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._orchestrations[record.orchestration_id] = (thread, stop)
        thread.start()
        return record.to_dict()

    def _orchestration_task(
        self,
        base: HarnessTask,
        orchestration_id: str,
        kind: str,
        index: int,
        steps: List[Dict[str, Any]],
        *,
        prior: Optional[List[Dict[str, Any]]] = None,
    ) -> HarnessTask:
        step = steps[index]
        # Every run in a workflow retrieves memory for the overall task and
        # reuses one snapshot, so stages and compared harnesses see the same
        # evidence.
        context = {
            "memory_query": base.context.get("memory_query") or base.task,
            "shared_context_key": base.context.get("shared_context_key")
            or f"orchestration:{orchestration_id}",
        }
        metadata = {
            "orchestration": {
                "id": orchestration_id,
                "kind": kind,
                "step": index,
                "steps": len(steps),
                "name": step.get("name"),
            }
        }
        if kind == "plan":
            return self._stage_task(
                base,
                HarnessStage.from_value(step),
                prior=prior or [],
                context=context,
                metadata=metadata,
            )
        payload = base.to_dict()
        payload.pop("run_id", None)
        payload["harness"] = step["harness"]
        payload["model"] = step.get("model") or base.model
        payload["context"] = {**base.context, **context}
        payload["metadata"] = {**base.metadata, **metadata}
        return HarnessTask.from_dict(payload)

    def _drive_orchestration(
        self,
        orchestration_id: str,
        kind: str,
        base: HarnessTask,
        steps: List[Dict[str, Any]],
        stop: threading.Event,
    ) -> None:
        store = self.run_store
        try:
            store.update_orchestration(
                orchestration_id,
                status=HarnessStatus.RUNNING.value,
                started_at=utcnow_iso(),
                heartbeat_at=utcnow_iso(),
            )
            if kind == "plan":
                self._drive_plan(orchestration_id, base, steps, stop)
            else:
                self._drive_compare(orchestration_id, base, steps, stop)
        except _OrchestrationStopped:
            self._finish_orchestration(
                orchestration_id,
                HarnessStatus.INTERRUPTED,
                error_code="host_stopped",
                error="The harness host shut down; later steps did not start.",
            )
        except Exception as exc:
            logger.exception("Harness workflow %s failed", orchestration_id)
            try:
                self._finish_orchestration(
                    orchestration_id,
                    HarnessStatus.FAILED,
                    error_code=type(exc).__name__,
                    error=str(exc) or "The workflow driver failed.",
                )
            except Exception:
                logger.exception("Could not record workflow %s", orchestration_id)
        finally:
            with self._lock:
                self._orchestrations.pop(orchestration_id, None)

    def _drive_plan(
        self,
        orchestration_id: str,
        base: HarnessTask,
        steps: List[Dict[str, Any]],
        stop: threading.Event,
    ) -> None:
        handoffs: List[Dict[str, Any]] = []
        total = len(steps)
        for index, step in enumerate(steps):
            label = f"Stage {index + 1} of {total} ({step.get('name')})"
            if stop.is_set():
                raise _OrchestrationStopped()
            if self._orchestration_cancel_requested(orchestration_id):
                self._finish_orchestration(
                    orchestration_id,
                    HarnessStatus.CANCELED,
                    error_code="canceled",
                    error=f"Canceled before {label.lower()} started.",
                )
                return
            stage_task = self._orchestration_task(
                base,
                orchestration_id,
                "plan",
                index,
                steps,
                prior=self._fit_prior(base, handoffs),
            )
            run = self.start(stage_task)
            steps[index]["run_id"] = run.run_id
            self.run_store.update_orchestration(
                orchestration_id, steps=steps, current_step=index
            )
            final = self._await_orchestration_runs(
                orchestration_id, [run.run_id], stop
            )[0]
            result = dict(final.result or {})
            handoff = self._record_step(
                orchestration_id, base, stage_task, steps, index
            )
            if handoff is not None:
                handoffs.append(handoff)
            if final.status != HarnessStatus.SUCCEEDED:
                status = {
                    HarnessStatus.CANCELED: HarnessStatus.CANCELED,
                    HarnessStatus.INTERRUPTED: HarnessStatus.INTERRUPTED,
                }.get(final.status, HarnessStatus.FAILED)
                detail = str(
                    result.get("error") or final.status.value.replace("_", " ")
                ).rstrip(". ")
                later = " Later stages did not run." if index + 1 < total else ""
                self._finish_orchestration(
                    orchestration_id,
                    status,
                    error_code=result.get("error_code") or final.status.value,
                    error=f"{label} on {final.harness or step.get('harness')}: "
                    f"{detail}.{later}",
                )
                return
        self._finish_orchestration(orchestration_id, HarnessStatus.SUCCEEDED)

    def _drive_compare(
        self,
        orchestration_id: str,
        base: HarnessTask,
        steps: List[Dict[str, Any]],
        stop: threading.Event,
    ) -> None:
        tasks: Dict[int, HarnessTask] = {}
        for index in range(len(steps)):
            if stop.is_set():
                raise _OrchestrationStopped()
            if self._orchestration_cancel_requested(orchestration_id):
                break
            tasks[index] = self._orchestration_task(
                base, orchestration_id, "compare", index, steps
            )
            run = self.start(tasks[index])
            steps[index]["run_id"] = run.run_id
            self.run_store.update_orchestration(orchestration_id, steps=steps)
        run_ids = [step["run_id"] for step in steps if step.get("run_id")]
        finals = (
            self._await_orchestration_runs(orchestration_id, run_ids, stop)
            if run_ids
            else []
        )
        for index, stage_task in tasks.items():
            self._record_step(orchestration_id, base, stage_task, steps, index)
        succeeded = [run for run in finals if run.status == HarnessStatus.SUCCEEDED]
        if len(succeeded) == len(steps):
            self._finish_orchestration(orchestration_id, HarnessStatus.SUCCEEDED)
        elif self._orchestration_cancel_requested(orchestration_id) or (
            finals and all(run.status == HarnessStatus.CANCELED for run in finals)
        ):
            self._finish_orchestration(
                orchestration_id,
                HarnessStatus.CANCELED,
                error_code="canceled",
                error="The comparison was canceled.",
            )
        else:
            self._finish_orchestration(
                orchestration_id,
                HarnessStatus.FAILED,
                error_code="comparison_incomplete",
                error=f"{len(succeeded)} of {len(steps)} harnesses succeeded.",
            )

    def _record_step(
        self,
        orchestration_id: str,
        base: HarnessTask,
        task: HarnessTask,
        steps: List[Dict[str, Any]],
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """Build a finished step's handoff, save it to memory and note where."""
        step = steps[index]
        handoff = self._stage_handoff(step["run_id"], str(step.get("name") or ""))
        if handoff is None:
            return None
        workflow = dict(task.metadata.get("orchestration") or {})
        memory_record_id = self._remember_handoff(
            task,
            handoff,
            workflow=workflow,
            thread_id=base.thread_id or f"harness-workflow-{orchestration_id}",
        )
        step["handoff"] = {
            "response_chars": handoff.get("response_chars"),
            "files_changed": handoff.get("files_changed") or [],
            "memory_record_id": memory_record_id,
        }
        self.run_store.update_orchestration(orchestration_id, steps=steps)
        return handoff

    def _orchestration_cancel_requested(self, orchestration_id: str) -> bool:
        record = self.run_store.get_orchestration(orchestration_id)
        return bool(record is None or record.cancel_requested)

    def _await_orchestration_runs(
        self,
        orchestration_id: str,
        run_ids: List[str],
        stop: threading.Event,
    ) -> List[HarnessRun]:
        """Wait for runs to finish, keeping the workflow's status and heartbeat
        current, forwarding cancellation and ending expired approvals."""
        cancel_sent: set[str] = set()
        last_beat = 0.0
        while True:
            if stop.is_set():
                raise _OrchestrationStopped()
            runs = [self.run_store.get(run_id) for run_id in run_ids]
            missing = [rid for rid, run in zip(run_ids, runs) if run is None]
            if missing:
                raise KeyError(f"Harness run disappeared: {missing[0]}")
            record = self.run_store.get_orchestration(orchestration_id)
            canceling = record is None or record.cancel_requested
            for run in runs:
                if run.status.terminal:
                    continue
                if canceling and run.run_id not in cancel_sent:
                    cancel_sent.add(run.run_id)
                    try:
                        self.cancel(run.run_id)
                    except Exception:
                        logger.exception("Could not cancel harness run %s", run.run_id)
                elif run.status == HarnessStatus.PENDING_APPROVAL:
                    self._end_expired_approval(run)
            if all(run.status.terminal for run in runs):
                return runs
            waiting = any(run.status == HarnessStatus.PENDING_APPROVAL for run in runs)
            working = any(
                run.status in {HarnessStatus.QUEUED, HarnessStatus.RUNNING}
                for run in runs
            )
            status = (
                HarnessStatus.PENDING_APPROVAL
                if waiting and not working
                else HarnessStatus.RUNNING
            )
            now = time.monotonic()
            if record is not None and (
                record.status != status
                or now - last_beat >= ORCHESTRATION_HEARTBEAT_SECONDS
            ):
                self.run_store.update_orchestration(
                    orchestration_id,
                    status=status.value,
                    heartbeat_at=utcnow_iso(),
                )
                last_beat = now
            stop.wait(ORCHESTRATION_POLL_SECONDS)

    def _end_expired_approval(self, run: HarnessRun) -> None:
        """A stage whose approval expired would wait forever; cancel it."""
        if not run.approval_proposal_id:
            return
        proposal = self.approval_store.get(run.approval_proposal_id)
        if proposal is None or not proposal.expired:
            return
        if proposal.status == ApprovalStatus.CONSUMED:
            return
        result = HarnessResult(
            run_id=run.run_id,
            harness=run.harness or "auto",
            status=HarnessStatus.CANCELED,
            error_code="approval_expired",
            error="The approval request expired before anyone approved it.",
        )
        self.run_store.update(
            run.run_id,
            status=result.status.value,
            finished_at=utcnow_iso(),
            result=result.to_dict(),
        )
        self._emit(
            run.run_id,
            HarnessEventType.COMPLETE,
            {"status": result.status.value, "error_code": result.error_code},
        )

    def _finish_orchestration(
        self,
        orchestration_id: str,
        status: HarnessStatus,
        *,
        error_code: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = self.run_store.get_orchestration(orchestration_id)
        runs = [
            self.run_store.get(run_id) for run_id in (record.run_ids if record else [])
        ]
        results = [dict(run.result or {}) for run in runs if run is not None]
        costs = [
            float(value["cost_usd"])
            for value in results
            if isinstance(value.get("cost_usd"), (int, float))
        ]
        summary = {
            "runs": len(results),
            "succeeded": sum(
                1
                for run in runs
                if run is not None and run.status == HarnessStatus.SUCCEEDED
            ),
            "verified": sum(1 for value in results if value.get("verified")),
            "cost_usd": round(sum(costs), 6) if costs else None,
        }
        updated = self.run_store.update_orchestration(
            orchestration_id,
            status=status.value,
            finished_at=utcnow_iso(),
            heartbeat_at=utcnow_iso(),
            error_code=error_code,
            error=error,
            summary=summary,
        )
        return updated.to_dict()

    def _orchestration_driver_alive(self, orchestration_id: str) -> bool:
        with self._lock:
            entry = self._orchestrations.get(orchestration_id)
        return bool(entry and entry[0].is_alive())

    def _orchestration_view(self, record: HarnessOrchestration) -> Dict[str, Any]:
        value = record.to_dict()
        if record.status.terminal or self._orchestration_driver_alive(
            record.orchestration_id
        ):
            return value
        beat = _parse_iso(record.heartbeat_at or record.updated_at)
        if beat is not None and time.time() - beat > ORCHESTRATION_STALE_SECONDS:
            # No driver has reported for a while: the host that ran this
            # workflow stopped. Report it without writing on a read.
            value.update(
                status=HarnessStatus.INTERRUPTED.value,
                stale=True,
                error_code=record.error_code or "host_restarted",
                error=record.error
                or "The harness host stopped during this workflow; later steps did not start.",
            )
        return value

    def get_orchestration(self, orchestration_id: str) -> Optional[Dict[str, Any]]:
        record = self._orchestration_store().get_orchestration(orchestration_id)
        return self._orchestration_view(record) if record else None

    def list_orchestrations(
        self, *, limit: int = 50, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        store = self._orchestration_store()
        return [
            self._orchestration_view(record)
            for record in store.list_orchestrations(limit=limit, status=status)
        ]

    def cancel_orchestration(self, orchestration_id: str) -> Dict[str, Any]:
        """Stop a workflow: cancel its active runs and start no further steps."""
        store = self._orchestration_store()
        record = store.get_orchestration(orchestration_id)
        if record is None:
            raise KeyError(f"Unknown harness workflow: {orchestration_id}")
        if record.status.terminal:
            return {
                "ok": False,
                "orchestration_id": orchestration_id,
                "status": record.status.value,
                "reason": "already_terminal",
            }
        store.update_orchestration(orchestration_id, cancel_requested=True)
        # Cancel active runs now rather than on the driver's next poll.
        for run_id in record.run_ids:
            run = self.run_store.get(run_id)
            if run is not None and not run.status.terminal:
                try:
                    self.cancel(run_id)
                except Exception:
                    logger.exception("Could not cancel harness run %s", run_id)
        if self._orchestration_view(record).get("stale"):
            # Nobody is driving it any more, so nobody else will close it.
            self._finish_orchestration(
                orchestration_id,
                HarnessStatus.CANCELED,
                error_code="canceled",
                error="The workflow was canceled after its host stopped.",
            )
            return {
                "ok": True,
                "orchestration_id": orchestration_id,
                "status": HarnessStatus.CANCELED.value,
            }
        return {"ok": True, "orchestration_id": orchestration_id, "status": "canceling"}

    def _deleting_store(self) -> Any:
        if not callable(getattr(self.run_store, "delete", None)):
            raise TypeError("This harness run store does not support deleting runs")
        return self.run_store

    def _still_working(self, run: HarnessRun) -> bool:
        if not run.status.terminal:
            return True
        with self._lock:
            thread = self._threads.get(run.run_id)
        # A finished run's worker may still be writing its last events; it
        # ends within moments.
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            return thread.is_alive()
        return False

    def _delegate_runs(self, run_ids: Iterable[str]) -> List[HarnessRun]:
        """Runs that a MemAgent's harness delegates made for these runs,
        including their own delegates' runs."""
        children: Dict[str, List[HarnessRun]] = {}
        for run in self.run_store.list(limit=10_000):
            parent = dict((run.task or {}).get("metadata") or {}).get("parent_run_id")
            if parent:
                children.setdefault(str(parent), []).append(run)
        seen = {str(run_id) for run_id in run_ids}
        queue, found = list(seen), []
        while queue:
            for child in children.get(queue.pop(), []):
                if child.run_id not in seen:
                    seen.add(child.run_id)
                    found.append(child)
                    queue.append(child.run_id)
        return found

    def _remove_unused_scratch(self, folders: Iterable[Any]) -> List[str]:
        """Remove MemoRizz scratch folders (made for a blank workspace) that no
        remaining run, workflow or delegate grant uses. Never touches other
        folders."""
        root = self._scratch_root().resolve()
        candidates = set()
        for folder in folders:
            if not folder or not self.is_scratch_workspace(folder):
                continue
            path = Path(str(folder)).resolve()
            if path.parent == root and path.is_dir():
                candidates.add(path)
        if not candidates:
            return []
        in_use = set()

        def note(folder: Any) -> None:
            if folder:
                try:
                    in_use.add(Path(str(folder)).resolve())
                except OSError:
                    pass

        for run in self.run_store.list(limit=10_000):
            note((run.task or {}).get("workspace"))
        if callable(getattr(self.run_store, "list_orchestrations", None)):
            for record in self.run_store.list_orchestrations(limit=10_000):
                note((record.task or {}).get("workspace"))
        try:
            grants = self.approval_store.list(status=ApprovalStatus.APPROVED, limit=500)
        except Exception:
            grants = []
        for grant in grants:
            if grant.tool_name == "metaharness.delegate_access" and not grant.expired:
                note((grant.arguments or {}).get("workspace"))
        removed = []
        for path in sorted(candidates - in_use):
            try:
                shutil.rmtree(path)
                removed.append(str(path))
            except OSError:
                logger.warning(
                    "Could not remove scratch folder %s", path, exc_info=True
                )
        return removed

    def delete_runs(
        self, run_ids: Iterable[str], *, remove_scratch: bool = True
    ) -> Dict[str, Any]:
        """Delete finished runs, their events, and the runs their MemAgent
        delegates made. Runs still working, and stages of a workflow (delete
        the workflow instead), are kept and listed with the reason. A scratch
        folder MemoRizz made for a deleted run goes too once nothing else uses
        it (``remove_scratch=False`` keeps it); files in your own folders and
        answers saved to memory stay."""
        store = self._deleting_store()
        chosen: List[str] = []
        kept: List[Dict[str, Any]] = []
        for run_id in dict.fromkeys(str(value) for value in run_ids if value):
            run = self.run_store.get(run_id)
            if run is None:
                kept.append({"run_id": run_id, "reason": "not_found"})
            elif self._still_working(run):
                kept.append({"run_id": run_id, "reason": "active"})
            elif dict((run.task or {}).get("metadata") or {}).get("orchestration"):
                workflow = dict(run.task["metadata"]["orchestration"])
                kept.append(
                    {
                        "run_id": run_id,
                        "reason": "workflow",
                        "orchestration_id": workflow.get("id"),
                    }
                )
            else:
                chosen.append(run_id)
        delegates = []
        folders = []
        for run in self._delegate_runs(chosen):
            if self._still_working(run):
                kept.append({"run_id": run.run_id, "reason": "active"})
            else:
                delegates.append(run.run_id)
                folders.append((run.task or {}).get("workspace"))
        for run_id in chosen:
            run = self.run_store.get(run_id)
            folders.append((run.task or {}).get("workspace") if run else None)
        store.delete(chosen + delegates)
        return {
            "ok": bool(chosen),
            "deleted": chosen,
            "delegates_deleted": delegates,
            "kept": kept,
            "scratch_removed": self._remove_unused_scratch(folders)
            if remove_scratch
            else [],
        }

    def delete_conversation(
        self, conversation_id: str, *, remove_scratch: bool = True
    ) -> Dict[str, Any]:
        """Delete every finished turn of a harness conversation."""
        runs = self.conversation(conversation_id)
        if not runs:
            raise KeyError(f"Unknown harness conversation: {conversation_id}")
        return self.delete_runs(
            [run["run_id"] for run in runs], remove_scratch=remove_scratch
        )

    def delete_run(self, run_id: str, *, remove_scratch: bool = True) -> Dict[str, Any]:
        """Delete one finished run (and its delegates' runs)."""
        result = self.delete_runs([run_id], remove_scratch=remove_scratch)
        if result["deleted"]:
            return result
        reason = result["kept"][0] if result["kept"] else {"reason": "not_found"}
        if reason["reason"] == "not_found":
            raise KeyError(f"Unknown harness run: {run_id}")
        if reason["reason"] == "active":
            raise ValueError("This run is still working. Cancel it before deleting it.")
        raise ValueError(
            "This run is a step of a workflow. Delete the workflow to remove it."
        )

    def delete_orchestration(
        self, orchestration_id: str, *, remove_scratch: bool = True
    ) -> Dict[str, Any]:
        """Delete a finished plan or comparison with all its runs (and its
        scratch folder, as ``delete_runs`` does)."""
        store = self._orchestration_store()
        self._deleting_store()
        record = store.get_orchestration(orchestration_id)
        if record is None:
            raise KeyError(f"Unknown harness workflow: {orchestration_id}")
        view = self._orchestration_view(record)
        runs = [run for run in map(self.run_store.get, record.run_ids) if run]
        if not HarnessStatus(view["status"]).terminal or any(
            self._still_working(run) for run in runs
        ):
            raise ValueError(
                "This workflow is still working. Cancel it before deleting it."
            )
        steps = [run.run_id for run in runs]
        delegate_runs = [
            run for run in self._delegate_runs(steps) if not self._still_working(run)
        ]
        delegates = [run.run_id for run in delegate_runs]
        folders = [(record.task or {}).get("workspace")] + [
            (run.task or {}).get("workspace") for run in [*runs, *delegate_runs]
        ]
        self.run_store.delete(steps + delegates)
        store.delete_orchestration(orchestration_id)
        return {
            "ok": True,
            "orchestration_id": orchestration_id,
            "deleted": steps,
            "delegates_deleted": delegates,
            "scratch_removed": self._remove_unused_scratch(folders)
            if remove_scratch
            else [],
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        # Stop workflow drivers first so none starts another run while the
        # active runs below are canceled.
        with self._lock:
            drivers = list(self._orchestrations.values())
        for _, stop in drivers:
            stop.set()
        for thread, _ in drivers:
            thread.join(timeout=5.0)
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
        if self._judge is not None:
            self._judge.close()
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

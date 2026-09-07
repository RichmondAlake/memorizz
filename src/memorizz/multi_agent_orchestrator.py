# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import concurrent.futures
import copy
import json
import logging
import threading
import time
import uuid
from contextvars import copy_context
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional

from .approval import ApprovalRequired
from .completion import CompletionCandidate, CompletionPolicy, CompletionRejectedError
from .coordination.shared_memory.shared_memory import SharedMemory
from .coordination.structured_results import (
    consolidate_structured_findings,
    finding_ids,
)
from .execution import AgentExecutionError, delegated_execution, recording_warning
from .streaming import (
    CancellationToken,
    StreamCancelled,
    current_cancellation,
    current_stream,
)
from .task_decomposition import SubTask, TaskDecomposer
from .tool_context import reset_tool_context, set_tool_context
from .tool_outcomes import ToolResult

if TYPE_CHECKING:
    from .memagent import MemAgent

logger = logging.getLogger(__name__)
_worker_slots = threading.BoundedSemaphore(32)
_supervised_workers = set()
_supervised_lock = threading.Lock()


class MultiAgentOrchestrator:
    """
    Orchestrates multi-agent workflows with hierarchical coordination support.

    This orchestrator supports both flat and hierarchical multi-agent scenarios:
    - Flat: Root agent coordinates directly with delegate agents
    - Hierarchical: Delegate agents can have their own sub-agents, all coordinated
      within a single shared memory session for complete visibility and control
    """

    def __init__(
        self,
        root_agent: "MemAgent",
        delegates: List["MemAgent"],
        *,
        delegation_plan: Any = None,
        persist_participants: bool = False,
        workflow_id: Optional[str] = None,
        consolidation_strategy: str = "model",
        primary_task_id: Optional[str] = None,
        evidence_context: bool = True,
        max_dependency_context_chars: int = 12_000,
        max_consolidation_result_chars: int = 12_000,
        required_finding_ids: Optional[Iterable[str]] = None,
        adaptive_escalation: Optional[Mapping[str, Any]] = None,
        allow_root_fallback: bool = False,
        thread_strategy: str = "task",
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        cancellation=None,
        on_task_event=None,
        on_task_result=None,
        task_completion_policy=None,
    ):
        self.root_agent = root_agent
        self.delegates = delegates
        if thread_strategy not in {"task", "shared"}:
            raise ValueError("thread_strategy must be task or shared")
        if max_workers is not None and (
            isinstance(max_workers, bool) or int(max_workers) < 1
        ):
            raise ValueError("max_workers must be positive")
        if timeout is not None and (
            float(timeout) <= 0 or not float(timeout) < float("inf")
        ):
            raise ValueError("timeout must be a finite positive number")
        for callback in (on_task_event, on_task_result):
            if callback is not None and not callable(callback):
                raise TypeError("Task callbacks must be callable")
        self.allow_root_fallback = bool(allow_root_fallback)
        self.thread_strategy = thread_strategy
        self.max_workers = min(32, int(max_workers or max(1, len(delegates))))
        self.timeout = float(timeout) if timeout is not None else None
        self.cancellation = (
            cancellation or current_cancellation.get() or CancellationToken()
        )
        self.on_task_event = on_task_event
        self.on_task_result = on_task_result
        self.task_completion_policy = CompletionPolicy.from_value(
            task_completion_policy
        )
        self._deadline = None
        self._cancel_reason = None
        self._recording_warnings = []
        self._recording_lock = threading.Lock()
        self._worker_metadata = {}
        self._child_threads = {}
        self._task_results = {}
        self.shared_memory = SharedMemory(root_agent.memory_provider)
        self.task_decomposer = TaskDecomposer(root_agent)
        self.shared_memory_id = None
        self.delegation_plan = delegation_plan
        self.persist_participants = bool(persist_participants)
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.consolidation_strategy = (
            str(consolidation_strategy or "model").strip().lower()
        )
        if self.consolidation_strategy not in {
            "model",
            "deterministic",
            "primary",
            "structured",
        }:
            raise ValueError(
                "consolidation_strategy must be model, deterministic, primary, or "
                "structured"
            )
        self.primary_task_id = str(primary_task_id or "").strip() or None
        self.evidence_context = bool(evidence_context)
        self.max_dependency_context_chars = max(
            1_000, min(int(max_dependency_context_chars), 100_000)
        )
        self.max_consolidation_result_chars = max(
            1_000, min(int(max_consolidation_result_chars), 100_000)
        )
        self.required_finding_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in (required_finding_ids or [])
                if str(item).strip()
            )
        )
        self.adaptive_escalation = dict(adaptive_escalation or {})
        self.adaptive_escalation["enabled"] = bool(
            self.adaptive_escalation.get("enabled", False)
        )
        if self.adaptive_escalation["enabled"] and not self.required_finding_ids:
            raise ValueError(
                "adaptive escalation requires at least one required_finding_id"
            )
        self.last_workflow_report: Optional[Dict[str, Any]] = None
        self._request_user_id: Optional[str] = None
        self._request_context: Optional[Dict[str, Any]] = None
        self._tool_context: Optional[Dict[str, Any]] = None
        self._trace_id: Optional[str] = None
        self._request_memory_id: Optional[str] = None
        self._request_thread_id: Optional[str] = None
        self._original_query: Optional[str] = None
        self._last_consolidation_report: Optional[Dict[str, Any]] = None
        self._last_execution_report: Optional[Dict[str, Any]] = None
        self._delegate_context_packs: Dict[str, Any] = {}
        self._delegate_context_lock = threading.Lock()
        self.is_nested_orchestrator = (
            False  # Flag to track if this is a sub-level orchestrator
        )
        self._owns_session_outcome = True

    def execute_multi_agent_workflow(
        self,
        user_query: str,
        memory_id: str = None,
        thread_id: str = None,
        *,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        delegation_plan: Any = None,
        return_report: bool = False,
    ) -> Any:
        """
        Execute a multi-agent workflow with hierarchical coordination support.

        The workflow intelligently handles nested agent scenarios:
        1. First checks if the root agent is already part of an active shared session
        2. If yes, joins that session as a sub-agent rather than creating a new one
        3. If no, creates a new root-level shared session
        4. Ensures all agent activities are tracked in a unified shared memory

        Parameters:
            user_query (str): The task to be executed
            memory_id (str): Memory context for the execution
            thread_id (str): Thread context

        Returns:
            str: The consolidated response from all agents
        """

        logger.info(f"Starting multi-agent workflow for query: {user_query}")
        logger.info(f"Root agent ID: {self.root_agent.agent_id}")
        logger.info(f"Number of delegates: {len(self.delegates)}")
        logger.info(f"Delegate IDs: {[agent.agent_id for agent in self.delegates]}")
        self._request_user_id = user_id
        self._request_context = dict(context or {})
        self._tool_context = dict(tool_context or {})
        self._trace_id = trace_id or str(uuid.uuid4())
        self._request_memory_id = memory_id
        self._request_thread_id = thread_id
        self._original_query = user_query
        self._request_context.setdefault("memory_query", user_query)
        self._request_context.setdefault(
            "shared_context_key", f"workflow:{self.workflow_id}"
        )
        # All panel stages retrieve procedural evidence under the root's
        # ownership. This enables one shared snapshot without exposing one
        # delegate's private skills or workflows to another delegate.
        self._request_context.setdefault(
            "memory_context_agent_id", self.root_agent.agent_id
        )
        self._request_context.setdefault("memory_context_strategy", "evidence_pack")
        self._request_context.setdefault("trace_id", self._trace_id)
        self._last_consolidation_report = None
        self._last_execution_report = None
        with self._delegate_context_lock:
            self._delegate_context_packs.clear()
        try:
            self._deadline = (
                time.monotonic() + self.timeout if self.timeout is not None else None
            )
            selected_plan = (
                delegation_plan if delegation_plan is not None else self.delegation_plan
            )
            if (selected_plan is None or callable(selected_plan)) and self._stopped():
                raise StreamCancelled()
            delegate_ids = [agent.agent_id for agent in self.delegates]
            if len(set(delegate_ids)) != len(delegate_ids):
                raise ValueError("Delegates must use unique agent IDs")
            sub_tasks = self.task_decomposer.decompose_task(
                user_query, self.delegates, plan=selected_plan
            )
            self.task_decomposer.validate_plan(sub_tasks, set(delegate_ids))
            if not sub_tasks:
                if (
                    selected_plan is None
                    and self.allow_root_fallback
                    and not self._stopped()
                ):
                    fallback_context = dict(tool_context or {})
                    fallback_context["_memorizz_skip_delegation"] = True
                    strict_token = delegated_execution.set(True)
                    try:
                        response = self.root_agent.run(
                            user_query,
                            memory_id,
                            thread_id,
                            user_id=user_id,
                            context=context,
                            tool_context=fallback_context,
                        )
                    finally:
                        delegated_execution.reset(strict_token)
                    return self._finish_workflow(
                        [], response, return_report=return_report, fallback=True
                    )
                raise ValueError(
                    "No executable delegation tasks; root fallback is disabled"
                )
            # Validate adaptive routing before any primary task can have effects.
            escalation_ids = set(
                self.adaptive_escalation.get("escalation_task_ids") or []
            )
            if self.adaptive_escalation.get("enabled"):
                if not escalation_ids <= {task.task_id for task in sub_tasks}:
                    raise ValueError("Unknown adaptive escalation task IDs")
                if escalation_ids and all(
                    task.task_id in escalation_ids for task in sub_tasks
                ):
                    raise ValueError(
                        "Adaptive escalation requires at least one primary task"
                    )
                if any(
                    set(task.dependencies) & escalation_ids
                    for task in sub_tasks
                    if task.task_id not in escalation_ids
                ):
                    raise ValueError(
                        "Primary tasks cannot depend on adaptive escalation tasks"
                    )
            if self.persist_participants:
                self._prepare_participants()
            # HIERARCHICAL COORDINATION: Check if we should join an existing session
            existing_session = self._find_or_create_shared_session()

            if existing_session:
                self.shared_memory_id = str(
                    existing_session.get("memory_id") or existing_session.get("_id")
                )
                logger.info(f"Joining existing shared session: {self.shared_memory_id}")
                self.is_nested_orchestrator = True
                self._owns_session_outcome = (
                    self.shared_memory._decode_payload(existing_session).get(
                        "root_agent_id"
                    )
                    == self.root_agent.agent_id
                )

                # Register our delegates as sub-agents in the existing session
                delegate_ids = [agent.agent_id for agent in self.delegates]
                if delegate_ids:
                    self._record(
                        "register_sub_agents",
                        self.shared_memory.register_sub_agents,
                        memory_id=self.shared_memory_id,
                        parent_agent_id=self.root_agent.agent_id,
                        sub_agent_ids=delegate_ids,
                    )

            # **FIX: Add shared memory ID to root agent's memory_ids array**
            # This ensures consistency between single-agent and multi-agent memory management
            if (
                self.persist_participants
                and self.shared_memory_id
                and self.root_agent.agent_id
                and self.shared_memory_id not in (self.root_agent.memory_ids or [])
            ):
                # Initialize memory_ids if it's None
                if self.root_agent.memory_ids is None:
                    self.root_agent.memory_ids = []

                # Add shared memory ID to agent's memory_ids array
                self.root_agent.memory_ids.append(self.shared_memory_id)
                logger.info(
                    f"Added shared memory ID {self.shared_memory_id} to root agent's memory_ids array"
                )

                # Persist the updated memory_ids to the memory provider
                if hasattr(
                    self.root_agent.memory_provider, "update_memagent_memory_ids"
                ) and self._participant_is_persisted(self.root_agent):
                    update_success = (
                        self.root_agent.memory_provider.update_memagent_memory_ids(
                            self.root_agent.agent_id, self.root_agent.memory_ids
                        )
                    )
                    logger.info(
                        f"Persisted memory_ids to storage: {'success' if update_success else 'failed'}"
                    )
                else:
                    logger.debug(
                        "Root participant is not persisted; skipping memory-id update"
                    )

            # **FIX: Also add shared memory ID to delegate agents' memory_ids arrays**
            # This ensures all participating agents have access to the shared memory
            for delegate in self.delegates if self.persist_participants else []:
                if (
                    self.shared_memory_id
                    and delegate.agent_id
                    and self.shared_memory_id not in (delegate.memory_ids or [])
                ):
                    # Initialize memory_ids if it's None
                    if delegate.memory_ids is None:
                        delegate.memory_ids = []

                    # Add shared memory ID to delegate's memory_ids array
                    delegate.memory_ids.append(self.shared_memory_id)
                    logger.info(
                        f"Added shared memory ID {self.shared_memory_id} to delegate agent {delegate.agent_id}'s memory_ids array"
                    )

                    # Persist the updated memory_ids to the memory provider
                    if hasattr(
                        delegate.memory_provider, "update_memagent_memory_ids"
                    ) and self._participant_is_persisted(delegate):
                        update_success = (
                            delegate.memory_provider.update_memagent_memory_ids(
                                delegate.agent_id, delegate.memory_ids
                            )
                        )
                        logger.info(
                            f"Persisted delegate {delegate.agent_id} memory_ids to storage: {'success' if update_success else 'failed'}"
                        )
                    else:
                        logger.debug(
                            "Delegate %s is not persisted; skipping memory-id update",
                            delegate.agent_id,
                        )

            # Log the start of multi-agent execution with hierarchy context
            self._record(
                "workflow_start",
                self.shared_memory.add_blackboard_entry,
                memory_id=self.shared_memory_id,
                agent_id=self.root_agent.agent_id,
                content={
                    "original_query": user_query,
                    "started_at": datetime.now().isoformat(),
                    "orchestrator_type": (
                        "nested" if self.is_nested_orchestrator else "root"
                    ),
                    "delegate_count": len(self.delegates),
                },
                entry_type="workflow_start",
            )

            # Log task decomposition
            self._record(
                "task_decomposition",
                self.shared_memory.add_blackboard_entry,
                memory_id=self.shared_memory_id,
                agent_id=self.root_agent.agent_id,
                content={"sub_tasks": [task.to_dict() for task in sub_tasks]},
                entry_type="task_decomposition",
            )
            self._record(
                "after_task_decomposition",
                self._after_task_decomposition,
                sub_tasks,
                user_query,
            )

            # 3. Execute sub-tasks in parallel
            logger.info(f"Executing {len(sub_tasks)} sub-tasks in parallel...")
            sub_task_results = self._execute_sub_tasks(sub_tasks, memory_id, thread_id)
            logger.info(
                f"Sub-task execution completed with {len(sub_task_results)} results"
            )

            # 4. Consolidate results
            logger.info("Starting result consolidation...")
            consolidation_inputs = [
                item for item in sub_task_results if item.get("status") == "completed"
            ]
            if self._stopped():
                consolidated_response = self._deterministic_consolidation(
                    user_query, consolidation_inputs
                )
                self._last_consolidation_report = {
                    "status": "cancelled",
                    "model_used": False,
                }
            elif not consolidation_inputs:
                consolidated_response = ""
                self._last_consolidation_report = {
                    "status": "failed",
                    "model_used": False,
                    "reason": "no_completed_tasks",
                }
            else:
                consolidated_response = self._consolidate_results(
                    user_query, consolidation_inputs
                )
            logger.info(
                f"Consolidation completed: {consolidated_response[:100]}..."
                if consolidated_response
                else "No consolidated response"
            )

            return self._finish_workflow(
                sub_task_results, consolidated_response, return_report=return_report
            )

        except (Exception, StreamCancelled) as e:
            logger.error(f"Error in multi-agent workflow: {e}", exc_info=True)
            if isinstance(e, StreamCancelled):
                self._cancel_reason = "cancelled"
            # Successful worker effects are never retried because orchestration
            # or recording failed. Preserve the one authoritative task ledger.
            results = list(self._task_results.values())
            response = self._deterministic_consolidation(
                user_query, [row for row in results if row["status"] == "completed"]
            )
            return self._finish_workflow(
                results, response, return_report=return_report, error=e
            )

    def _stopped(self) -> bool:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self._cancel_reason = "deadline_exceeded"
            self.cancellation.cancel()
        if self.cancellation.cancelled:
            self._cancel_reason = self._cancel_reason or "cancelled"
        return self._cancel_reason is not None

    def _warning(self, operation, error, task_id=None):
        warning = {
            "operation": operation,
            "task_id": task_id,
            "error_code": type(error).__name__,
            "error": str(error),
        }
        with self._recording_lock:
            self._recording_warnings.append(warning)
        logger.warning("Delegation recording failed (%s): %s", operation, error)

    def _record(self, operation, callback, *args, **kwargs):
        try:
            result = callback(*args, **kwargs)
            if result is False:
                raise RuntimeError("Recording operation returned False")
            return result
        except Exception as exc:
            self._warning(operation, exc)
            return None

    def _task_event(
        self, task, event_type, *, execution_context="coordinator", **values
    ):
        event = {
            "type": event_type,
            "workflow_id": self.workflow_id,
            "trace_id": self._trace_id,
            "task_id": task.task_id,
            "agent_id": task.assigned_agent_id,
            "user_id": self._request_user_id,
            "memory_id": self._request_memory_id,
            "parent_thread_id": self._request_thread_id,
            "thread_id": self._child_threads.get(task.task_id, self._request_thread_id),
            "execution_context": execution_context,
            **values,
        }
        if self.on_task_event is not None:
            # Runtime context is for the trusted host callback, not the shared
            # ledger. Copy it so one child/callback cannot mutate another's scope.
            try:
                self.on_task_event(
                    {
                        **copy.deepcopy(event),
                        "context": copy.deepcopy(self._request_context),
                    }
                )
            except Exception as exc:
                self._warning("on_task_event", exc, task.task_id)
        return event

    def _finish_workflow(
        self, tasks, response, *, return_report, error=None, fallback=False
    ):
        counts = {
            state: sum(row.get("status") == state for row in tasks)
            for state in (
                "completed",
                "failed",
                "blocked",
                "waiting",
                "cancelled",
                "cancelling",
                "skipped",
            )
        }
        counts["total"] = len(tasks)
        failures = [
            row for row in tasks if row.get("status") not in {"completed", "skipped"}
        ]
        consolidation = dict(self._last_consolidation_report or {})
        if self._stopped() or counts["cancelled"] or counts["cancelling"]:
            outcome = "cancelled"
        elif counts["waiting"]:
            outcome = "waiting"
        elif error is not None or failures or consolidation.get("status") == "failed":
            outcome = "partial" if counts["completed"] else "failed"
        else:
            outcome = "succeeded" if tasks or fallback else "failed"
        ok = outcome == "succeeded"
        status = "completed" if ok else outcome
        execution = dict(self._last_execution_report or {})
        execution.update(
            {
                "workers_still_running": [
                    row["task_id"] for row in tasks if row.get("status") == "cancelling"
                ],
                "cancellation_reason": self._cancel_reason,
            }
        )
        report = {
            "ok": ok,
            "partial": bool(counts["completed"] and not ok),
            "status": status,
            "outcome": outcome,
            "counts": counts,
            "fallback": fallback,
            "workflow_id": self.workflow_id,
            "shared_memory_id": self.shared_memory_id,
            "trace_id": self._trace_id,
            "parent_thread_id": self._request_thread_id,
            "user_id": self._request_user_id,
            "response": response,
            "tasks": tasks,
            "failures": failures,
            "consolidation": consolidation,
            "execution": execution,
            "error": str(error) if error is not None else None,
            "error_code": getattr(error, "code", type(error).__name__)
            if error is not None
            else None,
        }
        if self.shared_memory_id and self._owns_session_outcome:
            self._record(
                "workflow_outcome",
                self.shared_memory.update_session_status,
                self.shared_memory_id,
                status,
                outcome=outcome,
                counts=counts,
            )
        if self.shared_memory_id:
            self._record(
                "workflow_complete",
                self.shared_memory.add_blackboard_entry,
                self.shared_memory_id,
                self.root_agent.agent_id,
                {
                    "status": status,
                    "outcome": outcome,
                    "counts": counts,
                    "response": response,
                },
                "workflow_complete",
            )
        if self.on_task_event is not None:
            self._record(
                "on_task_event",
                self.on_task_event,
                {
                    "type": "workflow_finished",
                    "workflow_id": self.workflow_id,
                    "trace_id": self._trace_id,
                    "parent_thread_id": self._request_thread_id,
                    "user_id": self._request_user_id,
                    "status": status,
                    "outcome": outcome,
                    "counts": dict(counts),
                    "execution_context": "coordinator",
                },
            )
        self._attach_recording_warnings(report)
        self.last_workflow_report = report
        return report if return_report else response

    def _attach_recording_warnings(self, report):
        with self._recording_lock:
            report["recording_warnings"] = copy.deepcopy(self._recording_warnings)
        report["reconciliation_required"] = bool(
            report["recording_warnings"] or report["counts"]["cancelling"]
        )

    def _run_task(self, task, agent, memory_id, thread_id, dependency_results):
        """Worker boundary: propagate cancellation, but isolate answer streaming."""
        strict_token = delegated_execution.set(True)
        stream_token = current_stream.set(None)
        cancel_token = current_cancellation.set(self.cancellation)
        warning_token = recording_warning.set(
            lambda operation, error: self._warning(operation, error, task.task_id)
        )
        result = None
        status = "failed"
        error = None
        tool_token = None
        try:
            tool_token = set_tool_context(
                {
                    **copy.deepcopy(self._tool_context or {}),
                    "workflow_id": self.workflow_id,
                    "trace_id": self._trace_id,
                    "delegated_task_id": task.task_id,
                    "delegated_by": self.root_agent.agent_id,
                    "parent_thread_id": self._request_thread_id,
                    "thread_id": thread_id,
                    "user_id": self._request_user_id,
                }
            )
            self.cancellation.check()
            self._task_event(
                task, "task_started", execution_context="worker", status="in_progress"
            )
            self.cancellation.check()
            result = self._execute_single_task(
                task, agent, memory_id, thread_id, dependency_results
            )
            if isinstance(result, ToolResult):
                if not result.outcome.ok:
                    raise AgentExecutionError(
                        result.outcome.reason_code or result.outcome.status.value,
                        "Delegate returned an unsuccessful typed outcome",
                    )
                result = result.value
            if self.task_completion_policy.enabled:
                decision = self.task_completion_policy.evaluate(
                    CompletionCandidate(
                        query=task.description,
                        response=result
                        if isinstance(result, str)
                        else json.dumps(result, default=str),
                        iteration=1,
                        metadata={
                            "task_id": task.task_id,
                            "workflow_id": self.workflow_id,
                            "thread_id": thread_id,
                            "user_id": self._request_user_id,
                        },
                    )
                )
                if not decision.accepted:
                    raise CompletionRejectedError(decision, 1)
            status = "completed"
            return result
        except ApprovalRequired as exc:
            status, error = "waiting", exc
            raise
        except StreamCancelled as exc:
            status, error = "cancelled", exc
            raise
        except AgentExecutionError as exc:
            status, error = exc.status, exc
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            event = self._task_event(
                task,
                "task_settled",
                execution_context="worker",
                status=status,
                result=result,
                error_code=type(error).__name__ if error else None,
            )
            if self.on_task_result is not None:
                try:
                    metadata = self.on_task_result(
                        {
                            **copy.deepcopy(event),
                            "context": copy.deepcopy(self._request_context),
                        }
                    )
                    if metadata is not None:
                        if not isinstance(metadata, Mapping):
                            raise TypeError(
                                "on_task_result must return a mapping or None"
                            )
                        metadata = json.loads(
                            json.dumps(dict(metadata), allow_nan=False)
                        )
                        with self._recording_lock:
                            self._worker_metadata[task.task_id] = metadata
                        event["metadata"] = metadata
                except Exception as exc:
                    self._warning("on_task_result", exc, task.task_id)
            self._record(
                "task_settled",
                self.shared_memory.add_blackboard_entry,
                self.shared_memory_id,
                task.assigned_agent_id,
                event,
                "task_settled",
            )
            current_cancellation.reset(cancel_token)
            current_stream.reset(stream_token)
            delegated_execution.reset(strict_token)
            recording_warning.reset(warning_token)
            if tool_token is not None:
                reset_tool_context(tool_token)

    def _participant_is_persisted(self, agent: "MemAgent") -> bool:
        provider = getattr(agent, "memory_provider", None)
        lookup = getattr(provider, "retrieve_memagent", None)
        if not callable(lookup):
            return False
        try:
            return lookup(agent.agent_id) is not None
        except Exception:
            return False

    def _prepare_participants(self) -> None:
        """Persist previously unsaved participants when explicitly requested."""
        for agent in [self.root_agent, *self.delegates]:
            if self._participant_is_persisted(agent):
                continue
            if not getattr(agent, "memory_provider", None):
                raise ValueError(
                    f"Participant {agent.agent_id} has no memory provider to persist"
                )
            agent.save()

    def _execute_sub_tasks(
        self, sub_tasks: List[SubTask], memory_id: str, thread_id: str
    ) -> List[Dict[str, Any]]:
        """Execute a plan, escalating only when primary coverage is incomplete."""
        started = time.monotonic()
        config = self.adaptive_escalation
        escalation_ids = {
            str(item) for item in (config.get("escalation_task_ids") or []) if str(item)
        }
        if not config.get("enabled") or not escalation_ids:
            results = self._execute_sub_tasks_parallel(sub_tasks, memory_id, thread_id)
            self._last_execution_report = {
                "strategy": "parallel",
                "adaptive": False,
                "task_count": len(sub_tasks),
                "latency_ms": int((time.monotonic() - started) * 1_000),
            }
            return results

        known_ids = {task.task_id for task in sub_tasks}
        unknown = sorted(escalation_ids - known_ids)
        if unknown:
            raise ValueError(
                "Unknown adaptive escalation task IDs: " + ", ".join(unknown)
            )
        primary_tasks = [
            task for task in sub_tasks if task.task_id not in escalation_ids
        ]
        escalation_tasks = [
            task for task in sub_tasks if task.task_id in escalation_ids
        ]
        if not primary_tasks:
            raise ValueError("Adaptive escalation requires at least one primary task")

        primary_started = time.monotonic()
        primary_results = self._execute_sub_tasks_parallel(
            primary_tasks, memory_id, thread_id
        )
        primary_latency_ms = int((time.monotonic() - primary_started) * 1_000)
        covered_ids, parse_errors = finding_ids(primary_results)
        missing_ids = [
            criterion_id
            for criterion_id in self.required_finding_ids
            if criterion_id not in set(covered_ids)
        ]
        primary_failed = any(
            result.get("status") != "completed" for result in primary_results
        )
        escalated = bool(missing_ids or parse_errors or primary_failed)
        escalation_results: List[Dict[str, Any]] = []
        escalation_latency_ms = 0
        if escalated:
            descriptions = dict(config.get("criterion_descriptions") or {})
            missing_detail = [
                f"{criterion_id}: {descriptions.get(criterion_id, criterion_id)}"
                for criterion_id in missing_ids
            ]
            instruction = (
                "\n\nADAPTIVE ESCALATION\n"
                "The primary review did not return validated structured coverage for:\n- "
                + "\n- ".join(missing_detail or self.required_finding_ids)
                + "\nFocus on these gaps. Return the same structured finding schema; "
                "do not repeat covered criteria unless correcting them."
            )
            for task in escalation_tasks:
                task.description = task.description.rstrip() + instruction
            escalation_started = time.monotonic()
            escalation_results = self._execute_sub_tasks_parallel(
                escalation_tasks,
                memory_id,
                thread_id,
                initial_results=primary_results,
            )
            escalation_latency_ms = int((time.monotonic() - escalation_started) * 1_000)
        else:
            for task in escalation_tasks:
                task.status = "skipped"
                task.result = {
                    "reason": "primary_structured_coverage_complete",
                    "covered_ids": covered_ids,
                }
                escalation_results.append(task.to_dict())

        by_id = {
            result["task_id"]: result
            for result in [*primary_results, *escalation_results]
        }
        results = [by_id[task.task_id] for task in sub_tasks if task.task_id in by_id]
        final_covered, final_parse_errors = finding_ids(results)
        self._last_execution_report = {
            "strategy": "adaptive_coverage",
            "adaptive": True,
            "escalated": escalated,
            "primary_task_ids": [task.task_id for task in primary_tasks],
            "escalation_task_ids": [task.task_id for task in escalation_tasks],
            "required_ids": list(self.required_finding_ids),
            "primary_covered_ids": covered_ids,
            "primary_missing_ids": missing_ids,
            "covered_ids": final_covered,
            "missing_ids": [
                item for item in self.required_finding_ids if item not in final_covered
            ],
            "parse_errors": {**parse_errors, **final_parse_errors},
            "primary_latency_ms": primary_latency_ms,
            "escalation_latency_ms": escalation_latency_ms,
            "latency_ms": int((time.monotonic() - started) * 1_000),
        }
        return results

    def _execute_sub_tasks_parallel(
        self,
        sub_tasks: List[SubTask],
        memory_id: str,
        thread_id: str,
        *,
        initial_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a dependency graph and expose partial/blocked states."""
        agent_map = {agent.agent_id: agent for agent in self.delegates}
        sorted_tasks = sorted(sub_tasks, key=lambda item: item.priority)
        results: List[Dict[str, Any]] = []
        seeded_results = {
            str(item.get("task_id")): item
            for item in (initial_results or [])
            if item.get("task_id")
        }
        results_by_task: Dict[str, Dict[str, Any]] = dict(seeded_results)
        completed_tasks: set[str] = {
            task_id
            for task_id, value in seeded_results.items()
            if value.get("status") == "completed"
        }
        failed_tasks: set[str] = {
            task_id
            for task_id, value in seeded_results.items()
            if value.get("status") not in {"completed", "skipped"}
        }

        task_ids = [task.task_id for task in sorted_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Delegation plans must use unique task IDs")
        if set(task_ids) & set(seeded_results):
            raise ValueError("Initial delegation results duplicate planned task IDs")
        known_ids = set(task_ids) | set(seeded_results)

        def mark_unrunnable(task: SubTask, status: str, reason: str) -> None:
            task.status = status
            task.result = {
                "error": reason,
                "dependencies": list(task.dependencies),
            }
            failed_tasks.add(task.task_id)
            value = task.to_dict()
            value["thread_id"] = self._child_threads.get(task.task_id)
            value["parent_thread_id"] = thread_id
            results.append(value)
            self._task_results[task.task_id] = value
            self._task_event(task, "task_" + status, status=status, reason=reason)

        for task in sorted_tasks:
            missing = [dep for dep in task.dependencies if dep not in known_ids]
            if missing:
                mark_unrunnable(
                    task, "blocked", f"Unknown dependencies: {', '.join(missing)}"
                )
            elif task.assigned_agent_id not in agent_map:
                mark_unrunnable(
                    task,
                    "failed",
                    f"Unknown delegate: {task.assigned_agent_id}",
                )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            future_to_task: Dict[Any, SubTask] = {}
            future_started: Dict[Any, float] = {}
            while True:
                # Consume finished work first so cancellation never discards a
                # result that is already available. Pop timing exactly once.
                for future in [item for item in future_to_task if item.done()]:
                    task = future_to_task.pop(future)
                    duration_ms = int(
                        (time.monotonic() - future_started.pop(future)) * 1_000
                    )
                    try:
                        task.result = future.result()
                        task.status = "completed"
                        completed_tasks.add(task.task_id)
                    except ApprovalRequired as exc:
                        task.status = "waiting"
                        task.result = {
                            "error_code": "approval_required",
                            "proposal": exc.proposal.to_dict(include_arguments=False),
                        }
                        failed_tasks.add(task.task_id)
                    except (StreamCancelled, concurrent.futures.CancelledError):
                        task.status = "cancelled"
                        task.result = {"error_code": self._cancel_reason or "cancelled"}
                        failed_tasks.add(task.task_id)
                    except AgentExecutionError as exc:
                        task.status = exc.status
                        task.result = {
                            **exc.metadata,
                            "error": str(exc),
                            "error_code": exc.code,
                        }
                        failed_tasks.add(task.task_id)
                    except Exception as exc:
                        task.status = "failed"
                        task.result = {
                            "error": str(exc),
                            "error_code": getattr(exc, "code", type(exc).__name__),
                        }
                        failed_tasks.add(task.task_id)
                    task_value = task.to_dict()
                    task_value.update(
                        {
                            "duration_ms": duration_ms,
                            "thread_id": self._child_threads.get(task.task_id),
                            "parent_thread_id": thread_id,
                        }
                    )
                    with self._recording_lock:
                        if task.task_id in self._worker_metadata:
                            task_value["metadata"] = copy.deepcopy(
                                self._worker_metadata[task.task_id]
                            )
                    results.append(task_value)
                    results_by_task[task.task_id] = task_value
                    self._task_results[task.task_id] = task_value
                    self._record(
                        "task_completion",
                        self.shared_memory.add_blackboard_entry,
                        self.shared_memory_id,
                        task.assigned_agent_id,
                        task_value,
                        "task_completion",
                    )
                    if task.status == "completed":
                        self._record(
                            "after_task_completion",
                            self._after_task_completion,
                            task,
                            task.result,
                        )
                    self._task_event(
                        task, "task_finished", status=task.status, result=task.result
                    )

                if self._stopped():
                    for task in sorted_tasks:
                        if task.status == "pending":
                            mark_unrunnable(task, "cancelled", self._cancel_reason)
                    for future, task in future_to_task.items():
                        cancelled = future.cancel()
                        mark_unrunnable(
                            task,
                            "cancelled" if cancelled else "cancelling",
                            self._cancel_reason,
                        )
                        self._task_results[task.task_id][
                            "execution_still_running"
                        ] = not cancelled
                    break

                for task in sorted_tasks:
                    if task.status != "pending":
                        continue
                    if any(dep in failed_tasks for dep in task.dependencies):
                        mark_unrunnable(
                            task, "blocked", "One or more dependencies failed"
                        )
                        continue
                    if len(future_to_task) >= self.max_workers:
                        break
                    if all(dep in completed_tasks for dep in task.dependencies):
                        if not _worker_slots.acquire(blocking=False):
                            mark_unrunnable(
                                task, "blocked", "Delegation worker capacity exhausted"
                            )
                            continue
                        task.status = "in_progress"
                        child_thread = thread_id
                        if self.thread_strategy == "task":
                            identity = json.dumps(
                                [
                                    self.workflow_id,
                                    self._request_user_id,
                                    memory_id,
                                    thread_id,
                                    task.task_id,
                                ]
                            )
                            child_thread = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
                        self._child_threads[task.task_id] = child_thread
                        dependency_results = {
                            dependency: results_by_task[dependency].get("result")
                            for dependency in task.dependencies
                            if dependency in results_by_task
                        }
                        worker_context = copy_context()  # a distinct Context per future
                        try:
                            future = executor.submit(
                                worker_context.run,
                                self._run_task,
                                task,
                                agent_map[task.assigned_agent_id],
                                memory_id,
                                child_thread,
                                dependency_results,
                            )
                        except BaseException:
                            _worker_slots.release()
                            raise
                        with _supervised_lock:
                            _supervised_workers.add(future)

                        def release(done):
                            with _supervised_lock:
                                _supervised_workers.discard(done)
                            _worker_slots.release()

                        future.add_done_callback(release)
                        future_to_task[future] = task
                        future_started[future] = time.monotonic()

                if not future_to_task:
                    for task in sorted_tasks:
                        if task.status == "pending":
                            mark_unrunnable(
                                task,
                                "blocked",
                                "Dependency cycle or unsatisfied dependency",
                            )
                    break

                done, _ = concurrent.futures.wait(
                    future_to_task,
                    timeout=0.05,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
        finally:
            # Python cannot kill an arbitrary blocking tool. Running futures
            # retain their capacity slot and worker settlement callback until
            # they exit; the report explicitly identifies these unfinished tasks.
            executor.shutdown(wait=False, cancel_futures=True)

        by_id = {result["task_id"]: result for result in results}
        return [by_id[task.task_id] for task in sorted_tasks if task.task_id in by_id]

    def _execute_single_task(
        self,
        task: SubTask,
        agent: "MemAgent",
        memory_id: str,
        thread_id: str,
        dependency_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a single sub-task with an agent."""

        try:
            # Log task start
            self._record(
                "task_start",
                self.shared_memory.add_blackboard_entry,
                memory_id=self.shared_memory_id,
                agent_id=agent.agent_id,
                content={
                    "task_id": task.task_id,
                    "description": task.description,
                    "started_at": datetime.now().isoformat(),
                },
                entry_type="task_start",
            )

            request_context = copy.deepcopy(self._request_context or {})
            request_context["delegation"] = {
                "workflow_id": self.workflow_id,
                "trace_id": self._trace_id,
                "task_id": task.task_id,
                "dependencies": list(task.dependencies),
            }
            if task.task_id in self._child_threads:
                request_context["delegation"].update(
                    {
                        "parent_thread_id": self._request_thread_id,
                        "thread_id": thread_id,
                    }
                )
            if dependency_results:
                bounded_dependencies: Dict[str, Any] = {}
                used = 0
                for dependency, value in dependency_results.items():
                    text = str(value)
                    remaining = self.max_dependency_context_chars - used
                    if remaining <= 0:
                        break
                    if len(text) > remaining:
                        text = text[:remaining] + "\u2026"
                    bounded_dependencies[str(dependency)] = text
                    used += len(text)
                request_context["dependency_results"] = bounded_dependencies
                request_context["model_context"] = {
                    "dependency_results": bounded_dependencies,
                    "instruction": (
                        "Treat dependency results as prior-stage evidence. Verify them "
                        "against the workspace and retrieved memory; do not repeat "
                        "supported material unless the task asks for a rewrite."
                    ),
                }
            tool_context = copy.deepcopy(self._tool_context or {})
            tool_context.update(
                {
                    "workflow_id": self.workflow_id,
                    "trace_id": self._trace_id,
                    "delegated_task_id": task.task_id,
                    "delegated_by": self.root_agent.agent_id,
                    "parent_thread_id": self._request_thread_id,
                    "thread_id": thread_id,
                    "user_id": self._request_user_id,
                }
            )
            if delegated_execution.get():
                self.cancellation.check()
            if (
                getattr(agent, "meta_harness", None) is not None
                and getattr(agent, "meta_harness_mode", None) == "runtime"
            ):
                harness_result = agent.run_on_harness(
                    task.description,
                    memory_id=memory_id,
                    thread_id=thread_id,
                    user_id=self._request_user_id,
                    context=request_context,
                )
                if not harness_result.ok:
                    code = harness_result.error_code or harness_result.status.value
                    detail = harness_result.error or "runtime harness delegate failed"
                    status = {
                        "pending_approval": "waiting",
                        "queued": "waiting",
                        "running": "waiting",
                        "canceled": "cancelled",
                        "interrupted": "cancelled",
                    }.get(harness_result.status.value, "failed")
                    raise AgentExecutionError(
                        code,
                        f"Delegate harness failed ({code}): {detail}",
                        status=status,
                        metadata={"harness_run_id": harness_result.run_id},
                    )
                if harness_result.context_pack is not None:
                    with self._delegate_context_lock:
                        self._delegate_context_packs[
                            task.task_id
                        ] = harness_result.context_pack
                result = harness_result.final_response
            else:
                result = agent.run(
                    task.description,
                    memory_id,
                    thread_id,
                    user_id=self._request_user_id,
                    context=request_context,
                    tool_context=tool_context,
                )

            return result

        except Exception as e:
            logger.error(
                f"Error executing task {task.task_id} with agent {agent.agent_id}: {e}"
            )
            raise

    def _consolidate_results(
        self, original_query: str, sub_task_results: List[Dict[str, Any]]
    ) -> str:
        """Consolidate results from all sub-tasks into a final response."""

        started = time.monotonic()
        if self.consolidation_strategy == "primary":
            completed = [
                item for item in sub_task_results if item.get("status") == "completed"
            ]
            primary = next(
                (
                    item
                    for item in completed
                    if item.get("task_id") == self.primary_task_id
                ),
                completed[0] if completed and self.primary_task_id is None else None,
            )
            if primary is not None:
                self._last_consolidation_report = {
                    "status": "succeeded",
                    "strategy": "primary",
                    "model_used": False,
                    "primary_task_id": primary.get("task_id"),
                    "review_task_ids": [
                        item.get("task_id") for item in completed if item is not primary
                    ],
                    "reason": "configured_primary_response_preserves_delegate_evidence",
                    "latency_ms": int((time.monotonic() - started) * 1_000),
                }
                return str(primary.get("result") or "")
            self._last_consolidation_report = {
                "status": "failed",
                "strategy": "primary",
                "fallback_strategy": "deterministic",
                "fallback_used": True,
                "error_code": "primary_task_unavailable",
                "latency_ms": int((time.monotonic() - started) * 1_000),
            }
            return self._deterministic_consolidation(original_query, sub_task_results)

        if self.consolidation_strategy == "deterministic":
            self._last_consolidation_report = {
                "status": "succeeded",
                "strategy": "deterministic",
                "model_used": False,
                "reason": "configured_deterministic_consolidation",
                "latency_ms": int((time.monotonic() - started) * 1_000),
            }
            return self._deterministic_consolidation(original_query, sub_task_results)

        if self.consolidation_strategy == "structured":
            try:
                response, coverage = consolidate_structured_findings(
                    sub_task_results,
                    required_ids=self.required_finding_ids,
                )
                _, memory_context_metadata = self._coordinator_memory_context(
                    original_query
                )
                memory_context_metadata = {
                    **memory_context_metadata,
                    "used_for_synthesis": False,
                }
                self._last_consolidation_report = {
                    "status": "succeeded",
                    "strategy": "structured",
                    "model_used": False,
                    "reason": "deterministic_structured_finding_union",
                    "coverage": coverage,
                    "memory_context": memory_context_metadata,
                    "latency_ms": int((time.monotonic() - started) * 1_000),
                }
                return response
            except Exception as exc:
                logger.error(
                    "Structured result consolidation failed; preserving raw "
                    "delegate evidence (%s)",
                    type(exc).__name__,
                )
                self._last_consolidation_report = {
                    "status": "failed",
                    "strategy": "structured",
                    "model_used": False,
                    "fallback_strategy": "deterministic",
                    "fallback_used": True,
                    "error_code": type(exc).__name__,
                    "latency_ms": int((time.monotonic() - started) * 1_000),
                }
                return self._deterministic_consolidation(
                    original_query, sub_task_results
                )

        if not self.root_agent.model:
            initialization_error = getattr(self.root_agent, "_llm_init_error", None)
            if initialization_error:
                logger.error(
                    "The configured root LLMProvider is unavailable; using "
                    "deterministic result aggregation"
                )
                self._last_consolidation_report = {
                    "status": "failed",
                    "strategy": "model",
                    "fallback_strategy": "deterministic",
                    "fallback_used": True,
                    "error_code": "llm_initialization_failed",
                    "latency_ms": int((time.monotonic() - started) * 1_000),
                }
            else:
                logger.info(
                    "Root agent has no LLMProvider; using supported deterministic "
                    "result aggregation"
                )
                self._last_consolidation_report = {
                    "status": "succeeded",
                    "strategy": "deterministic",
                    "model_used": False,
                    "reason": "root_agent_has_no_llm_provider",
                    "latency_ms": int((time.monotonic() - started) * 1_000),
                }
            return self._deterministic_consolidation(original_query, sub_task_results)

        try:
            memory_context, memory_context_metadata = self._coordinator_memory_context(
                original_query
            )
            consolidation_prompt = self.task_decomposer.create_consolidation_prompt(
                original_query,
                sub_task_results,
                memory_context=memory_context,
                max_result_chars=self.max_consolidation_result_chars,
            )
            root_instruction = str(
                getattr(self.root_agent, "instruction", "") or ""
            ).strip()
            system_prompt = (
                "You are MemoRizz's evidence-preserving result coordinator. "
                "You may reorganize supported evidence, but you must not create "
                "new factual claims. Authoritative retrieved memory defines the "
                "scope and requirements.\n"
            )
            if root_instruction:
                system_prompt += (
                    "\nROOT COORDINATOR INSTRUCTION:\n" f"{root_instruction}\n"
                )

            # Use root agent to consolidate
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": consolidation_prompt},
            ]

            set_cache_key = getattr(self.root_agent.model, "set_prompt_cache_key", None)
            if callable(set_cache_key):
                set_cache_key(
                    ":".join(
                        item
                        for item in (
                            "memorizz-consolidation",
                            str(self.root_agent.agent_id or "root"),
                            str(self._request_memory_id or "default"),
                        )
                        if item
                    )
                )

            completion_decisions: List[Dict[str, Any]] = []
            aggregate_usage: Dict[str, int] = {}
            rejection_count = 0
            while True:
                response = self.root_agent.model.generate(messages, tools=None)
                response_text = self.task_decomposer._response_text(response).strip()
                if not response_text:
                    raise ValueError(
                        "The consolidation provider returned an empty response"
                    )
                usage_getter = getattr(self.root_agent.model, "get_last_usage", None)
                if callable(usage_getter):
                    for key, value in dict(usage_getter() or {}).items():
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            aggregate_usage[key] = int(
                                aggregate_usage.get(key, 0) + value
                            )
                policy = getattr(self.root_agent, "completion_policy", None)
                if policy is None or not getattr(policy, "enabled", False):
                    break
                candidate = CompletionCandidate(
                    query=original_query,
                    response=response_text,
                    iteration=rejection_count + 1,
                    metadata={
                        "workflow_id": self.workflow_id,
                        "phase": "multi_agent_consolidation",
                        "task_ids": [item.get("task_id") for item in sub_task_results],
                    },
                )
                decision = policy.evaluate(candidate)
                completion_decisions.append(decision.to_dict())
                if decision.accepted:
                    break
                rejection_count += 1
                if rejection_count > policy.max_rejections:
                    if policy.fail_closed:
                        raise CompletionRejectedError(decision, rejection_count)
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {"role": "user", "content": policy.retry_message(decision)},
                    ]
                )
            self._last_consolidation_report = {
                "status": "succeeded",
                "strategy": "model",
                "model_used": True,
                "provider": (
                    getattr(self.root_agent, "llm_provider", None)
                    or getattr(self.root_agent.model, "provider", None)
                    or type(self.root_agent.model).__name__
                ),
                "model": (
                    getattr(self.root_agent, "llm_model", None)
                    or getattr(self.root_agent.model, "model", None)
                ),
                "memory_context": memory_context_metadata,
                "completion_decisions": completion_decisions,
                "generation_attempts": rejection_count + 1,
                "latency_ms": int((time.monotonic() - started) * 1_000),
            }
            if aggregate_usage:
                self._last_consolidation_report["usage"] = aggregate_usage
            return response_text

        except Exception as e:
            logger.error(
                "Model-based result consolidation failed; using deterministic "
                "aggregation (%s)",
                type(e).__name__,
            )
            self._last_consolidation_report = {
                "status": "failed",
                "strategy": "model",
                "fallback_strategy": "deterministic",
                "fallback_used": True,
                "error_code": type(e).__name__,
                "latency_ms": int((time.monotonic() - started) * 1_000),
            }
            return self._deterministic_consolidation(original_query, sub_task_results)

    def _coordinator_memory_context(
        self, original_query: str
    ) -> tuple[str, Dict[str, Any]]:
        """Retrieve one bounded, scoped evidence pack for final synthesis."""
        if (
            not self.evidence_context
            or self.root_agent.memory_provider is None
            or self._request_memory_id is None
        ):
            return "", {"available": False, "reason": "disabled_or_unscoped"}

        # Runtime delegates already received a host-scoped context pack. Reuse
        # it for synthesis when every captured pack is identical and it fits
        # the root's evidence budget. This removes a duplicate Oracle/filesystem
        # retrieval while preserving the coordinator's stricter budget as a
        # fail-closed boundary.
        with self._delegate_context_lock:
            delegate_packs = list(self._delegate_context_packs.items())
        fingerprints = {
            str(getattr(pack, "fingerprint", "") or "")
            for _, pack in delegate_packs
            if pack is not None
        }
        if delegate_packs and len(fingerprints) == 1 and "" not in fingerprints:
            _, shared_pack = delegate_packs[0]
            root_plane = getattr(self.root_agent, "learning_control_plane", None)
            root_budget = int(
                getattr(
                    getattr(root_plane, "config", None),
                    "evidence_token_budget",
                    2_000,
                )
                or 2_000
            )
            if int(getattr(shared_pack, "token_estimate", 0) or 0) <= root_budget:
                metadata = dict(getattr(shared_pack, "metadata", None) or {})
                return str(getattr(shared_pack, "rendered", "") or ""), {
                    "available": bool(getattr(shared_pack, "source_ids", None)),
                    "strategy": "shared_harness_evidence_pack",
                    "pack_id": metadata.get("pack_id"),
                    "source_ids": list(getattr(shared_pack, "source_ids", None) or []),
                    "tokens_used": int(getattr(shared_pack, "token_estimate", 0) or 0),
                    "tokens_saved": int(metadata.get("tokens_saved", 0) or 0),
                    "candidate_count": metadata.get("candidate_count"),
                    "selected_count": metadata.get("selected_count"),
                    "context_snapshot_reused": True,
                    "delegate_task_ids": [task_id for task_id, _ in delegate_packs],
                }

        plane = getattr(self.root_agent, "learning_control_plane", None)
        if plane is not None and getattr(
            getattr(plane, "config", None), "retrieval_enabled", True
        ):
            try:
                pack = plane.retrieve_evidence(
                    original_query,
                    memory_id=self._request_memory_id,
                    user_id=self._request_user_id,
                    thread_id=self._request_thread_id,
                    run_id=self.workflow_id,
                    trace_id=self._trace_id,
                )
                return pack.render(), {
                    "available": bool(pack.items),
                    "strategy": "evidence_pack",
                    "pack_id": pack.pack_id,
                    "source_ids": [item.source_id for item in pack.items],
                    "tokens_used": pack.tokens_used,
                    "tokens_saved": max(0, pack.candidate_tokens - pack.tokens_used),
                    "candidate_count": pack.candidate_count,
                    "selected_count": len(pack.items),
                }
            except Exception as exc:
                logger.warning(
                    "Coordinator evidence-pack retrieval failed; using bounded "
                    "memory fallback (%s)",
                    type(exc).__name__,
                )

        try:
            from .metaharness.context import HarnessContextBuilder

            pack = HarnessContextBuilder(
                self.root_agent.memory_provider, max_chars=8_000, per_type=2
            ).build(
                original_query,
                memory_id=self._request_memory_id,
                user_id=self._request_user_id,
                thread_id=self._request_thread_id,
            )
            return pack.rendered, {
                "available": bool(pack.source_ids),
                "strategy": "bounded_retrieval",
                "source_ids": list(pack.source_ids),
                "tokens_used": pack.token_estimate,
                "truncated": pack.truncated,
            }
        except Exception as exc:
            logger.warning(
                "Coordinator memory retrieval failed (%s)", type(exc).__name__
            )
            return "", {
                "available": False,
                "reason": f"retrieval_failed:{type(exc).__name__}",
            }

    @staticmethod
    def _deterministic_consolidation(
        original_query: str, sub_task_results: List[Dict[str, Any]]
    ) -> str:
        """Preserve delegate evidence without pretending it was synthesized."""

        lines = [f"Results for: {original_query}", ""]
        for result in sub_task_results:
            lines.extend(
                [
                    f"Task: {result.get('description', 'Unknown')}",
                    f"Result: {result.get('result', 'No result')}",
                    "",
                ]
            )
        return "\n".join(lines)

    def get_shared_memory_context(self) -> str:
        """Get shared memory context for inclusion in agent prompts."""

        if not self.shared_memory_id:
            return ""

        try:
            # Get blackboard entries
            entries = self.shared_memory.get_blackboard_entries(self.shared_memory_id)

            if not entries:
                return ""

            context = "\n\n---------SHARED MEMORY CONTEXT---------\n"
            context += "Multi-agent coordination information:\n\n"

            for entry in entries[-10:]:  # Last 10 entries
                context += f"Agent: {entry.get('agent_id', 'Unknown')}\n"
                context += f"Type: {entry.get('entry_type', 'Unknown')}\n"
                context += f"Content: {entry.get('content', 'No content')}\n"
                context += f"Time: {entry.get('created_at', 'Unknown')}\n"
                context += "---\n"

            return context

        except Exception as e:
            logger.error(f"Error getting shared memory context: {e}")
            return ""

    def _find_or_create_shared_session(self) -> Optional[Dict[str, Any]]:
        """
        Intelligent session management for hierarchical multi-agent coordination.

        This method implements the core logic for hierarchical coordination:
        1. Checks if the root agent is already participating in an active shared session
        2. If found, returns that session to enable joining (hierarchical mode)
        3. If not found, creates a new root-level session (flat mode)

        This ensures that sub-agents don't create isolated sessions but instead
        join the existing coordination context, enabling true hierarchical workflows.

        Returns:
            Optional[Dict[str, Any]]: Existing session to join, or None if new session created
        """
        try:
            # Check if our root agent is already part of an active shared session
            existing_session = self.shared_memory.find_active_session_for_agent(
                self.root_agent.agent_id,
                workflow_id=self.workflow_id,
                user_id=self._request_user_id,
            )

            if existing_session:
                logger.info(
                    f"Found existing session for agent {self.root_agent.agent_id}"
                )
                logger.info(
                    f"Session hierarchy: {self.shared_memory.get_agent_hierarchy(str(existing_session.get('_id')))}"
                )
                return existing_session

            # No existing session found - create a new root-level session
            logger.info(
                "No existing session found, creating new root-level shared session"
            )
            delegate_ids = [agent.agent_id for agent in self.delegates]

            self.shared_memory_id = self.shared_memory.create_shared_session(
                root_agent_id=self.root_agent.agent_id,
                delegate_agent_ids=delegate_ids,
                workflow_id=self.workflow_id,
                user_id=self._request_user_id,
                trace_id=self._trace_id,
            )

            logger.info(f"Created new shared memory session: {self.shared_memory_id}")
            logger.info(f"Initial delegates: {delegate_ids}")

            # Return None to indicate we created a new session (not joining existing)
            return None

        except Exception as e:
            logger.error(f"Error in session management: {e}", exc_info=True)
            # Fallback: create new session
            delegate_ids = [agent.agent_id for agent in self.delegates]
            self.shared_memory_id = self.shared_memory.create_shared_session(
                root_agent_id=self.root_agent.agent_id,
                delegate_agent_ids=delegate_ids,
                workflow_id=self.workflow_id,
                user_id=self._request_user_id,
                trace_id=self._trace_id,
            )
            return None

    def _enhance_task_decomposition_with_hierarchy(
        self, user_query: str, *, plan: Any = None
    ) -> List[SubTask]:
        """
        Enhanced task decomposition that considers the complete agent hierarchy.

        Traditional decomposition only considers immediate delegates. This enhanced
        version looks at the full shared memory session to understand the complete
        agent capabilities available across all hierarchy levels.

        Parameters:
            user_query (str): The task to decompose

        Returns:
            List[SubTask]: Decomposed tasks optimized for the full hierarchy
        """
        try:
            # Get the complete agent hierarchy from shared memory
            hierarchy = self.shared_memory.get_agent_hierarchy(self.shared_memory_id)

            logger.info(f"Task decomposition considering hierarchy: {hierarchy}")

            # Standard task decomposition with immediate delegates
            sub_tasks = self.task_decomposer.decompose_task(
                user_query, self.delegates, plan=plan
            )

            # TODO: Future enhancement - analyze sub_agent capabilities for optimal task assignment
            # This could involve:
            # 1. Collecting capabilities from all sub-agents in the hierarchy
            # 2. Re-optimizing task assignments based on the full capability set
            # 3. Creating more granular tasks that leverage specific sub-agent strengths

            logger.info(
                f"Decomposed into {len(sub_tasks)} tasks for {hierarchy.get('total_agents', 0)} total agents"
            )

            return sub_tasks

        except Exception as e:
            logger.error(f"Error in enhanced task decomposition: {e}")
            # Fallback to standard decomposition
            return self.task_decomposer.decompose_task(
                user_query, self.delegates, plan=plan
            )

    # Hook methods for specialized orchestrators ---------------------------------
    def _after_task_decomposition(
        self, sub_tasks: List[SubTask], user_query: str
    ) -> None:
        """Optional hook for subclasses after decomposition."""
        return

    def _after_task_completion(self, task: SubTask, result: Any) -> None:
        """Optional hook after individual task completion."""
        return

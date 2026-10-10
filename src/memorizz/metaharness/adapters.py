"""First-party adapters for MemAgent, Codex, Claude Code, OpenHands, DeepSeek, pi
and Hermes."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import AdapterOutcome, AgentHarness, EventSink, SubprocessHarness
from .models import (
    HarnessCapabilities,
    HarnessContextPack,
    HarnessEvent,
    HarnessEventType,
    HarnessTask,
)
from .security import HarnessSecurityError, build_child_environment


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


def _json_or_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


@contextmanager
def _live_trace(agent: Any, run_id: str, emit: EventSink) -> Iterator[None]:
    """Report a MemAgent's model calls, reasoning, tool calls and delegate
    tasks as harness events while it runs, instead of only its answer."""

    # MemAgent sends a long tool result or reasoning in pieces ("append" on all
    # but the first, "complete" on the last); record each once, whole.
    pieces: Dict[str, List[str]] = {}

    def trace(event: Dict[str, Any]) -> None:
        if event.get("type") != "trace":
            return
        kind = event.get("trace_kind")
        call_id = event.get("tool_call_id") or event.get("span_id")
        if "complete" in event and kind in {"tool_result", "reasoning"}:
            key = f"{kind}:{event.get('trace_id') or call_id}"
            if not event.get("append"):
                pieces[key] = []
            pieces.setdefault(key, []).append(str(event.get("content") or ""))
            if not event.get("complete"):
                return
            event = {**event, "content": "".join(pieces.pop(key, []))[:12000]}
        if kind == "tool_call":
            data = {
                "id": call_id,
                "name": event.get("tool_name"),
                "input": _json_or_text(event.get("content")),
            }
            emit(HarnessEvent(run_id, HarnessEventType.TOOL_CALL, data))
        elif kind == "tool_result":
            status = str(event.get("status") or "")
            data = {
                "tool_use_id": call_id,
                "content": event.get("content") or status,
                "is_error": event.get("success") is False
                and status != "approval_required",
                "status": status,
                "duration_ms": event.get("duration_ms"),
            }
            if event.get("cache"):
                # A tool-cache hit, or a result stored for reuse.
                data["cache"] = event.get("cache")
            emit(HarnessEvent(run_id, HarnessEventType.TOOL_RESULT, data))
        elif kind == "reasoning" and event.get("content"):
            emit(
                HarnessEvent(
                    run_id,
                    HarnessEventType.REASONING,
                    {"id": event.get("span_id"), "text": event.get("content")},
                )
            )
        elif kind in {"model_call", "model_result"}:
            data = {
                "phase": kind,
                "id": event.get("span_id"),
                "iteration": event.get("iteration"),
                "stage": event.get("stage"),
            }
            if kind == "model_result":
                data.update(
                    success=event.get("success"),
                    duration_ms=event.get("duration_ms"),
                    error_code=event.get("error_code") or None,
                )
            emit(HarnessEvent(run_id, HarnessEventType.STATUS, data))

    # Delegate tasks name only the agent's id; show its name, and the
    # harness it runs on when it is harness-backed.
    delegate_names: Dict[str, str] = {}
    for delegate in getattr(agent, "delegates", None) or []:
        persona = getattr(delegate, "persona", None)
        name = getattr(delegate, "name", None) or getattr(persona, "name", None)
        agent_id = getattr(delegate, "agent_id", None)
        if not agent_id:
            continue
        label = str(name or str(agent_id)[:8])
        if getattr(delegate, "meta_harness_mode", None) == "runtime":
            label += f" ({getattr(delegate, 'default_harness', None) or 'harness'})"
        delegate_names[str(agent_id)] = label

    def delegation(event: Dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        task_id = event.get("task_id")
        if not task_id:
            return
        if kind == "task_started":
            agent_id = str(event.get("agent_id") or "")
            label = delegate_names.get(agent_id) or agent_id[:8] or "delegate"
            data = {
                "id": task_id,
                "name": label,
                "subagent": True,
                "input": {"description": label, "agent_id": agent_id},
            }
            emit(HarnessEvent(run_id, HarnessEventType.TOOL_CALL, data))
        elif kind == "task_progress" and event.get("harness_run_id"):
            # A harness delegate's own run started: the trace links to it.
            data = {
                "id": task_id,
                "subagent": True,
                "status": "in_progress",
                "harness": event.get("harness"),
                "harness_run_id": event.get("harness_run_id"),
            }
            emit(HarnessEvent(run_id, HarnessEventType.TOOL_CALL, data))
        elif kind == "task_finished":
            status = str(getattr(event.get("status"), "value", event.get("status")))
            result = event.get("result")
            text = getattr(result, "final_response", None) or result
            data = {
                "tool_use_id": task_id,
                "content": str(text or status)[:4000],
                "is_error": status not in {"succeeded", "completed", "success"},
                "status": status,
            }
            emit(HarnessEvent(run_id, HarnessEventType.TOOL_RESULT, data))

    setters = [
        getattr(agent, name, None)
        for name in ("set_stream_event_callback", "set_delegation_event_callback")
    ]
    if not all(callable(setter) for setter in setters):
        yield
        return
    previous = (
        getattr(agent, "_stream_event_callback", None),
        getattr(agent, "_delegation_event_callback", None),
    )
    agent.set_stream_event_callback(trace)
    agent.set_delegation_event_callback(delegation)
    try:
        yield
    finally:
        agent.set_stream_event_callback(previous[0])
        agent.set_delegation_event_callback(previous[1])


def _cost_limit_problem(
    task: HarnessTask, model: Dict[str, str]
) -> Optional[AdapterOutcome]:
    """A cost limit needs a model MemoRizz can price. A local model has no API
    charges, so any limit holds; a cloud model without a rate card can't be
    checked, so the run is refused rather than left unlimited."""
    if task.budget.max_cost_usd is None:
        return None
    provider = str(model.get("provider") or "").lower()
    name = str(model.get("model") or "")
    from ..llms.llm_factory import LOCAL_LLM_PROVIDERS
    from ..observability.pricing import DEFAULT_PRICING

    if provider in LOCAL_LLM_PROVIDERS:
        return None

    quote = DEFAULT_PRICING.quote(
        {
            "provider": provider,
            "model": name,
            "input_tokens": 1000,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 100,
        }
    )
    if quote.get("cost_status") == "calculated":
        return None
    label = (
        f"{provider}/{name}" if provider and name else (name or provider or "its model")
    )
    return AdapterOutcome(
        error_code="cost_budget_unpriced_model",
        error=(
            f"This MemAgent runs on {label}, which MemoRizz has no price for, "
            "so the run's cost limit can't be enforced."
        ),
        remediation=(
            "Remove the cost limit for this run, or give the agent a model "
            "MemoRizz can price."
        ),
    )


def _agent_model(agent: Any) -> Dict[str, str]:
    """The model a MemAgent runs on, and its provider, when it says."""
    config = getattr(agent, "llm_config", None) or {}
    if not isinstance(config, dict):
        config = {}
    provider = getattr(agent, "model", None)
    running = provider.get_config() if hasattr(provider, "get_config") else None
    if isinstance(running, dict) and running.get("model"):
        config = running
    name = (
        config.get("model")
        or config.get("deployment_name")
        or getattr(provider, "model", None)
        or getattr(provider, "model_name", None)
    )
    result: Dict[str, str] = {}
    if isinstance(name, str) and name:
        result["model"] = name
    if config.get("provider"):
        result["provider"] = str(config["provider"])
    return result


# Providers a MemAgent can run on, for "provider/model" names.
_MEMAGENT_PROVIDERS = {
    "openai",
    "anthropic",
    "ollama",
    "azure",
    "huggingface",
    "mlx",
    "local-openai",
    "deepseek",
}


def _saved_llm_config(provider: Any, agent_id: str) -> Dict[str, Any]:
    record = provider.retrieve_memagent(agent_id)
    saved = (
        record.get("llm_config")
        if isinstance(record, dict)
        else getattr(record, "llm_config", None)
    )
    return dict(saved or {})


def _memagent_model_config(provider: Any, agent_id: str, model: str) -> Dict[str, Any]:
    """The saved agent's LLM config with another model, for one run only.

    ``provider/model`` also switches provider. The saved window size belongs
    to the old model, so the new one gets its own default.
    """
    config = _saved_llm_config(provider, agent_id)
    name = str(model).strip()
    prefix, _, rest = name.partition("/")
    if rest and prefix.lower() in _MEMAGENT_PROVIDERS:
        if prefix.lower() != str(config.get("provider") or "").lower():
            config = {"provider": prefix.lower()}
        name = rest
    config["model"] = name
    config.pop("deployment_name", None)
    config.pop("context_window_tokens", None)
    return config


class _NoInternetProvider(RuntimeError):
    """Network is Full but MemoRizz has no way to reach the web."""


@contextmanager
def _run_web_access(agent: Any, network: str) -> Iterator[Optional[str]]:
    """Give a MemAgent web access for one run only when Network is Full.

    Yields the internet provider's name, or None when the run has no web
    access. A saved agent keeps its own provider; one without gets MemoRizz's
    default for the run. With Network off, the agent's web tools are hidden
    for the run and come back afterwards.
    """
    from ..internet_access.providers.offline import OfflineInternetProvider

    manager = getattr(agent, "internet_access_manager", None)
    own = getattr(manager, "provider", None) if manager is not None else None
    if network == "full":
        if own is not None and not isinstance(own, OfflineInternetProvider):
            yield manager.get_provider_name()
            return
        from .. import internet_access

        provider = internet_access.get_default_internet_access_provider()
        if isinstance(provider, OfflineInternetProvider):
            raise _NoInternetProvider(
                "Network is Full, but this agent has no internet provider and "
                "MemoRizz has no search key to lend it."
            )
        if own is None:
            agent.with_internet_access_provider(provider)
            try:
                yield agent.internet_access_manager.get_provider_name()
            finally:
                # Detaching closes the provider this run borrowed.
                agent.with_internet_access_provider(None)
            return
        # The agent's placeholder provider steps aside for the run.
        manager.provider = provider
        try:
            yield provider.get_provider_name()
        finally:
            manager.provider = own
            try:
                provider.close()
            except Exception:
                pass
        return
    if own is None:
        yield None
        return
    # Hide the tools without closing the agent's own provider.
    manager.provider = None
    agent._unregister_internet_access_tools()
    try:
        yield None
    finally:
        manager.provider = own
        agent._register_internet_access_tools()


def _memagent_usage(agent: Any, model: Dict[str, str]) -> tuple:
    """A MemAgent run's usage and cost.

    Whole-run totals (every model call, in the normalized input/output keys
    the budget checks read); the provider's last-call usage is only a
    fallback for agents without run accounting. Priced per call at the
    provider's list rates (what an API key is billed); left unpriced when any
    call couldn't be, such as a local model.
    """
    run_usage = getattr(agent, "get_last_run_usage", None)
    usage: Dict[str, Any] = dict(run_usage() or {}) if callable(run_usage) else {}
    usage_getter = getattr(getattr(agent, "model", None), "get_last_usage", None)
    if not usage and callable(usage_getter):
        try:
            candidate = usage_getter() or {}
        except Exception:
            candidate = {}
        if isinstance(candidate, dict):
            usage = dict(candidate)
    if model.get("model"):
        usage.setdefault("model", model["model"])
    cost = usage.pop("cost_usd", None)
    if cost is not None and not usage.get("unpriced_calls"):
        usage["cost_basis"] = "list_rate"
    else:
        cost = None
    return usage, cost


@contextmanager
def _cancellable(cancel_event: Any) -> Iterator[threading.Event]:
    """Run a MemAgent turn that stops when the harness run is canceled.

    MemAgent checks the current cancellation token between model calls and
    tool calls, and a delegate running on a harness cancels its own run when
    the token fires, so canceling a coordinator stops its delegates too. The
    yielded event is set once the turn was stopped this way.
    """
    from ..streaming import CancellationToken, StreamCancelled, current_cancellation

    token = CancellationToken()
    stopped = threading.Event()
    finished = threading.Event()

    def watch() -> None:
        # Cancel requests from another process arrive on cancel_event only.
        while not finished.is_set():
            if cancel_event.is_set():
                token.cancel()
                return
            finished.wait(0.2)

    watcher = threading.Thread(
        target=watch, name="memorizz-memagent-cancel", daemon=True
    )
    watcher.start()
    scope = current_cancellation.set(token)
    try:
        yield stopped
    except StreamCancelled:
        stopped.set()
    finally:
        finished.set()
        current_cancellation.reset(scope)
    if token.cancelled:
        stopped.set()


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
                "cost_reporting": usage_reporting,
                "token_reporting": usage_reporting,
                "network_policy": "internet_provider",
                "network_modes": ["none", "full"],
                "task_tool_policy": False,
                "agent_id": getattr(self.agent, "agent_id", None),
                "cancellation": "cooperative",
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
        provider = getattr(self.agent, "memory_provider", None)
        own_id = getattr(self.agent, "agent_id", None)
        wanted = getattr(task, "agent_id", None)
        if (
            wanted
            and own_id
            and str(wanted) != str(own_id)
            and callable(getattr(provider, "retrieve_memagent", None))
        ):
            # The task names another saved agent (a chat's /compare memagent
            # step): run that one, as a standalone run would, not the agent
            # this harness happens to be attached to.
            return PersistedMemAgentHarness(provider).run(
                task,
                workspace=workspace,
                context_pack=context_pack,
                emit=emit,
                cancel_event=cancel_event,
            )
        refused = _cost_limit_problem(task, _agent_model(self.agent))
        if refused is not None:
            return refused
        try:
            web = _run_web_access(self.agent, task.permissions.network)
        except _NoInternetProvider as exc:
            return AdapterOutcome(
                error_code="internet_provider_unavailable",
                error=str(exc),
                remediation=(
                    "Give the agent an internet provider on its Agents page, or "
                    "set TAVILY_API_KEY or FIRECRAWL_API_KEY for MemoRizz."
                ),
            )
        model = _agent_model(self.agent)
        with web as provider_name:
            emit(
                HarnessEvent(
                    task.run_id,
                    HarnessEventType.STATUS,
                    {"status": "running", "web_access": provider_name, **model},
                )
            )
            context = dict(task.context)
            if context_pack.rendered:
                context["harness_memory_context"] = context_pack.rendered
                context["harness_memory_source_ids"] = context_pack.source_ids
            try:
                with _live_trace(self.agent, task.run_id, emit), _cancellable(
                    cancel_event
                ) as stopped:
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
                            # Delegates that run on harnesses share this run's
                            # workspace and what it was approved for.
                            "harness_parent": {
                                "run_id": task.run_id,
                                "workspace": str(workspace),
                                "permissions": task.permissions.to_dict(),
                            },
                        },
                        observability_context={
                            "run_id": task.run_id,
                            "memory_id": task.memory_id,
                            "thread_id": task.thread_id,
                            "user_id": task.user_id,
                        },
                    )
            except Exception as exc:
                from .security import redact

                usage, cost = _memagent_usage(self.agent, model)
                return AdapterOutcome(
                    error_code="memagent_run_failed",
                    error=f"{type(exc).__name__}: {redact(str(exc))}",
                    usage=usage,
                    cost_usd=cost,
                    exit_code=1,
                )
            if stopped.is_set():
                # What it spent before stopping still counts.
                usage, cost = _memagent_usage(self.agent, model)
                return AdapterOutcome(
                    error_code="canceled",
                    error="Harness run was canceled",
                    usage=usage,
                    cost_usd=cost,
                )
        emit(
            HarnessEvent(
                task.run_id,
                HarnessEventType.MESSAGE,
                {"role": "assistant", "text": str(response)},
            )
        )
        usage, cost = _memagent_usage(self.agent, model)
        return AdapterOutcome(
            final_response=str(response),
            usage=usage,
            cost_usd=cost,
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
                None
                if available and not error
                else "Create and save a MemAgent, then select it with agent_id."
                if available
                else "Configure a memory provider that supports saved MemAgents."
            ),
            structured_events=True,
            resume=False,
            mcp=True,
            per_action_approvals=True,
            requires_external_isolation=False,
            # A run reports its whole-run tokens, and its cost when the model
            # can be priced (a cost limit on an unpriced cloud model is
            # refused when the run starts; local models have no API charges).
            usage_reporting=True,
            metadata={
                "cost_reporting": True,
                "token_reporting": True,
                "network_policy": "internet_provider",
                "network_modes": ["none", "full"],
                "task_tool_policy": False,
                "requires_agent_id": True,
                "explicit_only": True,
                "saved_agent_count": saved_agent_count,
                "cancellation": "cooperative",
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

            overrides: Dict[str, Any] = {}
            if task.model:
                overrides["llm_config"] = _memagent_model_config(
                    self.memory_provider, task.agent_id, task.model
                )
            agent = MemAgent.load(
                task.agent_id,
                memory_provider=self.memory_provider,
                meta_harness=False,
                meta_harness_mode=None,
                capture_memory_history=True,
                capture_context_snapshots=True,
                **overrides,
            )
            if overrides:
                # Saves during this run (a new memory registered, say) keep
                # the agent's own model, not the one borrowed for the run.
                agent._pinned_llm_config = _saved_llm_config(
                    self.memory_provider, task.agent_id
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


def codex_list_cost(model: Optional[str], usage: Dict[str, Any]) -> Optional[float]:
    """A Codex turn's cost at OpenAI list rates, or None when it can't be priced."""
    if not model:
        return None
    from ..observability.pricing import DEFAULT_PRICING

    quote = DEFAULT_PRICING.quote(
        {
            "provider": "openai",
            "model": model,
            "service_tier": usage.get("service_tier"),
            "input_tokens": usage.get("input_tokens"),
            "cached_tokens": usage.get("cached_input_tokens") or 0,
            "cache_write_tokens": usage.get("cache_write_input_tokens") or 0,
            # Reasoning tokens are already part of output_tokens.
            "output_tokens": usage.get("output_tokens"),
        }
    )
    if quote.get("cost_status") != "calculated":
        return None
    return float(quote["cost_usd"])


class _CodexSubagentWatcher:
    """Follow the sub-agents a Codex run starts, from their session files.

    ``codex exec --json`` reports only that the parent waits; each sub-agent
    writes its own rollout under ``$CODEX_HOME/sessions`` naming its parent.
    This reads those while the run goes and reports each sub-agent and its
    searches, reasoning and answer, nested under it.
    """

    def __init__(self, run_id: str, emit: EventSink, sessions: Path, started: float):
        self.run_id = run_id
        self.emit = emit
        self.sessions = sessions
        self.started = started
        self.threads: set = set()
        self.children: Dict[str, Dict[str, Any]] = {}
        self.skipped: set = set()
        self.spawns: set = set()  # sub-agent spawn calls Codex reported
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def saw_spawn(self, item_id: Any) -> None:
        self.spawns.add(str(item_id))

    def start(self, thread_id: Any) -> None:
        if self._thread is not None:
            return
        self.threads.add(str(thread_id))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            try:
                self.poll()
            except Exception:
                pass
        if self.spawns and not self.children:
            # Session files are Codex internals; say so rather than show
            # sub-agents as if they did nothing.
            self.emit(
                HarnessEvent(
                    self.run_id,
                    HarnessEventType.STATUS,
                    {
                        "status": "subagent_details_unavailable",
                        "notice": (
                            f"Codex started {len(self.spawns)} sub-agent"
                            f"{'' if len(self.spawns) == 1 else 's'}, but their session "
                            f"files under {self.sessions} were missing or in a format "
                            "this MemoRizz version doesn't read, so their own steps "
                            "aren't shown."
                        ),
                    },
                )
            )

    def _loop(self) -> None:
        while not self._stop.wait(1.5):
            try:
                self.poll()
            except Exception:
                pass

    def _candidates(self) -> List[Path]:
        # Dated folders, in local time or UTC, around midnight too.
        now, utc = datetime.now(), datetime.now(timezone.utc)
        days = {
            (moment + timedelta(days=shift)).strftime("%Y/%m/%d")
            for moment in (now, utc)
            for shift in (-1, 0, 1)
        }
        found: List[Path] = []
        for day in days:
            folder = self.sessions / day
            if folder.is_dir():
                found.extend(folder.glob("rollout-*.jsonl"))
        if not found and self.sessions.is_dir():
            # Another layout: look anywhere under sessions.
            found.extend(self.sessions.rglob("rollout-*.jsonl"))
        recent = []
        for path in found:
            try:
                if (
                    str(path) not in self.skipped
                    and path.stat().st_mtime >= self.started - 5
                ):
                    recent.append(path)
            except OSError:
                continue
        # Names start with the creation time, so a sub-agent comes before its own.
        return sorted(recent)

    def poll(self) -> None:
        for path in self._candidates():
            key = str(path)
            if key not in self.children:
                meta = self._meta(path)
                spawn = self._spawn(meta or {})
                parent = str(spawn.get("parent_thread_id") or "")
                if not meta or parent not in self.threads:
                    if meta and not parent:  # not a sub-agent at all
                        self.skipped.add(key)
                    continue
                child = str(meta.get("id") or "")
                self.threads.add(child)
                self.children[key] = {
                    "id": child,
                    "offset": 0,
                    "line": 0,
                    "start": int(meta.get("subagent_history_start_ordinal") or 0),
                }
                role = str(spawn.get("agent_path") or "").rstrip("/").split("/")[-1]
                name = spawn.get("agent_nickname") or meta.get("agent_nickname")
                label = " · ".join(str(part) for part in (name, role) if part)
                data = {
                    "id": child,
                    "name": "spawn_agent",
                    "type": "codex_subagent",
                    "subagent": True,
                    "status": "in_progress",
                    "input": {"description": label or "sub-agent"},
                }
                if parent in {c["id"] for c in self.children.values()}:
                    data["parent_id"] = parent
                self.emit(HarnessEvent(self.run_id, HarnessEventType.TOOL_CALL, data))
            self._read(path, self.children[key])

    @staticmethod
    def _spawn(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Where a sub-agent's session names its parent. An ordinary session's
        source is a plain string such as "exec"."""
        source = meta.get("source")
        subagent = source.get("subagent") if isinstance(source, dict) else None
        for found in (
            subagent.get("thread_spawn") if isinstance(subagent, dict) else None,
            subagent,
            source if isinstance(source, dict) else None,
            meta,
        ):
            if isinstance(found, dict) and found.get("parent_thread_id"):
                return found
        return {}

    @staticmethod
    def _kind(value: Any) -> str:
        """Compare record kinds whatever their case: AgentMessage, agent_message."""
        return re.sub(r"[^a-z]", "", str(value or "").lower())

    @staticmethod
    def _meta(path: Path) -> Optional[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                first = json.loads(handle.readline() or "{}")
        except (OSError, ValueError):
            return None
        if _CodexSubagentWatcher._kind(first.get("type")) != "sessionmeta":
            return None
        return first.get("payload") or {}

    def _read(self, path: Path, child: Dict[str, Any]) -> None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(child["offset"])
                while True:
                    line = handle.readline()
                    if not line or not line.endswith("\n"):
                        break
                    child["offset"] = handle.tell()
                    index = child["line"]
                    child["line"] += 1
                    if index < child["start"]:
                        continue  # the parent's history the sub-agent was forked with
                    try:
                        self._record(child["id"], json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            return

    def _record(self, child: str, row: Dict[str, Any]) -> None:
        payload = row.get("payload") or {}
        if self._kind(row.get("type")) != "eventmsg" or not isinstance(payload, dict):
            return
        kind = self._kind(payload.get("type"))
        if kind == "taskcomplete":
            self.emit(
                HarnessEvent(
                    self.run_id,
                    HarnessEventType.TOOL_RESULT,
                    {
                        "tool_use_id": child,
                        "content": str(payload.get("last_agent_message") or "done")[
                            :4000
                        ],
                        "is_error": False,
                    },
                )
            )
            return
        if kind != "itemcompleted" or str(payload.get("thread_id")) != child:
            return
        item = payload.get("item") or {}
        item_type = self._kind(item.get("type"))
        if item_type == "agentmessage":
            text = "\n".join(
                str(part.get("text") or "")
                for part in item.get("content") or []
                if isinstance(part, dict)
            ).strip()
            if text:
                self.emit(
                    HarnessEvent(
                        self.run_id,
                        HarnessEventType.MESSAGE,
                        {"role": "assistant", "text": text, "parent_id": child},
                    )
                )
        elif item_type == "reasoning":
            text = "\n".join(
                str(part) for part in item.get("summary_text") or []
            ).strip()
            self.emit(
                HarnessEvent(
                    self.run_id,
                    HarnessEventType.REASONING,
                    {"text": text, "hidden": not text, "parent_id": child},
                )
            )
        elif item_type in {"extension", "commandexecution", "mcptoolcall", "websearch"}:
            name = item.get("kind") or item.get("type")
            data = {
                "id": item.get("id"),
                "name": "web_search" if name == "web.search" else name,
                "type": "web_search" if name == "web.search" else item_type,
                "status": "completed",
                "parent_id": child,
            }
            if item.get("query"):
                data["query"] = item["query"]
            if item.get("command"):
                data["input"] = {"command": item["command"]}
            self.emit(HarnessEvent(self.run_id, HarnessEventType.TOOL_CALL, data))


class CodexHarness(SubprocessHarness):
    name = "codex"
    executable = "codex"
    node_cli = True
    auth_environment = ("CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_HOME")
    authentication_remediation = (
        "Set OPENAI_API_KEY or CODEX_API_KEY, or authenticate the CLI with "
        "`codex login`, then rerun `memorizz harness doctor codex`."
    )
    # Model catalogs read from `codex debug models`, per executable path.
    _catalogs: Dict[str, List[str]] = {}
    _catalog_lock = threading.Lock()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._run_models: Dict[str, Optional[str]] = {}
        self._run_models_lock = threading.Lock()
        # Runs that may start sub-agents, followed from their session files.
        self._watchers: Dict[str, "_CodexSubagentWatcher"] = {}

    def model_catalog(self) -> List[str]:
        """Models this Codex login can use, preferred first, read locally.

        Runs ignore the user's Codex config, so without an explicit model the
        CLI picks the catalog's first entry. Knowing it lets MemoRizz name the
        model and price the run.
        """
        command = shutil.which(self.command) or self.command
        with self._catalog_lock:
            if command in self._catalogs:
                return list(self._catalogs[command])
        slugs: List[str] = []
        try:
            result = subprocess.run(
                [command, "debug", "models"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                env=self._probe_environment(),
            )
            entries = json.loads(result.stdout or "{}").get("models") or []
            listed = [
                entry
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("slug")
                and entry.get("visibility", "list") != "hide"
            ]
            listed.sort(key=lambda entry: entry.get("priority", 1_000))
            slugs = [str(entry["slug"]) for entry in listed]
        except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError):
            slugs = []
        with self._catalog_lock:
            self._catalogs[command] = slugs
        return list(slugs)

    def _resolve_model(self, task: HarnessTask) -> Optional[str]:
        if task.model or self.default_model:
            return str(task.model or self.default_model)
        catalog = self.model_catalog()
        return catalog[0] if catalog else None

    def run(self, task: HarnessTask, **kwargs: Any) -> AdapterOutcome:
        with self._run_models_lock:
            self._run_models[task.run_id] = self._resolve_model(task)
        watcher = None
        if task.permissions.allow_subagents and callable(kwargs.get("emit")):
            home = self._child_environment(task).get("CODEX_HOME") or os.environ.get(
                "CODEX_HOME"
            )
            sessions = Path(home or Path.home() / ".codex").expanduser() / "sessions"
            watcher = _CodexSubagentWatcher(
                task.run_id, kwargs["emit"], sessions, time.time()
            )
            with self._run_models_lock:
                self._watchers[task.run_id] = watcher
        try:
            return super().run(task, **kwargs)
        finally:
            with self._run_models_lock:
                self._run_models.pop(task.run_id, None)
                self._watchers.pop(task.run_id, None)
            if watcher is not None:
                watcher.stop()

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
                env=self._probe_environment(),
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
            models=self.model_catalog() if values["available"] else [],
            metadata={
                "cost_reporting": True,
                "cost_basis": (
                    "Estimated at OpenAI list rates from reported token usage. On a "
                    "ChatGPT plan Codex is billed by the plan, not per token."
                ),
                "token_reporting": True,
                "network_policy": "codex_web_search",
                "network_modes": ["none", "full"],
                "subagents": "codex_multi_agent",
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
        # Sub-agents hang off a saved parent session; an ephemeral run's
        # spawns fail with "no thread with id".
        if not task.metadata.get("persist_session", False) and not (
            task.permissions.allow_subagents
        ):
            command.insert(3, "--ephemeral")
        with self._run_models_lock:
            model = self._run_models.get(task.run_id)
        model = model or self._resolve_model(task)
        if model:
            command.extend(["--model", model])
        output_schema_path = task.metadata.get("_memorizz_output_schema_path")
        if output_schema_path:
            command.extend(["--output-schema", str(output_schema_path)])
        # Trusted SDK callers may supply non-policy Codex tuning, but the
        # host-owned safety layer is appended afterwards so it cannot be
        # weakened. Project-scoped Codex config can otherwise enable hooks,
        # web search, extra MCP servers, egress, or additional writable roots.
        for value in task.metadata.get("codex_config", []) or []:
            command.extend(["--config", str(value)])
        # Network Full turns on Codex's live web search and lets commands in
        # an editing run reach the network; otherwise both stay off.
        web = task.permissions.network == "full"
        policy_overrides = [
            'approval_policy="never"',
            'web_search="live"' if web else 'web_search="disabled"',
            f"tools.web_search={'true' if web else 'false'}",
            "features.skill_mcp_dependency_install=false",
            "hooks={}",
            "mcp_servers={}",
            f"sandbox_workspace_write.network_access={'true' if web else 'false'}",
            # Codex starts sub-agents only when the run allows subagents.
            f"features.multi_agent={'true' if task.permissions.allow_subagents else 'false'}",
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
            with self._run_models_lock:
                model = self._run_models.get(run_id)
                watcher = self._watchers.get(run_id)
            if watcher is not None and payload.get("thread_id"):
                watcher.start(payload["thread_id"])
            status = {**checkpoint, **({"model": model} if model else {})}
            events.append(HarnessEvent(run_id, HarnessEventType.STATUS, status))
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
                # Codex reports these when they start and when they finish.
                if not item.get("status"):
                    item = {
                        **item,
                        "status": "completed"
                        if event_type == "item.completed"
                        else "in_progress",
                    }
                events.append(HarnessEvent(run_id, HarnessEventType.TOOL_CALL, item))
            elif kind == "reasoning":
                text = str(item.get("text") or _text_blocks(item))
                if text:
                    events.append(
                        HarnessEvent(
                            run_id,
                            HarnessEventType.REASONING,
                            {"id": item.get("id"), "text": text},
                        )
                    )
            elif kind == "collab_tool_call":
                # Codex starting or talking to one of its own subagents.
                if "spawn" in str(item.get("tool") or ""):
                    with self._run_models_lock:
                        watcher = self._watchers.get(run_id)
                    if watcher is not None:
                        watcher.saw_spawn(item.get("id") or len(watcher.spawns))
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.TOOL_CALL,
                        {
                            **item,
                            "name": str(item.get("tool") or "subagent"),
                            "subagent": True,
                            "input": {
                                key: item.get(key)
                                for key in ("prompt", "receiver_thread_ids")
                                if item.get(key)
                            },
                        },
                    )
                )
            else:
                events.append(HarnessEvent(run_id, HarnessEventType.LOG, payload))
        elif event_type == "turn.completed":
            usage = dict(payload.get("usage") or {})
            with self._run_models_lock:
                model = self._run_models.get(run_id)
            if model:
                usage["model"] = model
            cost = codex_list_cost(model, usage)
            if cost is not None:
                updates["cost_usd"] = cost
                usage["cost_basis"] = "list_rate_estimate"
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
    node_cli = True
    name = "claude-code"
    executable = "claude"
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
    # Flags whose presence selects restricted mode (see _restricted_mode).
    _RESTRICTED_FLAGS = ("--restricted", "--tools", "--strict-mcp-config")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cli_flags: Optional[frozenset] = None
        # The model each run reported at start-up, for its usage record.
        self._models: Dict[str, str] = {}
        self._models_lock = threading.Lock()

    def _detect_cli_flags(self, command: Optional[str]) -> frozenset:
        """Which isolation flags this Claude Code build supports, from --help."""
        if not command:
            return frozenset()
        try:
            result = subprocess.run(
                [command, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env=build_child_environment(node_command=command),
            )
        except (OSError, subprocess.TimeoutExpired):
            return frozenset()
        text = f"{result.stdout}\n{result.stderr}"
        return frozenset(
            flag for flag in self._RESTRICTED_FLAGS if re.search(rf"{flag}\b", text)
        )

    def _restricted_mode(self) -> bool:
        """--bare loads only Bash, Edit and Read, so Glob, Grep and the web tools
        never exist there. Builds with --restricted and --tools load exactly the
        named tools, confine file tools to the workspace and ignore user,
        project and local settings; MemoRizz uses that mode when available and
        falls back to --bare otherwise."""
        return bool(self._cli_flags) and set(self._RESTRICTED_FLAGS) <= set(
            self._cli_flags
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
        if values["available"] and not probe_error:
            self._cli_flags = self._detect_cli_flags(values.get("command"))
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
                "subagents": "task_tool",
                "forbidden_tools": ["bash"],
                "task_tool_policy": True,
                "output_schema": True,
                "authentication_configured": authenticated,
                "authentication_mode": "environment_or_cloud_provider",
                "required_environment_any": list(self.auth_environment),
                "tool_isolation": "restricted" if self._restricted_mode() else "bare",
            },
        )

    def _child_environment(self, task: HarnessTask) -> Dict[str, str]:
        environment = super()._child_environment(task)
        if self._restricted_mode():
            # Outside --bare, Claude Code would read the operator's own settings,
            # login, plugins and memory; a throwaway config directory keeps the
            # run isolated and authenticated only by the allowed environment.
            run_dir = task.metadata.get("_memorizz_run_dir")
            base = Path(run_dir) if run_dir else Path(tempfile.mkdtemp())
            config_dir = base / "claude-config"
            config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            environment["CLAUDE_CONFIG_DIR"] = str(config_dir)
            environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return environment

    def build_command(
        self, task: HarnessTask, *, workspace: Path, prompt: str
    ) -> List[str]:
        restricted = self._restricted_mode()
        command = [
            self.command,
            *(["--restricted", "--strict-mcp-config"] if restricted else ["--bare"]),
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
        # Subagents run with the same tools as the run itself.
        if task.permissions.allow_subagents and "Task" not in allowed:
            allowed.append("Task")
        if task.permissions.mcp_access != "none":
            allowed.append("mcp__memorizz__*")
        if restricted:
            # Load exactly these built-in tools; MCP tools stay permission rules.
            command.extend(
                [
                    "--tools",
                    ",".join(tool for tool in allowed if not tool.startswith("mcp__")),
                ]
            )
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
            command.extend(["--mcp-config", str(mcp_config)])
            if not restricted:
                command.append("--strict-mcp-config")
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
            if payload.get("model"):
                with self._models_lock:
                    self._models[run_id] = str(payload["model"])
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
            # Events from a subagent carry the Task/Agent call that started it.
            parent = (
                {"parent_id": payload.get("parent_tool_use_id")}
                if payload.get("parent_tool_use_id")
                else {}
            )
            content = message.get("content") if isinstance(message, dict) else []
            for block in content or []:
                if isinstance(block, dict) and block.get("type") in {
                    "thinking",
                    "redacted_thinking",
                }:
                    # Claude Code often sends the block with its text left out;
                    # the step still shows that the model reasoned there.
                    thinking = str(block.get("thinking") or "").strip()
                    events.append(
                        HarnessEvent(
                            run_id,
                            HarnessEventType.REASONING,
                            {"text": thinking, "hidden": not thinking, **parent},
                        )
                    )
            text = _text_blocks(message)
            if text:
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.MESSAGE,
                        {"role": "assistant", "text": text, **parent},
                    )
                )
            for block in content or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    data = {**block, **parent}
                    if block.get("name") in {"Task", "Agent"}:
                        data["subagent"] = True
                    events.append(
                        HarnessEvent(run_id, HarnessEventType.TOOL_CALL, data)
                    )
        elif kind == "user":
            events.append(HarnessEvent(run_id, HarnessEventType.TOOL_RESULT, payload))
        elif kind == "result":
            response = str(payload.get("result") or "")
            updates["final_response"] = response
            updates["cost_usd"] = payload.get("total_cost_usd")
            updates["usage"] = dict(payload.get("usage") or {})
            with self._models_lock:
                model = self._models.pop(run_id, None)
            # `usage` covers only the last model call; `modelUsage` has the
            # whole run per model (Claude Code also calls a small model itself).
            per_model = {
                name: value
                for name, value in (payload.get("modelUsage") or {}).items()
                if isinstance(value, dict)
            }
            if per_model:

                def total(key: str) -> int:
                    return sum(
                        int(value.get(key) or 0)
                        for value in per_model.values()
                        if str(value.get(key) or 0).isdigit()
                    )

                updates["usage"] = {
                    "input_tokens": total("inputTokens"),
                    "output_tokens": total("outputTokens"),
                    "cache_read_input_tokens": total("cacheReadInputTokens"),
                    "cache_creation_input_tokens": total("cacheCreationInputTokens"),
                    "models": {
                        name: {
                            "input_tokens": value.get("inputTokens"),
                            "output_tokens": value.get("outputTokens"),
                            "cache_read_input_tokens": value.get(
                                "cacheReadInputTokens"
                            ),
                            "cache_creation_input_tokens": value.get(
                                "cacheCreationInputTokens"
                            ),
                            "cost_usd": value.get("costUSD"),
                        }
                        for name, value in per_model.items()
                    },
                }
                model = model or max(
                    per_model,
                    key=lambda name: float(per_model[name].get("costUSD") or 0),
                )
            if model:
                updates["usage"]["model"] = model
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
                # A wrapper container still reaches the model API over the
                # internet, so only "full" can be honoured there.
                "network_modes": ["full"]
                if self.external_isolation
                else ["none", "restricted", "full"],
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

    def _child_environment(self, task: HarnessTask) -> Dict[str, str]:
        environment = super()._child_environment(task)
        model = task.model or self.default_model
        if model:
            environment["LLM_MODEL"] = str(model)
        # The isolation wrapper mounts the workspace read-only unless edits
        # were approved for this run.
        environment["MEMORIZZ_WORKSPACE_WRITABLE"] = (
            "1" if task.writes_workspace else "0"
        )
        return environment

    @staticmethod
    def _content_text(content: Any) -> str:
        return "\n".join(
            str(part.get("text") or "")
            for part in content or []
            if isinstance(part, dict)
        ).strip()

    def _parse_sdk_event(
        self, run_id: str, payload: Dict[str, Any]
    ) -> tuple[List[HarnessEvent], Dict[str, Any]]:
        """Events from the OpenHands CLI 1.x (``kind``-tagged SDK events)."""
        kind = str(payload.get("kind") or "")
        events: List[HarnessEvent] = []
        updates: Dict[str, Any] = {}
        if kind == "MessageEvent" and payload.get("source") == "agent":
            message = payload.get("llm_message") or {}
            if message.get("reasoning_content"):
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.REASONING,
                        {"text": message["reasoning_content"]},
                    )
                )
            text = self._content_text(message.get("content"))
            if text:
                updates["final_response"] = text
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.MESSAGE,
                        {"role": "assistant", "text": text},
                    )
                )
        elif kind == "ActionEvent":
            if payload.get("reasoning_content"):
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.REASONING,
                        {"text": payload["reasoning_content"]},
                    )
                )
            action = {
                key: value
                for key, value in (payload.get("action") or {}).items()
                if key != "kind" and value not in (None, "")
            }
            tool = str(payload.get("tool_name") or "")
            if tool in {"terminal", "execute_bash", "bash"}:
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.COMMAND,
                        {
                            "id": payload.get("tool_call_id"),
                            "command": action.get("command"),
                            "status": "in_progress",
                        },
                    )
                )
            else:
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.TOOL_CALL,
                        {
                            "id": payload.get("tool_call_id"),
                            "name": tool,
                            "input": action,
                        },
                    )
                )
        elif kind == "ObservationEvent":
            observation = payload.get("observation") or {}
            events.append(
                HarnessEvent(
                    run_id,
                    HarnessEventType.TOOL_RESULT,
                    {
                        "tool_use_id": payload.get("tool_call_id"),
                        "content": self._content_text(observation.get("content"))[
                            :4000
                        ],
                        "is_error": bool(observation.get("is_error")),
                    },
                )
            )
        elif kind == "ConversationErrorEvent":
            error = str(
                payload.get("detail") or payload.get("code") or "OpenHands run failed"
            )
            updates.update(error_code="openhands_failed", error=error)
            events.append(
                HarnessEvent(run_id, HarnessEventType.ERROR, {"error": error})
            )
        else:
            events.append(HarnessEvent(run_id, HarnessEventType.LOG, payload))
        return events, updates

    def parse_event(
        self, run_id: str, payload: Dict[str, Any]
    ) -> tuple[List[HarnessEvent], Dict[str, Any]]:
        if payload.get("kind") and not payload.get("type"):
            return self._parse_sdk_event(run_id, payload)
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


# ---------------------------------------------------------------- DeepSeek

DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"
# USD per million tokens as (cache hit, cache miss, output), at peak and
# off-peak rates. Source: https://api-docs.deepseek.com/quick_start/pricing,
# effective 2026-09-10. Peak is 01:00-04:00 and 06:00-10:00 UTC on weekdays;
# Chinese public holidays (off-peak) are not modelled, so estimates err high.
DEEPSEEK_PRICES_AS_OF = "2026-09-10"
DEEPSEEK_PRICES = {
    "deepseek-flash": {"peak": (0.006, 0.30, 1.20), "off_peak": (0.003, 0.15, 0.60)},
    "deepseek-v4-pro": {
        "peak": (0.044, 1.32, 3.96),
        "off_peak": (0.022, 0.66, 1.98),
    },
}
# Retired names DeepSeek still accepts and bills as Flash.
DEEPSEEK_MODEL_ALIASES = {
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
}


def deepseek_peak(at: datetime) -> bool:
    at = at.astimezone(timezone.utc)
    return at.weekday() < 5 and (1 <= at.hour < 4 or 6 <= at.hour < 10)


def deepseek_cost(
    model: Any, usage: Dict[str, Any], *, at: Optional[datetime] = None
) -> Optional[float]:
    """Estimate a DeepSeek bill from Anthropic-format usage; None if unpriced."""
    name = re.sub(r"\[[^\]]*\]$", "", str(model or "")).strip().lower()
    prices = DEEPSEEK_PRICES.get(DEEPSEEK_MODEL_ALIASES.get(name, name))
    if prices is None:
        return None

    def count(key: str) -> int:
        try:
            return max(0, int(usage.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    hit, miss, output = prices[
        "peak" if deepseek_peak(at or datetime.now(timezone.utc)) else "off_peak"
    ]
    total = (
        count("cache_read_input_tokens") * hit
        + (count("input_tokens") + count("cache_creation_input_tokens")) * miss
        + count("output_tokens") * output
    )
    return round(total / 1_000_000, 8)


class DeepSeekHarness(ClaudeCodeHarness):
    """Claude Code's agent loop on DeepSeek models.

    DeepSeek documents running Claude Code against its Anthropic-compatible
    endpoint. This reuses Claude Code's structured events and MemoRizz's tool
    policy (Bash denied, edits only when approved, MCP via --strict-mcp-config)
    while the model calls go to DeepSeek. Only DEEPSEEK_API_KEY reaches the
    process: Anthropic and cloud-provider credentials are never forwarded.
    Claude Code's own cost figure uses Anthropic prices, so cost is estimated
    from DeepSeek's price table instead.
    """

    name = "deepseek"
    executable = "claude"
    auth_environment = ("DEEPSEEK_API_KEY",)
    authentication_remediation = (
        "Set DEEPSEEK_API_KEY to a DeepSeek platform key, then rerun "
        "`memorizz harness doctor deepseek`."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.default_model = self.default_model or DEEPSEEK_DEFAULT_MODEL
        self._usage: Dict[str, Dict[str, Any]] = {}
        self._usage_lock = threading.Lock()

    def _deepseek_key(self) -> str:
        return str(
            self.extra_env.get("DEEPSEEK_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or ""
        ).strip()

    def _authentication_configured(self) -> bool:
        return bool(self._deepseek_key())

    def _capabilities(self, **values: Any) -> HarnessCapabilities:
        capability = super()._capabilities(**values)
        missing_key = capability.error_code == "authentication_required"
        unavailable = capability.error_code == "harness_unavailable"
        return replace(
            capability,
            models=list(DEEPSEEK_PRICES),
            error="DEEPSEEK_API_KEY is not set." if missing_key else capability.error,
            remediation=(
                self.authentication_remediation
                if missing_key
                else (
                    f"Install Claude Code ({self.command!r}); the deepseek harness "
                    "runs its agent loop on DeepSeek models. Then rerun "
                    "`memorizz harness doctor deepseek`."
                )
                if unavailable
                else capability.remediation
            ),
            metadata={
                **capability.metadata,
                "runtime": "claude-code",
                "api_base_url": DEEPSEEK_ANTHROPIC_BASE_URL,
                "model_default": self.default_model,
                "cost_reporting": True,
                "cost_basis": (
                    f"Estimated from DeepSeek's {DEEPSEEK_PRICES_AS_OF} price table "
                    "(peak and off-peak by UTC hour)"
                ),
                "authentication_mode": "environment",
                "required_environment_any": ["DEEPSEEK_API_KEY"],
            },
        )

    def _child_environment(self, task: HarnessTask) -> Dict[str, str]:
        environment = super()._child_environment(task)
        key = self._deepseek_key()
        if not key:
            raise HarnessSecurityError("DEEPSEEK_API_KEY is required for deepseek")
        # A task's allowed_env must not smuggle Anthropic or cloud credentials
        # to a third-party endpoint.
        for name in list(environment):
            if name.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")) or name in {
                "DEEPSEEK_API_KEY",
                "AWS_BEARER_TOKEN_BEDROCK",
            }:
                environment.pop(name)
        model = str(task.model or self.default_model)
        environment.update(
            {
                "ANTHROPIC_BASE_URL": DEEPSEEK_ANTHROPIC_BASE_URL,
                "ANTHROPIC_API_KEY": key,
                "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": DEEPSEEK_DEFAULT_MODEL,
                "CLAUDE_CODE_SUBAGENT_MODEL": DEEPSEEK_DEFAULT_MODEL,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        )
        return environment

    def build_command(
        self, task: HarnessTask, *, workspace: Path, prompt: str
    ) -> List[str]:
        command = super().build_command(task, workspace=workspace, prompt=prompt)
        # Claude Code would enforce the budget with Anthropic prices; MemoRizz
        # enforces it with the DeepSeek estimate as usage streams in.
        if "--max-budget-usd" in command[: command.index("--")]:
            index = command.index("--max-budget-usd")
            del command[index : index + 2]
        return command

    def run(self, task: HarnessTask, **kwargs: Any) -> AdapterOutcome:
        with self._usage_lock:
            self._usage[task.run_id] = {
                "model": str(task.model or self.default_model),
                "messages": {},
            }
        try:
            return super().run(task, **kwargs)
        finally:
            with self._usage_lock:
                self._usage.pop(task.run_id, None)

    def parse_event(
        self, run_id: str, payload: Dict[str, Any]
    ) -> tuple[List[HarnessEvent], Dict[str, Any]]:
        events, updates = super().parse_event(run_id, payload)
        kind = str(payload.get("type") or "")
        with self._usage_lock:
            state = self._usage.setdefault(run_id, {"model": None, "messages": {}})
            if kind == "assistant":
                message = payload.get("message")
                message = message if isinstance(message, dict) else {}
                usage = message.get("usage")
                if isinstance(usage, dict) and message.get("id"):
                    # Claude Code repeats one API message per content block.
                    state["messages"][str(message["id"])] = usage
                    totals: Dict[str, int] = {}
                    for item in state["messages"].values():
                        for key, value in item.items():
                            if isinstance(value, int):
                                totals[key] = totals.get(key, 0) + value
                    updates["usage"] = totals
                    cost = deepseek_cost(state["model"], totals)
                    if cost is not None:
                        updates["cost_usd"] = cost
            elif kind == "result":
                usage = dict(payload.get("usage") or {})
                # Replace Claude Code's Anthropic-priced figure.
                updates["cost_usd"] = deepseek_cost(state["model"], usage)
        return events, updates


# ---------------------------------------------------------------------- pi

# Credential variables per pi provider; only the task's provider is forwarded.
PI_PROVIDER_KEYS = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "xai": ("XAI_API_KEY",),
}
# Used when MemoRizz has no pi provider or model configured: pi's own default
# provider (Google) usually has no key here, and its default Claude is older.
PI_DEFAULT_MODELS = (
    ("anthropic", "anthropic/claude-sonnet-5-5"),
    ("openai", "openai/gpt-5.5"),
    ("deepseek", None),
    ("google", None),
    ("openrouter", None),
    ("mistral", None),
    ("groq", None),
    ("xai", None),
)
PI_READ_TOOLS = ("read", "grep", "find", "ls")
PI_EDIT_TOOLS = ("edit", "write")
PI_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}


class PiHarness(SubprocessHarness):
    """pi (pi.dev): a minimal coding agent for many model providers.

    pi has no sandbox, permission prompts or MCP client, so MemoRizz enables
    only its file tools: read, grep, find and ls, plus edit and write for an
    approved edit run. ``bash`` is never enabled, so pi cannot run commands or
    reach the network beyond its model API. Its edit and write tools are not
    confined to the workspace, so edit runs also require an operator-attested
    isolation wrapper. Memory reaches pi through the prompt's context pack.
    """

    name = "pi"
    executable = "pi"
    node_cli = True
    auth_environment = ("PI_CODING_AGENT_DIR",)
    authentication_remediation = (
        "Set the provider's API key (for example DEEPSEEK_API_KEY) or sign in with "
        "`pi` and /login, then rerun `memorizz harness doctor pi`."
    )

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        external_isolation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.provider = str(provider or "").strip().lower() or None
        self.external_isolation = bool(external_isolation)
        # No update checks, telemetry or other start-up network calls.
        self.extra_env = {
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
            "PI_OFFLINE": "1",
            **self.extra_env,
        }
        self._usage: Dict[str, Dict[str, Any]] = {}
        self._usage_lock = threading.Lock()
        self._catalog: Optional[List[str]] = None

    def _has_key(self, provider: str) -> bool:
        return any(
            os.environ.get(name) or self.extra_env.get(name)
            for name in PI_PROVIDER_KEYS.get(provider, ())
        )

    def _auto_model(self) -> Optional[str]:
        """A model from the first provider with a key, when none is set."""
        if self.default_model or self.provider:
            return None
        for provider, model in PI_DEFAULT_MODELS:
            if self._has_key(provider):
                return model or None
        return None

    def _effective_model(self, task: Optional[HarnessTask] = None) -> Optional[str]:
        return (
            str((task.model if task else None) or self.default_model or "").strip()
            or self._auto_model()
        )

    def model_catalog(self) -> List[str]:
        """pi's models for the providers with a key, current ones first."""
        if self._catalog is not None:
            return self._catalog
        keys = tuple(
            name
            for provider in PI_PROVIDER_KEYS
            if self._has_key(provider)
            for name in PI_PROVIDER_KEYS[provider]
        )
        models: List[str] = []
        if keys:
            try:
                result = subprocess.run(
                    [shutil.which(self.command) or self.command, "--list-models"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                    env={
                        **build_child_environment(
                            allowed_names=keys, node_command=self.command
                        ),
                        **{
                            k: v
                            for k, v in self.extra_env.items()
                            if k.startswith("PI_")
                        },
                    },
                )
                for line in result.stdout.splitlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] in PI_PROVIDER_KEYS:
                        models.append(f"{parts[0]}/{parts[1]}")
            except (OSError, subprocess.TimeoutExpired):
                models = []
        current = re.compile(r"claude-(opus|sonnet|haiku|fable)-5|gpt-5\.[5-9]|gpt-6")
        preferred = [model for _, model in PI_DEFAULT_MODELS if model in models]
        models = preferred + sorted(
            (m for m in models if m not in preferred),
            key=lambda m: (not current.search(m), m),
        )
        self._catalog = models
        return models

    def _provider_for(self, model: Optional[str]) -> Optional[str]:
        model = str(model or "").strip()
        if "/" in model:
            return model.split("/", 1)[0].lower()
        return self.provider

    def _provider_keys(self, provider: Optional[str]) -> tuple[str, ...]:
        if provider is None:
            # pi falls back to its own configured provider, which may use any
            # of these; each process still only reads the one it needs.
            return tuple(key for keys in PI_PROVIDER_KEYS.values() for key in keys)
        return PI_PROVIDER_KEYS.get(provider, ())

    def _auth_environment(self, task: HarnessTask) -> tuple[str, ...]:
        provider = self._provider_for(self._effective_model(task))
        return (*self.auth_environment, *self._provider_keys(provider))

    def _capabilities(self, **values: Any) -> HarnessCapabilities:
        probe_error = values.get("error")
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
                "Install pi with `npm install -g @earendil-works/pi-coding-agent`, "
                "then rerun `memorizz harness doctor pi`."
                if probe_error and not values["available"]
                else None
            ),
            structured_events=True,
            resume=False,
            mcp=False,
            per_action_approvals=False,
            requires_external_isolation=False,
            usage_reporting=True,
            metadata={
                "cost_reporting": True,
                "token_reporting": True,
                "network_policy": "no_network_tools",
                "network_modes": ["none"],
                "forbidden_tools": ["bash"],
                "task_tool_policy": True,
                "output_schema": False,
                "mcp_fallback": "context_pack",
                "provider": self.provider,
                "write_requires_external_isolation": not self.external_isolation,
            },
        )

    def probe(self) -> HarnessCapabilities:
        capability = super().probe()
        if not capability.available or capability.error or not capability.command:
            return capability
        if capability.metadata.get("authentication_checked"):
            return capability
        default = self._effective_model()
        provider = self._provider_for(default)
        metadata = dict(
            capability.metadata,
            authentication_checked=True,
            default_model=default,
        )
        capability = replace(capability, models=self.model_catalog())
        updated = capability
        if provider is None:
            metadata.update(
                authentication_status="unknown",
                authentication_note=(
                    "No provider is configured for MemoRizz, so pi uses its own "
                    "default provider and stored login."
                ),
            )
            updated = replace(capability, metadata=metadata)
        else:
            try:
                # Reports whether credentials exist without calling the model.
                result = subprocess.run(
                    [
                        str(capability.command),
                        "auth",
                        "check",
                        "--provider",
                        provider,
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                    env=build_child_environment(
                        allowed_names=(
                            *self.auth_environment,
                            *self._provider_keys(provider),
                        ),
                        overrides=self.extra_env,
                        node_command=str(capability.command),
                    ),
                )
                status = str(json.loads(result.stdout or "{}").get("status") or "")
            except (OSError, subprocess.TimeoutExpired, ValueError):
                status = ""
            if status == "ready":
                metadata.update(
                    authentication_configured=True,
                    authentication_status="authenticated",
                )
                updated = replace(capability, metadata=metadata)
            elif status in {"not_ready", "invalid"}:
                metadata.update(
                    authentication_configured=False, authentication_status="missing"
                )
                updated = replace(
                    capability,
                    error_code="authentication_required",
                    error=f"pi has no usable credentials for {provider}.",
                    remediation=self.authentication_remediation,
                    metadata=metadata,
                )
            else:
                metadata.update(authentication_status="unknown")
                updated = replace(capability, metadata=metadata)
        with self._process_lock:
            self._probe_cache = updated
        return updated

    def build_command(
        self, task: HarnessTask, *, workspace: Path, prompt: str
    ) -> List[str]:
        if task.writes_workspace and not self.external_isolation:
            raise HarnessSecurityError(
                "pi's edit tools are not confined to the workspace; run edits "
                "through an isolated pi wrapper and set "
                "MEMORIZZ_PI_EXTERNAL_ISOLATION=true"
            )
        tools = list(PI_READ_TOOLS) + (
            list(PI_EDIT_TOOLS) if task.writes_workspace else []
        )
        requested = [
            str(item).strip().lower()
            for item in task.permissions.allowed_tools
            if str(item).strip()
        ]
        if requested:
            # A task may narrow the tool set, never widen it.
            tools = [tool for tool in tools if tool in requested]
        denied = {str(item).strip().lower() for item in task.permissions.denied_tools}
        tools = [tool for tool in tools if tool not in denied]
        command = [
            self.command,
            "--mode",
            "json",
            "--offline",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--no-approve",
        ]
        if not task.metadata.get("persist_session", False):
            command.append("--no-session")
        model = str(self._effective_model(task) or "").strip()
        if model and "/" not in model and self.provider:
            command.extend(["--provider", self.provider])
        elif not model and self.provider:
            command.extend(["--provider", self.provider])
        if model:
            command.extend(["--model", model])
        thinking = task.metadata.get("pi_thinking")
        if thinking is not None:
            level = str(thinking).strip().lower()
            if level not in PI_THINKING_LEVELS:
                raise ValueError(
                    "pi_thinking must be one of "
                    + ", ".join(sorted(PI_THINKING_LEVELS))
                )
            command.extend(["--thinking", level])
        if tools:
            command.extend(["--tools", ",".join(tools)])
        else:
            command.append("--no-tools")
        # The prompt stays an opaque positional after every option.
        command.extend(["--", prompt])
        return command

    def run(self, task: HarnessTask, **kwargs: Any) -> AdapterOutcome:
        try:
            return super().run(task, **kwargs)
        finally:
            with self._usage_lock:
                self._usage.pop(task.run_id, None)

    @staticmethod
    def _count(usage: Dict[str, Any], key: str) -> int:
        try:
            return max(0, int(usage.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    def _add_usage(self, run_id: str, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Accumulate one assistant message's usage; pi reports no run total."""
        cache_read = self._count(usage, "cacheRead")
        cache_write = self._count(usage, "cacheWrite") + self._count(
            usage, "cacheWrite1h"
        )
        with self._usage_lock:
            state = self._usage.setdefault(
                run_id,
                {
                    "usage": {
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_output_tokens": 0,
                    },
                    "cost": None,
                },
            )
            total = state["usage"]
            # pi's ``input`` excludes cache reads and writes; count all prompt
            # tokens so input-token budgets see the full prompt size.
            total["input_tokens"] += (
                self._count(usage, "input") + cache_read + cache_write
            )
            total["cached_input_tokens"] += cache_read
            total["cache_creation_input_tokens"] += cache_write
            total["output_tokens"] += self._count(usage, "output")
            total["reasoning_output_tokens"] += self._count(usage, "reasoning")
            cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
            if isinstance(cost.get("total"), (int, float)):
                state["cost"] = round((state["cost"] or 0.0) + float(cost["total"]), 8)
            return {"usage": dict(total), "cost_usd": state["cost"]}

    def parse_event(
        self, run_id: str, payload: Dict[str, Any]
    ) -> tuple[List[HarnessEvent], Dict[str, Any]]:
        kind = str(payload.get("type") or "")
        events: List[HarnessEvent] = []
        updates: Dict[str, Any] = {}
        if kind == "session":
            checkpoint = {"session_id": payload.get("id")}
            updates["checkpoint"] = checkpoint
            events.append(HarnessEvent(run_id, HarnessEventType.STATUS, checkpoint))
        elif kind == "message_end":
            message = payload.get("message")
            message = message if isinstance(message, dict) else {}
            # System, user and tool-result messages repeat what the events
            # below already record.
            if message.get("role") != "assistant":
                return events, updates
            text = _text_blocks(message)
            if text:
                updates["final_response"] = text
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.MESSAGE,
                        {"role": "assistant", "text": text},
                    )
                )
            usage = message.get("usage")
            if isinstance(usage, dict):
                totals = self._add_usage(run_id, usage)
                updates["usage"] = totals["usage"]
                if totals["cost_usd"] is not None:
                    updates["cost_usd"] = totals["cost_usd"]
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.USAGE,
                        {
                            **totals["usage"],
                            "cost_usd": totals["cost_usd"],
                            "provider": message.get("provider"),
                            "model": message.get("model"),
                        },
                    )
                )
            stop = str(message.get("stopReason") or "")
            if stop in {"error", "aborted"}:
                # pi exits 0 after a failed model call in JSON mode.
                error = str(message.get("errorMessage") or f"pi stopped: {stop}")
                updates.update(
                    error_code="pi_failed" if stop == "error" else "pi_aborted",
                    error=error,
                )
                events.append(
                    HarnessEvent(run_id, HarnessEventType.ERROR, {"error": error})
                )
        elif kind == "tool_execution_start":
            name = str(payload.get("toolName") or "")
            args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
            data = {"call_id": payload.get("toolCallId"), "name": name, "input": args}
            if name == "bash":
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.COMMAND,
                        {**data, "command": args.get("command")},
                    )
                )
            elif name in PI_EDIT_TOOLS:
                events.append(
                    HarnessEvent(
                        run_id,
                        HarnessEventType.FILE_CHANGE,
                        {**data, "path": args.get("path")},
                    )
                )
            else:
                events.append(HarnessEvent(run_id, HarnessEventType.TOOL_CALL, data))
        elif kind == "tool_execution_end":
            result = payload.get("result")
            events.append(
                HarnessEvent(
                    run_id,
                    HarnessEventType.TOOL_RESULT,
                    {
                        "call_id": payload.get("toolCallId"),
                        "name": payload.get("toolName"),
                        "is_error": bool(payload.get("isError")),
                        "text": _text_blocks(result)[:4_000],
                    },
                )
            )
        elif kind in {
            "auto_retry_start",
            "auto_retry_end",
            "compaction_start",
            "compaction_end",
        }:
            events.append(
                HarnessEvent(
                    run_id,
                    HarnessEventType.STATUS,
                    {key: payload.get(key) for key in payload if key != "messages"},
                )
            )
            if kind == "auto_retry_end" and payload.get("success") is False:
                error = str(payload.get("finalError") or "pi exhausted its retries")
                updates.update(error_code="pi_failed", error=error)
                events.append(
                    HarnessEvent(run_id, HarnessEventType.ERROR, {"error": error})
                )
        # Streaming deltas (message_update) and lifecycle records (turn_*,
        # agent_*, message_start) duplicate the content above and are skipped.
        return events, updates


# ------------------------------------------------------------------ Hermes

# First Hermes Agent release with `chat --format stream-json`.
HERMES_MIN_VERSION = (0, 21, 4)
# Credential variables per Hermes provider; only the run's provider is forwarded.
HERMES_PROVIDER_KEYS = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"),
    "openai-api": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "xai": ("XAI_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY",),
    "ollama-cloud": ("OLLAMA_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "custom": (),
}
HERMES_COMMAND_TOOLS = {"terminal", "process_manage", "execute_code"}
HERMES_EDIT_TOOLS = {"write_file", "patch"}
# Toolsets MemoRizz never enables: they run commands or code, drive a browser,
# spawn agents, or let Hermes rewrite its own memory and skills.
HERMES_FORBIDDEN_TOOLSETS = [
    "terminal",
    "code_execution",
    "browser",
    "delegation",
    "memory",
    "skills",
]


def _hermes_profiles_root() -> Optional[Path]:
    """Where per-run Hermes homes must live to share the installed runtime.

    Hermes 0.21.5 and later keep their runtime and install state under
    ``~/.hermes`` and treat a ``HERMES_HOME`` outside it as a fresh install,
    re-pointing their launcher at it; deleting that home afterwards breaks
    the install. Homes under ``~/.hermes/profiles`` share the runtime.
    """
    root = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
    if not (root / "hermes-agent").is_dir():
        return None
    profiles = root / "profiles"
    profiles.mkdir(parents=True, exist_ok=True, mode=0o700)
    return profiles


class HermesHarness(SubprocessHarness):
    """Nous Research's Hermes Agent (``hermes chat --format stream-json``).

    Every run gets its own generated ``HERMES_HOME``: no update checks, no
    borrowed Claude Code, Codex or GitHub logins, no extra title-generation
    calls, dangerous commands denied, and only the run's provider key in the
    environment. MemoRizz enables the ``file`` toolset, ``web`` with web
    access and the MemoRizz MCP server; never a terminal. Hermes has no
    read-only file toolset, so read-only runs point ``HERMES_WRITE_SAFE_ROOT``
    at an empty folder and every write fails. ``hermes -z`` is never used: it
    approves dangerous commands automatically.
    """

    name = "hermes"
    executable = "hermes"
    auth_environment = ()
    authentication_remediation = (
        "Set the provider's API key (for example OPENROUTER_API_KEY or "
        "ANTHROPIC_API_KEY) or MEMORIZZ_HERMES_PROVIDER, then rerun "
        "`memorizz harness doctor hermes`."
    )
    _authentication_failure = re.compile(
        SubprocessHarness._authentication_failure.pattern
        + r"|isn't configured yet|not connected to any ai provider",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_url = str(base_url or "").strip() or None
        provider = str(provider or "").strip().lower() or None
        # An OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama) is "custom".
        self.provider = provider or ("custom" if self.base_url else None)
        self._probe_home: Optional[Path] = None
        self._state: Dict[str, Dict[str, Any]] = {}
        self._state_lock = threading.Lock()

    # -- configuration -------------------------------------------------------

    def _provider_keys(self, provider: Optional[str]) -> tuple[str, ...]:
        if provider is None:
            # Hermes auto-detects its provider from whichever key is set.
            return tuple(k for keys in HERMES_PROVIDER_KEYS.values() for k in keys)
        return HERMES_PROVIDER_KEYS.get(provider, ())

    def _auth_environment(self, task: HarnessTask) -> tuple[str, ...]:
        return self._provider_keys(self.provider)

    def _config(
        self, *, model: Optional[str], mcp_server: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "updates": {"check": False},
            "auth": {"adopt_external_logins": False},
            "security": {"tirith_enabled": False},
            "auxiliary": {
                "title_generation": {"enabled": False},
                "background_review": {"enabled": False},
            },
            "tools": {"tool_search": {"enabled": "off"}},
            "approvals": {"single_query_mode": "deny"},
            "curator": {"enabled": False},
            "memory": {"memory_enabled": False},
        }
        block: Dict[str, Any] = {}
        if self.provider:
            block["provider"] = self.provider
        if model:
            block["default"] = model
        if self.base_url:
            block["base_url"] = self.base_url
            # Local servers ignore the key, but Hermes needs one to be set.
            block["api_key"] = "memorizz-local"
        if block:
            config["model"] = block
        if mcp_server:
            config["mcp_servers"] = {"memorizz": mcp_server}
        return config

    @staticmethod
    def _write_home(path: Path, config: Dict[str, Any]) -> Path:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        # JSON is valid YAML, so no YAML dependency is needed.
        (path / "config.yaml").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        return path

    def _probe_environment(self) -> Dict[str, str]:
        # Without its own home, `hermes --version` creates ~/.hermes and runs
        # a network update check.
        if self._probe_home is None:
            import atexit
            import shutil

            probe_dir = Path(
                tempfile.mkdtemp(
                    prefix="memorizz-hermes-probe-", dir=_hermes_profiles_root()
                )
            )
            # One per process; gone when the process ends.
            atexit.register(shutil.rmtree, probe_dir, True)
            self._probe_home = self._write_home(
                probe_dir, self._config(model=None, mcp_server=None)
            )
        environment = build_child_environment(
            allowed_names=self.auth_environment, overrides=self.extra_env
        )
        environment["HERMES_HOME"] = str(self._probe_home)
        return environment

    def _run_dir(self, task: HarnessTask) -> Path:
        run_dir = task.metadata.get("_memorizz_run_dir")
        if run_dir:
            return Path(run_dir)
        with self._state_lock:
            state = self._state.setdefault(task.run_id, {"text": []})
            if "dir" not in state:
                state["dir"] = tempfile.mkdtemp(prefix="memorizz-hermes-")
            return Path(state["dir"])

    @staticmethod
    def _mcp_server(task: HarnessTask) -> Optional[Dict[str, Any]]:
        """The generated MemoRizz MCP server, as a Hermes mcp_servers entry."""
        path = task.metadata.get("mcp_config_path")
        if task.permissions.mcp_access == "none" or not path:
            return None
        try:
            servers = json.loads(Path(path).read_text(encoding="utf-8"))["mcpServers"]
            server = dict(servers["memorizz"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return {key: server[key] for key in ("command", "args", "env") if key in server}

    # -- capability ----------------------------------------------------------

    def _capabilities(self, **values: Any) -> HarnessCapabilities:
        available = bool(values["available"])
        error = values.get("error")
        error_code = "harness_unavailable" if error and not available else None
        remediation = (
            "Install Hermes Agent (https://hermes-agent.nousresearch.com) and "
            "rerun `memorizz harness doctor hermes`."
            if error_code
            else None
        )
        version = str(values.get("version") or "")
        if available and not error:
            match = re.match(r"Hermes Agent v(\d+)\.(\d+)\.(\d+)", version)
            if match is None:
                # Meta's JavaScript engine also installs a `hermes` binary.
                available, error_code = False, "harness_unavailable"
                error = f"{values.get('command')} is not Nous Research's Hermes Agent"
                remediation = (
                    "Install Hermes Agent or set MEMORIZZ_HERMES_COMMAND to it."
                )
            elif tuple(int(part) for part in match.groups()) < HERMES_MIN_VERSION:
                available, error_code = False, "harness_unavailable"
                error = "Hermes Agent 0.21.4 or newer is needed for structured output"
                remediation = "Run `hermes update`."
        authenticated = self.provider == "custom" or any(
            str(os.environ.get(key) or self.extra_env.get(key) or "").strip()
            for key in self._provider_keys(self.provider)
            if key.endswith(("_KEY", "_TOKEN"))
        )
        if available and not error and not authenticated:
            error_code = "authentication_required"
            error = (
                f"No API key is set for Hermes provider {self.provider}."
                if self.provider
                else "No API key is set for any Hermes provider."
            )
            remediation = self.authentication_remediation
        return HarnessCapabilities(
            name=self.name,
            available=available,
            command=values.get("command"),
            version=version or None,
            error_code=error_code,
            error=error,
            remediation=remediation,
            structured_events=True,
            resume=False,
            mcp=True,
            per_action_approvals=False,
            requires_external_isolation=False,
            usage_reporting=True,
            metadata={
                "cost_reporting": False,
                "token_reporting": True,
                "network_policy": "toolset_allowlist",
                "network_modes": ["none", "full"],
                "forbidden_tools": HERMES_FORBIDDEN_TOOLSETS,
                "task_tool_policy": True,
                "output_schema": False,
                "provider": self.provider,
                "authentication_configured": authenticated,
                "write_confinement": "HERMES_WRITE_SAFE_ROOT",
            },
        )

    # -- execution -----------------------------------------------------------

    def _child_environment(self, task: HarnessTask) -> Dict[str, str]:
        environment = super()._child_environment(task)
        run_dir = self._run_dir(task)
        profiles = _hermes_profiles_root()
        if profiles is not None:
            with self._state_lock:
                state = self._state.setdefault(task.run_id, {"text": []})
                if "home" not in state:
                    state["home"] = tempfile.mkdtemp(
                        prefix=f"memorizz-{task.run_id[:8]}-", dir=profiles
                    )
                home_dir = Path(state["home"])
        else:
            home_dir = run_dir / "hermes-home"
        home = self._write_home(
            home_dir,
            self._config(
                model=task.model or self.default_model,
                mcp_server=self._mcp_server(task),
            ),
        )
        environment["HERMES_HOME"] = str(home)
        if task.writes_workspace:
            safe_root = Path(task.workspace).expanduser().resolve()
        else:
            # Hermes has no read-only file toolset: point writes at an empty
            # folder so every write_file and patch is refused.
            safe_root = run_dir / "hermes-no-writes"
            safe_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        environment["HERMES_WRITE_SAFE_ROOT"] = str(safe_root)
        return environment

    def build_command(
        self, task: HarnessTask, *, workspace: Path, prompt: str
    ) -> List[str]:
        toolsets = ["file"] + (["web"] if task.permissions.network == "full" else [])
        requested = {
            str(item).strip().lower()
            for item in task.permissions.allowed_tools
            if str(item).strip()
        }
        if requested:
            toolsets = [name for name in toolsets if name in requested]
        denied = {str(item).strip().lower() for item in task.permissions.denied_tools}
        toolsets = [name for name in toolsets if name not in denied]
        if self._mcp_server(task):
            toolsets.append("memorizz")
        if not toolsets:
            # Without -t, Hermes would enable its default toolsets.
            raise ValueError("A hermes run needs at least one allowed toolset")
        # The prompt goes in a private file, not argv.
        query = self._run_dir(task) / "hermes-query.md"
        query.write_text(prompt, encoding="utf-8")
        query.chmod(0o600)
        command = [
            self.command,
            "chat",
            "--query-file",
            str(query),
            "--format",
            "stream-json",
            "--ignore-rules",
            "--source",
            "tool",
            "--max-turns",
            str(task.budget.max_steps),
            "--run-budget",
            str(int(task.budget.max_wall_time_seconds)),
            "-t",
            ",".join(toolsets),
        ]
        model = task.model or self.default_model
        if model:
            command.extend(["-m", str(model)])
        if self.provider:
            command.extend(["--provider", self.provider])
        return command

    def run(self, task: HarnessTask, **kwargs: Any) -> AdapterOutcome:
        try:
            return super().run(task, **kwargs)
        finally:
            with self._state_lock:
                state = self._state.pop(task.run_id, None) or {}
            import shutil

            for key in ("dir", "home"):
                if state.get(key):
                    shutil.rmtree(state[key], ignore_errors=True)

    def _flush_text(self, run_id: str) -> List[HarnessEvent]:
        with self._state_lock:
            state = self._state.setdefault(run_id, {"text": []})
            text = "".join(state["text"]).strip()
            state["text"] = []
        if not text:
            return []
        return [
            HarnessEvent(
                run_id, HarnessEventType.MESSAGE, {"role": "assistant", "text": text}
            )
        ]

    def parse_event(
        self, run_id: str, payload: Dict[str, Any]
    ) -> tuple[List[HarnessEvent], Dict[str, Any]]:
        kind = str(payload.get("type") or "")
        events: List[HarnessEvent] = []
        updates: Dict[str, Any] = {}
        if kind == "system" and payload.get("subtype") == "init":
            checkpoint = {"session_id": payload.get("session_id")}
            updates["checkpoint"] = checkpoint
            events.append(
                HarnessEvent(
                    run_id,
                    HarnessEventType.STATUS,
                    {**checkpoint, "model": payload.get("model")},
                )
            )
        elif kind == "text":
            # Text streams in chunks; it becomes one message per tool turn.
            with self._state_lock:
                self._state.setdefault(run_id, {"text": []})["text"].append(
                    str(payload.get("text") or "")
                )
        elif kind == "tool_use":
            events.extend(self._flush_text(run_id))
            name = str(payload.get("name") or "")
            tool_input = payload.get("input")
            tool_input = tool_input if isinstance(tool_input, dict) else {}
            data = {
                "name": name,
                "input": tool_input,
                "call_id": payload.get("tool_call_id"),
            }
            if name in HERMES_COMMAND_TOOLS:
                data["command"] = tool_input.get("command") or tool_input.get("code")
                events.append(HarnessEvent(run_id, HarnessEventType.COMMAND, data))
            elif name in HERMES_EDIT_TOOLS:
                data["path"] = tool_input.get("path")
                events.append(HarnessEvent(run_id, HarnessEventType.FILE_CHANGE, data))
            else:
                events.append(HarnessEvent(run_id, HarnessEventType.TOOL_CALL, data))
        elif kind == "tool_result":
            output = str(payload.get("output") or "")
            events.append(
                HarnessEvent(
                    run_id,
                    HarnessEventType.TOOL_RESULT,
                    {
                        "name": payload.get("name"),
                        "is_error": bool(payload.get("is_error")),
                        "duration_ms": payload.get("duration_ms"),
                        "text": output[:4_000],
                    },
                )
            )
        elif kind == "result":
            events.extend(self._flush_text(run_id))
            text = str(payload.get("text") or "")
            if text:
                updates["final_response"] = text
            tokens = payload.get("tokens")
            if isinstance(tokens, dict):
                usage = {
                    "input_tokens": tokens.get("input"),
                    "output_tokens": tokens.get("output"),
                    "cached_input_tokens": tokens.get("cache_read"),
                    "cache_creation_input_tokens": tokens.get("cache_write"),
                    "total_tokens": tokens.get("total"),
                }
                usage = {
                    key: value for key, value in usage.items() if value is not None
                }
                updates["usage"] = usage
                events.append(HarnessEvent(run_id, HarnessEventType.USAGE, usage))
            exit_code = payload.get("exit_code")
            if exit_code not in (0, None):
                error = str(
                    payload.get("error") or text or f"Hermes exited with {exit_code}"
                )
                updates.update(
                    error_code=(
                        "hermes_interrupted" if exit_code == 130 else "hermes_failed"
                    ),
                    error=error,
                )
                events.append(
                    HarnessEvent(run_id, HarnessEventType.ERROR, {"error": error})
                )
        else:
            events.append(HarnessEvent(run_id, HarnessEventType.LOG, payload))
        return events, updates


__all__ = [
    "ClaudeCodeHarness",
    "CodexHarness",
    "DeepSeekHarness",
    "HermesHarness",
    "NativeMemAgentHarness",
    "OpenHandsHarness",
    "PersistedMemAgentHarness",
    "PiHarness",
    "deepseek_cost",
]

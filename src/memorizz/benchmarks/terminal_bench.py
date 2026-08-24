"""Harbor adapter for evaluating MemoRizz on Terminal-Bench.

The MemoRizz process and model provider run on the host. Commands execute only
inside Harbor's isolated task environment through the ``terminal_exec`` tool.
Harbor is an optional dependency; install ``memorizz[terminal-bench]`` before
importing this module.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import AgentAuthenticationError
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)

from memorizz._env_io import load_layered_env
from memorizz.benchmarks.pricing import (
    estimate_openai_text_cost,
    resolve_openai_text_pricing,
)
from memorizz.llms.openai import OpenAI
from memorizz.tooling import governed_tool


class BenchmarkSpendLimitExceeded(RuntimeError):
    """Raised after a model call causes a trial to exceed its spend guard."""


class BenchmarkRunCancelled(RuntimeError):
    """Raised in the worker thread after Harbor cancels an agent run."""


def _bounded_text(value: str | None, limit: int) -> str:
    """Keep both ends of command output while bounding model context growth."""

    raw = value or ""
    if len(raw) <= limit:
        return raw
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    omitted = len(raw) - head - tail
    return f"{raw[:head]}\n...[{omitted} characters omitted]...\n{raw[-tail:]}"


def estimate_terra_cost(usage: dict[str, Any]) -> float:
    """Estimate GPT-5.6 Terra token cost for one API response.

    Reasoning tokens are already included in completion tokens. OpenAI applies
    long-context multipliers to the complete request once prompt input exceeds
    272K tokens, so the estimate applies those multipliers per call.
    """

    return estimate_openai_text_cost("gpt-5.6-terra", usage)


def _safe_json(value: Any) -> Any:
    """Convert provider objects into deterministic JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _safe_json(model_dump(exclude_none=True))
        except Exception:
            pass
    return str(value)


class _TrackedOpenAI(OpenAI):
    """OpenAI provider that records every benchmark call and enforces a budget."""

    def __init__(
        self,
        *,
        max_cost_usd: float,
        deadline_monotonic: float | None = None,
        finalization_reserve_sec: float = 120.0,
        stop_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> None:
        self.pricing = resolve_openai_text_pricing(str(kwargs.get("model") or ""))
        super().__init__(**kwargs)
        self.max_cost_usd = max(0.01, float(max_cost_usd))
        self.deadline_monotonic = deadline_monotonic
        self.finalization_reserve_sec = max(15.0, float(finalization_reserve_sec))
        self.stop_event = stop_event or threading.Event()
        self.forced_finalization = False
        self.call_records: list[dict[str, Any]] = []
        self._pending_tool_call_ids: deque[str] = deque()
        self.total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        self.estimated_cost_usd = 0.0

    def _check_budget_before_call(self) -> None:
        if self.stop_event.is_set():
            raise BenchmarkRunCancelled("Harbor cancelled the benchmark agent run")
        if self.estimated_cost_usd >= self.max_cost_usd:
            raise BenchmarkSpendLimitExceeded(
                "Terminal-Bench trial spend guard reached "
                f"(${self.estimated_cost_usd:.4f} >= ${self.max_cost_usd:.2f})"
            )

    def seconds_remaining(self) -> float | None:
        """Return the agent's own remaining wall-clock budget, when configured."""

        if self.deadline_monotonic is None:
            return None
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def _apply_deadline_policy(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        tool_choice: str,
    ) -> tuple[list[dict[str, Any]], Optional[list[dict[str, Any]]], str, bool]:
        """Warn near the deadline and force a tool-free final response in reserve."""

        remaining = self.seconds_remaining()
        if remaining is None or not tools:
            return messages, tools, tool_choice, False

        effective_messages = list(messages)
        if self.forced_finalization or remaining <= self.finalization_reserve_sec:
            self.forced_finalization = True
            effective_messages.append(
                {
                    "role": "developer",
                    "content": (
                        "The benchmark wall-clock budget is nearly exhausted. "
                        "Do not call another tool. Give the final concise response now, "
                        "based on the work already completed in the container."
                    ),
                }
            )
            return effective_messages, None, "auto", True

        if remaining <= self.finalization_reserve_sec * 2:
            effective_messages.append(
                {
                    "role": "developer",
                    "content": (
                        f"Approximately {int(remaining)} seconds remain in this "
                        "benchmark run. Prioritize completing and verifying the "
                        "implementation, then stop before the deadline."
                    ),
                }
            )
        return effective_messages, tools, tool_choice, False

    @staticmethod
    def _response_payload(response: Any) -> tuple[str, list[dict[str, Any]]]:
        if isinstance(response, str):
            return response, []
        try:
            message = response.choices[0].message
        except Exception:
            return str(response), []

        tool_calls: list[dict[str, Any]] = []
        for call in getattr(message, "tool_calls", None) or []:
            raw_arguments = getattr(call.function, "arguments", "{}") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, ValueError):
                arguments = {"_raw": str(raw_arguments)}
            tool_calls.append(
                {
                    "id": str(getattr(call, "id", "") or ""),
                    "name": str(getattr(call.function, "name", "") or "unknown"),
                    "arguments": _safe_json(arguments),
                }
            )
        content = getattr(message, "content", None) or ""
        return str(content), tool_calls

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        self._check_budget_before_call()
        remaining_at_start = self.seconds_remaining()
        messages, tools, tool_choice, forced_final = self._apply_deadline_policy(
            messages,
            tools,
            tool_choice,
        )
        started_at = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            response = super().generate(
                messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as exc:
            self.call_records.append(
                {
                    "timestamp": timestamp,
                    "duration_ms": round((time.monotonic() - started_at) * 1000),
                    "message": "[provider error]",
                    "tool_calls": [],
                    "usage": {},
                    "cost_usd": 0.0,
                    "seconds_remaining_at_start": remaining_at_start,
                    "forced_finalization": forced_final,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise

        usage = dict(self.get_last_usage() or {})
        call_cost = estimate_openai_text_cost(self.model, usage)
        for key in self.total_usage:
            self.total_usage[key] += max(0, int(usage.get(key) or 0))
        self.estimated_cost_usd += call_cost

        content, tool_calls = self._response_payload(response)
        for tool_call in tool_calls:
            if tool_call["id"]:
                self._pending_tool_call_ids.append(tool_call["id"])
        self.call_records.append(
            {
                "timestamp": timestamp,
                "duration_ms": round((time.monotonic() - started_at) * 1000),
                "message": content,
                "tool_calls": tool_calls,
                "tool_results": {},
                "usage": usage,
                "cost_usd": call_cost,
                "seconds_remaining_at_start": remaining_at_start,
                "forced_finalization": forced_final,
                "error": None,
            }
        )

        if self.estimated_cost_usd > self.max_cost_usd:
            raise BenchmarkSpendLimitExceeded(
                "Terminal-Bench trial spend guard exceeded after the latest call "
                f"(${self.estimated_cost_usd:.4f} > ${self.max_cost_usd:.2f})"
            )
        return response

    def record_tool_result(self, result: dict[str, Any]) -> None:
        """Attach the next executed tool result to its originating model call."""

        call_id = (
            self._pending_tool_call_ids.popleft() if self._pending_tool_call_ids else ""
        )
        for record in reversed(self.call_records):
            if any(call.get("id") == call_id for call in record.get("tool_calls", [])):
                record.setdefault("tool_results", {})[call_id] = _safe_json(result)
                return


class HarborTerminalBridge:
    """Synchronous MemoRizz tool backed by Harbor's async container API."""

    def __init__(
        self,
        environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
        provider: _TrackedOpenAI,
        *,
        default_timeout_sec: int,
        max_output_chars: int,
    ) -> None:
        self._environment = environment
        self._loop = loop
        self._provider = provider
        self._default_timeout_sec = default_timeout_sec
        self._max_output_chars = max_output_chars
        self.calls: list[dict[str, Any]] = []

    @governed_tool(
        deterministic=False,
        side_effects=True,
        requires_approval=False,
        domains=("benchmark_workspace",),
    )
    def terminal_exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Execute a non-interactive command in the isolated task container.

        Args:
            command: Complete shell command to run inside the task container.
            cwd: Optional container working directory.
            timeout_seconds: Timeout from 1 to 300 seconds.

        Returns:
            Return code, bounded stdout/stderr, truncation flags and duration.
        """

        requested_timeout = timeout_seconds or self._default_timeout_sec
        timeout = max(1, min(int(requested_timeout), 300))
        remaining = self._provider.seconds_remaining()
        if self._provider.stop_event.is_set():
            raise BenchmarkRunCancelled("Harbor cancelled the benchmark agent run")
        if remaining is not None:
            executable_seconds = int(
                max(0.0, remaining - self._provider.finalization_reserve_sec)
            )
            if executable_seconds < 1:
                public_result = {
                    "return_code": 124,
                    "stdout": "",
                    "stderr": (
                        "terminal_exec skipped: benchmark deadline reserve has begun; "
                        "return a final response now"
                    ),
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                    "duration_ms": 0,
                }
                self.calls.append(
                    {
                        "command": command,
                        "cwd": cwd,
                        "timeout_seconds": 0,
                        "verification": False,
                        **public_result,
                    }
                )
                self._provider.record_tool_result(public_result)
                return public_result
            timeout = min(timeout, executable_seconds)
        started_at = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(command=command, cwd=cwd, timeout_sec=timeout),
            self._loop,
        )
        result: ExecResult = future.result(timeout=timeout + 15)
        stdout = _bounded_text(result.stdout, self._max_output_chars)
        stderr = _bounded_text(result.stderr, self._max_output_chars)
        public_result = {
            "return_code": result.return_code,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": len(result.stdout or "") > len(stdout),
            "stderr_truncated": len(result.stderr or "") > len(stderr),
            "duration_ms": round((time.monotonic() - started_at) * 1000),
        }
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout_seconds": timeout,
                "verification": False,
                **public_result,
            }
        )
        self._provider.record_tool_result(public_result)
        return public_result

    @governed_tool(
        deterministic=False,
        side_effects=True,
        requires_approval=False,
        domains=("benchmark_workspace",),
    )
    def terminal_verify(
        self,
        command: str,
        description: str,
        cwd: str | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Run the final, task-specific acceptance check in the task container.

        Args:
            command: A substantive test/assertion command whose exit status
                represents the task's acceptance criteria.
            description: Human-readable statement of what the command proves.
            cwd: Optional container working directory.
            timeout_seconds: Timeout from 1 to 300 seconds.

        Returns:
            The command result plus explicit verification evidence.
        """
        result = self.terminal_exec(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        normalized = " ".join(str(command).strip().lower().split())
        trivial = normalized in {"true", ":", "pwd", "ls", "echo ok", "echo pass"}
        result.update(
            {
                "verification": True,
                "verification_description": str(description).strip(),
                "substantive": not trivial and len(normalized) >= 4,
                "verified": result.get("return_code") == 0 and not trivial,
            }
        )
        if self.calls:
            self.calls[-1].update(
                {
                    "verification": True,
                    "verification_description": str(description).strip(),
                    "substantive": result["substantive"],
                    "verified": result["verified"],
                }
            )
        return result


class MemorizzHarborAgent(BaseAgent):
    """Run MemoRizz host-side with an allowlisted Harbor terminal tool."""

    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        reasoning_effort: str = "max",
        max_steps: int = 50,
        max_cost_usd: float = 4.0,
        max_wall_time_seconds: float | None = 840.0,
        finalization_reserve_seconds: float = 120.0,
        tool_timeout_sec: int = 120,
        max_output_chars: int = 12_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.reasoning_effort = str(reasoning_effort)
        self._max_steps = max(1, int(max_steps))
        self._max_cost_usd = max(0.01, float(max_cost_usd))
        self._max_wall_time_seconds = (
            None
            if max_wall_time_seconds is None
            else max(60.0, float(max_wall_time_seconds))
        )
        self._finalization_reserve_seconds = max(
            15.0,
            float(finalization_reserve_seconds),
        )
        if (
            self._max_wall_time_seconds is not None
            and self._finalization_reserve_seconds >= self._max_wall_time_seconds
        ):
            raise ValueError(
                "finalization_reserve_seconds must be less than "
                "max_wall_time_seconds"
            )
        self._tool_timeout_sec = max(1, min(int(tool_timeout_sec), 300))
        self._max_output_chars = max(1_000, int(max_output_chars))

    @staticmethod
    def name() -> str:
        return "memorizz"

    def version(self) -> str | None:
        try:
            import memorizz

            return memorizz.__version__
        except (ImportError, AttributeError):
            return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """MemoRizz and its model provider run outside the task container."""

    def _build_model(
        self,
        *,
        deadline_monotonic: float | None,
        stop_event: threading.Event,
    ) -> tuple[str, _TrackedOpenAI]:
        load_layered_env()
        requested = self.model_name or "openai/gpt-5.6-terra"
        if "/" in requested:
            provider_name, model_name = requested.split("/", maxsplit=1)
        else:
            provider_name, model_name = "openai", requested
        if provider_name.lower() != "openai":
            raise ValueError(
                "The MemoRizz Terminal-Bench adapter currently supports "
                f"openai/<model>; received {requested!r}."
            )
        api_key = self._get_env("OPENAI_API_KEY")
        if not api_key:
            raise AgentAuthenticationError(
                "OPENAI_API_KEY is required for the MemoRizz Terminal-Bench adapter."
            )
        provider = _TrackedOpenAI(
            api_key=api_key,
            model=model_name,
            context_window_tokens=1_050_000,
            max_tokens=32_000,
            reasoning_effort=self.reasoning_effort,
            api_mode="responses",
            max_cost_usd=self._max_cost_usd,
            deadline_monotonic=deadline_monotonic,
            finalization_reserve_sec=self._finalization_reserve_seconds,
            stop_event=stop_event,
        )
        return requested, provider

    def _write_trajectory(
        self,
        *,
        instruction: str,
        requested_model: str,
        provider: _TrackedOpenAI,
    ) -> None:
        steps: list[Step] = [Step(step_id=1, source="user", message=instruction)]
        for record in provider.call_records:
            tool_calls = [
                ToolCall(
                    tool_call_id=call["id"],
                    function_name=call["name"],
                    arguments=call["arguments"],
                )
                for call in record.get("tool_calls", [])
                if call.get("id")
            ]
            tool_results = record.get("tool_results", {})
            observations = [
                ObservationResult(
                    source_call_id=call.tool_call_id,
                    content=json.dumps(
                        tool_results[call.tool_call_id],
                        ensure_ascii=False,
                        default=str,
                    ),
                )
                for call in tool_calls
                if call.tool_call_id in tool_results
            ]
            usage = record.get("usage", {})
            message = record.get("message") or (
                "[tool call: "
                + ", ".join(call.function_name for call in tool_calls)
                + "]"
                if tool_calls
                else "[empty response]"
            )
            if record.get("error"):
                message = f"{message}\n{record['error']}"
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=record.get("timestamp"),
                    source="agent",
                    model_name=requested_model,
                    reasoning_effort=self.reasoning_effort,
                    message=message,
                    tool_calls=tool_calls or None,
                    observation=(
                        Observation(results=observations) if observations else None
                    ),
                    metrics=Metrics(
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        cached_tokens=usage.get("cached_tokens"),
                        cost_usd=record.get("cost_usd"),
                    ),
                    llm_call_count=1,
                )
            )

        trajectory = Trajectory(
            schema_version="ATIF-v1.7",
            session_id=str(self.context_id or self.session_id or "trial"),
            agent=Agent(
                name="MemoRizz",
                version=self.version() or "unknown",
                model_name=requested_model,
                extra={
                    "reasoning_effort": self.reasoning_effort,
                    "memory_provider": "filesystem",
                    "max_steps": self._max_steps,
                    "max_cost_usd": self._max_cost_usd,
                    "max_wall_time_seconds": self._max_wall_time_seconds,
                    "finalization_reserve_seconds": (
                        self._finalization_reserve_seconds
                    ),
                    "pricing": provider.pricing.to_dict(),
                },
            ),
            steps=steps,
            notes="Generated natively by the MemoRizz Harbor adapter.",
            final_metrics=FinalMetrics(
                total_prompt_tokens=provider.total_usage["prompt_tokens"],
                total_completion_tokens=provider.total_usage["completion_tokens"],
                total_cached_tokens=provider.total_usage["cached_tokens"],
                total_cost_usd=provider.estimated_cost_usd,
                total_steps=len(steps),
            ),
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        target = self.logs_dir / "trajectory.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(trajectory.to_json_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(target)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from memorizz import CompletionPolicy, ContextPolicy, MemAgent
        from memorizz.enums.memory_type import MemoryType
        from memorizz.memory_provider.filesystem import (
            FileSystemConfig,
            FileSystemProvider,
        )

        started_at = time.monotonic()
        stop_event = threading.Event()
        deadline_monotonic = (
            None
            if self._max_wall_time_seconds is None
            else started_at + self._max_wall_time_seconds
        )
        requested_model, provider = self._build_model(
            deadline_monotonic=deadline_monotonic,
            stop_event=stop_event,
        )
        loop = asyncio.get_running_loop()
        bridge = HarborTerminalBridge(
            environment,
            loop,
            provider,
            default_timeout_sec=self._tool_timeout_sec,
            max_output_chars=self._max_output_chars,
        )
        memory_provider = FileSystemProvider(
            FileSystemConfig(
                root_path=self.logs_dir / "memory",
                lazy_vector_indexes=True,
            )
        )

        agent_instruction = """You are an autonomous terminal engineering agent in
an isolated Terminal-Bench task container. Use terminal_exec repeatedly to inspect
the environment, edit files, run tests, and verify the requested outcome. Commands
must be non-interactive. Begin by inspecting the task environment and relevant
files. Treat every non-zero return code or failing test as evidence to diagnose and
fix. Avoid dumping binary files or huge dependency trees. The grader evaluates the
container's files and behavior, so perform the work rather than merely describing
commands. Your last tool call before proposing a final response MUST be
terminal_verify with a task-specific command that checks every measurable acceptance
criterion. A zero exit code from an unrelated or trivial command is not sufficient.
If terminal_verify fails, keep working and run it again after the fix. Do not access
Terminal-Bench websites, repositories, task solutions, or grader internals."""

        def _terminal_completion(candidate):
            latest = bridge.calls[-1] if bridge.calls else None
            if not latest:
                return {
                    "accepted": False,
                    "code": "missing_terminal_evidence",
                    "reason": "No terminal execution evidence was produced.",
                }
            evidence = {
                "command": latest.get("command"),
                "description": latest.get("verification_description"),
                "return_code": latest.get("return_code"),
                "verified": latest.get("verified", False),
                "substantive": latest.get("substantive", False),
            }
            if not latest.get("verification"):
                return {
                    "accepted": False,
                    "code": "verification_not_last",
                    "reason": (
                        "The last tool call was not terminal_verify. Run a final "
                        "task-specific acceptance command."
                    ),
                    "metadata": evidence,
                }
            if not latest.get("substantive"):
                return {
                    "accepted": False,
                    "code": "trivial_verification",
                    "reason": "The final verification command was trivial.",
                    "metadata": evidence,
                }
            if not latest.get("verified"):
                return {
                    "accepted": False,
                    "code": "verification_failed",
                    "reason": (
                        "The final acceptance command failed with return code "
                        f"{latest.get('return_code')}."
                    ),
                    "metadata": evidence,
                }
            return {
                "accepted": True,
                "code": "terminal_evidence_passed",
                "reason": "The host observed a successful final acceptance command.",
                "metadata": evidence,
            }

        completion_policy = CompletionPolicy(
            enabled=True,
            max_rejections=3,
            fail_closed=True,
            require_tool_calls=True,
            validator=_terminal_completion,
            validator_name="terminal_bench_final_verification",
            forbidden_response_patterns=(
                r"\b(?:still|currently)\s+(?:exceeds?|fails?|above|over)\b",
                r"\btests?\s+(?:are\s+)?(?:still\s+)?failing\b",
                r"\b(?:not|wasn't|isn't)\s+(?:fully\s+)?(?:complete|fixed|passing)\b",
            ),
        )

        benchmark_memory_id = f"terminal-bench:{self.context_id or self.session_id}"
        benchmark_thread_id = str(self.context_id or self.session_id or "trial")

        agent = MemAgent(
            model=provider,
            llm_config={
                "provider": "openai",
                "model": requested_model.split("/", maxsplit=1)[-1],
                "reasoning_effort": self.reasoning_effort,
                "api_mode": "responses",
                "context_window_tokens": 1_050_000,
            },
            tools=[bridge.terminal_exec, bridge.terminal_verify],
            instruction=agent_instruction,
            memory_provider=memory_provider,
            memory_types=[
                MemoryType.CONVERSATION_MEMORY,
                MemoryType.WORKFLOW_MEMORY,
            ],
            max_steps=self._max_steps,
            semantic_cache=True,
            semantic_cache_config={
                "similarity_threshold": 0.98,
                "scope": "session",
                "ttl_hours": 1.0,
                "admission_policy": "read_only_deterministic",
            },
            automations_enabled=False,
            context_policy=ContextPolicy(
                progressive_tool_disclosure=False,
                max_tool_invocations_per_turn=self._max_steps,
                max_tool_attempts_per_call=2,
            ),
            completion_policy=completion_policy,
            name="MemoRizz Terminal-Bench Agent",
            auto_register=True,
            learning_control_plane={
                "enabled": True,
                # A Terminal-Bench trial starts with an empty isolated store;
                # capture the run without paying for pointless first-turn recall.
                "retrieval_enabled": False,
                "evidence_sources": [
                    "conversation_memory",
                    "workflow_memory",
                    "learning_artifacts",
                ],
            },
        )

        response: str | None = None
        error: str | None = None
        cancelled = False
        try:
            response = await asyncio.to_thread(
                agent.run,
                instruction,
                memory_id=benchmark_memory_id,
                thread_id=benchmark_thread_id,
                user_id="terminal-bench",
                context={
                    "benchmark": "terminal-bench-2.1",
                    "environment": environment.environment_name,
                },
                tool_context={"benchmark_container": True},
            )
            if isinstance(response, str) and response.startswith(
                (
                    "I apologize, but I encountered an error:",
                    "I encountered an error while processing your request:",
                )
            ):
                raise RuntimeError(response)
        except asyncio.CancelledError:
            cancelled = True
            stop_event.set()
            error = "CancelledError: Harbor cancelled the agent at its hard timeout"
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration_ms = round((time.monotonic() - started_at) * 1000)
            outcome_record = None
            compiler_report = None
            latest_verification = bridge.calls[-1] if bridge.calls else {}
            try:
                verified = bool(
                    latest_verification.get("verification")
                    and latest_verification.get("verified")
                )
                trace_context = agent.get_trace_context()
                if trace_context.get("root_trace_id"):
                    outcome_record = agent.record_task_outcome(
                        "success" if verified else "failure",
                        verified=verified,
                        source="terminal_bench_host_verification",
                        score=1.0 if verified else 0.0,
                        metrics={
                            "return_code": latest_verification.get("return_code"),
                            "cancelled": cancelled,
                        },
                        trace_context=trace_context,
                        external_id=(
                            "terminal-bench:" f"{trace_context.get('root_trace_id')}"
                        ),
                    )
                    compiler_report = agent.compile_memory(
                        memory_id=benchmark_memory_id,
                        user_id="terminal-bench",
                        thread_id=benchmark_thread_id,
                    )
            except Exception as exc:
                outcome_record = {"error": f"{type(exc).__name__}: {exc}"}
            self._write_trajectory(
                instruction=instruction,
                requested_model=requested_model,
                provider=provider,
            )
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            try:
                cache_report = agent.semantic_cache_stats()
            except Exception as exc:
                cache_report = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                observability = agent.observability_summary(
                    benchmark_memory_id, "terminal-bench"
                )
            except Exception as exc:
                observability = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                learning_report = agent.learning_report(
                    memory_id=benchmark_memory_id, user_id="terminal-bench"
                )
            except Exception as exc:
                learning_report = {"error": f"{type(exc).__name__}: {exc}"}
            run_report = {
                "memorizz_version": self.version(),
                "model": requested_model,
                "reasoning_effort": self.reasoning_effort,
                "max_steps": self._max_steps,
                "max_cost_usd": self._max_cost_usd,
                "max_wall_time_seconds": self._max_wall_time_seconds,
                "finalization_reserve_seconds": self._finalization_reserve_seconds,
                "forced_finalization": provider.forced_finalization,
                "cancelled": cancelled,
                "duration_ms": duration_ms,
                "tool_call_count": len(bridge.calls),
                "tool_calls": bridge.calls,
                "model_call_count": len(provider.call_records),
                "usage": provider.total_usage,
                "estimated_cost_usd": provider.estimated_cost_usd,
                "pricing": provider.pricing.to_dict(),
                "completion_policy": agent.completion_policy_report(),
                "semantic_cache": cache_report,
                "observability": observability,
                "verified_outcome": outcome_record,
                "memory_compiler": compiler_report,
                "learning_control_plane": learning_report,
                "response": response,
                "error": error,
            }
            (self.logs_dir / "memorizz-run.json").write_text(
                json.dumps(run_report, indent=2, default=str),
                encoding="utf-8",
            )
            context.n_input_tokens = provider.total_usage["prompt_tokens"]
            context.n_cache_tokens = provider.total_usage["cached_tokens"]
            context.n_output_tokens = provider.total_usage["completion_tokens"]
            context.cost_usd = provider.estimated_cost_usd
            context.metadata = {
                "memorizz_version": self.version(),
                "model": requested_model,
                "reasoning_effort": self.reasoning_effort,
                "tool_call_count": len(bridge.calls),
                "model_call_count": len(provider.call_records),
                "duration_ms": duration_ms,
                "spend_guard_usd": self._max_cost_usd,
                "pricing": provider.pricing.to_dict(),
                "max_wall_time_seconds": self._max_wall_time_seconds,
                "forced_finalization": provider.forced_finalization,
                "cancelled": cancelled,
                "final_response": response,
                "error": error,
            }
            if not cancelled:
                try:
                    agent.close(close_model_provider=True)
                except Exception:
                    pass

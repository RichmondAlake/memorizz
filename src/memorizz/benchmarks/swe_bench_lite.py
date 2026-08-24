"""Memory-first MemoRizz agent runner for official SWE-bench Lite instances."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from memorizz import CompletionPolicy, ContextPolicy, MemAgent
from memorizz.benchmarks.common import (
    create_benchmark_memory_provider,
    normalize_memory_backend,
)
from memorizz.enums import MemoryType
from memorizz.llms.openai import OpenAI
from memorizz.tooling import governed_tool

MINI_INPUT_USD_PER_MILLION = 0.75
MINI_CACHED_INPUT_USD_PER_MILLION = 0.075
MINI_OUTPUT_USD_PER_MILLION = 4.50


def estimate_gpt_5_4_mini_cost(usage: Dict[str, Any]) -> float:
    prompt = max(0, int(usage.get("prompt_tokens") or 0))
    cached = min(prompt, max(0, int(usage.get("cached_tokens") or 0)))
    output = max(0, int(usage.get("completion_tokens") or 0))
    return (
        (prompt - cached) * MINI_INPUT_USD_PER_MILLION
        + cached * MINI_CACHED_INPUT_USD_PER_MILLION
        + output * MINI_OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _jsonable(dump(exclude_none=True))
    return str(value)


def _bounded(value: Any, limit: int) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= limit:
        return text, False
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return (
        f"{text[:head]}\n...[{len(text) - head - tail} characters omitted]...\n"
        f"{text[-tail:]}",
        True,
    )


class BenchmarkSpendLimitExceeded(RuntimeError):
    pass


class TrackedOpenAI(OpenAI):
    """Record API evidence and enforce a per-instance dollar guard."""

    def __init__(self, *, max_cost_usd: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_cost_usd = max(0.01, float(max_cost_usd))
        self.estimated_cost_usd = 0.0
        self.call_records: List[Dict[str, Any]] = []
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._pending_tool_call_ids: deque[str] = deque()

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        if self.estimated_cost_usd >= self.max_cost_usd:
            raise BenchmarkSpendLimitExceeded(
                f"SWE-bench instance cost guard reached ${self.max_cost_usd:.2f}"
            )
        started = time.monotonic()
        response = super().generate(messages, tools=tools, tool_choice=tool_choice)
        usage = dict(self.get_last_usage() or {})
        cost = estimate_gpt_5_4_mini_cost(usage)
        self.estimated_cost_usd += cost
        for key in self.total_usage:
            self.total_usage[key] += max(0, int(usage.get(key) or 0))
        content = ""
        calls: List[Dict[str, Any]] = []
        if isinstance(response, str):
            content = response
        else:
            try:
                message = response.choices[0].message
                content = str(getattr(message, "content", None) or "")
                for call in getattr(message, "tool_calls", None) or []:
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except (TypeError, ValueError):
                        arguments = {"_raw": str(call.function.arguments)}
                    call_id = str(getattr(call, "id", "") or "")
                    calls.append(
                        {
                            "id": call_id,
                            "name": str(call.function.name),
                            "arguments": _jsonable(arguments),
                        }
                    )
                    if call_id:
                        self._pending_tool_call_ids.append(call_id)
            except Exception:
                content = str(response)
        self.call_records.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.monotonic() - started) * 1000),
                "content": content,
                "tool_calls": calls,
                "tool_results": {},
                "usage": usage,
                "cost_usd": cost,
            }
        )
        if self.estimated_cost_usd > self.max_cost_usd:
            raise BenchmarkSpendLimitExceeded(
                "SWE-bench instance cost guard exceeded after the latest call: "
                f"${self.estimated_cost_usd:.4f} > ${self.max_cost_usd:.2f}"
            )
        return response

    def record_tool_result(self, result: Dict[str, Any]) -> None:
        call_id = (
            self._pending_tool_call_ids.popleft() if self._pending_tool_call_ids else ""
        )
        for record in reversed(self.call_records):
            if any(call.get("id") == call_id for call in record["tool_calls"]):
                record["tool_results"][call_id] = _jsonable(result)
                break


class DockerWorkspaceBridge:
    """Allowlisted command bridge into an official SWE-bench instance image."""

    def __init__(
        self,
        container: Any,
        provider: TrackedOpenAI,
        *,
        max_output_chars: int = 16_000,
        default_timeout_seconds: int = 120,
    ) -> None:
        self.container = container
        self.provider = provider
        self.max_output_chars = max(1_000, int(max_output_chars))
        self.default_timeout_seconds = max(1, min(int(default_timeout_seconds), 300))
        self.calls: List[Dict[str, Any]] = []

    @governed_tool(
        deterministic=False,
        side_effects=True,
        requires_approval=False,
        domains=("swe_bench_workspace",),
    )
    def terminal_exec(
        self,
        command: str,
        cwd: str = "/testbed",
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a bounded shell command in the isolated SWE-bench image."""
        timeout = max(
            1,
            min(int(timeout_seconds or self.default_timeout_seconds), 300),
        )
        started = time.monotonic()
        wrapped = [
            "timeout",
            "--signal=TERM",
            str(timeout),
            "bash",
            "-lc",
            str(command),
        ]
        result = self.container.exec_run(wrapped, workdir=cwd, demux=True)
        stdout_raw, stderr_raw = result.output or (b"", b"")
        stdout, stdout_truncated = _bounded(
            (stdout_raw or b"").decode("utf-8", errors="replace"),
            self.max_output_chars,
        )
        stderr, stderr_truncated = _bounded(
            (stderr_raw or b"").decode("utf-8", errors="replace"),
            self.max_output_chars,
        )
        public = {
            "return_code": int(result.exit_code),
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "verification": False,
        }
        self.calls.append(
            {"command": command, "cwd": cwd, "timeout_seconds": timeout, **public}
        )
        self.provider.record_tool_result(public)
        return public

    @governed_tool(
        deterministic=False,
        side_effects=True,
        requires_approval=False,
        domains=("swe_bench_workspace",),
    )
    def terminal_verify(
        self,
        command: str,
        description: str,
        cwd: str = "/testbed",
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the final task-specific acceptance test and retain its evidence."""
        result = self.terminal_exec(command, cwd=cwd, timeout_seconds=timeout_seconds)
        normalized = " ".join(str(command).strip().lower().split())
        trivial = normalized in {"true", ":", "pwd", "ls", "echo ok", "echo pass"}
        result.update(
            {
                "verification": True,
                "verification_description": str(description).strip(),
                "substantive": not trivial and len(normalized) >= 4,
                "verified": result["return_code"] == 0 and not trivial,
            }
        )
        if self.calls:
            self.calls[-1].update(result)
        return result


def _completion_validator(bridge: DockerWorkspaceBridge):
    def validate(_candidate):
        latest = bridge.calls[-1] if bridge.calls else None
        if not latest:
            return False, "No workspace evidence exists."
        metadata = {
            "command": latest.get("command"),
            "return_code": latest.get("return_code"),
            "description": latest.get("verification_description"),
        }
        if not latest.get("verification"):
            return {
                "accepted": False,
                "code": "verification_not_last",
                "reason": "The final tool call must be terminal_verify.",
                "metadata": metadata,
            }
        if not latest.get("verified"):
            return {
                "accepted": False,
                "code": "verification_failed",
                "reason": "The task-specific final verification did not pass.",
                "metadata": metadata,
            }
        return {
            "accepted": True,
            "code": "workspace_verification_passed",
            "reason": "The host observed a passing final verification command.",
            "metadata": metadata,
        }

    return validate


def run_swe_bench_instance(
    instance: Dict[str, Any],
    *,
    output_dir: Path,
    model_name: str = "gpt-5.4-mini",
    reasoning_effort: str = "high",
    max_steps: int = 36,
    max_cost_usd: float = 3.0,
    memory_backend: str = "filesystem",
    learning_control_plane: bool = True,
) -> Dict[str, Any]:
    """Run one instance and emit an official prediction plus audit report."""
    try:
        import docker
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "docker Python SDK is required for SWE-bench evaluation"
        ) from exc

    instance_id = str(instance["instance_id"])
    image = str(instance.get("image") or "")
    if not image:
        raise ValueError(
            "The SWE-bench dataset row must include its official instance image"
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_backend = normalize_memory_backend(memory_backend)
    provider = TrackedOpenAI(
        api_key=api_key,
        model=model_name,
        api_mode="responses",
        reasoning_effort=reasoning_effort,
        max_tokens=24_000,
        context_window_tokens=400_000,
        max_cost_usd=max_cost_usd,
    )
    client = docker.from_env(timeout=1_800)
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        client.images.pull(image)
    container_name = f"memorizz.swebench.{instance_id.lower()}.{uuid.uuid4().hex[:8]}"
    container = client.containers.create(
        image=image,
        name=container_name,
        user="root",
        detach=True,
        command="tail -f /dev/null",
        network_disabled=True,
        platform="linux/amd64",
        labels={"memorizz.benchmark": "swe-bench-lite", "instance_id": instance_id},
        security_opt=["no-new-privileges"],
    )
    container.start()
    bridge = DockerWorkspaceBridge(container, provider)
    memory_provider = create_benchmark_memory_provider(
        memory_backend,
        filesystem_root=output_dir / "memory",
    )
    completion_policy = CompletionPolicy(
        enabled=True,
        max_rejections=3,
        fail_closed=True,
        require_tool_calls=True,
        validator=_completion_validator(bridge),
        validator_name="swe_bench_final_verification",
        forbidden_response_patterns=(
            r"\btests?\s+(?:are\s+)?(?:still\s+)?failing\b",
            r"\b(?:not|isn't|wasn't)\s+(?:fully\s+)?(?:fixed|passing|complete)\b",
        ),
        retry_instruction=(
            "The host rejected that proposed final response: {reason} "
            "If verification was not last, call terminal_verify next—not "
            "terminal_exec—with the strongest task-specific check. If a prior "
            "verification failed, fix the failure first and then call "
            "terminal_verify again. Do not propose completion until it passes."
        ),
    )
    memory_id = (
        f"swe-bench-lite:{instance_id}:"
        f"{uuid.uuid5(uuid.NAMESPACE_URL, str(output_dir))}"
    )
    thread_id = instance_id
    agent = MemAgent(
        model=provider,
        llm_config={
            "provider": "openai",
            "model": model_name,
            "api_mode": "responses",
            "reasoning_effort": reasoning_effort,
        },
        tools=[bridge.terminal_exec, bridge.terminal_verify],
        instruction=(
            "You are an autonomous software engineer working inside an isolated "
            "official SWE-bench instance image. Inspect /testbed, diagnose the issue, "
            "edit only the implementation needed for the requested fix, and run "
            "focused tests. Do not seek or inspect gold patches, hidden tests, benchmark "
            "websites, or grader internals. The container has no network. Preserve "
            "unrelated behavior. Keep the tool loop efficient by batching related "
            "read-only checks into information-dense terminal calls. Your final tool "
            "call must be terminal_verify with the strongest task-specific test "
            "available; do not call terminal_exec after it. If verification fails, "
            "continue fixing and verify again."
        ),
        memory_provider=memory_provider,
        memory_types=[
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.WORKFLOW_MEMORY,
            MemoryType.SUMMARIES,
            MemoryType.TOOL_LOG,
        ],
        max_steps=max_steps,
        semantic_cache=True,
        semantic_cache_config={
            "similarity_threshold": 0.99,
            "scope": "session",
            "ttl_hours": 4.0,
            "admission_policy": "read_only_deterministic",
            "freshness_by_domain": {"swe_bench_workspace": 0.0},
        },
        automations_enabled=False,
        context_policy=ContextPolicy(
            progressive_tool_disclosure=False,
            # CompletionPolicy can ask for one initial verification plus its
            # bounded retries after the ordinary work-loop budget is spent.
            max_tool_invocations_per_turn=(
                max_steps + completion_policy.max_rejections + 1
            ),
            max_tool_attempts_per_call=2,
        ),
        completion_policy=completion_policy,
        name=f"MemoRizz SWE-bench {instance_id}",
        auto_register=True,
        learning_control_plane=(
            {
                "enabled": True,
                "evidence_sources": [
                    "conversation_memory",
                    "summaries",
                    "workflow_memory",
                    "learning_artifacts",
                ],
                "evidence_token_budget": 2200,
                "evidence_max_items": 6,
                "evidence_candidates_per_source": 6,
                "evidence_max_per_source": 2,
            }
            if learning_control_plane
            else False
        ),
    )
    response: Optional[str] = None
    error: Optional[str] = None
    summary_ids: List[str] = []
    reflection_cache: Dict[str, Any] = {}
    reflection = None
    outcome_record = None
    compiler_report = None
    started = time.monotonic()
    try:
        task_prompt = (
            f"Repository: {instance.get('repo')}\n"
            f"Base commit: {instance.get('base_commit')}\n"
            f"Issue:\n{instance.get('problem_statement')}"
        )
        response = agent.run(
            task_prompt,
            memory_id=memory_id,
            thread_id=thread_id,
            user_id="swe-bench-lite",
            context={
                "benchmark": "SWE-bench Lite",
                "instance_id": instance_id,
                "cache_domain": "swe_bench_workspace",
                "cache_data_version": str(instance.get("base_commit") or "unknown"),
            },
            tool_context={"benchmark_container": True},
        )
        summary_ids = agent.generate_summaries(
            days_back=1,
            max_memories_per_summary=12,
            memory_id=memory_id,
            user_id="swe-bench-lite",
            thread_id=thread_id,
        )
        reflection = MemAgent(
            model=provider,
            instruction=(
                "Read the compacted benchmark memory and summarize the reusable "
                "repository-specific lessons. Do not use tools or propose edits."
            ),
            memory_provider=memory_provider,
            memory_ids=[memory_id],
            memory_types=[MemoryType.CONVERSATION_MEMORY, MemoryType.SUMMARIES],
            semantic_cache=True,
            semantic_cache_config={
                "similarity_threshold": 1.0,
                "scope": "session",
                "ttl_hours": 4.0,
            },
            automations_enabled=False,
            name=f"MemoRizz SWE reflection {instance_id}",
            auto_register=False,
            learning_control_plane=(
                {
                    "enabled": True,
                    "evidence_sources": ["conversation_memory", "summaries"],
                    "evidence_token_budget": 1600,
                    "evidence_max_items": 4,
                    "evidence_max_per_source": 2,
                }
                if learning_control_plane
                else False
            ),
        )
        reflection_query = "Summarize the reusable engineering lessons from this run."
        reflection_first = reflection.run(
            reflection_query,
            memory_id=memory_id,
            thread_id=f"{thread_id}:reflection",
            user_id="swe-bench-lite",
            context={"cache_data_version": instance.get("base_commit")},
        )
        stats_before = reflection.semantic_cache_stats()
        reflection_second = reflection.run(
            reflection_query,
            memory_id=memory_id,
            thread_id=f"{thread_id}:reflection",
            user_id="swe-bench-lite",
            context={"cache_data_version": instance.get("base_commit")},
        )
        reflection_cache = {
            "first_response_chars": len(reflection_first),
            "repeat_response_match": reflection_first == reflection_second,
            "stats_before_repeat": stats_before,
            "stats_after_repeat": reflection.semantic_cache_stats(),
        }
        latest_verification = bridge.calls[-1] if bridge.calls else {}
        verified = bool(
            latest_verification.get("verification")
            and latest_verification.get("verified")
        )
        main_trace = agent.get_trace_context()
        outcome_record = agent.record_task_outcome(
            "success" if verified else "failure",
            verified=True,
            source="swe_bench_terminal_verification",
            score=1.0 if verified else 0.0,
            metrics={
                "instance_id": instance_id,
                "command": latest_verification.get("command"),
                "return_code": latest_verification.get("return_code"),
                "official_resolved": None,
            },
            trace_context=main_trace,
            external_id=(f"swe-bench:{instance_id}:{main_trace.get('root_trace_id')}"),
        )
        if learning_control_plane:
            compiler_report = agent.compile_memory(
                memory_id=memory_id,
                user_id="swe-bench-lite",
                thread_id=thread_id,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        diff_exec = container.exec_run(
            [
                "bash",
                "-lc",
                "git add -N . >/dev/null 2>&1 || true; "
                "git -c core.fileMode=false diff --binary --no-ext-diff",
            ],
            workdir="/testbed",
            demux=True,
        )
        diff_stdout, _diff_stderr = diff_exec.output or (b"", b"")
        patch = (diff_stdout or b"").decode("utf-8", errors="replace")
        try:
            observability = agent.observability_summary(memory_id, "swe-bench-lite")
        except Exception as exc:
            observability = {"error": f"{type(exc).__name__}: {exc}"}
        report = {
            "benchmark": "SWE-bench Lite",
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "image": image,
            "model": model_name,
            "reasoning_effort": reasoning_effort,
            "duration_seconds": time.monotonic() - started,
            "response": response,
            "error": error,
            "patch_chars": len(patch),
            "usage": provider.total_usage,
            "estimated_cost_usd": provider.estimated_cost_usd,
            "completion_policy": agent.completion_policy_report(),
            "tool_calls": bridge.calls,
            "summary_ids": summary_ids,
            "semantic_cache": agent.semantic_cache_stats(),
            "reflection_cache_probe": reflection_cache,
            "observability": observability,
            "memory_backend": memory_backend,
            "verified_outcome": outcome_record,
            "memory_compiler": compiler_report,
            "learning_control_plane": agent.learning_report(
                memory_id=memory_id, user_id="swe-bench-lite"
            ),
            "features": {
                "filesystem_memory": memory_backend == "filesystem",
                "oracle_memory": memory_backend == "oracle",
                "semantic_cache": True,
                "summarization_and_compaction": True,
                "workflow_memory": True,
                "tool_log_offloading": True,
                "host_completion_gate": True,
                "bounded_evidence_pack": bool(learning_control_plane),
                "immutable_learning_events": bool(learning_control_plane),
            },
        }
        (output_dir / "memorizz-run.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": f"memorizz/{model_name}",
            "model_patch": patch,
        }
        (output_dir / "predictions.jsonl").write_text(
            json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        try:
            container.stop(timeout=5)
        finally:
            container.remove(force=True)
        try:
            if reflection is not None:
                reflection.close(close_memory_provider=False)
            agent.close(close_model_provider=True)
        except Exception:
            pass
    return {"prediction": prediction, "report": report}


__all__ = [
    "DockerWorkspaceBridge",
    "TrackedOpenAI",
    "estimate_gpt_5_4_mini_cost",
    "run_swe_bench_instance",
]

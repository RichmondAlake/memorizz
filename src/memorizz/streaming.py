"""Versioned ordered answer events with bounded delivery and cooperative cancellation.

Each invocation owns one worker and copied Context for its entire lifetime.
Provider reasoning, tool arguments/results and rejected drafts are not public
events. Cancellation never caches a partial answer. Blocking third-party tools
must cooperate; their worker remains supervised until it returns.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import threading
import time
import uuid
import weakref
from contextvars import ContextVar, copy_context

EVENT_VERSION = 1
MAX_EVENT_BYTES = 65536
current_cancellation = ContextVar("memorizz_stream_cancellation", default=None)
current_stream = ContextVar("memorizz_ordered_stream", default=None)
_workers = set()
_workers_lock = threading.Lock()
_worker_slots = threading.BoundedSemaphore(32)
_execution_locks = weakref.WeakValueDictionary()
_execution_locks_guard = threading.Lock()


def execution_lock(key):
    """Share locks only while a waiter/owner holds a strong reference."""
    with _execution_locks_guard:
        lock = _execution_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _execution_locks[key] = lock
        return lock


class StreamCancelled(BaseException):
    """Internal cooperative stop; not an answer or a provider failure."""


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks = set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def check(self):
        if self.cancelled:
            raise StreamCancelled()

    def cancel(self):
        self._event.set()
        with self._lock:
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    def register(self, callback):
        with self._lock:
            self._callbacks.add(callback)
        if self.cancelled:
            callback()

        def unregister():
            with self._lock:
                self._callbacks.discard(callback)

        return unregister


class StreamEvent(dict):
    """JSON-safe v1 envelope. Sequence numbers belong to an invocation, not SSE."""

    def to_dict(self):
        return dict(self)

    def to_json(self):
        return json.dumps(self, ensure_ascii=False, allow_nan=False)


def session_for(agent):
    session = current_stream.get()
    return session if session is not None and session.agent is agent else None


def check_cancelled():
    token = current_cancellation.get()
    if token:
        token.check()


def start_owned_worker(target, *, name):
    """Supervise cooperative provider workers; abandoned work retains capacity."""
    if not _worker_slots.acquire(blocking=False):
        raise RuntimeError("Streaming worker capacity reached")
    context = copy_context()

    def run():
        try:
            context.run(target)
        finally:
            with _workers_lock:
                _workers.discard(threading.current_thread())
            _worker_slots.release()

    worker = threading.Thread(target=run, name=name, daemon=True)
    with _workers_lock:
        _workers.add(worker)
    try:
        worker.start()
    except BaseException:
        with _workers_lock:
            _workers.discard(worker)
        _worker_slots.release()
        raise
    return worker


class EventStream:
    """Closeable synchronous event iterator; use ``with`` or ``finally: close``."""

    def __init__(self, producer, *, cancellation=None, queue_size=64):
        if not 1 <= queue_size <= 1024:
            raise ValueError("queue_size must be between 1 and 1024")
        if not _worker_slots.acquire(blocking=False):
            raise RuntimeError("Streaming worker capacity reached")
        self.cancellation = cancellation or CancellationToken()
        self.queue = queue.Queue(maxsize=queue_size)
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.terminal = None
        self._terminal_delivered = False
        context = copy_context()

        def run():
            token = current_cancellation.set(self.cancellation)
            try:
                producer(self)
            finally:
                current_cancellation.reset(token)
                self.finished.set()
                with _workers_lock:
                    _workers.discard(threading.current_thread())
                _worker_slots.release()

        self.worker = threading.Thread(
            target=lambda: context.run(run), name="memorizz-event-stream", daemon=True
        )
        with _workers_lock:
            _workers.add(self.worker)
        try:
            self.worker.start()
        except BaseException:
            with _workers_lock:
                _workers.discard(self.worker)
            _worker_slots.release()
            raise

    def put(self, event):
        while not self.closed.is_set():
            self.cancellation.check()
            try:
                self.queue.put(event, timeout=0.05)
                return
            except queue.Full:
                continue
        raise StreamCancelled()

    def __iter__(self):
        return self

    def __next__(self):
        return self.poll()

    def poll(self, timeout=None):
        """Return the next event, or None on timeout (for transport heartbeats)."""
        if self.closed.is_set() or self._terminal_delivered:
            raise StopIteration
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.closed.is_set():
                raise StopIteration
            try:
                return self.queue.get(timeout=0.05)
            except queue.Empty:
                if self.finished.is_set():
                    self._terminal_delivered = True
                    if self.terminal is not None:
                        return self.terminal
                    raise StopIteration
                if deadline is not None and time.monotonic() >= deadline:
                    return None

    def close(self):
        self.closed.set()
        self.cancellation.cancel()
        if threading.current_thread() is not self.worker:
            self.worker.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    async def async_events(self):
        sentinel = object()
        try:
            while True:
                event = await asyncio.to_thread(next, self, sentinel)
                if event is sentinel:
                    return
                yield event
        finally:
            await asyncio.to_thread(self.close)


class _Session:
    def __init__(self, agent, stream, mode, identity):
        self.agent, self.stream, self.mode = agent, stream, mode
        self.identity = identity
        self.seq = 0
        self.event_lock = threading.RLock()
        self.sealed = False
        self.started = time.monotonic()
        self.answer_id = str(uuid.uuid4())
        self.attempt = 1
        self.answer = ""
        self.answer_finished = False
        self.outcome = "completed"
        self.error_code = None
        self.approval = None
        self.persistence = {
            "conversation": "unknown",
            "cache": "not_requested",
            "workflow": "not_requested",
            "trace": "unknown",
        }

    def envelope(self, kind, **payload):
        self.seq += 1
        event = StreamEvent(
            version=EVENT_VERSION,
            run_id=self.identity["run_id"],
            seq=self.seq,
            type=kind,
            elapsed_ms=round((time.monotonic() - self.started) * 1000, 3),
            **payload,
        )
        if len(event.to_json().encode("utf-8")) > MAX_EVENT_BYTES:
            raise ValueError("Stream event exceeds size limit")
        return event

    def emit(self, kind, **payload):
        with self.event_lock:
            if self.sealed:
                return
            if kind == "approval.required":
                self.approval = payload.get("proposal")
            event = self.envelope(kind, **payload)
            try:
                self.stream.put(event)
            except BaseException:
                self.seq -= 1
                raise

    def delta(self, text):
        if not isinstance(text, str):
            raise TypeError("Answer deltas must be strings")
        if self.answer_finished:
            raise RuntimeError("Answer already ended")
        # Bound transport frames without delaying the first provider fragment.
        for offset in range(0, len(text), 4096):
            delta = text[offset : offset + 4096]
            self.emit(
                "answer.delta",
                answer_id=self.answer_id,
                attempt=self.attempt,
                delta=delta,
            )
            self.answer += delta

    def answer_done(self):
        if self.answer_finished or self.outcome != "completed":
            return
        self.answer_finished = True
        self.emit(
            "answer.done",
            answer_id=self.answer_id,
            attempt=self.attempt,
            sha256=hashlib.sha256(self.answer.encode("utf-8")).hexdigest(),
            chars=len(self.answer),
            validation="accepted"
            if self.agent.completion_policy.enabled
            else "not_required",
        )

    def callback(self, event):
        kind = event.get("type")
        if kind == "completion_check":
            self.attempt = event.get("iteration") or 1
            self.emit(
                "completion.check",
                answer_id=self.answer_id,
                attempt=self.attempt,
                **{
                    k: event[k]
                    for k in (
                        "accepted",
                        "code",
                        "response_sha256",
                        "response_chars",
                        "tool_call_count",
                    )
                    if k in event
                },
            )
        elif kind == "error":
            self.outcome = "error"
            self.error_code = event.get("error_code") or "stream_error"
        elif kind == "trace" and event.get("trace_kind") in {
            "tool_call",
            "tool_result",
        }:
            self.emit(
                "tool.started"
                if event["trace_kind"] == "tool_call"
                else "tool.completed",
                tool_name=str(
                    event.get("tool_name") or event.get("logical_tool_name") or "tool"
                )[:240],
                status=event.get("status"),
                span_id=event.get("span_id"),
            )
        elif kind == "stream_start":
            self.emit(
                "status",
                stage="agent_ready",
                **{
                    k: event[k]
                    for k in ("memory_id", "thread_id", "root_trace_id", "turn_id")
                    if k in event
                },
            )
        elif kind == "trace" and event.get("trace_kind") == "model_call":
            self.emit(
                "status",
                stage="provider_request",
                span_id=event.get("span_id"),
                attempt=event.get("iteration"),
            )


def agent_event_stream(
    agent,
    query,
    *,
    delivery_mode=None,
    cancellation=None,
    queue_size=64,
    _prepare=None,
    _finalize=None,
    _agent_id=None,
    **kwargs,
):
    """Create one stream; transport loaders may prepare/finalize in its owned context.

    Deferred loaders receive the session (including an ExitStack for scoped locks)
    and return (agent, run_kwargs). They must not publish private configuration.
    """
    from contextlib import ExitStack

    from .llms.streaming import streaming_capabilities

    mode = delivery_mode or (
        agent.completion_policy.delivery_mode
        if agent and agent.completion_policy.enabled
        else "final_stream"
    )
    if agent:
        agent.completion_policy.validate_delivery_mode(mode)
    identity = {
        "agent_id": agent.agent_id if agent else _agent_id,
        "run_id": str(uuid.uuid4()),
        "root_trace_id": str(uuid.uuid4()),
        "turn_id": str(uuid.uuid4()),
        "memory_id": kwargs.get("memory_id")
        or (agent.get_current_memory_id() if agent else None)
        or (agent.memory_ids[0] if agent and agent.memory_ids else str(uuid.uuid4())),
    }
    identity["thread_id"] = (
        kwargs.get("thread_id")
        or (agent._thread_ids_by_memory.get(identity["memory_id"]) if agent else None)
        or str(uuid.uuid4())
    )
    if any(
        not isinstance(value, str) or len(value) > 512 for value in identity.values()
    ):
        raise ValueError("Stream identifiers must be strings of at most 512 characters")

    def settings(active_agent, run_kwargs):
        selected = delivery_mode or (
            active_agent.completion_policy.delivery_mode
            if active_agent.completion_policy.enabled
            else "final_stream"
        )
        active_agent.completion_policy.validate_delivery_mode(selected)
        capabilities = streaming_capabilities(active_agent.model)
        fallback = capabilities["fallback_reason"]
        if (
            active_agent.meta_harness is not None
            and active_agent.meta_harness_mode == "runtime"
            and not (run_kwargs.get("tool_context") or {}).get(
                "_memorizz_harness_native"
            )
        ):
            fallback = "runtime_harness"
        elif (
            active_agent.delegates
            and active_agent.delegation_config.get("enabled", True)
            and active_agent.delegation_config.get("mode", "auto")
            in {"auto", "deterministic"}
            and not (run_kwargs.get("tool_context") or {}).get(
                "_memorizz_skip_delegation"
            )
        ):
            fallback = "delegation"
        return selected, {
            "delivery_mode": "buffered" if fallback else selected,
            "buffering_reason": fallback
            or ("completion_validation" if selected == "buffered" else None),
            "capabilities": capabilities,
        }

    def produce(stream):
        session = _Session(agent, stream, mode, identity)
        current = current_stream.set(session)
        iterator = None
        with ExitStack() as stack:
            session.stack = stack
            try:
                run_kwargs = dict(kwargs)
                if _prepare:
                    session.emit(
                        "run.started",
                        **{k: v for k, v in identity.items() if k != "run_id"},
                        delivery_mode="initializing",
                        buffering_reason="agent_loading",
                        capabilities=None,
                    )
                    session.emit("status", stage="preparing_agent")
                    active_agent, prepared_kwargs = _prepare(session)
                    stream.cancellation.check()
                    run_kwargs.update(prepared_kwargs or {})
                    session.agent = active_agent
                    session.mode, info = settings(active_agent, run_kwargs)
                    session.emit("status", stage="agent_ready", **info)
                else:
                    active_agent = agent
                    session.mode, info = settings(active_agent, run_kwargs)
                    session.emit(
                        "run.started",
                        **{k: v for k, v in identity.items() if k != "run_id"},
                        **info,
                    )
                run_kwargs.update(
                    memory_id=identity["memory_id"], thread_id=identity["thread_id"]
                )
                callback = run_kwargs.pop("event_callback", None)

                def event_callback(event):
                    session.callback(event)
                    if callback:
                        callback(event)

                iterator = active_agent.run_stream(
                    query, event_callback=event_callback, **run_kwargs
                )
                for chunk in iterator:
                    stream.cancellation.check()
                    if session.outcome == "completed":
                        session.delta(chunk)
                session.answer_done()
            except (StreamCancelled, GeneratorExit):
                session.outcome, session.error_code = "cancelled", "cancelled"
            except BaseException as exc:
                session.outcome = (
                    "cancelled" if stream.cancellation.cancelled else "error"
                )
                session.error_code = (
                    "cancelled"
                    if stream.cancellation.cancelled
                    else getattr(exc, "code", "stream_error")
                )
            finally:
                if iterator is not None:
                    try:
                        iterator.close()
                    except BaseException:
                        session.persistence["trace"] = "unknown"
                if stream.cancellation.cancelled:
                    session.outcome, session.error_code = "cancelled", "cancelled"
                if _finalize and session.agent is not None:
                    try:
                        _finalize(session)
                    except Exception:
                        session.persistence["adapter_state"] = "failed"
                # Finish after adapter writes, while its authorization/agent lock is held.
                with session.event_lock:
                    session.sealed = True
                    stream.terminal = session.envelope(
                        "run.done",
                        status=session.outcome,
                        error_code=str(session.error_code)[:240]
                        if session.error_code
                        else None,
                        persistence=session.persistence,
                        **{k: v for k, v in identity.items() if k != "run_id"},
                    )
                current_stream.reset(current)

    return EventStream(produce, cancellation=cancellation, queue_size=queue_size)

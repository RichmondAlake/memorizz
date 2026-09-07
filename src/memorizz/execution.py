"""Runtime-only execution contracts shared by the agent and its coordinator."""

from contextvars import ContextVar

delegated_execution = ContextVar("memorizz_delegated_execution", default=False)
recording_warning = ContextVar("memorizz_recording_warning", default=None)


def report_recording_warning(operation, error):
    """Expose best-effort agent persistence errors without changing execution."""
    callback = recording_warning.get()
    if callback is not None:
        callback(operation, error)


class AgentExecutionError(RuntimeError):
    """An incomplete execution, never a successful conversational answer."""

    def __init__(
        self, code: str, message: str, *, status: str = "failed", metadata=None
    ):
        self.code = code
        self.status = status
        self.metadata = dict(metadata or {})
        super().__init__(message)

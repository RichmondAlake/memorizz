from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ScenarioResult:
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class TestingAgent:
    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        self.model = model
        self.config = kwargs


class Scenario:
    _testing_agent: Optional[TestingAgent] = None
    _config: Dict[str, Any] = {}

    def __init__(
        self,
        description: str,
        *,
        agent: Callable[[str, Dict[str, Any]], Any],
        success_criteria: Optional[List[str]] = None,
        failure_criteria: Optional[List[str]] = None,
    ):
        self.description = description
        self.agent = agent
        self.success_criteria = success_criteria or []
        self.failure_criteria = failure_criteria or []

    @classmethod
    def configure(cls, **kwargs: Any) -> None:
        testing_agent = kwargs.get("testing_agent")
        if testing_agent is not None:
            cls._testing_agent = testing_agent
        cls._config.update(kwargs)

    async def run(self) -> ScenarioResult:
        prompt = self.description or "Provide a helpful response."
        context = {"description": self.description}
        result = self.agent(prompt, context)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            message = result.get("message") or ""
        else:
            message = str(result)
        success = bool(message and message.strip())
        return ScenarioResult(success=success, details={"message": message})

"""Deterministic capability, policy, health, and outcome routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from .base import AgentHarness
from .models import HarnessCapabilities, HarnessTask


class HarnessReadinessError(RuntimeError):
    """Typed, secret-free failure raised before an adapter process starts."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        harness: Optional[str] = None,
        remediation: Optional[str] = None,
        candidates: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.harness = harness
        self.remediation = remediation
        self.candidates = list(candidates or [])
        rendered = self.message
        if self.remediation and self.remediation not in rendered:
            rendered = f"{rendered} {self.remediation}"
        super().__init__(rendered)

    def to_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "harness": self.harness,
            "remediation": self.remediation,
            "candidates": self.candidates,
        }


@dataclass
class HarnessRoutingDecision:
    selected: str
    explicit: bool
    candidates: List[Dict[str, object]]
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "selected": self.selected,
            "explicit": self.explicit,
            "candidates": self.candidates,
            "reason": self.reason,
        }


class HarnessRouter:
    """Stable router; an LLM never authorizes or overrides this policy."""

    def __init__(
        self,
        *,
        preference: Iterable[str] = ("memagent", "codex", "claude-code", "openhands"),
        allowlist: Optional[Iterable[str]] = None,
    ) -> None:
        self.preference = [str(item).lower() for item in preference]
        self.allowlist = (
            {str(item).lower() for item in allowlist} if allowlist is not None else None
        )

    def select(
        self,
        task: HarnessTask,
        adapters: Mapping[str, AgentHarness],
        *,
        metrics: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> tuple[AgentHarness, HarnessRoutingDecision]:
        requested = task.harness
        explicit = requested not in {"", "auto"}
        if explicit and requested not in adapters:
            known = ", ".join(sorted(adapters)) or "none"
            raise HarnessReadinessError(
                "harness_not_registered",
                f"Harness {requested!r} is not registered (registered: {known}).",
                harness=requested,
                remediation="Run `memorizz harness list` and select a registered harness.",
            )
        candidates: List[Dict[str, object]] = []
        accepted: List[tuple[float, str, AgentHarness, HarnessCapabilities]] = []
        for name, adapter in sorted(adapters.items()):
            capability = adapter.probe()
            reasons: List[str] = []
            if self.allowlist is not None and name not in self.allowlist:
                reasons.append("not_allowlisted")
            if not capability.available:
                reasons.append("unavailable")
            if capability.error:
                reasons.append("not_ready")
            if capability.metadata.get("explicit_only") and not explicit:
                reasons.append("explicit_selection_required")
            if capability.metadata.get("requires_agent_id") and not task.agent_id:
                reasons.append("agent_id_required")
            if name == "memagent":
                wrapped_agent_id = capability.metadata.get("agent_id")
                if task.metadata.get("exclude_native") or (
                    wrapped_agent_id
                    and task.agent_id
                    and str(wrapped_agent_id) == str(task.agent_id)
                ):
                    reasons.append("native_recursion_guard")
            if task.permissions.mcp_access != "none" and not capability.mcp:
                reasons.append("mcp_unsupported")
            if task.output_schema and not capability.metadata.get(
                "output_schema", False
            ):
                reasons.append("output_schema_unsupported")
            supported_network_modes = capability.metadata.get("network_modes")
            if (
                isinstance(supported_network_modes, list)
                and task.permissions.network not in supported_network_modes
            ):
                reasons.append("network_policy_unsupported")
            requested_tools = {
                str(item).strip().lower() for item in task.permissions.allowed_tools
            }
            forbidden_tools = {
                str(item).strip().lower()
                for item in capability.metadata.get("forbidden_tools", []) or []
            }
            if requested_tools.intersection(forbidden_tools):
                reasons.append("requested_tool_requires_external_isolation")
            if (
                task.permissions.allowed_tools or task.permissions.denied_tools
            ) and not capability.metadata.get("task_tool_policy", False):
                reasons.append("task_tool_policy_unsupported")
            if capability.metadata.get("authentication_configured") is False:
                reasons.append("authentication_missing")
            cost_reporting = bool(
                capability.metadata.get("cost_reporting", capability.usage_reporting)
            )
            token_reporting = bool(
                capability.metadata.get("token_reporting", capability.usage_reporting)
            )
            if task.budget.max_cost_usd is not None and not cost_reporting:
                reasons.append("cost_budget_telemetry_unsupported")
            if (
                task.budget.max_input_tokens is not None
                or task.budget.max_output_tokens is not None
            ) and not token_reporting:
                reasons.append("token_budget_telemetry_unsupported")
            isolation = str(task.metadata.get("execution_backend") or "local").lower()
            if capability.requires_external_isolation:
                if isolation not in {"docker", "remote", "sandbox"}:
                    reasons.append("external_isolation_required")
                if not capability.metadata.get("external_isolation_configured", False):
                    reasons.append("external_isolation_not_configured")
            if explicit and name != requested:
                reasons.append("not_requested")

            score = 0.0
            try:
                score += max(0, len(self.preference) - self.preference.index(name)) * 10
            except ValueError:
                pass
            history = dict((metrics or {}).get(name) or {})
            score += float(history.get("verified_success_rate") or 0.0) * 20
            score -= min(float(history.get("average_cost_usd") or 0.0), 100.0)
            candidates.append(
                {
                    "name": name,
                    "available": capability.available,
                    "version": capability.version,
                    "error_code": capability.error_code,
                    "error": capability.error,
                    "remediation": capability.remediation,
                    "score": round(score, 4),
                    "rejected_reasons": reasons,
                }
            )
            if not reasons:
                accepted.append((score, name, adapter, capability))

        if not accepted:
            detail = "; ".join(
                f"{item['name']}: {','.join(item['rejected_reasons']) or 'rejected'}"
                for item in candidates
            )
            if explicit:
                target = next(item for item in candidates if item["name"] == requested)
                reasons = list(target["rejected_reasons"])
                code = str(target.get("error_code") or "")
                if not code:
                    if "authentication_missing" in reasons:
                        code = "authentication_required"
                    elif "output_schema_unsupported" in reasons:
                        code = "output_schema_unsupported"
                    elif "unavailable" in reasons:
                        code = "harness_unavailable"
                    else:
                        code = "harness_not_ready"
                message = str(target.get("error") or "").strip() or (
                    f"Harness {requested!r} is not ready: {', '.join(reasons)}."
                )
                raise HarnessReadinessError(
                    code,
                    message,
                    harness=requested,
                    remediation=(
                        str(target.get("remediation"))
                        if target.get("remediation")
                        else f"Run `memorizz harness doctor {requested}` for setup details."
                    ),
                    candidates=candidates,
                )
            reported = [
                f"{item['name']}: {item['error']}"
                for item in candidates
                if item.get("error")
            ]
            message = f"No eligible harness is available ({detail})."
            if reported:
                message += " " + " ".join(reported[:3])
            raise HarnessReadinessError(
                "no_eligible_harness",
                message,
                remediation="Run `memorizz harness doctor` and configure at least one eligible harness.",
                candidates=candidates,
            )
        accepted.sort(key=lambda item: (-item[0], item[1]))
        _, name, adapter, _ = accepted[0]
        decision = HarnessRoutingDecision(
            selected=name,
            explicit=explicit,
            candidates=candidates,
            reason=(
                "explicit caller override passed capability and policy checks"
                if explicit
                else "highest deterministic capability, policy, health, and history score"
            ),
        )
        return adapter, decision


__all__ = [
    "HarnessReadinessError",
    "HarnessRouter",
    "HarnessRoutingDecision",
]

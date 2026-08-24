"""Deterministic, coverage-aware consolidation for multi-agent findings."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _strings(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    items = value if isinstance(value, (list, tuple, set)) else [value]
    result: List[str] = []
    for item in items:
        if isinstance(item, Mapping):
            text = json.dumps(dict(item), ensure_ascii=False, sort_keys=True)
        else:
            text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _json_value(text: str) -> Any:
    candidate = _FENCE_RE.sub("", str(text or "").strip())
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as original:
        decoder = json.JSONDecoder()
        for index, character in enumerate(candidate):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                return value
            except json.JSONDecodeError:
                continue
        raise original


@dataclass
class StructuredFinding:
    """One provider-neutral finding emitted by a delegate harness."""

    criterion_id: str
    title: str
    finding: str
    severity: str = "unspecified"
    evidence: List[str] = field(default_factory=list)
    missing_tests: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    contributors: List[str] = field(default_factory=list)

    @classmethod
    def from_value(
        cls, value: Mapping[str, Any], *, contributor: str
    ) -> "StructuredFinding":
        criterion_id = str(
            value.get("criterion_id")
            or value.get("finding_id")
            or value.get("id")
            or ""
        ).strip()
        if not _ID_RE.fullmatch(criterion_id):
            raise ValueError("A structured finding requires a safe criterion_id")
        title = str(
            value.get("title") or value.get("summary") or value.get("finding") or ""
        ).strip()
        finding = str(
            value.get("finding")
            or value.get("description")
            or value.get("summary")
            or ""
        ).strip()
        if not title or not finding:
            raise ValueError(
                f"Structured finding {criterion_id!r} requires title and finding"
            )
        return cls(
            criterion_id=criterion_id,
            title=title,
            finding=finding,
            severity=str(value.get("severity") or "unspecified").strip().lower(),
            evidence=_strings(value.get("evidence")),
            missing_tests=_strings(value.get("missing_tests") or value.get("tests")),
            source_ids=_strings(value.get("source_ids") or value.get("sources")),
            contributors=[str(contributor)],
        )

    @property
    def richness(self) -> int:
        return sum(
            len(value)
            for value in [
                self.title,
                self.finding,
                *self.evidence,
                *self.missing_tests,
                *self.source_ids,
            ]
        )

    def merge(self, other: "StructuredFinding") -> "StructuredFinding":
        """Merge duplicate criteria while keeping the richer explanation."""
        if self.criterion_id != other.criterion_id:
            raise ValueError("Only findings with the same criterion_id can be merged")
        richer, secondary = (
            (other, self) if other.richness > self.richness else (self, other)
        )
        return StructuredFinding(
            criterion_id=self.criterion_id,
            title=richer.title,
            finding=richer.finding,
            severity=(
                richer.severity
                if richer.severity != "unspecified"
                else secondary.severity
            ),
            evidence=_strings([*self.evidence, *other.evidence]),
            missing_tests=_strings([*self.missing_tests, *other.missing_tests]),
            source_ids=_strings([*self.source_ids, *other.source_ids]),
            contributors=_strings([*self.contributors, *other.contributors]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "title": self.title,
            "finding": self.finding,
            "severity": self.severity,
            "evidence": list(self.evidence),
            "missing_tests": list(self.missing_tests),
            "source_ids": list(self.source_ids),
            "contributors": list(self.contributors),
        }


def parse_structured_findings(
    text: Any, *, contributor: str
) -> List[StructuredFinding]:
    """Parse a fenced or plain JSON finding envelope."""
    payload = _json_value(str(text or ""))
    if isinstance(payload, Mapping):
        values = payload.get("findings")
    else:
        values = payload
    if not isinstance(values, list):
        raise ValueError("Structured delegate output requires a findings list")
    findings: List[StructuredFinding] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("Every structured finding must be an object")
        findings.append(StructuredFinding.from_value(value, contributor=contributor))
    return findings


def finding_ids(
    results: Iterable[Mapping[str, Any]]
) -> Tuple[List[str], Dict[str, str]]:
    """Return stable finding IDs plus parse errors for completed task results."""
    ids: List[str] = []
    errors: Dict[str, str] = {}
    for result in results:
        if result.get("status") != "completed":
            continue
        task_id = str(result.get("task_id") or "unknown")
        try:
            findings = parse_structured_findings(
                result.get("result"), contributor=task_id
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors[task_id] = f"{type(exc).__name__}: {exc}"
            continue
        for finding in findings:
            if finding.criterion_id not in ids:
                ids.append(finding.criterion_id)
    return ids, errors


def _render(findings: Sequence[StructuredFinding], missing: Sequence[str]) -> str:
    lines = ["# Findings", ""]
    for finding in findings:
        severity = finding.severity.upper()
        lines.extend(
            [
                f"## [{finding.criterion_id}] {finding.title} — {severity}",
                "",
                finding.finding,
                "",
            ]
        )
        if finding.evidence:
            lines.append("Evidence:")
            lines.extend(f"- {item}" for item in finding.evidence)
            lines.append("")
        if finding.missing_tests:
            lines.append("Missing tests:")
            lines.extend(f"- {item}" for item in finding.missing_tests)
            lines.append("")
        if finding.source_ids:
            lines.append("Sources: " + ", ".join(finding.source_ids))
            lines.append("")
    if missing:
        lines.extend(
            [
                "## Coverage warning",
                "",
                "No validated delegate finding was returned for: "
                + ", ".join(missing)
                + ".",
                "",
            ]
        )
    return "\n".join(lines).strip()


def consolidate_structured_findings(
    results: Iterable[Mapping[str, Any]],
    *,
    required_ids: Iterable[str] = (),
) -> Tuple[str, Dict[str, Any]]:
    """Merge delegate JSON without a synthesis model or evidence loss."""
    required = list(
        dict.fromkeys(str(item).strip() for item in required_ids if str(item).strip())
    )
    allowed_ids = set(required)
    merged: Dict[str, StructuredFinding] = {}
    parse_errors: Dict[str, str] = {}
    ignored_ids: List[str] = []
    parsed_count = 0
    accepted_count = 0
    completed_count = 0
    for result in results:
        if result.get("status") != "completed":
            continue
        completed_count += 1
        task_id = str(result.get("task_id") or "unknown")
        try:
            findings = parse_structured_findings(
                result.get("result"), contributor=task_id
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            parse_errors[task_id] = f"{type(exc).__name__}: {exc}"
            continue
        parsed_count += len(findings)
        for finding in findings:
            if allowed_ids and finding.criterion_id not in allowed_ids:
                if finding.criterion_id not in ignored_ids:
                    ignored_ids.append(finding.criterion_id)
                continue
            accepted_count += 1
            current = merged.get(finding.criterion_id)
            merged[finding.criterion_id] = (
                current.merge(finding) if current else finding
            )

    order = {criterion_id: index for index, criterion_id in enumerate(required)}
    findings = sorted(
        merged.values(),
        key=lambda item: (order.get(item.criterion_id, len(order)), item.criterion_id),
    )
    covered = [item.criterion_id for item in findings]
    missing = [criterion_id for criterion_id in required if criterion_id not in merged]
    report = {
        "completed_task_count": completed_count,
        "parsed_finding_count": parsed_count,
        "accepted_finding_count": accepted_count,
        "consolidated_finding_count": len(findings),
        "deduplicated_finding_count": max(0, accepted_count - len(findings)),
        "required_ids": required,
        "covered_ids": covered,
        "missing_ids": missing,
        "coverage_complete": not missing,
        "parse_errors": parse_errors,
        "ignored_ids": ignored_ids,
        "findings": [item.to_dict() for item in findings],
    }
    if not findings:
        raise ValueError("No valid structured findings were returned by delegates")
    return _render(findings, missing), report


__all__ = [
    "StructuredFinding",
    "consolidate_structured_findings",
    "finding_ids",
    "parse_structured_findings",
]

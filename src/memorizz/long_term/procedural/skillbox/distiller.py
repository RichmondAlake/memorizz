# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""LLM distillation of workflow trajectories into SKILL.md documents,
plus the machine validation gate that stands between distillation output
and context authority.

Distillation is a generalization step and can be wrong. A wrong skill with
elevated prompt position *authoritatively misleads* — so no skill is stored
without passing every check in :meth:`SkillDistiller.validate`.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

INSUFFICIENT = "INSUFFICIENT"

_RESULT_TRUNCATE_CHARS = 300
_MIN_LEAK_CHECK_LENGTH = 8

DISTILLATION_INSTRUCTIONS = """\
You are converting repeated successful agent trajectories into a reusable
skill for the same agent.

You will receive {success_count} successful run(s) and {failure_count}
failed run(s) of the same procedure. Each run lists: the user query, the
ordered tool calls with argument names/values, truncated results, and any
errors.

OUTPUT: one SKILL.md document, and nothing else. Constraints:
1. YAML frontmatter with exactly these keys: name, description,
   preconditions (list), tools (list).
2. `description` states WHEN to use this skill — the triggering intent —
   in one sentence. It must not enumerate the tool sequence.
3. `preconditions` are checkable facts about the query/state that must
   hold. Derive them from what varied vs. stayed constant across the runs.
   If no precondition can be stated, the skill is too specific — output
   exactly INSUFFICIENT and stop.
4. Procedure steps reference tools by name with argument SHAPES (which
   fields, sourced from where). Never copy literal argument or result
   values from the runs.
5. Include a "When NOT to apply" line under the "## When to apply"
   section, derived from the failure runs when present.
6. A "## Failure modes observed" section may list only failures actually
   present in the input runs.
7. Total length under {max_chars} characters.
8. If the runs do not support a generalization (contradictory sequences,
   unclear intent), output exactly: INSUFFICIENT

Document skeleton (follow it):

---
name: <short imperative name>
description: <one sentence: when to use this>
preconditions:
  - <checkable fact>
tools: [<tool_name>, ...]
---

# <same name>

## When to apply
<2-3 sentences restating applicability and when NOT to apply>

## Procedure
1. Call `<tool>` with <argument shape>. Expect <fields>.
...

## Failure modes observed
- <only if present in input runs>

## Worked example
<one concrete argument/result trace, values redacted to shapes>
"""


@dataclass
class ParsedSkillMd:
    """Frontmatter + body extracted from a distilled SKILL.md document."""

    name: str = ""
    description: str = ""
    preconditions: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    body: str = ""
    content: str = ""


@dataclass
class ValidationVerdict:
    ok: bool
    reasons: List[str] = field(default_factory=list)


def _parse_frontmatter_block(block: str) -> Dict[str, Any]:
    """Parse the constrained frontmatter grammar without requiring PyYAML.

    Handles scalar values, inline lists (``tools: [a, b]``) and dash lists.
    Falls back to ``yaml.safe_load`` when PyYAML is importable, since LLM
    output occasionally quotes values.
    """
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(block)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        pass
    except Exception:
        # Malformed YAML — fall through to the tolerant manual parser.
        pass

    result: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        dash = re.match(r"^\s+-\s+(.*)$", line)
        if dash and current_list_key:
            result[current_list_key].append(dash.group(1).strip().strip("\"'"))
            continue
        keyed = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if not keyed:
            continue
        key, value = keyed.group(1), keyed.group(2).strip()
        if not value:
            result[key] = []
            current_list_key = key
        elif value.startswith("[") and value.endswith("]"):
            items = [
                item.strip().strip("\"'")
                for item in value[1:-1].split(",")
                if item.strip()
            ]
            result[key] = items
            current_list_key = None
        else:
            result[key] = value.strip("\"'")
            current_list_key = None
    return result


def parse_skill_md(content: str) -> Optional[ParsedSkillMd]:
    """Split a SKILL.md document into frontmatter fields and body.

    Returns ``None`` when the document has no parseable frontmatter.
    """
    if not content or not content.strip():
        return None
    match = re.match(r"^\s*---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        return None
    front = _parse_frontmatter_block(match.group(1))
    if not isinstance(front, dict):
        return None

    def _as_list(value) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    return ParsedSkillMd(
        name=str(front.get("name") or "").strip(),
        description=str(front.get("description") or "").strip(),
        preconditions=_as_list(front.get("preconditions")),
        tools=_as_list(front.get("tools")),
        body=match.group(2),
        content=content.strip(),
    )


def _format_run(doc: Dict[str, Any]) -> str:
    lines = [f"User query: {doc.get('user_query') or '(none recorded)'}"]
    lines.append(f"Outcome: {doc.get('outcome', 'success')}")
    for step_key, step in (doc.get("steps") or {}).items():
        if not isinstance(step, dict):
            continue
        arguments = step.get("arguments")
        result = str(step.get("result", ""))[:_RESULT_TRUNCATE_CHARS]
        error = step.get("error")
        lines.append(f"  {step_key}")
        lines.append(f"    arguments: {arguments}")
        lines.append(f"    result (truncated): {result}")
        if error:
            lines.append(f"    error: {error}")
    return "\n".join(lines)


def _longest_argument_values(docs: List[Dict[str, Any]], count: int = 5) -> List[str]:
    """The longest literal argument values across sample runs — the values
    most likely to betray a copy-paste leak into the distilled skill."""
    values: List[str] = []
    for doc in docs:
        for step in (doc.get("steps") or {}).values():
            if not isinstance(step, dict):
                continue
            arguments = step.get("arguments")
            if not isinstance(arguments, dict):
                continue
            for value in arguments.values():
                text = str(value).strip()
                if len(text) >= _MIN_LEAK_CHECK_LENGTH:
                    values.append(text)
    values.sort(key=len, reverse=True)
    return values[:count]


class SkillDistiller:
    """Distill sampled runs into a SKILL.md and validate the result.

    ``resolve_tool`` is a callable ``(tool_name) -> bool`` answering whether
    the name resolves in the agent's toolbox — injected rather than bound to
    a Toolbox instance so the engine works against ToolManager, Toolbox, or
    a bare provider.
    """

    def __init__(
        self,
        llm_provider,
        resolve_tool: Callable[[str], bool],
        max_content_chars: int = 4000,
    ):
        self.llm_provider = llm_provider
        self.resolve_tool = resolve_tool
        self.max_content_chars = max_content_chars

    # ---------------------------------------------------------- distillation

    def distill(
        self,
        success_docs: List[Dict[str, Any]],
        failure_docs: List[Dict[str, Any]],
        previous_demotion_reason: Optional[str] = None,
    ) -> str:
        """One LLM call: sampled runs in, SKILL.md (or INSUFFICIENT) out."""
        sections = []
        for index, doc in enumerate(success_docs, start=1):
            sections.append(f"SUCCESSFUL RUN {index}\n{_format_run(doc)}")
        for index, doc in enumerate(failure_docs, start=1):
            sections.append(f"FAILED RUN {index}\n{_format_run(doc)}")
        prompt = "\n\n".join(sections)
        if previous_demotion_reason:
            prompt += (
                "\n\nNOTE: a previous version of this skill was demoted for: "
                f"{previous_demotion_reason}. The new version must address it."
            )
        instructions = DISTILLATION_INSTRUCTIONS.format(
            success_count=len(success_docs),
            failure_count=len(failure_docs),
            max_chars=self.max_content_chars,
        )
        output = self.llm_provider.generate_text(prompt, instructions=instructions)
        return (output or "").strip()

    # ------------------------------------------------------- validation gate

    def validate(
        self,
        skill_md: str,
        sample_docs: List[Dict[str, Any]],
        use_llm_judge: bool = True,
        failure_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> "tuple[ValidationVerdict, Optional[ParsedSkillMd]]":
        """Machine checks — ALL must pass before a skill gains authority."""
        reasons: List[str] = []
        failed_samples = list(failure_docs or [])
        all_samples = list(sample_docs) + failed_samples

        if not skill_md or skill_md.strip().upper().startswith(INSUFFICIENT):
            return (
                ValidationVerdict(False, ["distillation returned INSUFFICIENT"]),
                None,
            )

        parsed = parse_skill_md(skill_md)
        if parsed is None:
            return (
                ValidationVerdict(False, ["frontmatter failed to parse"]),
                None,
            )

        # 1. Required fields
        if not parsed.name:
            reasons.append("frontmatter missing 'name'")
        if not parsed.description:
            reasons.append("frontmatter missing 'description'")
        if not parsed.preconditions:
            reasons.append("frontmatter has no preconditions")

        # 2. Every declared tool resolves by name. A skill referencing a
        #    deleted/renamed tool is rejected here — and the same check runs
        #    for the skill's whole life in SkillMonitor.staleness_sweep.
        for tool_name in parsed.tools:
            try:
                resolvable = bool(self.resolve_tool(tool_name))
            except Exception:
                resolvable = False
            if not resolvable:
                reasons.append(f"tool not found in toolbox: {tool_name}")

        # 3. No hallucinated tools in procedure steps: every backticked
        #    "Call `x`" target must be declared in the frontmatter tools.
        declared = set(parsed.tools)
        for called in re.findall(r"[Cc]all(?:ing)?\s+`([^`]+)`", parsed.body):
            if called not in declared:
                reasons.append(f"procedure calls undeclared tool: {called}")

        # 4. Description must state intent, not dump the tool sequence.
        description_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", parsed.description)
        if description_tokens:
            tool_tokens = [token for token in description_tokens if token in declared]
            if len(tool_tokens) / len(description_tokens) > 0.5:
                reasons.append("description is a tool-sequence dump")

        # 5a. Length cap.
        if len(parsed.content) > self.max_content_chars:
            reasons.append(
                f"content exceeds {self.max_content_chars} chars "
                f"({len(parsed.content)})"
            )

        # 5b. Literal-value leak: the longest argument values from the
        #     source runs must not appear verbatim (privacy + it means the
        #     'generalization' memorized an instance).
        for value in _longest_argument_values(all_samples):
            if value in parsed.content:
                reasons.append(
                    f"literal argument value leaked into skill: {value[:40]!r}"
                )
                break

        if reasons:
            return ValidationVerdict(False, reasons), parsed

        # 6. LLM-as-judge — one cheap call. Reject on anything but a clear
        #    YES; distillation is stochastic, the trajectory stays eligible
        #    next cycle.
        if use_llm_judge and self.llm_provider is not None:
            try:
                judge_prompt = (
                    "SKILL DOCUMENT:\n"
                    + parsed.content
                    + "\n\nOBSERVED SUCCESSFUL RUNS:\n"
                    + "\n\n".join(_format_run(doc) for doc in sample_docs[:3])
                )
                if failed_samples:
                    judge_prompt += "\n\nOBSERVED FAILED RUNS:\n" + "\n\n".join(
                        _format_run(doc) for doc in failed_samples[:3]
                    )
                judge_answer = self.llm_provider.generate_text(
                    judge_prompt,
                    instructions=(
                        "Would this instruction document, applied to a query it "
                        "matches, plausibly reproduce the observed successful "
                        "runs while avoiding or correctly handling the observed "
                        "failed runs? Answer with YES or NO on the first line, "
                        "then one sentence of reasoning."
                    ),
                )
                first_line = (judge_answer or "").strip().splitlines()[0].upper()
                if "YES" not in first_line:
                    return (
                        ValidationVerdict(
                            False, [f"LLM judge rejected: {judge_answer[:200]}"]
                        ),
                        parsed,
                    )
            except Exception as exc:
                logger.warning("LLM judge unavailable, skipping check: %s", exc)

        return ValidationVerdict(True, []), parsed

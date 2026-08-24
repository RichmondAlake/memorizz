# Deep Research Mode

Deep research mode equips agents to explore large corpora, call specialized tools, and collaborate with peers before producing a final report. All agents participating in this workflow should be instantiated with `ApplicationMode.DEEP_RESEARCH` so they inherit the shared-memory-heavy stack and procedural tooling the mode enables.

## Quick Start (Workflow Wrapper)

Use `DeepResearchWorkflow` when you want a fully configured root/delegate/synthesis stack in a single call:

```python
from pathlib import Path

from memorizz.memagent.orchestrators import DeepResearchWorkflow
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

memory_provider = FileSystemProvider(
    FileSystemConfig(root_path=Path("~/.memorizz/memory").expanduser())
)

workflow = DeepResearchWorkflow.from_config(
    memory_provider=memory_provider,
    delegate_instructions=[
        "Financial researcher: gather revenue/profit/cash-flow trends.",
        "Market analyst: capture competitors, market share, and macro forces.",
        "Risk analyst: surface regulatory or operational risks.",
    ],
)

report = workflow.run(
    "Produce a 3-year stock analysis for Apple Inc. highlighting financial, market, and risk factors."
)
print(report)
```

The wrapper:

- Builds Deep Research MemAgents for the root, each delegate, and the synthesis stage.
- Automatically attaches the default internet access provider and context-window summarization tools.
- Returns a memo routed through the `DeepResearchOrchestrator` under the hood (so you still get shared-memory logs, `COMMAND/STATUS/REPORT` messages, etc.).

Use this path for most workflows. Drop down to manual wiring when you need per-agent customizations beyond the instruction text (e.g., bespoke tools per delegate, advanced prompting, or human-in-the-loop supervision).

## Memory Stack

- `MemoryType.TOOLBOX` — dynamic research tools (web search, document loaders, code runners).
- `MemoryType.SHARED_MEMORY` — blackboard coordination layer so agents can exchange directives, progress, and research artifacts.
- `MemoryType.KNOWLEDGE_BASE` — research knowledge base for source documents, citations, and reusable findings.
- `MemoryType.SHORT_TERM_MEMORY` — rolling scratchpad for intra-session reasoning plus summary references.

## Agent Topology

1. **Root Coordinator (MemAgent orchestrator)**
   - Breaks the incoming query into parallelizable sub-questions using its working memory.
   - Issues structured task briefs to delegate agents via shared memory. Briefs include deliverable schema, dependency graph handles, and the “no sub-spawning” constraint the package enforces.
   - Owns the global task ledger, context budgets, and escalation routing.

2. **Delegate Researchers (MemAgent instances bound to the same shared session)**
   - One delegate per sub-question. They follow the package pattern of reading the shared board for `COMMAND` instructions, posting `STATUS` updates, and persisting intermediate artifacts into short-term memory.
   - When blocked, they push a `STATUS:blocker` message to the root so it can re-scope or redirect; they never spawn new agents themselves.

3. **Synthesis Agent**
   - Subscribes to the shared session but only acts once `REPORT` entries exist for every sub-task.
   - Consolidates delegate reports, resolves conflicts using knowledge base lookups, and produces the final narrative plus confidence notes before sending it back to the root (and optionally the user).

This mirrors MemoRizz’s existing multi-agent pattern (see `docs/memory-types/shared.md`): root + delegates coordinated through `SharedMemory`.

## Communication & Coordination

- **Instruction-first protocol**: All shared-mem payloads are framed as `COMMAND`, `STATUS`, `REPORT`, or `QUESTION`. Delegates interpret the latest `COMMAND` snapshot as the canonical spec for their work.
- **Progress signaling**: Delegates log compact `STATUS` payloads (progress %, blockers, immediate next step) so the root can adjust directions mid-flight.
- **Report routing**: On completion, delegates post a structured `REPORT` (summary, evidence pointers, unresolved gaps) addressed to the synthesis agent while CC’ing the root.
- **Root directives**: The root agent processes every `STATUS`, updates the dependency map, and either issues new `COMMAND` instructions or requests clarifications via targeted `QUESTION`s.
- **Synthesis hand-off**: Once all reports are in, the root emits a `COMMAND:assemble` to the synthesis agent. The synthesizer compiles the master report, stores it in the knowledge base, and posts the final response/log bundle.

## Internet Access Defaults

Deep Research agents automatically attach an internet toolchain:

- If you set `MEMORIZZ_DEFAULT_INTERNET_PROVIDER` (optionally with `MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY`) or `FIRECRAWL_API_KEY`, the agent spins up that provider and exposes `internet_search` / `open_web_page` tools to the LLM.
- If no live provider is configured, MemoRizz falls back to an offline provider so the tools still exist and respond with guidance on how to enable real access.
- Call `agent.search_internet(...)` or `agent.fetch_url(...)` directly in Python to trigger the same functionality that the tools invoke.

## Auto-Summarization & Context Hygiene

- The DeepMemAgent registers the `context_window_stats` tool automatically, so LLMs can inspect their latest usage snapshot mid-turn.
- Once usage crosses ~85% of the configured window, the agent calls `generate_summaries()` for the recent conversation window, stores `{summary_id, short_description, token_estimate}` in a registry, and posts those IDs to shared memory. Use `list_context_summaries()` or the `list_summary_registry` tool to fetch the catalog, then call `fetch_summary(summary_id=...)` when you need the full text.
- This policy keeps prompts trim while ensuring the synthesizer can fetch the exact detail set it needs late in the workflow.

## Shared Memory Message Protocol

All coordination flows through typed messages enforced by `memorizz.coordination.shared_memory`:

- `COMMAND` – emitted by the root for every decomposed task (`command_id`, `target_agent_id`, `instructions`, `dependencies`).
- `STATUS` – periodic delegate updates (`progress`, `blockers`, `summary_ids`). Use `shared.list_messages(..., message_type=STATUS)` to audit execution and `get_latest_status` for per-agent snapshots.
- `REPORT` – final deliverables from delegates to the synthesis agent (findings, citations, summary references).

The `SharedMemory` helper now exposes `post_command`, `post_status`, `post_report`, plus guardrails so only the root can register additional sub-agents. Delegates read the canonical instructions directly from the shared board and never spawn peers on their own.

## Manual Wiring (Custom Agents)

```python
from pathlib import Path

from memorizz.memagent.builders import create_deep_research_agent
from memorizz.memagent.orchestrators import DeepResearchOrchestrator
from memorizz.internet_access import get_default_internet_access_provider
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

memory_provider = FileSystemProvider(
    FileSystemConfig(root_path=Path("~/.memorizz/memory").expanduser())
)
internet_provider = get_default_internet_access_provider()

root_agent = (create_deep_research_agent(
        "Root coordinator: break queries down, assign tasks, keep context under control."
    , internet_provider=internet_provider)
    .with_memory_provider(memory_provider)
    .build())

delegate_web = (create_deep_research_agent(
        "Web researcher focusing on fresh information",
        internet_provider=internet_provider,
    )
    .with_memory_provider(memory_provider)
    .build())

delegate_data = (create_deep_research_agent(
        "Data analyst producing structured tables",
        internet_provider=internet_provider,
    )
    .with_memory_provider(memory_provider)
    .build())

synthesis_agent = (create_deep_research_agent(
        "Synthesis expert with citation discipline",
        internet_provider=internet_provider,
    )
    .with_memory_provider(memory_provider)
    .build())

orchestrator = DeepResearchOrchestrator(
    root_agent=root_agent,
    delegates=[delegate_web, delegate_data],
    synthesis_agent=synthesis_agent,
)

response = orchestrator.execute("Compare the latest LLM safety benchmarks.")
print(response)
```

Each delegate repeats the builder call with role-specific instructions (researcher, analyst, writer) but the same session. The synthesizer’s prompt stresses conflict resolution, citation hygiene, and final logging.

> See `examples/deep_research/deep_research_memagent.ipynb` for a runnable deep-research walkthrough.

## Prompt Templates

- **Root coordinator prompt** – Mission statement (“decompose, assign, guardrails”), explicit communication grammar (`COMMAND/STATUS/REPORT/QUESTION`), rules for dependency tracking, escalation policy, and reminder that only the root can spawn new agents.
- **Delegate prompt** – Sub-question definition, deliverable schema, progress-report cadence, summary-ID policy, context-window check reminder, and the prohibition on spawning additional delegates (must escalate to root).
- **Synthesis prompt** – Input report checklist, gap-analysis instructions, reference validation steps against the knowledge base, expectation to archive the final response plus logs in shared memory and to output the user-facing deliverable.

## When to Use

- Competitive analysis and due diligence
- Multi-step investigations that span hours or days
- Reports that must capture reasoning traces and memory IDs for auditability

Deep research mode embraces deliberation: expect heavy use of short-term memory summarization, semantic cache invalidation, and explicit shared-memory coordination so multi-agent teams stay aligned without inflating the LLM context window.

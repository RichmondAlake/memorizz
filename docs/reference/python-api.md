# Python SDK API

This page covers the stable, high-value Python surface. Import public types from
`memorizz` unless a guide documents a provider-specific module.

## MemAgent

::: memorizz.memagent.core.MemAgent
    options:
      members:
        - run
        - run_stream
        - validate_configuration
        - capability_report
        - semantic_cache_stats
        - inspect_semantic_cache
        - invalidate_semantic_cache
        - generate_summaries
        - observability_summary
        - get_trace_context
        - build_personalization_context
        - last_memory_context_evidence
        - last_tool_outcomes
        - record_feedback
        - record_task_outcome
        - has_automations
        - create_automation
        - list_automations
        - get_automation
        - pause_automation
        - resume_automation
        - trigger_automation
        - list_automation_runs
        - delete_automation
        - list_approval_proposals
        - approve
        - reject
        - cancel_approval
        - resume_approval
        - close
        - lifecycle
      show_source: false
      show_root_heading: true

### Persistence

`agent.save()` stores the current definition in its configured memory provider.
`MemAgent.load(agent_id, memory_provider=provider, **overrides)` restores it;
omit `memory_provider` only when the definition lives in the default filesystem
provider.

## Personalization context

`PersonalizationContext` is the provider-neutral boundary between durable
memory and a host application's generation prompt. Cross-thread conversation
recall is disabled by default and becomes available only when the host opts in
with an exact `memory_id` and authenticated `user_id`.

```python
from memorizz import PersonalizationPolicy

personalization = agent.build_personalization_context(
    "Draft a LinkedIn post about memory observability",
    memory_id="workspace-7",
    user_id="user-42",
    exclude_thread_id="current-thread",
    preferences={"preferred_tone": "concise and technical"},
    policy=PersonalizationPolicy(
        conversation_recall=True,
        min_relevance_score=0.70,
        max_conversation_memories=2,
    ),
)

response = agent.run(
    "Draft the post.",
    memory_id="workspace-7",
    user_id="user-42",
    context={"personalization_context": personalization.to_dict()},
)
```

The rendered prompt tells the model to use memories only when they naturally
improve the current answer. `personalization.trace_summary()` and
`agent.last_memory_context_evidence()` contain counts, scores, attribute names,
and hashed references, not the underlying profile values or conversation text.

### Canonical user entities and duplicate cleanup

Use a stable application-owned `identity_key` for profile writes. Name changes
then update one deterministic entity instead of creating a second profile.
This field is deliberately absent from the model-facing `entity_memory_upsert`
tool: only a trusted host or application may bind a canonical identity.
Existing cross-name duplicates are never merged heuristically: preview an exact
selection first, inspect conflicts, and only then apply a reversible soft
supersession.

```python
from memorizz.long_term.semantic.entity_memory import EntityMemory

entities = EntityMemory(provider)
preview = entities.consolidate_duplicate_entities(
    memory_id="workspace-7",
    user_id="user-42",
    entity_ids=["reviewed-id-1", "reviewed-id-2"],
    canonical_identity_key="authenticated_user",
)

# After operator review of preview["groups"]:
applied = entities.consolidate_duplicate_entities(
    memory_id="workspace-7",
    user_id="user-42",
    entity_ids=["reviewed-id-1", "reviewed-id-2"],
    canonical_identity_key="authenticated_user",
    apply=True,
)
```

The apply path writes and verifies the merged canonical row before setting
`metadata.superseded_by` on aliases. It does not delete records. Anonymous
legacy rows require the separate explicit `migrate_legacy_scope` operator step;
authenticated runtime reads never adopt them automatically.

## MemAgentBuilder

::: memorizz.memagent.builders.agent_builder.MemAgentBuilder
    options:
      members:
        - with_name
        - with_instruction
        - with_model
        - with_llm_config
        - with_persona
        - with_favorite
        - with_memory_provider
        - with_memory_ids
        - with_application_mode
        - with_tools
        - with_tool
        - with_toolbox
        - with_mcp_servers
        - with_internet_access_provider
        - with_skills_marketplace
        - with_skill_paths
        - with_sandbox
        - with_sandbox_provider
        - with_browser_control
        - with_browser_control_provider
        - with_meta_harness
        - with_execution_harness
        - with_semantic_cache
        - with_embedding_provider
        - with_retrieval_policy
        - with_context_policy
        - with_tool_result_policy
        - with_approval_store
        - with_skills
        - with_skill_retrieval
        - with_skillbox
        - with_continual_learning
        - with_learning_control_plane
        - with_workflow_outcome_evaluator
        - with_delegation
        - with_delegates
        - with_semantic_layer
        - with_completion_policy
        - with_automations_enabled
        - with_default_timezone
        - with_self_aware
        - with_entity_memory
        - with_tool_access
        - with_max_steps
        - with_verbose
        - with_oracle_from_env
        - with_e2b_from_env
        - build
        - build_and_save
      show_source: false
      show_root_heading: true

## Meta-harness

::: memorizz.metaharness.service.MetaHarness
    options:
      members:
        - from_env
        - register
        - list_harnesses
        - probe
        - run
        - start
        - stream
        - approve
        - reject
        - resume_approval
        - resume_approval_start
        - cancel
        - retry
        - run_plan
        - compare
        - recover_interrupted_runs
        - close
      show_source: false
      show_root_heading: true

::: memorizz.metaharness.models.HarnessTask
    options:
      show_source: false
      show_root_heading: true

::: memorizz.metaharness.base.AgentHarness
    options:
      show_source: false
      show_root_heading: true

See the [memory-first meta-harness guide](../guides/meta-harness.md) for the
adapter security matrix and complete SDK, CLI, UI, and MCP workflows.

## Runtime policies

::: memorizz.tooling.ToolPolicy
    options:
      show_source: false
      show_root_heading: true

::: memorizz.tooling.ToolResultPolicy
    options:
      show_source: false
      show_root_heading: true

::: memorizz.tool_outcomes.ToolOutcome
    options:
      show_source: false
      show_root_heading: true

::: memorizz.tool_outcomes.ToolOutcomeStatus
    options:
      show_source: false
      show_root_heading: true

::: memorizz.tool_outcomes.ToolResult
    options:
      show_source: false
      show_root_heading: true

::: memorizz.tooling.ContextPolicy
    options:
      show_source: false
      show_root_heading: true

::: memorizz.retrieval.RetrievalPolicy
    options:
      show_source: false
      show_root_heading: true

::: memorizz.completion.CompletionPolicy
    options:
      show_source: false
      show_root_heading: true

## Model providers

::: memorizz.llms.llm_provider.LLMProvider
    options:
      show_source: false
      show_root_heading: true
      members:
        - generate
        - generate_stream
        - get_config
        - get_last_usage
        - get_context_window_tokens

## Memory providers

::: memorizz.memory_provider.base.MemoryProvider
    options:
      show_source: false
      show_root_heading: true
      members:
        - store
        - retrieve_by_query
        - retrieve_by_id
        - list_all
        - retrieve_conversation_history_ordered_by_timestamp
        - query_observability_records
        - invalidate_semantic_cache
        - store_memagent
        - delete_memagent
        - list_memagents
        - close

See the [custom provider contract](../memory-providers/custom.md) before
implementing this interface, particularly the three-state `user_id` filter.

## Memory evaluation

The normalized memory-suite SDK keeps diagnostic runs separate from exact
paper reproduction. Use `get_protocol_manifest()` before a run and inspect the
returned report's `comparison_label`, `paper_comparable`, and
`comparability_reasons` fields before publishing a comparison.

```python
from memorizz.benchmarks.memory_suite import (
    get_protocol_manifest,
    run_memory_suite,
    verify_dataset,
)

readiness = verify_dataset("longmemeval-v2", data_path="./datasets/lme-v2")
protocol = get_protocol_manifest("longmemeval-v2")

report = run_memory_suite(
    "longmemeval-v2",
    "./datasets/lme-v2",
    variant="standard-100",
    profile="smoke",
    workspace="./.memorizz-eval",
    memory_backend="filesystem",  # use "oracle" for Oracle AI Database
    candidate_pool_size=256,
    lexical_ratio=0.35,
    oracle_reader=True,
)
```

`MemorySuiteRunner` additionally accepts injected model, judge, embedding, and
memory-provider objects for deterministic tests. See the
[evaluation-suite guide](../evaluation-suite.md) for official-source sync,
strict comparability, CLI equivalents, and result interpretation.

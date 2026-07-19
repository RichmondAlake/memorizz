# Skillbox Module

Learned skills for MemoRizz agents: SKILL.md documents that live in the
database (`MemoryType.SKILLBOX`) with a lifecycle attached. Skills are
distilled from repeated successful workflow trajectories by the
`PromotionEngine`, injected into matching runs at a persisted user or reviewed
developer authority, monitored by the `SkillMonitor`, and demoted when they
drift. Optional passive evaluation observes later workflows for SHADOW skills
without injecting the skill, rerunning tools, or calling an LLM judge.

## Quick Start

```python
from pathlib import Path
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.long_term.procedural.skillbox import (
    Skillbox, PromotionEngine, PromotionConfig, SkillInjectionRole,
    SkillMonitor, SkillStatus,
)

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))
skillbox = Skillbox(provider, agent_id="my-agent")

engine = PromotionEngine(
    provider, skillbox,
    llm_provider=my_llm,                       # anything with .generate_text()
    config=PromotionConfig(
        require_shadow=True,
        skill_injection_role=SkillInjectionRole.DEVELOPER,
    ),
    agent_id="my-agent",
)
report = engine.run_promotion_cycle()
print(report.promoted, report.rejected, report.skipped)

for hit in skillbox.retrieve_skills_by_query("refund my order", limit=2):
    print(hit.similarity, hit.skill.name)
```

## With MemAgent

```python
agent = MemAgent(
    memory_provider=provider,
    continual_learning=True,
    continual_learning_config={
        "require_shadow": True,
        "skill_injection_role": "developer",
        "shadow_evaluation_enabled": True,
    },
)
# agent.continual_learning_manager owns the Skillbox/engine/monitor
```

## Notes

- Skill embeddings encode **applicability only** (name + description +
  preconditions + trigger queries) — never the procedure body or tool
  sequence. Retrieval must answer *when to do it*, not *how it was done*.
- `retrieve_skills_by_query` defaults to `min_similarity=0.70`, stricter
  than any other retrieval in the codebase: skills carry instruction
  authority, so false positives cost more than misses.
- `injection_role="user"` is the backward-compatible default.
  `injection_role="developer"` is rejected unless `require_shadow=True`;
  a reviewer must explicitly activate the shadow skill before it is used.
  Legacy records without the field load as `user`.
- `retrieve_shadow_skills_by_query` is a separate, background-only channel.
  First-party providers apply `status=shadow`, `agent_id`, and `user_id`
  before final top-k. Shadows never appear in normal prompt retrieval or
  `skills_activated`.
- `ShadowEvaluator` receives a small immutable post-store workflow snapshot,
  writes deterministic, versioned evidence to
  `Workflow.shadow_evaluations`, and reconciles
  `Skill.stats["shadow"]`. Its bounded queue uses `put_nowait`; worker,
  provider, and persistence failures never reach the user-facing run.
- `get_shadow_readiness(skill_id)` is advisory. It never activates a skill,
  and its observational metrics do not replace a reviewed canary or A/B test.
- Official OpenAI requests preserve the developer role. Anthropic places the
  reviewed instruction in its top-level system parameter, and providers
  without a native developer role use a system-equivalent mapping.
- Filesystem, MongoDB, and Oracle persist the same field. Existing Oracle
  schemas can run
  `memory_provider/oracle/migrations/002_add_skill_injection_role.sql` and
  `memory_provider/oracle/migrations/003_add_shadow_evaluations.sql`.
- `update_skill` patches lifecycle/stat fields only; identity and content
  are immutable in place (re-promotion creates a new version).
- Lifecycle: `candidate | shadow | active | deprecated | demoted`. Every
  exit from ACTIVE writes `demoted_at` + `demotion_reason` and clears the
  `promoted_skill_id` stamps on the source workflows so they retrieve
  again.
- See `docs/guides/continual-learning.md` for the full loop.

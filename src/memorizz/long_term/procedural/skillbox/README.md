# Skillbox Module

Learned skills for MemoRizz agents: SKILL.md documents that live in the
database (`MemoryType.SKILLBOX`) with a lifecycle attached. Skills are
distilled from repeated successful workflow trajectories by the
`PromotionEngine`, injected into matching runs as strong priors, monitored
by the `SkillMonitor`, and demoted when they drift.

## Quick Start

```python
from pathlib import Path
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.long_term.procedural.skillbox import (
    Skillbox, PromotionEngine, PromotionConfig, SkillMonitor, SkillStatus,
)

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))
skillbox = Skillbox(provider, agent_id="my-agent")

engine = PromotionEngine(
    provider, skillbox,
    llm_provider=my_llm,                       # anything with .generate_text()
    config=PromotionConfig(require_shadow=True),
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
    continual_learning_config={"require_shadow": True},
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
- `update_skill` patches lifecycle/stat fields only; identity and content
  are immutable in place (re-promotion creates a new version).
- Lifecycle: `candidate | shadow | active | deprecated | demoted`. Every
  exit from ACTIVE writes `demoted_at` + `demotion_reason` and clears the
  `promoted_skill_id` stamps on the source workflows so they retrieve
  again.
- See `docs/guides/continual-learning.md` for the full loop.

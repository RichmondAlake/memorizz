"""Create two MemAgents that hand parts of each request to harnesses.

1. Code review crew: Codex hunts bugs, Claude Code designs tests, pi checks the
   docs against the code, and the coordinator writes one review.
2. Research desk: Codex and Claude Code research different companies on the
   web at the same time, and the coordinator writes one comparison.

Each delegate is a saved agent in "run complete turns on a harness" mode, so
its share of a request runs on that harness. Running this script again reuses
agents it already created (matched by name).

    python examples/metaharness/multi_harness_delegates/setup_examples.py
    python .../setup_examples.py --memory-root ~/.memorizz/memory \\
        --coordinator-model anthropic/claude-sonnet-5-5

Restart ``memorizz ui`` afterwards if it is running against the same store, so
it lists the new agents.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from memorizz import MemAgent
from memorizz._env_io import load_layered_env, memory_root
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

HERE = Path(__file__).resolve().parent
SAMPLE_PROJECT = HERE / "sample_project"

EXAMPLES = {
    "code_review_crew": {
        "coordinator": {
            "name": "Code review crew",
            "instruction": (
                "You lead a code review. Give each delegate one job and let "
                "them work at the same time: the Codex bug hunter finds bugs "
                "and proves each one by running a failing input; the Claude "
                "Code test designer proposes pytest cases that would catch "
                "those kinds of bugs, without editing files; the pi doc checker "
                "compares README.md and the docstrings with the code and lists "
                "every mismatch. Then write one review with three sections: "
                "bugs (with the input that proves each), tests to add, and "
                "documentation fixes. Name which delegate found what."
            ),
        },
        "delegates": [
            {
                "name": "Codex bug hunter",
                "harness": "codex",
                "model": None,
                "instruction": (
                    "Finds bugs in Python code and proves each one by running "
                    "the code with an input that fails. Runs on Codex."
                ),
            },
            {
                "name": "Claude Code test designer",
                "harness": "claude-code",
                "model": "claude-sonnet-5-5",
                "instruction": (
                    "Designs pytest cases that would catch bugs, reading the "
                    "code and existing tests; never edits files. Runs on "
                    "Claude Code."
                ),
            },
            {
                "name": "pi doc checker",
                "harness": "pi",
                "model": None,
                "instruction": (
                    "Checks README files and docstrings against the code and "
                    "lists every mismatch. Runs on pi."
                ),
            },
        ],
        "try": (
            "Review this project: find the bugs in inventory.py, say which "
            "tests would catch them, and check README.md against the code."
        ),
        "workspace": str(SAMPLE_PROJECT),
        "network": "none",
    },
    "research_desk": {
        "coordinator": {
            "name": "Research desk",
            "instruction": (
                "You run a research desk. Split each question into parts by "
                "company or topic and give each delegate its own part so they "
                "work at the same time: the Codex researcher and the Claude "
                "Code researcher both search the web and cite sources. Then "
                "write one answer: a comparison table with a source for every "
                "figure, then a two-sentence takeaway. Say which delegate "
                "found each figure, and flag any figure the sources disagree "
                "on."
            ),
        },
        "delegates": [
            {
                "name": "Codex researcher",
                "harness": "codex",
                "model": None,
                "instruction": (
                    "Researches companies and markets on the web and cites a "
                    "source for every figure. Runs on Codex."
                ),
            },
            {
                "name": "Claude Code researcher",
                "harness": "claude-code",
                "model": "claude-sonnet-5-5",
                "instruction": (
                    "Researches companies and markets on the web and cites a "
                    "source for every figure. Runs on Claude Code."
                ),
            },
        ],
        "try": (
            "Compare Oracle, Microsoft and Alphabet: latest share price, "
            "change over the past month, and one headline from this week, "
            "each with a source."
        ),
        "workspace": "",
        "network": "full",
    },
}


def _llm_config(model: str) -> dict:
    provider, _, name = model.partition("/")
    return {"provider": provider, "model": name} if name else {"model": model}


def _by_name(provider) -> dict:
    found = {}
    for agent in provider.list_memagents() or []:
        name = (
            agent.get("name")
            if isinstance(agent, dict)
            else getattr(agent, "name", None)
        )
        agent_id = (
            agent.get("agent_id")
            if isinstance(agent, dict)
            else getattr(agent, "agent_id", None)
        )
        if name and agent_id:
            found[name] = agent_id
    return found


def setup(memory_path: Path, coordinator_model: str) -> dict:
    provider = FileSystemProvider(
        FileSystemConfig(root_path=memory_path, lazy_vector_indexes=True)
    )
    existing = _by_name(provider)
    llm = _llm_config(coordinator_model)
    created = {}
    try:
        for key, example in EXAMPLES.items():
            delegates = []
            for spec in example["delegates"]:
                if spec["name"] in existing:
                    delegates.append(
                        MemAgent.load(existing[spec["name"]], memory_provider=provider)
                    )
                    continue
                harness_config = {"model": spec["model"]} if spec["model"] else {}
                delegate = MemAgent(
                    llm_config=llm,
                    memory_provider=provider,
                    name=spec["name"],
                    instruction=spec["instruction"],
                    meta_harness=True,
                    meta_harness_mode="runtime",
                    default_harness=spec["harness"],
                    harness_config=harness_config,
                )
                delegate.save()
                delegates.append(delegate)
            name = example["coordinator"]["name"]
            if name in existing:
                coordinator_id = existing[name]
            else:
                coordinator = MemAgent(
                    llm_config=llm,
                    memory_provider=provider,
                    name=name,
                    instruction=example["coordinator"]["instruction"],
                    delegates=delegates,
                    delegation={
                        "enabled": True,
                        "mode": "auto",
                        "max_workers": len(delegates),
                        "allow_root_fallback": True,
                    },
                )
                coordinator.save()
                coordinator_id = coordinator.agent_id
            created[key] = {
                "coordinator": {"name": name, "agent_id": coordinator_id},
                "delegates": [
                    {
                        "name": d.name,
                        "harness": d.default_harness,
                        "agent_id": d.agent_id,
                    }
                    for d in delegates
                ],
                "try": example["try"],
                "workspace": example["workspace"],
                "network": example["network"],
            }
    finally:
        provider.close()
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--memory-root", default=os.environ.get("MEMORIZZ_MEMORY_ROOT"))
    parser.add_argument("--coordinator-model", default="anthropic/claude-sonnet-5-5")
    args = parser.parse_args()
    load_layered_env()
    root = Path(args.memory_root).expanduser() if args.memory_root else memory_root()
    for key, example in setup(root, args.coordinator_model).items():
        coordinator = example["coordinator"]
        print(f"\n{coordinator['name']}  ({coordinator['agent_id']})")
        for delegate in example["delegates"]:
            print(f"  delegate: {delegate['name']} on {delegate['harness']}")
        print(
            f"  workspace: {example['workspace'] or 'blank (a fresh scratch folder)'}"
        )
        print(f"  network: {example['network']}")
        print(f"  try: {example['try']}")


if __name__ == "__main__":
    main()

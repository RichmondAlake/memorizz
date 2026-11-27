# One Memory, Three Kinds of Agent

Three notebooks on shared memory: how agents that were never introduced to
each other end up giving the same answer, because they all look in the same
place. The first shares one memory between agents you build, the second
between coding agents you don't, and the third with the personal assistants
you only configure.

## The use case: Fern & Ember

All three run on one world. **Fern & Ember** is a fictional online plant shop
planning a **Spring Sale**. One decision (the dates, the discounts, what is
excluded) has to reach the copywriter, the support desk, the shop's code and
the owner's phone, and it changes once along the way.

| | |
|---|---|
| **The decision** | The Spring Sale runs 3 to 9 March 2027, later extended to 12 March. 20% off houseplants, 10% off pots, gift cards excluded |
| **Agents** | Mara (merchandiser, owns the decision), Cole (copywriter), Sam (support), Ona (shop manager, coordinates the other three) |
| **Harnesses** | pi writes the brief, Codex builds `sale_price()`, Claude Code reviews it |
| **Assistants** | A stand-in called Pocket, then Muse, Grok Bot and a ChatGPT dot |
| **The memory** | One memory ID, `fern-ember`, in one store: the folder `fern_ember_store` beside the notebooks |

## The notebooks

Run them in order. Each adds to the store the one before it left.

| | Notebook | What it shows | Needs |
|---|---|---|---|
| 1 | [Agents sharing one memory](01_agents_sharing_one_memory.ipynb) | Three agents with three memories know 0 of 3 facts; in one memory ID they know 3 of 3. What "shared" is made of, how a change reaches every agent, readers that cannot rewrite a record, what stays private, a coordinator's board, and who wrote what | An OpenAI key |
| 2 | [Harnesses sharing one memory](02_harnesses_sharing_one_memory.ipynb) | Codex, Claude Code and pi put the sale into the shop's code from the record Mara wrote, with nobody pasting a rule. Approval before an edit, a host check after, a refine cycle that turns answers into findings, and Sam answering from what a harness did | Notebook 1, and at least one of Codex, Claude Code or pi signed in |
| 3 | [Assistants on the same memory](03_assistants_on_the_same_memory.ipynb) | The memory behind an OAuth sign-in as a remote MCP server. A stand-in assistant signs in step by step, reads the decision and saves a note that Sam and a harness then use. Then the steps for Muse, Grok Bot and dots | Notebook 1. A public HTTPS tunnel only for the real apps |

Notebook 1 starts the shop from empty each time it runs (`FRESH = True`).
Notebooks 2 and 3 never wipe the store; each removes only what its own
earlier runs left.

## See it in the console

Every notebook prints the command. It opens the same store with the same
embedding model:

```bash
MEMORIZZ_HOME=$PWD/fern_ember_home \
MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai \
MEMORIZZ_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small \
MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=256 \
memorizz ui --backend filesystem --path $PWD/fern_ember_store
```

Open **Memory types → Shared → `fern-ember`** for the whole cast on one page:
who writes the memory's records, who reads them, the harnesses that ran in
it, the apps that saved to it, the Spring Sale record and who last changed
it.

## Version

The notebooks need MemoRizz 0.17.27 or later. Three things they use arrived
in that release: readers of entity memory (`policy={"writes": False}`),
harnesses sharing a memory with a refine cycle, and `memorizz mcp apps serve
--memory-id`. Each notebook installs MemoRizz from PyPI if it is missing;
upgrade an older install with `pip install -U "memorizz[ui,filesystem,mcp]"`.

The diagrams are Mermaid, kept as `.mmd` sources beside their rendered `.png`
in `diagrams/`.

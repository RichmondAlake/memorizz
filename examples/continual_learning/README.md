# Continual Learning in Token Space

Five notebooks on how a MemoRizz agent gets better at its job without anyone
touching the model's weights. What changes is the *tokens* the harness puts in
front of the model (skills, persona, facts, tool descriptions), which makes
every lesson something you can read, review, measure and undo.

## The use case: Calderwood Insurance

All five run on one world. **Calderwood Insurance** is a fictional UK insurer,
and **Ada** is its internal claims support assistant: handlers ask her about
status codes, error codes and payment limits, and she has tools into the
claims systems. Everything about Calderwood is invented, so no model knows any
of it before it is told.

| | |
|---|---|
| **People** | Priya Shah (Grade 2 handler, the main user), Tom Okafor (Grade 1, the newcomer), Senior Handlers Gwen and Ruth, Desk Leads Dev and Callum |
| **Systems** | Beacon (claims, `CW-` references, `Q` status codes), Atlas (policies), Tally (payments), the Ringfence (sanctions screening), Drop then Compass (customer uploads) |
| **Teams** | Counter Fraud Unit (CFU), Financial Crime, Large Loss Unit (LLU), Settlement Review Team (SRT) |
| **House terms** | TPA (Total Payout Authority), Amber Pack, HSR, RVF, IBNR, CAT |

Calderwood's knowledge arrives in rounds, and the notebooks sit inside them:

| Round | When | What changes |
|---|---|---|
| 1 | September 2026 | The baseline: Drop in use, Grade 1 TPA £2,500, an Amber Pack due within 2 working days |
| 2 | January 2027 | Compass replaces Drop; Grade 1 TPA rises to £3,000; Amber Pack deadline becomes 3 working days; payments over £25,000 need an SRT clearance |
| 3 | April 2027 | Motor total loss deadline falls from 10 to 7 working days |

The people, systems, codes and limits live in one small module,
[`calderwood_world.py`](calderwood_world.py), so a fact has one spelling
across the series. Each notebook keeps its own fixtures (the claims on the
desk that day, the requests a handler makes) in its cells, where you can read
them.

## The notebooks

| # | Notebook | Calderwood scene | What evolves | From what evidence |
|---|---|---|---|---|
| 01 | [Workflows to skills](01_workflows_to_skills.ipynb) | Paying a new payee: look up the claim, check the handler's TPA, screen the payee, then release or stop | A procedure that keeps succeeding becomes a `SKILL.md`; it is demoted when the January rule change breaks it | Workflow memory |
| 02 | [Persona evolution](02_persona_evolution.ipynb) | Handlers keep asking for plain-English customer updates; a Desk Lead keeps correcting drafts that mention a fraud referral | One evidenced goals sentence, and a standing rule proposed from two handlers' corrections and approved by a Desk Lead | The episodic log |
| 03 | [Entity evolution](03_entity_evolution.ipynb) | Grade 1 TPA: a rumour, then the real January change; Drop retired; the CFU filed under two names | Values confirmed, superseded with their history kept, and dated twice; duplicates merged; notes that still state an old value flagged | Conversations, under a source rule |
| 04 | [Toolbox augmentation and evolution](04_toolbox_evolution.ipynb) | Handlers say "this is a large loss" and "one for Counter Fraud"; Ada passes claim numbers without `CW-` and uploads files over 25MB | Each tool's description: enriched at registration, then revised from real usage | The tool log |
| 05 | [The refine cycle](05_refine_cycle.ipynb) | One January 2027 day, tidied overnight | All of the above, as one ordered offline pass of eleven jobs with one report | Transcripts and existing memory |

Read them in order, or start with 05 for the whole picture and go back for
the detail. Each one stands on its own.

`continual_learning_guide.ipynb` is the earlier, longer guide to the
workflow-to-skill loop on Oracle AI Database, with a three-arm comparison of
raw workflows against skills at user and developer authority.

## Run them

From this folder, in any Python 3.10+ environment with Jupyter:

```bash
export OPENAI_API_KEY="..."      # or put it in a .env file in this folder or above it
jupyter lab
```

- **MemoRizz is installed from PyPI.** Each notebook's first cell runs
  `pip install --upgrade "memorizz[ui,filesystem]"`, so there is nothing to
  set up first. The notebooks need MemoRizz 0.17.25 or later. Keep
  `calderwood_world.py` beside them.
- **One store for the whole series.** Every notebook writes to
  `calderwood_store/`, a folder beside the notebooks (gitignored). Delete it
  to start again. Your own MemoRizz home is left alone.
- **One small, current model.** The chat model and the background model both
  default to `gpt-6-luna`, the cheapest of OpenAI's current models. Override
  them with `MEMORIZZ_NB_CHAT_MODEL` and `MEMORIZZ_NB_BACKGROUND_MODEL`.
- **Cost and time.** Each notebook makes real model calls and runs in a few
  minutes.

## See the results in the MemoRizz UI

Because the notebooks share one store, the local UI can open it. From this
folder:

```bash
MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai \
MEMORIZZ_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small \
MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=256 \
memorizz ui --backend filesystem --path calderwood_store
```

Each notebook ends with a "See it in the MemoRizz UI" section that names the
agents it made and the pages to open: the conversation in the playground, and
the learning pages that show what changed (Workflows and Skills, Persona
evolution, Entities and Memory evolution, the Learning control plane). The
screenshots in the notebooks are in `images/` and were taken from that UI
after the saved run.

The flowcharts and sequence diagrams are embedded in the notebooks
themselves, so they show in any notebook viewer. Their Mermaid sources are
in `images/diagrams/` as `.mmd` files.

Every notebook also runs a twin agent with the learning switched off through
the same work, so each result has something to be compared against.

## Reading the saved outputs

The notebooks are saved with the outputs of a real run. Model replies vary,
so your numbers will differ from the saved ones; the markdown says where a
result depends on the run. These are demonstrations of mechanism on small
fixtures, not benchmarks. Each notebook has a section on what its numbers do
and don't show, including the points MemoRizz does not cover yet.

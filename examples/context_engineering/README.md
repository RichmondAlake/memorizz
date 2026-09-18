# Persistent notes and searchable history

Two educational notebooks implement the memory behavior described in the
user-supplied context-management paragraph from the
[GPT-6 Astra announcement](https://openai.com/index/gpt-6-astra/): maintain notes
across context windows and search original messages and tool outputs for details
the notes omitted. They distinguish that description from our implementation
choices and from the internal mechanisms the paragraph does not disclose.

| Notebook | Storage | Requirements |
|---|---|---|
| [Filesystem edition](persistent_notes_filesystem.ipynb) | JSON files and exact vector search | Python 3.10+, Jupyter, OpenAI API key |
| [Oracle AI Database edition](persistent_notes_oracle.ipynb) | Oracle records and exact VECTOR-distance search | OpenAI API key, existing Memorizz Oracle schema, database connection, compatible vector dimensions |

Both editions contain the same three-window debugging story, detailed Markdown,
Mermaid architecture and sequence diagrams, note revisions, a provider restart,
an omitted-detail recovery exercise, bounded source expansion, assertions, and
a GPT-6-written next-step note. Each notebook is self-contained: download the
single `.ipynb` file and run its cells in order. All implementation code appears
in short teaching cells of at most 25 lines, with no companion Python module.

## Run

Open the downloaded notebook in Jupyter. Its first cell installs the published
PyPI package into the active kernel:

```bash
pip install memorizz
```

For Oracle, use `pip install "memorizz[oracle]"`. The notebook installation cells
pin `0.10.0` for reproducibility. No editable install, repository checkout,
`sys.path` modification, or unreleased source changes are needed. If you need
Jupyter, install `jupyterlab` in your environment first.

Both notebooks read `OPENAI_API_KEY` from the environment or a private `.env`
beside the notebook, with a hidden-input prompt if it is missing. Never put a key
in notebook source or saved outputs. Running the lessons makes paid OpenAI API
requests.

The Oracle edition also reads `ORACLE_USER`, `ORACLE_PASSWORD`, and `ORACLE_DSN` from
the environment, local `.env`, or hidden input. It does not start a database, provision
users, or fall back to another provider. Preflight and vector-dimension validation
must pass. Both editions request 384-dimensional OpenAI embeddings by default; set
`MEMORIZZ_CONTEXT_DIMENSIONS` to match the prepared schema. Provider construction
may initialize missing Memorizz tables.

## Models and the teaching scenario

Every agent turn uses **OpenAI GPT-6 Astra (`gpt-6-astra`)** through Memorizz's
Responses API adapter, with `reasoning_effort="low"` and a 4,096-token output
limit per request. All indexing and query vectors use **`text-embedding-3-small`**.
The notebooks contain no scripted model or substitute embedding implementation.
The filesystem edition still needs internet access for generation and embeddings.

The 384-vector length uses OpenAI's supported `dimensions` parameter and matches
the prepared Oracle schema. New embeddings must use the same model and dimension
for indexing and querying. Changing either requires a fresh scope and reindexing.
See the [GPT-6 API guide](https://developers.openai.com/api/docs/guides/latest-model)
and [embedding dimensions guide](https://developers.openai.com/api/docs/guides/embeddings#reducing-embedding-dimensions).

The regression output and build log are sample artifacts, read by tools without
running a test suite. The initial notes are curated to omit a diagnostic. GPT-6
must then recover it through search and source expansion before writing its own
next-step note. `RecordedOpenAI` only observes real requests and responses; it
does not choose tools, generate answers, or fabricate usage. Assertions verify
the observed calls, facts, citations, persistence, and isolation. Model wording
and search queries can vary, and a successful lesson is not a general quality
benchmark or a reconstruction of Codex internals.

## Package compatibility

The published 0.10.0 wheel supplies the providers, embeddings, scoped vector
search, and Responses adapter. The notebooks implement their journal using
public `store`, `retrieve_by_id`, `search_memory`, and `delete_by_id` methods.
Notes and complete sources use private shared-memory records; searchable chunks
use the knowledge base.

Memorizz 0.11.0 includes the improvements below. The notebooks retain their
0.10.0 pin and self-contained implementation for reproducibility; these newer
APIs are useful alternatives, rather than requirements for running the lessons:

- `persist_all_results` and direct `search_tool_logs` are absent from 0.10.0.
  Notebook work tools capture full results explicitly before returning, and
  search uses a source-linked vector index.
- The 0.11.0 Oracle metadata and physical tool-log ID read fixes are not required.
  Canonical source reads use shared-memory logical IDs; chunk metadata comes
  from search results.
- GPT-6 context mappings are absent from 0.10.0. The notebooks set the documented
  context size and output limit explicitly and select the Responses API.

The note schema and rollover controller remain example policy. Their limitations
are explicit: one writer, no cross-record transaction, no complete request
snapshot capture, and no automatic rollover inside an in-flight tool loop.
Large histories need ranking evaluation, index repair, and efficient pagination.
See each notebook's gap table and evaluation section.

## Data lifecycle

Each full run creates a unique task/user scope. The final cell removes only the
lesson's recorded archive, index, note, checkpoint, conversation, and tool-log
rows and closes the provider. Set `MEMORIZZ_CONTEXT_KEEP_DATA=1` to retain them
for inspection. Separate Memorizz observability records follow their own retention
policy. Interrupted runs may leave tutorial records; retain the printed task/user
IDs and the filesystem path, when applicable, to reopen and clean up that run.

Re-run the whole notebook for a new lesson. Re-running a note-writing cell with
an obsolete expected revision intentionally fails instead of rewriting history.

# Tool learning: descriptions that improve with use

A tool's entry in toolbox memory is written once, from its docstring. Three
things then go wrong that the docstring's author could not have seen:

- **Calling.** The model reaches the tool and passes arguments it refuses,
  such as an id without its prefix. It reads the error and tries again, on
  every request.
- **Selection.** A request that should reach the tool does not, because the
  team's word for the task is not in the description.
- **Confusion.** A request keeps going to this tool and belongs to a
  neighbour.

MemoRizz handles both in two steps. Augmentation enriches the entry when the
tool is registered. Tool learning revises it later, from how the tool was
called.

## Augmentation at registration

```python
from memorizz import Toolbox

toolbox = Toolbox.from_functions(
    [lookup_claim, start_recovery],
    memory_provider=provider,
    llm_provider=model,
    agent_id="claims-desk",
    augment=True,
)
```

With `augment=True`, each tool costs one model call, which writes:

- a **description** of what the tool is for, in the words people use when
  they ask;
- six **example requests** it should handle.

Both are stored beside the docstring, which is kept as written. The argument
schema always comes from the function's signature, never from the model. The
reply has to be JSON with a description and a list of requests; otherwise the
tool is registered with its plain docstring and no examples.

A tool row is embedded from its name, description, argument names and
example requests, so routing can find it by any of them.

!!! note "Changed behaviour"
    `augment=True` used to make three model calls and store their replies as
    they came, which could put conversational text in the description and
    the example requests. It now makes one call and checks the reply. Rows
    registered earlier are left as they are. The text a tool row is embedded
    from now includes its example requests and prefers the description to
    the docstring; rows already embedded keep their vector until they are
    next written.

## Revising an entry from the tool log

```python
review = agent.review_tools()             # propose; dry_run=True stores nothing
for revision in review.revisions:
    print(revision.tool, revision.notes_added, revision.queries_added)
    agent.apply_tool_revision(revision.revision_id, approved_by="operator-7")

agent.undo_tool_revision("lookup_claim", approved_by="operator-7")
agent.tool_revisions()                    # history, newest first
```

A review reads the agent's recent tool calls and, for each tool with
something to fix, asks the [background model](memory-consolidation.md#the-background-model)
for three things. The code adds a fourth itself:

| | What it fixes | What it rests on |
|---|---|---|
| A **usage note** | A calling error that repeats | At least `min_failures` (2) failed calls |
| An **example request** | A wording that should have reached the tool | At least `min_misses` (2) requests that did not |
| A **"Not for" line** | Requests that went to this tool and another served | At least `min_confusions` (2) such requests, all served by the tool it names |
| An **example call** | The shape of a call that works | The calls that worked, for a tool with `min_failures` refused ones |

A request "did not reach" a tool in one of two ways, both found by the code:

- the tool served it, although routing would not have offered the tool for
  it (the agent had to go looking);
- the user asked, the run did not use the tool, and the next request in the
  same conversation did. The first wording is the one that was missed.

A request counts for the last tool its run called successfully. A lookup
made on the way to a change is preparation, not what the request was for.

### Requests that went to the wrong tool

A request "went to" a tool and was served by another in one of two ways,
again both found by the code:

- the run used the tool, the user asked again, and the next request reached
  the other tool;
- routing ranked the tool first and left the tool that served the request
  out of those it offered. A tool the run called on the way (a lookup before
  a change) does not count.

Two such requests bring a tool to review even when it has fewer than
`min_calls` calls. The line the model writes has one shape: `Not for <kind of
request>; use <tool>.` It is shown to the model with the tool and kept out of
the text routing searches, because a description that lists what a tool is
not for attracts exactly those requests.

```python
revision.avoid_added     # ["Not for large loss referrals; use refer_llu."]
revision.evidence        # [{"kind": "confused_request", "served_by": "refer_llu", "how": "re_asked", ...}]
```

### The example call

For a tool whose calls keep being refused, the code builds one line from the
calls that worked, with no model involved:

```
Example call: {"claim_ref": "CW-######", "file_mb": "<number>"}
```

Digits are masked. A string is shown as a format only when every successful
call has the same masked form and little of it is fixed (a prefix, a
separator); anything else is `<text>`. No argument value is kept. The line
is in `revision.call_example`, and is applied and undone with the rest.

### What the code checks

The model proposes. The code decides what is written:

- A note cites the failed calls it rests on, and an example cites the missed
  requests it rests on and shares a word with each.
- Neither may contain an argument value from the calls (an id, a name, an
  amount), or text aimed at the model.
- A "Not for" line starts with those words, names a registered tool other
  than its own, cites requests that tool served, and shares a word with each.
- A note or "Not for" line is one line of at most 220 characters. A tool
  keeps at most 4 notes, 2 "Not for" lines, one example call and 12 example
  requests.
- Nothing the author wrote is rewritten. Notes go under a "Learned from use"
  heading after the description, and the schema is never touched.
- When a revision is applied, routing is checked again for the requests
  other tools served. If the new examples push another tool out of the tools
  offered for one of them, the examples are withdrawn and only the notes
  stay (`queries_withdrawn`, with the requests in `displaced`).

What the model proposed and the code refused is returned with the reason, in
`revision.refused` and `review.refused`.

### Who reads what

| | Routing searches | The model is shown |
|---|---|---|
| The author's docstring | When there is no description | Yes |
| The augmented description and its examples | Yes | No |
| Learned usage notes, "Not for" lines, the example call | No | Yes |
| Learned example requests | Yes | No, unless `show_learned_examples` is on |

A usage note says how to call the tool, so it is kept out of the text
routing searches. An example request is the reverse: it is a retrieval key,
written in a user's words, and a user's words do not belong in the prompt as
if they were instructions. So by default the model never reads one.

!!! note "Changed behaviour"
    Learned example requests used to be shown to the model too, on an "Also
    asked for as" line. They are now hidden by default. To show them:

    ```python
    agent.configure_tool_learning(show_learned_examples=True)
    ```

    The setting is stored with the agent's learning records and applies to
    every later session. When the model needs to know that a house term
    belongs to a tool, a "Not for" line on the neighbour it was confused
    with usually does the job without quoting anybody.

### Where an example came from

Every example request on a tool is one of three kinds
(`ToolLearner(agent).example_sources(tool)`):

| Kind | Written by | Can give way |
|---|---|---|
| `synthetic` | Augmentation at registration | Yes |
| `learned` | A revision, from real requests | Only by undoing the revision |
| `unknown` | Something else (a row edited by hand, an old row) | No |

Real wording displaces invented wording:

- a learned example that says the same as a synthetic one replaces it;
- at the cap (`max_queries`), a new learned example takes the place of the
  oldest synthetic one. With no synthetic one left, it is refused as
  `no_room`;
- one that repeats a learned example is refused as `already_kept`.

What gave way is in `revision.queries_displaced`, and undo puts it back.
Rows registered from now on carry `queries_source: "augmentation"`. An older
row counts as augmented when its description differs from its docstring.

### Tools nobody reaches

Each turn in which routing chooses among the tools, MemoRizz counts which
ones it offered (one small record per agent). With the tool calls, that
gives two flags:

```python
agent.tool_usage_flags()
# [{"tool": "set_rvf", "flag": "retrieved_never_called", "offered_turns": 9, "calls": 0, ...},
#  {"tool": "request_cat_export", "flag": "never_retrieved", "offered_turns": 0, ...}]
```

| Flag | Means | Usual cause |
|---|---|---|
| `retrieved_never_called` | Offered in `min_offers` (5) turns or more, never called | A description that matches requests the tool is not for |
| `never_retrieved` | Never offered and never called, over `min_turns` (20) routed turns | A description nobody's wording matches, or a tool nobody needs |

A review carries the same list in `review.flags`. A flag is for a person to
read: nothing is proposed or changed from it. Nothing is counted while every
tool is sent on every turn (a small toolbox under
`ContextPolicy.stable_tool_list_tokens`), because routing chose nothing.

### Review, apply, undo

A revision is stored as `proposed` and waits for
`apply_tool_revision(..., approved_by=...)`. `review_tools(auto_apply=True)`
applies it at once as `memorizz:tool_learning`. Each revision keeps:

- `evidence`: the requests and errors it rests on, with argument names but
  not argument values;
- `ranks`: where routing ranked the tool for each missed request, before and
  after;
- `approved_by` and `status` (`proposed`, `applied`, `dismissed`, `undone`).

`undo_tool_revision(tool, approved_by=...)` takes back the newest applied
revision of a tool. It refuses when the entry was edited since. A tool with
a revision waiting is not reviewed again, and a tool with no new calls since
its last review costs no model call.

## Settings

Pass `config=` to `review_tools`, as a dict or a `ToolLearningConfig`:

| Setting | Default | Meaning |
|---|---|---|
| `min_calls` | 3 | Calls of a tool before it is reviewed |
| `min_failures` | 2 | Failed calls a usage note must rest on |
| `min_misses` | 2 | Missed requests an example request must rest on |
| `max_notes` | 4 | Usage notes kept on one tool |
| `max_queries` | 12 | Example requests kept on one tool |
| `max_new_queries` | 4 | Example requests one review may add |
| `max_calls_per_tool` | 24 | Most recent calls read for one tool |
| `max_tools` | 8 | Tools reviewed in one pass, those with the most failed calls first |
| `min_confusions` | 2 | Wrongly routed requests a "Not for" line must rest on |
| `max_avoid` | 2 | "Not for" lines kept on one tool |
| `min_offers` | 5 | Turns a tool is offered in, uncalled, before it is flagged |
| `min_turns` | 20 | Routed turns observed before a tool never offered is flagged |
| `show_learned_examples` | `False` | Show the model learned example requests |

`agent.configure_tool_learning(...)` keeps any of these for the agent;
`config=` on one call overrides what is kept.

## What it needs, and its limits

- **Workflow memory** gives each call the request that led to it. An agent
  that keeps only the tool log (`ToolResultPolicy(persist_all_results=True)`)
  can learn usage notes, but not example requests.
- It revises tools registered through a `Toolbox` for that agent. Tools
  shared by several agents, the rows saving an agent writes, and a connected
  MCP server's tools are left alone.
- A request that went to the wrong tool and was never corrected leaves no
  trace. Finding those takes an outcome your application reports; see
  [verified outcomes](continual-learning.md#verified-outcomes).
- "The user asked again" also matches a conversation that moved on to a new
  task. The two-request rule, the shared-word rule and the background
  model's judgement reduce that; review before applying catches the rest.
- A "Not for" line needs the neighbour to have served the request in the
  end. Two tools that are confused and never corrected leave no pair.
- The usage flags count what routing offered, not what the model was shown:
  tools kept visible from an earlier turn are not counted again.
- The [refine cycle](refine-cycle.md) runs a tool review as one of its
  jobs, and proposes revisions that wait for a person (`refine_tool_review`).
  Outside the cycle it is an SDK feature: it has no page in the local UI,
  CLI command or MCP tool yet.
- It has been run on the filesystem provider and on Oracle (register with
  augmentation, review, apply, a reload and undo). A revision writes only
  the `description`, `queries` and `embedding` of a tool row, which every
  provider stores, but it has not been run on MongoDB. "Not for" lines, the
  example call, example sources, the usage flags and the stored settings
  have been run on the filesystem provider only. The `queries_source` marker
  is an extra field on the tool row; a provider that drops unknown fields
  falls back to comparing the description with the docstring.

The notebook
[`examples/continual_learning/04_toolbox_evolution.ipynb`](https://github.com/RichmondAlake/memorizz/blob/main/examples/continual_learning/04_toolbox_evolution.ipynb)
walks through all of it on a claims-desk toolbox.

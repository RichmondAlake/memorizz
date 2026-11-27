# An agent's own source

A self-aware agent can be given its own copy of the MemoRizz code it runs on.
It can then see what went wrong in its own turns, have a coding model
recommend a fix, and make that fix in its copy, where every change kept is a
version you can read.

```python
agent = MemAgent(
    model=chat_model,
    self_aware=True,
    self_aware_config={
        "own_source": True,
        # Optional: a different model for reading and editing code.
        "coding_model": {
            "provider": "ollama",
            "model": "flint",
            "host": "http://192.168.1.20:11434",
        },
    },
)
```

## What the agent gets

- **A copy of the source**, at `<MEMORIZZ_HOME>/self_aware/<agent>/source`.
  It is a git repository. Version 0 is MemoRizz as installed; each fix kept
  is one more commit.
- **An error log**, `errors.jsonl` beside the copy. Traces never store an
  exception's message, so this local file does: the message, and the frames
  inside MemoRizz as `memorizz/<file>:<line>`. It holds this agent's errors
  only and stays on the machine.
- **Its own traces.** Failed steps are read from the agent's trace events.

## Recommend, then fix

```python
agent.self_errors()                 # the error log and failed trace steps
review = agent.recommend_self_fix() # Problem, Cause, Fix, Risk. Changes nothing.
result = agent.fix_self(review["recommendation"])

result["outcome"]   # "fixed", "no_change" or "check_failed"
result["version"]   # the new version, when fixed
result["diff"]      # what changed
```

`fix_self` edits the copy, then checks it in a separate Python process: the
changed files must compile and MemoRizz must import from the copy. If the
check fails the coding model gets one more try with the check's output; if it
still fails, the edits are dropped. A version is therefore always importable.

Not every failure is a bug in MemoRizz. For a missing key, an unreachable
model or a fault in your own tool, the coding model is told to say so and
change nothing (`"no_change"`).

In a conversation the same steps are tools: `self_recent_errors`,
`self_read_trace`, `self_recommend_fix`, `self_apply_fix`,
`self_source_status` and `self_source_diff`. `self_apply_fix` needs approval.

## What a version does and does not do

The installed package is never edited, so a running agent is not changed
from under itself. A version stays in the agent's copy until you take it out:

```python
source = agent.self_source()
source.versions()            # newest first
source.diff(version_id)      # one version's change
source.patch()               # everything kept, as a patch against the install
source.restore("installed")  # go back; itself a new version
```

To run on a fix, apply the patch to a MemoRizz checkout and install that.
The agent does not load its own copy.

## In the console

`/agents/<id>/source` shows the versions with their diffs, the recent
errors, the coding model, and two buttons: **Find the cause** and **Fix it**.
While the model works, its current step shows on the page and beside the
agent's name in the sidebar.

`/agents/<id>/source/diff` is the full view of a change, drawn like a code
editor's diff: side by side or in one column, with unchanged lines folded. It
can show one version, everything that differs from the install, or edits not
yet kept. Click a line number, a fold or **Open file** to read the whole file
at that line, before or after the change, with the changed lines marked.

**Preview this version**, on the source page and on a version's diff, runs
that version as its own console: a second MemoRizz started from the
version's code on a free local port, opened in a new tab. It reads the same
store but is read-only, so the agent's code can be looked at and cannot
write anything. It stops by itself after 45 minutes, or when you select
**Stop** on the source page. This is how to see a change an agent made to
the console itself, since the console you are using runs the installed code.

Above the diff of a version the agent made, **How this change came about**
tells it in order:

1. **It noticed an error**: the message, the line of code and the file and
   line it came from, as the coding model read them. That line is marked with
   a red dot in the diff below.
2. **It found the cause**: the diagnosis, when the fix was made from one.
3. **It was asked to fix it**: the prompt sent to the coding model, and the
   standing instructions sent with it.
4. **It made the change**: each step it took, in order (reading its errors,
   reading and editing files, checking).
5. **It explained the change**: once the version is kept, the coding model
   is shown the error and its own diff and asked why. Its answer is kept as
   plain fields: a summary, the root cause, why this fix and not a larger
   one, the risk, how to confirm it, and the reason for each edit. Those
   reasons are also shown above each changed file in the diff, each leading
   to its line. An outside coding agent (Droid) gives no explanation.
6. **The copy passed its check and was kept**: what the check ran.

The coding model's own work is traced. Steps 2 and 4 link to it in Traces:
every model call and tool call it made, in a thread of its own under the
agent, tagged `work=self_repair`. Those turns never count as the agent's
own failures. `result["explanation"]` from `fix_self` carries the same
explanation, and the story has it as `explanation` and `work_trace`.

This account is saved beside the agent's source when the version is kept
(`<MEMORIZZ_HOME>/self_aware/<agent>/fixes/<commit>.json`), and
`agent.self_source().story(version_id)` returns it. A version made by hand or
by going back to an earlier one has no account.

## The coding model

Any provider `create_llm_provider` knows works; the model must support tool
calling. Only the provider, the model and, for Ollama, the address are
saved. Keys come from the environment.

A model on OpenRouter needs only its name; the key is read from
`OPENROUTER_API_KEY`, and `routing` is accepted as on any OpenRouter model:

```python
agent.with_self_aware(True, {
    "own_source": True,
    "coding_model": {"provider": "openrouter", "model": "recursive/flint-v1.0:free"},
})
```

### Factory's Droid instead of a coding model

[Droid](https://docs.factory.com/cli) is Factory's coding agent. Set it as
the coding agent and it does the diagnosis and the fix in place of a model
driven by MemoRizz:

```python
agent.with_self_aware(True, {"own_source": True, "coding_agent": "droid"})
# or name a model Droid knows, including one of your own:
agent.with_self_aware(True, {
    "own_source": True,
    "coding_agent": {"kind": "droid", "model": "custom:Flint-0"},
})
```

It runs `droid exec` in the agent's copy of the source: without `--auto` to
find the cause, and with `--auto low` to make the fix. Either way Droid is
left only its file tools (Read, LS, Grep, Glob, and for a fix Edit,
ApplyPatch and Create), so it cannot run commands, search the web or message
anyone. Set `"autonomy": "medium"` on the coding agent to give it its own
tools back, so it can run tests; that also lets it run other commands. The recent errors are written into
its prompt, since Droid has none of MemoRizz's tools. MemoRizz then runs the
same check, keeps the version and saves the account of the fix.

It needs Factory's CLI on the `PATH` and `FACTORY_API_KEY` where MemoRizz
runs. Droid's process is given that key and not the keys of other services.
Your own models are set up in Factory's `~/.factory/settings.json`. The
source and errors are sent to Factory and to whichever model Droid uses.

For a model someone else serves over the OpenAI protocol (vLLM, LM Studio, a
company gateway), give the endpoint and the name of the variable that holds
its key:

```python
agent.with_self_aware(True, {
    "own_source": True,
    "coding_model": {
        "provider": "openai",
        "model": "recursive/flint-v1.0:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
})
```

Only the variable's name is saved. An endpoint that is not OpenAI's own is
never sent `OPENAI_API_KEY`; without `api_key_env` it gets a placeholder key,
which is what local servers expect.

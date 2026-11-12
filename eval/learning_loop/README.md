# Learning-loop evaluation

Checks for the changes in the workflow-to-skills, persona, entity and refine-cycle
spec: the verified-outcome gate, the skill holdout, the duplicate check,
guardrails, dated entity history, the source rule, aliases, and the refine
cycle with its report.

## What is here

`mechanisms.py` runs the library's own code twice on the same generated
scenario: once configured as 0.17.24 behaved, once with the change on.

```bash
python eval/learning_loop/mechanisms.py            # prints a before/after table
python eval/learning_loop/mechanisms.py --json results.json
```

It calls no model and no embedding API, so it runs anywhere in a few
seconds. That is also its limit: it says what each mechanism does to stored
memory, not what an agent's answers gain. The scenarios are generated, and
"the right value" in the source-rule check is defined by the rule itself
(the host outranks a user remark, which outranks an inference).

One result to know about: the holdout flag trades missed harm against false
alarms, because one thread in ten is a small control group. Flagging any
lower figure (`holdout_flag_z=0`) catches nearly every harmful skill and
also flags a third to a half of the skills that make no difference. The
default margin of one standard error catches about nine in ten harmful
skills at 600 threads a skill and flags about one in five of the skills that
make no difference; with 2,000 threads it catches all of them. A flag asks a
person to look; it demotes nothing.

## What is not here yet

The round-based suite the spec describes: a session corpus split by time,
one refine cycle after each round, held-out questions scored for three arms
(stateless, recall only, everything on), a retention set scored every round
and a fact changed on purpose between rounds. It needs a chat model and a
corpus with tool-calling sessions, and it is what would show whether answers
improve.

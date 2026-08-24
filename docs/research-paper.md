# Research Papers

The local `research/` workspace contains two companion systems-paper drafts:

1. **Memorizz: A Memory-First Agent Harness for Efficient, Persistent Agentic
   Applications** covers memory taxonomy and unit shapes, provider abstraction,
   prompt and semantic caching, context tokenomics, and harness boundaries.
2. **Memorizz: An Agent Memory and Continual Learning Platform** covers
   immutable events, verified outcomes, bounded evidence, deterministic
   compilation, workflow-to-skill promotion, instruction authority, and
   governed forgetting.

The drafts are intentionally ignored by Git and are not part of the published
documentation or package. This protects the private authoring workspace; it
also means GitHub links to `research/*` are not valid until the authors choose
a separate publication location.

## Build and open the local PDFs

From the repository root:

```bash
cd research
tectonic paper1-memory-first-agent-harness.tex
tectonic paper2-agent-memory-continual-learning-platform.tex
```

Tectonic writes each PDF into the current directory. It does not open a viewer
automatically.

=== "macOS"

    ```bash
    open paper1-memory-first-agent-harness.pdf
    open paper2-agent-memory-continual-learning-platform.pdf
    ```

=== "Linux"

    ```bash
    xdg-open paper1-memory-first-agent-harness.pdf
    xdg-open paper2-agent-memory-continual-learning-platform.pdf
    ```

Underfull `\\hbox` messages are layout warnings, not build failures. A final
line such as `Writing paper1-memory-first-agent-harness.pdf` confirms that the
PDF was created.

## Evidence and attribution

The platform paper credits the recency–importance–relevance retrieval mechanism
in [Generative Agents](https://arxiv.org/abs/2304.03442) as inspiration for
separating freshness, trust/utility, and relevance. Memorizz differs by keeping
relevance as a recall signal and requiring reviewed, reversible retention
decisions.

The drafts treat local benchmark runs as engineering evidence, not leaderboard
submissions. Results should retain their dataset revision, task count, provider,
model, scope, token/cost/latency method, official-grader status, and known
limitations. Do not publish local `eval/results/` artifacts without reviewing
them for credentials, proprietary dataset content, tenant identifiers, and
unsupported statistical claims.

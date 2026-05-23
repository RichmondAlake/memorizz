# Memorizz Documentation

Memorizz is a memory-first framework for Python AI agents. The docs in this directory are versioned with the code so examples and API references stay in sync with each release.

## Looking For The App UI?

The docs website runs on `http://localhost:8000` (MkDocs).
The Memorizz web app UI is separate and runs on `http://127.0.0.1:8765`.

```bash
pip install "memorizz[ui]"
memorizz run local
```

Open `http://127.0.0.1:8765` after starting the command above.

## What Memorizz Covers

- **Memory architecture**: semantic, episodic, procedural, short-term, and shared memory systems.
- **Application modes**: pre-configured memory stacks for assistant, workflow, and deep research agents.
- **Provider abstraction**: Oracle, MongoDB, filesystem, or custom `MemoryProvider` implementations.
- **Agent runtime features**: tool calling, semantic cache, context-window stats, and summary generation.
- **Optional integrations**: internet access providers, sandbox code execution, skills marketplace (Vercel Agent Skills, SkillsMP), and local web UI.

## Docs Map

- **Getting Started**: installation, first agent setup, and the [Local UI Guide](getting-started/local-ui.md).
- **Memory Types**: detailed behavior of each memory subsystem.
- **Memory Providers**: persistence backend setup and tradeoffs.
- **Use Cases**: mode-specific patterns for assistants, workflows, and deep research.
- **Skills Marketplace**: discover and use Vercel Agent Skills or SkillsMP skills at runtime.
- **Internet Access + Sandbox**: optional runtime capabilities for web research and isolated code execution.

## Local Preview

From the project root:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Visit <http://localhost:8000>.

Before publishing docs changes:

```bash
mkdocs build --strict
```

## Release Checklist (Docs)

1. Ensure `README.md` and docs quickstarts reference existing files and valid API names.
2. Verify new features are documented under the relevant provider/memory/use-case page.
3. Run `mkdocs build --strict` to catch broken links/snippets.

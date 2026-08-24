# Memorizz Documentation Contributor Guide

The public site is built from this directory with MkDocs Material. Organize
content around a developer's task, keep examples aligned with the public SDK,
and state security or deployment boundaries next to the feature they govern.

## Information architecture

| Section | Reader question |
|---|---|
| Start Here | How do I install Memorizz and run the right interface? |
| Build Agents | How do I add modes, tools, MCP, browser, internet, sandbox, skills, or external agent harnesses? |
| Memory | What is stored, how is it scoped, and which backend should I use? |
| Continual Learning | How are workflows evaluated, promoted, and forgotten? |
| Operate | How do I configure, observe, secure, preflight, and troubleshoot it? |
| Evaluate & Research | How do I reproduce measurements and interpret their limits? |
| Reference | What is the exact public SDK surface? |

Put conceptual material before detailed reference, link rather than duplicate
configuration tables, and keep internal roadmaps or issue reports out of the
published navigation.

## Run locally

From the repository root:

```bash
python -m pip install -e ".[docs,dev]"
mkdocs serve
```

Open <http://localhost:8000>. The application UI is a separate process at
`http://127.0.0.1:8765` after `memorizz ui`.

## Validate before review

```bash
python -m pytest -q tests/unit/test_docs_quality.py
mkdocs build --strict
```

The static tests require every published page in `mkdocs.yml`, parse every
Python fence, resolve local links, and guard the main entry points against
known stale guidance.

## Writing rules

- Import stable user-facing types from `memorizz`; use internal paths only when
  explaining implementation or a provider-specific extension.
- Make code runnable, or label an excerpt/pseudocode explicitly and explain
  what the reader must supply.
- Include `memory_id`, `user_id`, and `thread_id` in shared-service examples.
- Distinguish host authority from model proposals. Never teach model-controlled
  `approved` or `confirm` booleans.
- Describe cache similarity separately from source freshness.
- State whether a provider or integration is a security boundary, execution
  provider, development tool, or experimental feature.
- Use placeholder credentials only and never paste real secrets or production
  trace content.
- Link configuration and troubleshooting from feature guides rather than
  repeating environment-variable lists everywhere.

## Release documentation checklist

1. Update the relevant task guide and curated API reference.
2. Update `README.md` only when installation or the top-level capability map
   changes.
3. Add the page to `mkdocs.yml` or explicitly exclude internal material.
4. Test code snippets against the supported Python versions and at least the
   default filesystem provider.
5. Add provider-specific behavior and migration/preflight implications.
6. Update `CHANGELOG.md` and run the strict documentation checks.

The deeper package architecture reference remains in
`src/memorizz/MEMORY_ARCHITECTURE.md`; public docs should explain stable
developer concepts rather than mirror the source tree.

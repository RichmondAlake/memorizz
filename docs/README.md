# Memorizz Docs (Contributor Guide)

This directory contains the Markdown source for the Memorizz documentation site (MkDocs Material).

## Run Docs Locally

From the project root (same directory as `mkdocs.yml`):

```bash
pip install -e ".[docs]"
make docs-serve
# or: mkdocs serve
```

Open <http://localhost:8000>.

## Build/Validate Before Push

```bash
make docs-build
# or: mkdocs build --strict
```

Use strict mode before merging to catch broken links/nav references.

## What to Keep Updated for Releases

- `README.md` (root): installation, quickstart, and examples
- `docs/getting-started/*`: first-run path for new developers
- `docs/memory-providers/*`: provider-specific setup
- `docs/use-cases/*`: mode-specific workflows
- `src/memorizz/MEMORY_ARCHITECTURE.md`: package architecture reference

## Optional Runtime Features

### Local Web UI

```bash
pip install "memorizz[ui]"
memorizz ui
```

Default URL: <http://127.0.0.1:8765>

### Sandbox Providers

```bash
pip install "memorizz[sandbox-e2b]"      # E2B
pip install "memorizz[sandbox-daytona]"  # Daytona
```

GraalPy is local-runtime based and does not require a pip extra from Memorizz.

## Navigation

Docs navigation is configured in `mkdocs.yml` under `nav:`.
When adding a page, update both:

1. the Markdown file under `docs/`
2. the `nav` entry in `mkdocs.yml`

## Style Expectations

- Keep code snippets runnable or clearly marked as pseudo-code.
- Prefer imports from public package paths that exist in this repo.
- Link only to files that actually exist in the repository.
- Keep terminology consistent: `Memorizz` (project/package name).

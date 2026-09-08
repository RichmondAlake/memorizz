# Preparing a release

Keep `pyproject.toml`, `src/memorizz/__init__.py` and
`packaging/npm/package.json` at the same version and add a dated section to
`CHANGELOG.md`. The tag must be `v` followed by that version. Commit the reviewed
changes before tagging; local build success is not publication.

## Verification

Run the unit/integration suite with `.[dev,ui,mcp]`, the pre-commit checks,
`node packaging/npm/test/launcher.test.js`, the synthetic browser runner in
[tests/README.md](../tests/README.md), and a strict documentation build. Live
database tests are opt-in; report any skipped live coverage separately.

Build with `uv build`, validate the wheel/source archive with `twine check`,
and test clean installs of the wheel with base dependencies and with UI/MCP
extras. The base install must support `memorizz --version` and `--help` without
pulling in the heavyweight local-ML or optional MCP stack. Use `npm pack
--ignore-scripts` in `packaging/npm` to inspect the launcher tarball. Audit
reports, credentials, local databases and generated files must not ship.

The Test workflow checks Python 3.10/3.12 and browser acceptance. The Publish
workflow independently gates distribution builds on Python tests and the same
browser suites, then checks clean wheel installs before publishing.

## Publishing channels

Pushing the matching version tag starts `.github/workflows/publish.yml`.
Its `workflow_dispatch` input can resume an existing release tag. Run it only
when publication is intended; PyPI releases cannot be overwritten with different
artifacts under the same version.

- PyPI uses the `pypi` environment and its configured trusted publisher.
- GitHub release notes come from the matching changelog section, with wheel and
  source archive attached.
- The npm wrapper requires `NPM_TOKEN` and must match the Python release version.
- The Homebrew tap requires `TAP_PUSH_TOKEN` with write access to
  `RichmondAlake/homebrew-memorizz`. Without it, the job skips the tap update;
  a green Publish workflow alone does not prove that every channel was updated.

## Homebrew prerequisite for 0.10.0

The existing formula updater changes only the main source URL and SHA-256. It
does not refresh Python resource dependencies or rebuild bottles. Before using
it for 0.10.0, update the separate tap formula to include the new `packaging`
base dependency and its verified source checksum. Remove the old-version bottle
block or replace it with bottles built and tested for the new version. The
Windows-only `tzdata` dependency is not needed on Homebrew's macOS/Linux targets.

Verify source installation and the formula's tests in the tap, then verify any
new bottles on their target platforms. Configure `TAP_PUSH_TOKEN` or update the
tap through its own authorized release workflow; do not copy credentials into
this repository. Finally, check the published PyPI, npm, GitHub and Homebrew
versions individually.

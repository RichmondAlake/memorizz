# Memorizz Desktop (Tauri shell — Phase 1)

A thin Tauri wrapper around the existing Memorizz local UI. In Phase 1 this
does one thing: render `http://127.0.0.1:8765` inside a native macOS window so
double-clicking an app is a valid entrypoint.

Phase 1 deliberately does **not** bundle Python. You run the server yourself in
a separate terminal. Phase 2 will embed `python-build-standalone` and manage
the FastAPI process as a Tauri sidecar.

## Prerequisites (one-time)

1. **Rust** (required by Tauri):

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source "$HOME/.cargo/env"
   ```

2. **Xcode Command Line Tools** (for the macOS linker):

   ```bash
   xcode-select --install
   ```

3. **Node 18+** (you already have `v24.15.0`).

## Running locally

```bash
# terminal 1 — Memorizz server
pip install -e "..[ui]"        # from the memorizz repo root
memorizz ui                     # serves http://127.0.0.1:8765

# terminal 2 — desktop shell
cd apps/desktop
npm install
npm run dev                     # == tauri dev
```

On first run, `npm run dev` compiles the Rust crate in `src-tauri/` (a few
minutes the first time, cached after). The window opens pointing at the live
UI.

## Packaging a `.app`

```bash
npm run build    # == tauri build
# output: src-tauri/target/release/bundle/macos/Memorizz.app
#         src-tauri/target/release/bundle/dmg/Memorizz_0.1.0_*.dmg
```

Unsigned bundles will fail Gatekeeper on other machines. Code signing and
notarization come in Phase 1b once the Developer ID is sorted.

## Icons

The shipped PNGs are solid-color placeholders so `tauri dev` runs. Replace
them by dropping a 1024×1024 source PNG and regenerating:

```bash
npm run icon path/to/memorizz-logo.png
```

This writes the full icon set (including `.icns` and `.ico`) into
`src-tauri/icons/`.

## How the window loads content

- **Dev mode** (`npm run dev`): Tauri loads `build.devUrl` from
  `tauri.conf.json` directly, i.e. `http://127.0.0.1:8765`. If the server
  isn't up, you'll see a "connection refused" error page — start the server
  and reload (⌘R).
- **Bundled app**: the window loads the static fallback in `dist/index.html`,
  which polls `127.0.0.1:8765` every 1.5s and redirects to the live UI once
  the server is reachable. This makes "open the app, then the server starts"
  a valid sequence.

## Known gaps (tracked for Phase 2)

- Python runtime not bundled — users still need `pip install memorizz[ui]`.
- No sidecar management of the FastAPI process.
- No auto-update channel.
- CSP is disabled (`security.csp: null`) because we load a remote origin. Fine
  for localhost; tighten once the server is embedded.
- No code signing / notarization pipeline.
- `.icns` icon is absent — bundled builds will fall back to Tauri defaults
  until `npm run icon` is run with a real source.

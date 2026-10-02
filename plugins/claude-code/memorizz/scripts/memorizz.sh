#!/bin/sh
# Run the MemoRizz CLI for the plugin: the one `memorizz plugin install`
# recorded, else MEMORIZZ_BIN, else `memorizz` on PATH (or a common install
# location), else uvx from PyPI.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -f "$here/memorizz-bin" ]; then
  recorded=$(head -n 1 "$here/memorizz-bin")
  [ -x "$recorded" ] && exec "$recorded" "$@"
fi
[ -n "$MEMORIZZ_BIN" ] && [ -x "$MEMORIZZ_BIN" ] && exec "$MEMORIZZ_BIN" "$@"
command -v memorizz >/dev/null 2>&1 && exec memorizz "$@"
for candidate in "$HOME/.local/bin/memorizz" /opt/homebrew/bin/memorizz /usr/local/bin/memorizz; do
  [ -x "$candidate" ] && exec "$candidate" "$@"
done
for uvx in uvx "$HOME/.local/bin/uvx" /opt/homebrew/bin/uvx; do
  if command -v "$uvx" >/dev/null 2>&1 || [ -x "$uvx" ]; then
    exec "$uvx" --quiet --from 'memorizz[mcp]' memorizz "$@"
  fi
done
echo "MemoRizz isn't installed. Install it with: pip install 'memorizz[mcp]' (or install uv: https://docs.astral.sh/uv/)." >&2
exit 127

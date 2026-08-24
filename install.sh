#!/usr/bin/env sh
# memorizz CLI installer.
#
#   curl -fsSL https://raw.githubusercontent.com/RichmondAlake/memorizz/main/install.sh | sh
#
# Installs the `memorizz` interactive CLI via uv (a standalone binary that
# manages its own Python — no pre-existing Python required). Falls back to pipx,
# then pip --user. Safe to re-run; upgrades in place.

set -u

INFO() { printf '\033[36m[memorizz]\033[0m %s\n' "$1"; }
WARN() { printf '\033[33m[memorizz]\033[0m %s\n' "$1" >&2; }
ERR() { printf '\033[31m[memorizz]\033[0m %s\n' "$1" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }

# Resolve a uv binary: already on PATH, in its default install dir, or freshly
# installed via the official standalone installer.
find_uv() {
  if have uv; then echo "uv"; return 0; fi
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin" "$HOME/.local/share/uv/bin"; do
    if [ -x "$d/uv" ]; then echo "$d/uv"; return 0; fi
  done
  return 1
}

UV="$(find_uv || true)"
if [ -z "${UV:-}" ]; then
  INFO "Installing uv (standalone; no Python required)…"
  if have curl; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif have wget; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    ERR "Need curl or wget to install uv. Install one and re-run."
    exit 1
  fi
  UV="$(find_uv || true)"
fi

if [ -n "${UV:-}" ]; then
  # Force a modern Python: memorizz needs >=3.10, and without this uv may fall
  # back to an older interpreter and silently resolve a stale memorizz release.
  INFO "Installing memorizz via uv (Python 3.12)…"
  if "$UV" tool install --force --python 3.12 memorizz; then
    "$UV" tool update-shell >/dev/null 2>&1 || true
    INFO "Installed. Start it with:  memorizz"
    INFO "(If 'memorizz' isn't found, restart your shell or add ~/.local/bin to PATH.)"
    exit 0
  fi
  WARN "uv install failed — trying pipx, then pip."
fi

if have pipx; then
  INFO "Installing memorizz via pipx…"
  if pipx install --force memorizz; then
    INFO "Installed (pipx). Start it with:  memorizz"
    exit 0
  fi
fi

if have python3; then
  INFO "Installing memorizz via pip --user…"
  if python3 -m pip install --user --upgrade memorizz; then
    INFO "Installed (pip --user). Start it with:  memorizz"
    exit 0
  fi
fi

ERR "Could not install memorizz automatically."
ERR "Install uv from https://astral.sh/uv then run:  uv tool install --python 3.12 memorizz"
exit 1

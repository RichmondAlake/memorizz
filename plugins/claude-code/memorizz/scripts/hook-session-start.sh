#!/bin/sh
# SessionStart (Codex or Claude Code): tell the agent this project's MemoRizz
# memory ID and its recent memories. Never fails the session.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"$here/memorizz.sh" plugin hook session-start 2>/dev/null || true

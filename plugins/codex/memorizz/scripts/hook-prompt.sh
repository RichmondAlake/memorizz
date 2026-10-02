#!/bin/sh
# UserPromptSubmit (Codex or Claude Code). Keeps the prompt so the Stop hook
# can save the whole turn, and, when MEMORIZZ_PROMPT_RECALL is true, prints
# project memories related to it. Fast when recall is off; never blocks.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/settings.sh"
payload=$(cat)
if [ "$(memorizz_setting MEMORIZZ_SESSION_CAPTURE turns)" != "off" ]; then
  session=$(printf '%s' "$payload" | sed -n 's/.*"session_id"[[:space:]]*:[[:space:]]*"\([A-Za-z0-9._:-]*\)".*/\1/p' | head -n 1)
  if [ -n "$session" ]; then
    dir="${MEMORIZZ_HOME:-$HOME/.memorizz}/plugin-sessions"
    mkdir -p "$dir" && chmod 700 "$dir" 2>/dev/null
    (umask 077; printf '%s\n' "$payload" > "$dir/$session.prompt.json")
  fi
fi
case "$(memorizz_setting MEMORIZZ_PROMPT_RECALL false)" in
  1|true|yes|on)
    printf '%s' "$payload" | "$here/memorizz.sh" plugin hook prompt 2>/dev/null || true
    ;;
esac
exit 0

#!/bin/sh
# Stop (Codex or Claude Code): save the turn that just ended (the prompt and
# the final answer, secrets removed) to the project's conversation memory.
# The save runs detached, so the agent never waits and `claude -p` or
# `codex exec` exiting doesn't cancel it. Off with MEMORIZZ_SESSION_CAPTURE=off.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/settings.sh"
[ "$(memorizz_setting MEMORIZZ_SESSION_CAPTURE turns)" = "off" ] && exit 0
payload=$(cat)
session=$(printf '%s' "$payload" | sed -n 's/.*"session_id"[[:space:]]*:[[:space:]]*"\([A-Za-z0-9._:-]*\)".*/\1/p' | head -n 1)
dir="${MEMORIZZ_HOME:-$HOME/.memorizz}/plugin-sessions"
[ -n "$session" ] && [ -f "$dir/$session.prompt.json" ] || exit 0
# Set the prompt aside so the next one can't be paired with this answer; the
# .saving file also tells the session summary to wait for this save.
turn=$(mktemp "$dir/$session.saving.XXXXXX") || exit 0
mv -f "$dir/$session.prompt.json" "$turn.prompt" || { rm -f "$turn"; exit 0; }
printf '%s\n' "$payload" > "$turn"
nohup sh -c '"$1" plugin hook stop --prompt-file "$3" < "$2"; rm -f "$2" "$3"' \
  sh "$here/memorizz.sh" "$turn" "$turn.prompt" >/dev/null 2>&1 &
exit 0

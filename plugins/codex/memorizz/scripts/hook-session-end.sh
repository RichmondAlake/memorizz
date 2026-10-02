#!/bin/sh
# SessionEnd and PreCompact (Codex or Claude Code). Summarizing takes longer
# than these hooks may run, so start it detached and return at once.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/settings.sh"
[ "$(memorizz_setting MEMORIZZ_SESSION_CAPTURE turns)" = "off" ] && exit 0
case "$(memorizz_setting MEMORIZZ_SESSION_SUMMARY true)" in
  0|false|no|off) exit 0 ;;
esac
payload_file=$(mktemp "${TMPDIR:-/tmp}/memorizz-session.XXXXXX") || exit 0
cat > "$payload_file"
nohup sh -c '"$1" plugin hook summarize < "$2"; rm -f "$2"' sh "$here/memorizz.sh" "$payload_file" >/dev/null 2>&1 &
exit 0

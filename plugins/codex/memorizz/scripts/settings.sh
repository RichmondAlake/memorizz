#!/bin/sh
# Shared by the hook scripts: read a MemoRizz setting from the environment, or
# from ~/.memorizz/.env, without starting Python.
memorizz_setting() {
  name=$1
  default=$2
  value=$(printenv "$name" 2>/dev/null)
  if [ -z "$value" ]; then
    env_file="${MEMORIZZ_HOME:-$HOME/.memorizz}/.env"
    value=$(sed -n "s/^[[:space:]]*$name[[:space:]]*=[[:space:]]*[\"']\{0,1\}\([^\"'#]*\).*/\1/p" "$env_file" 2>/dev/null | tail -n 1)
  fi
  value=$(printf '%s' "$value" | tr -d '[:space:]' | tr 'A-Z' 'a-z')
  printf '%s' "${value:-$default}"
}

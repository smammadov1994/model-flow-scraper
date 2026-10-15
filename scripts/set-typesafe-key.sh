#!/usr/bin/env bash
# Save your TypeSafe API key to .env (gitignored) without echoing it to the screen or shell history.
set -euo pipefail
cd "$(dirname "$0")/.."
printf "Paste your TypeSafe API key and press Enter (it won't be shown): "
IFS= read -rs key
echo
if [ -z "$key" ]; then echo "Nothing entered — no changes made."; exit 1; fi
touch .env && chmod 600 .env
grep -v '^TYPESAFE_API_KEY=' .env > .env.tmp || true
printf 'TYPESAFE_API_KEY=%s\n' "$key" >> .env.tmp
mv .env.tmp .env && chmod 600 .env
unset key
echo "Saved to $(pwd)/.env (only you can read it; git ignores it). You can close this tab."

#!/bin/sh
# Install local git hooks (commit-msg guard).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cp "$ROOT/scripts/hooks/commit-msg" "$ROOT/.git/hooks/commit-msg"
chmod +x "$ROOT/.git/hooks/commit-msg"
echo "Installed .git/hooks/commit-msg"

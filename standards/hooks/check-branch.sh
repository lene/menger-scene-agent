#!/bin/sh
# Reject commits made directly on main (Sprint 28.2 agentic policy check).
set -eu

BRANCH=$(git branch --show-current)
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
    echo "POLICY: committing directly on '$BRANCH' is not allowed." >&2
    echo "Create a feature branch first: git switch -c feature/<topic>" >&2
    exit 1
fi
exit 0

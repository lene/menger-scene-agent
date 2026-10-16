#!/bin/sh
# Fast hygiene checks on the staged diff (Sprint 28.2). Target: well under 1s.
# - leftover merge-conflict markers and whitespace errors (git diff --check)
# - debug println in production Scala sources
# - accidentally staged large files (>5 MB)
set -u

STATUS=0

if ! git diff --cached --check; then
    echo "POLICY: staged changes contain conflict markers or whitespace errors." >&2
    STATUS=1
fi

DEBUG_PRINTS=$(git diff --cached -U0 -- '*/src/main/*.scala' 2>/dev/null \
    | grep '^+' | grep -v '^+++' | grep -cE '\bprintln\(' || true)
if [ "${DEBUG_PRINTS:-0}" -gt 0 ]; then
    echo "POLICY: staged production Scala code adds println() calls:" >&2
    git diff --cached -U0 -- '*/src/main/*.scala' | grep -nE '^\+.*\bprintln\(' >&2
    echo "Use the logging framework, or stage tests/tools separately." >&2
    STATUS=1
fi

MAX_BYTES=5242880
for f in $(git diff --cached --name-only --diff-filter=ACM); do
    size=$(git cat-file -s ":$f" 2>/dev/null || echo 0)
    if [ "$size" -gt "$MAX_BYTES" ]; then
        echo "POLICY: staged file exceeds 5 MB: $f ($size bytes)." >&2
        echo "Large binaries do not belong in the repository." >&2
        STATUS=1
    fi
done

exit $STATUS

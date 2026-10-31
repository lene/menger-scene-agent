#!/bin/sh
# Suite: unit -- pytest with a coverage ratchet: total line coverage may not fall below
# coverage-baseline.txt (raise the baseline when coverage rises; never lower it to pass).
set -u
. ./standards/hooks/lib.sh

BASELINE=$(cat coverage-baseline.txt)
LOG=$(mktemp)
uv run --locked pytest -q --cov=core --cov=adapters --cov=gauntlet --cov=cli \
  --cov-report=term --cov-fail-under="$BASELINE" > "$LOG" 2>&1
RC=$?
cat "$LOG"
FAILED=$(grep -E '^(FAILED|ERROR) ' "$LOG" | sed 's/^[A-Z]* //; s/ - .*//')
rm -f "$LOG"
if [ "$RC" -eq 0 ]; then
  suite_pass unit
elif [ -n "$FAILED" ]; then
  suite_fail unit "$(printf '%s\n' "$FAILED" | grep -c .)" "$(printf '%s\n' "$FAILED" | paste -sd';' -)"
  exit 1
else
  suite_fail unit 1 "coverage below $BASELINE% or pytest error (rc $RC)"
  exit 1
fi

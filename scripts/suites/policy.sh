#!/bin/sh
# Suite: policy (Sprint 36 B1). Runs four vendored policy gates against $QA_RANGES:
# test-justification, rendering-discipline, log-verbosity, and issue-link. See
# menger-toplevel/docs/QA_INCIDENTS.md for the incident history behind these gates.
#
# Reads $QA_RANGES: newline-separated rev-ranges, set by the caller (pre-push hook or
# CI wrapper) from standards/hooks/lib.sh's push_ranges.
set -u
. ./standards/hooks/lib.sh

if [ -z "${QA_RANGES:-}" ]; then
  suite_skip policy "no push ranges available"
  exit 0
fi

STATUS=0
FAILED=""
if ! ./standards/hooks/check-test-justification.sh $QA_RANGES; then
  STATUS=1
  FAILED="test-justification"
fi
if ! ./standards/hooks/check-rendering-discipline.sh $QA_RANGES; then
  STATUS=1
  FAILED="${FAILED:+$FAILED;}rendering-discipline"
fi
if ! ./standards/hooks/check-log-verbosity.sh $QA_RANGES; then
  STATUS=1
  FAILED="${FAILED:+$FAILED;}log-verbosity"
fi
if ! ./standards/hooks/check-issue-link.sh $QA_RANGES; then
  STATUS=1
  FAILED="${FAILED:+$FAILED;}issue-link"
fi

if [ "$STATUS" -eq 0 ]; then
  suite_pass policy
else
  suite_fail policy 1 "$FAILED"
fi
exit "$STATUS"

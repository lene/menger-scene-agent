#!/bin/sh
# Suite: parity (Sprint 36 B5). Thin wrapper around check-suite-parity.sh so the
# orphan/dangling-suite check runs on every push, not just when someone remembers to
# call it by hand.
set -u
. ./standards/hooks/lib.sh

if ./scripts/check-suite-parity.sh; then
  suite_pass parity
  exit 0
fi

suite_fail parity 1 "check-suite-parity.sh"
exit 1

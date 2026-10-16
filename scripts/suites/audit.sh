#!/bin/sh
# Suite: audit -- known vulnerabilities in the locked dependency set (pip-audit).
set -u
. ./standards/hooks/lib.sh

if uv run --locked pip-audit --skip-editable; then
  suite_pass audit
else
  suite_fail audit 1 "pip-audit"
  exit 1
fi

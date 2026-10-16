#!/bin/sh
# Suite: types -- mypy against mypy-baseline.txt, which may only shrink (scripts/mypy-ratchet.py).
set -u
. ./standards/hooks/lib.sh

if uv run --locked python scripts/mypy-ratchet.py; then
  suite_pass types
else
  suite_fail types 1 "new mypy errors"
  exit 1
fi

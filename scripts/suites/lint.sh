#!/bin/sh
# Suite: lint -- ruff with the rule set pinned in pyproject.toml.
set -u
. ./standards/hooks/lib.sh

if uv run --locked ruff check .; then
  suite_pass lint
else
  suite_fail lint 1 "ruff"
  exit 1
fi

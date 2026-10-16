#!/bin/sh
# Agentic policy check (Sprint 28.2): commits that MODIFY or DELETE existing
# test files must justify it with a "Test-Change: <reason>" trailer in the
# commit message. Newly added tests need no trailer. Enforces the test-failure
# protocol (AGENTS.md): tests are never rewritten to pass without investigation.
#
# usage: check-test-justification.sh <rev-range> [<rev-range>...]
set -u

HOOKS_DIR=$(dirname "$0")
. "$HOOKS_DIR/lib.sh"

# The sbt layout by default; a repo with another layout lists its test-path patterns in
# standards/test-paths.txt, like rendering-paths.txt and hot-paths.txt (the scene agent keeps
# its tests in tests/, which the default never matched: review 2026-10-14, msa#29).
TEST_PATTERN='(^|/)src/test/'
TEST_PATHS_FILE="${TEST_PATHS_FILE:-$HOOKS_DIR/../test-paths.txt}"
if [ -r "$TEST_PATHS_FILE" ]; then
    TEST_PATTERN=$(grep -v '^[[:space:]]*\(#\|$\)' "$TEST_PATHS_FILE" | paste -sd'|' -)
fi
STATUS=0

for range in "$@"; do
    for commit in $(commits_in_range "$range"); do
        # Skip commits already on origin/main (absorbed during rebase)
        git merge-base --is-ancestor "$commit" "$(main_ref)" 2>/dev/null && continue
        is_bootstrapped_commit "$commit" && continue
        commit_is_wip "$commit" && continue
        touched=$(commit_modified_files "$commit" | grep -E "$TEST_PATTERN" || true)
        [ -z "$touched" ] && continue
        if ! commit_has_trailer "$commit" "Test-Change"; then
            echo "POLICY: commit modifies existing tests without justification:" >&2
            echo "  $(short_ref "$commit")" >&2
            echo "$touched" | sed 's/^/    /' >&2
            echo "Add a 'Test-Change: <why the expectation changed>' trailer to the" >&2
            echo "commit message after investigating per the test-failure protocol." >&2
            STATUS=1
        fi
    done
done

[ $STATUS -eq 0 ] && echo "test-change justification: ok"
exit $STATUS

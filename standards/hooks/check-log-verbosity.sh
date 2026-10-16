#!/bin/sh
# Agentic policy check (Sprint 36 #23): commits that ADD a new .info(/.warn(
# log call inside a hot-path file (one that fires every frame or every input
# event, not once per user action) must justify it with a
# "Hot-Path-Log-OK: <reason>" trailer. Enforces that "correct behavior logged
# too loudly on a hot path" doesn't reach a branch silently again
# (QA_INCIDENTS.md 2026-08-21: TesseractEdgeSceneBuilder flooded the console
# during interactive drag because a per-rebuild confirmation was logged at
# INFO instead of DEBUG).
#
# Hot-path files are grep -E patterns in standards/hot-paths.txt (repo-specific;
# override with HOT_PATHS_FILE). Skips cleanly if that file is absent, exactly
# like check-rendering-discipline.sh's rendering-paths.txt.
#
# usage: check-log-verbosity.sh <rev-range> [<rev-range>...]
set -u

HOOKS_DIR=$(dirname "$0")
. "$HOOKS_DIR/lib.sh"

PATHS_FILE="${HOT_PATHS_FILE:-$HOOKS_DIR/../hot-paths.txt}"

if [ ! -r "$PATHS_FILE" ]; then
    echo "log verbosity: skipped (no $PATHS_FILE in this repo)"
    exit 0
fi
PATTERNS=$(grep -v '^[[:space:]]*\(#\|$\)' "$PATHS_FILE")
[ -z "$PATTERNS" ] && { echo "log verbosity: skipped (empty pattern list)"; exit 0; }

STATUS=0

for range in "$@"; do
    for commit in $(commits_in_range "$range"); do
        # Skip commits already on origin/main (absorbed during rebase)
        git merge-base --is-ancestor "$commit" "$(main_ref)" 2>/dev/null && continue
        commit_is_wip "$commit" && continue
        hot_files=$(commit_files "$commit" | grep -E "$(echo "$PATTERNS" | paste -sd'|' -)" || true)
        [ -z "$hot_files" ] && continue
        added_loud=""
        for f in $hot_files; do
            loud=$(git diff-tree -p --no-commit-id "$commit" -- "$f" 2>/dev/null \
                | grep -E '^\+[^+].*\.(info|warn)\(' || true)
            [ -n "$loud" ] && added_loud="$added_loud$f
"
        done
        [ -z "$added_loud" ] && continue
        commit_has_trailer "$commit" "Hot-Path-Log-OK" && continue
        echo "POLICY: commit adds .info(/.warn( logging to a hot-path file:" >&2
        echo "  $(short_ref "$commit")" >&2
        echo "$added_loud" | sed 's/^/    /' >&2
        echo "Hot paths fire every frame/input-event; use .debug(/.trace( instead," >&2
        echo "or add a 'Hot-Path-Log-OK: <reason>' trailer if INFO+ is intentional" >&2
        echo "(e.g. deliberate per-frame progress output the user is meant to see)." >&2
        STATUS=1
    done
done

[ $STATUS -eq 0 ] && echo "log verbosity: ok"
exit $STATUS

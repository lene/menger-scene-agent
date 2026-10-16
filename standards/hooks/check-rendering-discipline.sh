#!/bin/sh
# Agentic policy check (Sprint 28.2): pushes that change rendering-relevant
# code must update reference images in the same push, or each such commit must
# carry a "No-Render-Impact: <reason>" trailer. Enforces the rendering-change
# discipline (AGENTS.md): the pre-push integration suite must not be the
# discovery mechanism for stale references.
#
# Rendering-relevant paths are grep -E patterns in standards/rendering-paths.txt
# (repo-specific; override with RENDERING_PATHS_FILE).
#
# usage: check-rendering-discipline.sh <rev-range> [<rev-range>...]
set -u

HOOKS_DIR=$(dirname "$0")
. "$HOOKS_DIR/lib.sh"

PATHS_FILE="${RENDERING_PATHS_FILE:-$HOOKS_DIR/../rendering-paths.txt}"
REFERENCE_PATTERN='^scripts/reference-images/'

if [ ! -r "$PATHS_FILE" ]; then
    echo "rendering discipline: skipped (no $PATHS_FILE in this repo)"
    exit 0
fi
PATTERNS=$(grep -v '^[[:space:]]*\(#\|$\)' "$PATHS_FILE")
[ -z "$PATTERNS" ] && { echo "rendering discipline: skipped (empty pattern list)"; exit 0; }

STATUS=0

for range in "$@"; do
    range_touches_refs=0
    offenders=""
    for commit in $(commits_in_range "$range"); do
        # Skip commits already on origin/main (absorbed during rebase)
        git merge-base --is-ancestor "$commit" "$(main_ref)" 2>/dev/null && continue
        files=$(commit_files "$commit")
        if echo "$files" | grep -qE "$REFERENCE_PATTERN"; then
            range_touches_refs=1
        fi
        rendering=$(echo "$files" | grep -E "$(echo "$PATTERNS" | paste -sd'|' -)" || true)
        [ -z "$rendering" ] && continue
        commit_is_wip "$commit" && continue
        commit_has_trailer "$commit" "No-Render-Impact" && continue
        offenders="$offenders$commit
"
    done
    if [ -n "$offenders" ] && [ "$range_touches_refs" -eq 0 ]; then
        echo "POLICY: rendering-relevant changes without reference-image updates" >&2
        echo "in push range $range:" >&2
        echo "$offenders" | while read -r c; do
            [ -n "$c" ] && echo "  $(short_ref "$c")" >&2
        done
        echo "Run the integration suite on this commit and commit any reference" >&2
        echo "diffs in the same push, or add a 'No-Render-Impact: <reason>'" >&2
        echo "trailer to commits that provably do not change pixel output." >&2
        STATUS=1
    fi
done

[ $STATUS -eq 0 ] && echo "rendering discipline: ok"
exit $STATUS

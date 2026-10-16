#!/bin/sh
# Agentic policy check (S2 C-3): commits pushed on a feat/sprint-* branch must carry a
# "Refs: owner/repo#N" trailer linking the GitHub issue the commit implements, or
# "Fixes: owner/repo#N" when the commit completes it (the issue is closed once the fix is
# confirmed; see the shared agent rules). Escape hatch:
# a "No-Issue: <reason>" trailer. See docs/superpowers/specs/2026-08-26-intent-achievement-
# spine-design.md (menger-toplevel) — closes audit findings C4/D2/F2 (findings had no stable
# ID; the fidelity check had no referent; checkpoint decisions evaporated).
#
# usage: check-issue-link.sh <rev-range> [<rev-range>...]
set -u

HOOKS_DIR=$(dirname "$0")
. "$HOOKS_DIR/lib.sh"

BRANCH=${GITHUB_HEAD_REF:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)}
case "$BRANCH" in
    feat/sprint-*) ;;
    *)
        echo "issue-link: ok (not a feat/sprint-* branch: $BRANCH)"
        exit 0
        ;;
esac

REFS_RE='^(Refs|Fixes): [A-Za-z0-9._-]+/[A-Za-z0-9._-]+#[0-9]+'
STATUS=0
MAIN_REF=$(main_ref)

for range in "$@"; do
    for commit in $(commits_in_range "$range"); do
        git merge-base --is-ancestor "$commit" "$MAIN_REF" 2>/dev/null && continue
        is_bootstrapped_commit "$commit" && continue
        commit_is_wip "$commit" && continue
        commit_has_trailer "$commit" "No-Issue" && continue
        if ! git log -1 --format=%B "$commit" | grep -qE "$REFS_RE"; then
            echo "POLICY: commit on $BRANCH has no issue link:" >&2
            echo "  $(short_ref "$commit")" >&2
            echo "Add a 'Refs: owner/repo#N' (or 'Fixes: owner/repo#N') trailer, or 'No-Issue: <reason>' if this" >&2
            echo "commit genuinely has no tracking issue." >&2
            STATUS=1
        fi
    done
done

[ $STATUS -eq 0 ] && echo "issue-link: ok"
exit $STATUS

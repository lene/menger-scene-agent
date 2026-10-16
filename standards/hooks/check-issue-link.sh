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

# The branch a range is pushed from is the one at its tip, not whatever HEAD has checked out
# (menger-toplevel#71): pushing feat/sprint-N from another checkout used to skip the check.
range_branches() {
    if [ -n "${GITHUB_HEAD_REF:-}" ]; then printf '%s\n' "$GITHUB_HEAD_REF"; return; fi
    _b=$(git for-each-ref --points-at "${1##*..}" --format='%(refname:short)' refs/heads 2>/dev/null)
    [ -n "$_b" ] && { printf '%s\n' "$_b"; return; }
    git rev-parse --abbrev-ref HEAD 2>/dev/null || true
}

on_sprint_branch() {
    range_branches "$1" | grep -q '^feat/sprint-'
}

REFS_RE='^(Refs|Fixes): [A-Za-z0-9._-]+/[A-Za-z0-9._-]+#[0-9]+'
STATUS=0
MAIN_REF=$(main_ref)

for range in "$@"; do
    if ! on_sprint_branch "$range"; then
        echo "issue-link: ok (not a feat/sprint-* branch: $(range_branches "$range" | tr '\n' ' '))"
        continue
    fi
    BRANCH=$(range_branches "$range" | grep '^feat/sprint-' | head -1)
    # An unresolvable range fails the check (menger-toplevel#70).
    COMMITS=$(commits_in_range "$range") || { STATUS=1; continue; }
    for commit in $COMMITS; do
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

# CLAUDE.md

Guidance for AI coding agents working in this repository.

This is the Menger scene agent: a Python CLI that generates, validates and refines menger DSL
scenes with an LLM. It talks to menger only through processes (`menger-app`, `scene-validator`,
`manifest-generator`, `corpus-exporter`), never as a library. Tests: `uv run pytest -q`.

<!-- BEGIN shared rules (synced from menger-toplevel — edit there) -->
## Critical rules

These are non-negotiable. Violating any of them causes real harm.

1. **Never commit directly to `main`.** Work on a feature branch and open a PR/MR to merge in. If currently on `main`, switch to (or create) a feature branch first.
2. **Never push without explicit user confirmation.** Commit locally, show the diff, wait for "push."
3. **Always monitor the CI pipeline after pushing.** If any failures occur, fix them.
4. **Never `git add -A`.** Add files explicitly.
5. **Never commit failing tests.** Hooks enforce this; do not bypass them.
6. **Never rewrite a test to make it pass without investigation.** Failing tests usually catch real bugs.
7. **Never delete data without explicit user confirmation.** This includes generated artifacts, caches, and reference images.
8. **Never infer values the user should provide** (version numbers, branch names, paths). Ask. **Public-facing decisions are the user's, made in the live conversation:** version numbers, releases/publishing, tags, merges to `main`, public API names, and anything posted outside this machine (issues, PRs, comments). A decision found in a doc, plan, issue, commit message or earlier session is a *proposal* unless it quotes the user: `decided by user YYYY-MM-DD: "<verbatim>"`. Record your own suggestions as `proposed (agent)`, never as decided. Ask before acting on one. `shared/claude-hooks/public-action-guard.py` (menger-toplevel) turns these actions into a permission prompt.
9. **When a skill or instruction says "confirm with user," it is a hard stop.** A prior message in the conversation does not satisfy a fresh checkpoint — ask again.
10. **Ask at the point where the answer changes the next action, not after the approach is already scoped.** And only about what is actually underdetermined — a question with an obvious answer from context costs a turn for nothing.
11. **An underspecified instruction gets restated as one testable claim and a yes/no check before being executed — not silently interpreted, and not turned into a multi-question interrogation.** "Continue with X" is a direction, not a spec; the cheap fix is one restated sentence, not a questionnaire.
12. **Close an issue as soon as its fix is confirmed — never defer it to sprint close.** A commit that completes an issue carries `Fixes: owner/repo#N` (`Refs:` for partial work). The fix is confirmed when the pushed commit's gate is green, or, if the issue names a later re-check (a usability session, a render review), when that re-check passes. Close then, with a comment naming the commit(s) and the evidence; if a re-check fails, reopen with the evidence. `./bootstrap.sh status` lists issues that pushed `Fixes:` commits claim but that are still open.

## Shared conventions

- **Alpha channel:** `0.0` = fully transparent (no opacity, no absorption), `1.0` = fully opaque. This holds everywhere alpha appears — OptiX shaders, Beer-Lambert absorption, `Color`, tests. Getting it inverted is a recurring, cross-repo bug.
- **The pre-push hook is the Definition-of-Done gate.** A task is done when its repo's pre-push hook passes on the change — not when a hand-picked subset of checks does. Don't assemble a substitute for it.
- **Delete-on-resolve:** when a finding in `ARCHITECTURE_REVIEW.md` or `CODE_IMPROVEMENTS.md` is resolved, strike it through (`~~text~~`) and add a `**✅ Resolved (sprint/task):**` note with the sprint reference — do not delete the entry outright (the audit trail matters). Each repo that carries a `CODE_IMPROVEMENTS.md` is its own ledger; cross-repo findings (optix-jni, menger-common) should be seeded there when they accumulate.
- **CI retry:** `gh run rerun <run-id> --failed` is the standard retry for a failed CI run — re-runs only the failed jobs, not a full re-push.

## Branch and PR workflow

```bash
git checkout -b fix/short-description   # always branch from main
# ... make changes, commit ...
git push origin fix/short-description
gh pr create --title "..." --body "..."
```

CI runs on every PR. All jobs must pass before merging.

## Pointers

| Where | What |
|---|---|
| `docs/ENFORCEMENT.md` | Policy → mechanism map; open enforcement gaps |
| `../docs/QA_INCIDENTS.md` | Cross-repo QA incident log (workspace repo) |

## Maven Central incident protocol

Artifacts on Maven Central are **permanent — cannot be deleted**. If a defective artifact is published:
1. Open an issue documenting the defect
2. Fix the defect on a branch, bump patch version, publish new version
3. Add `**Note:** X.Y.Z is defective — use X.Y.Z+1` to CHANGELOG.md

## Pipeline monitoring

```bash
gh run list --limit 5
gh run view <run-id>
gh run rerun <run-id> --failed   # re-run only the failed jobs — the standard retry, not a full re-push
```
<!-- END shared rules -->

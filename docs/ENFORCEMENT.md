# Enforcement Audit: Policy → Mechanism Map

Living document, reviewed at sprint close; format as in `menger/docs/ENFORCEMENT.md`. Created
2026-10-16 when the agent joined the workspace standards (review 2026-10-14, msa#29): until
then CLAUDE.md said "Hooks enforce this" and pointed here, but neither existed.

**Status legend:** ✅ enforced · ⚠️ partial · 🤖 AI policy (behavioural, unenforceable by design)
· ❌ unenforced (open issue)

## Commit hygiene

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| No commits directly on `main` | CLAUDE.md §Critical rules | `standards/hooks/check-branch.sh` (pre-commit); `main` ruleset | ✅ |
| No conflict markers, whitespace errors, files > 5 MB staged | CLAUDE.md §Critical rules | `standards/hooks/check-staged-hygiene.sh` (pre-commit) | ✅ |
| Never `git add -A`; never push without confirmation | CLAUDE.md §Critical rules | 🤖 AI policy | 🤖 |
| Commits link an issue (`Fixes:`/`Refs:`/`No-Issue:`) | CLAUDE.md §Critical rules 12 | `policy` suite: `check-issue-link.sh` (pre-push, PR CI) | ✅ |

## Test and code discipline

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| Tests pass before push | CLAUDE.md §Critical rules | `unit` suite (pre-push, CI `test`) | ✅ |
| Modified/deleted tests carry `Test-Change:` | CLAUDE.md §Critical rules 6 | `check-test-justification.sh` with `standards/test-paths.txt` (`^tests/`) | ✅ |
| Coverage may not fall | review 2026-10-14 (msa#40) | `unit` suite: `--cov-fail-under` = `coverage-baseline.txt` | ✅ |
| Lint | review 2026-10-14 (msa#40) | `lint` suite: ruff, rules pinned in `pyproject.toml` | ✅ |
| No new type errors | review 2026-10-14 (msa#40) | `types` suite: mypy vs `mypy-baseline.txt` (may only shrink) | ⚠️ 82 baselined errors |
| No known-vulnerable dependencies | review 2026-10-14 (msa#40) | `audit` suite: pip-audit on the locked set | ✅ |
| CI installs exactly the locked set | review 2026-10-14 (msa#40) | `uv sync --locked` in every CI job | ✅ |
| Supported Pythons work | `requires-python >= 3.10` | CI `python-versions` matrix (3.10, 3.11) + `test` (3.12) | ✅ |
| Suite scripts, manifest and hook agree | workspace standards | `parity` suite: `check-suite-parity.sh` | ✅ |
| Model-output quality does not regress | review 2026-10-14 (msa#30) | none yet | ❌ msa#30 |

## Security

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| Generated scenes run only in containers | review 2026-10-14 (menger#77) | launcher gate in `cli.py` (`MENGER_AGENT_HOST_RENDER=1` to override); menger's `test-containment.sh` | ✅ |
| Children get no credentials | review 2026-10-14 (msa#27) | `adapters/child_env.py`, `tests/test_child_env.py` | ✅ |

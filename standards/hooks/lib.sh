# shellcheck shell=sh
# Shared helpers for hook policy checks (Sprint 28.2). POSIX sh; source me.

# The source-of-truth main ref. Post GitHub migration (Sprint 35) `origin` still
# points at the stale GitLab mirror in long-lived clones, and merge-base against a
# months-old origin/main counts the entire sprint's diff as "changed" — firing the
# native gates on branches that touch no native code. Prefer github/main when the
# clone has it; fall back to origin/main, then local main.
main_ref() {
    for _r in github/main origin/main main; do
        git rev-parse --verify --quiet "refs/remotes/$_r" >/dev/null 2>&1 && { printf '%s' "$_r"; return 0; }
        git rev-parse --verify --quiet "refs/heads/$_r" >/dev/null 2>&1 && { printf '%s' "$_r"; return 0; }
    done
    printf 'origin/main'
}

commits_in_range() {
    # A base unknown here (e.g. the remote SHA of a force push) falls back to where the tip
    # left main. A range that still cannot be resolved is an error: the policy checks used to
    # see zero commits and pass (menger-toplevel#70).
    git rev-list --no-merges "$1" 2>/dev/null && return 0
    _tip=${1##*..}
    _base=$(git merge-base "$(main_ref)" "$_tip" 2>/dev/null) \
        && git rev-list --no-merges "$_base..$_tip" 2>/dev/null && return 0
    echo "POLICY: cannot resolve the commit range '$1'" >&2
    return 1
}

commit_files() {
    git diff-tree -r --no-commit-id --name-only "$1"
}

commit_modified_files() {
    # modified or deleted only — newly added files are not "changes to existing"
    git diff-tree -r --no-commit-id --name-only --diff-filter=MD "$1"
}

commit_subject() {
    git log -1 --format=%s "$1"
}

commit_is_wip() {
    # Exempt while the work is in progress on a branch, never in a PR: there a WIP commit is
    # about to reach main and gets the same checks as any other (menger-toplevel#71).
    [ -n "${GITHUB_BASE_REF:-}" ] && return 1
    case "$(commit_subject "$1")" in
        WIP:*) return 0 ;;
        *) return 1 ;;
    esac
}

commit_has_trailer() {
    # trailer must carry a non-empty value: "Key: reason"
    git log -1 --format=%B "$1" | grep -q "^$2: ..*"
}

short_ref() {
    git log -1 --format='%h %s' "$1"
}

# --- policy bootstrap exemption ---
#
# A commit-range policy check (issue-link, test-justification, ...) can only enforce
# history written after it went live. standards/hooks/bootstrap-sha.txt is repo-local
# and NOT vendored (every repo's policy history starts at a different commit): it holds
# the SHA the check was live as of. Commits at or before it are grandfathered; commits
# after must comply. Absent file or empty value = no exemption, all history enforced.
# The value is read from main, not the working tree: a commit in the range being checked
# could otherwise move it forward and exempt the commits before it (optix-jni#38).
is_bootstrapped_commit() {
    _bsha_path="$(cd "$HOOKS_DIR" && git rev-parse --show-prefix)bootstrap-sha.txt"
    _bsha=$(git show "$(main_ref):$_bsha_path" 2>/dev/null)
    [ -n "$_bsha" ] || return 1
    git merge-base --is-ancestor "$1" "$_bsha" 2>/dev/null
}

# --- pre-push change detection ----------------------------------------------
#
# git runs pre-push with "<local_ref> <local_sha> <remote_ref> <remote_sha>" lines
# on stdin. Three invocation contexts have to work, and testing only whether stdin
# is a terminal distinguishes just two of them:
#
#   real push          stdin is a pipe carrying the ref lines -> use them
#   manual, terminal   stdin is a TTY -> must NOT read; `read` would block waiting
#                      for a human. Compare the branch against origin/main instead.
#   manual, no TTY     an agent, script or CI job. stdin is empty and closed, so
#                      `read` hits EOF immediately and yields nothing -> also fall
#                      back to comparing against origin/main.
#
# The third case is the trap. `[ -t 0 ]` is false there, so a hook that treats
# "not a TTY" as "git gave me refs" takes the protocol branch, reads nothing, and
# hands every caller an empty change set. The caller then skips all its checks and
# exits 0 — a false pass that is indistinguishable from a clean run. That is how a
# push containing CI-config changes sailed through the gate untested.
#
# So the TTY test stays, but only as a hang guard: what makes this correct is the
# fallback when the read produces nothing. Do not "simplify" this back into a
# single [ -t 0 ] branch.
#
# Emits "<base>..<local_sha>" ranges, one per line.
# Returns 2 when the push only deletes a branch, so callers can exit deliberately.
push_ranges() {
    _ranges=""
    _deleting=0

    if [ ! -t 0 ]; then
        while read -r _local_ref _local_sha _remote_ref _remote_sha; do
            [ -z "$_local_sha" ] && continue
            case "$_local_sha" in
                0000000000000000000000000000000000000000) _deleting=1; continue ;;
            esac
            case "$_remote_sha" in
                ''|0000000000000000000000000000000000000000)
                    # New branch on the remote: diff from where it left main.
                    _base=$(git merge-base "$(main_ref)" "$_local_sha" 2>/dev/null \
                            || git rev-list --max-parents=0 "$_local_sha" | tail -n 1) ;;
                *)  _base="$_remote_sha" ;;
            esac
            _ranges="${_ranges}${_base}..${_local_sha}
"
        done
    fi

    if [ -z "$_ranges" ]; then
        [ "$_deleting" = "1" ] && return 2
        _base=$(git merge-base "$(main_ref)" HEAD 2>/dev/null)
        [ -n "$_base" ] && _ranges="${_base}..HEAD
"
    fi

    printf '%s' "$_ranges"
}

# Files touched by the given ranges, deduplicated.
changed_files() {
    for _range in "$@"; do
        [ -n "$_range" ] && git diff --name-only "$_range"
    done | sort -u
}

# --- tested tree == pushed tree (review 2026-10-14, menger-toplevel#66) ---------------
#
# The suites build and test the files on disk, and qa-runner caches a PASS against HEAD's
# tree. Both are only sound when what is on disk is exactly what is being pushed: no
# uncommitted changes to tracked files, no untracked file a build would pick up, and every
# pushed ref at HEAD. Otherwise an uncommitted fix makes the gate pass a commit that lacks
# it, and `git push origin <other-branch>` tests the checked-out branch instead.
# Untracked files that are not build inputs (notes, scratch output) are fine.
BUILD_INPUT_ERE='(^|/)(src|project|scripts|standards|\.git_hooks|docker)/|\.(sbt|scala|java|kt|py|sh|cpp|cu|cuh|h|hpp|cmake)$|(^|/)CMakeLists\.txt$|^qa-manifest\.yaml$'

# $1 = newline-separated "<base>..<sha>" ranges (QA_RANGES). Prints what differs; returns 1.
assert_tested_tree_is_pushed() {
    _problems=""
    _dirty=$(git status --porcelain --untracked-files=no 2>/dev/null)
    if [ -n "$_dirty" ]; then
        _problems="${_problems}  uncommitted changes to tracked files:
$(printf '%s\n' "$_dirty" | sed 's/^/    /')
"
    fi
    _untracked=$(git ls-files --others --exclude-standard 2>/dev/null \
        | grep -E "$BUILD_INPUT_ERE" || true)
    if [ -n "$_untracked" ]; then
        _problems="${_problems}  untracked files the build would use:
$(printf '%s\n' "$_untracked" | sed 's/^/    /')
"
    fi
    _head=$(git rev-parse HEAD 2>/dev/null)
    for _range in $1; do
        _tip=${_range##*..}
        _tip_sha=$(git rev-parse --verify --quiet "$_tip^{commit}" 2>/dev/null)
        if [ "$_tip_sha" != "$_head" ]; then
            _problems="${_problems}  pushing $(short_ref "$_tip") but HEAD is $(short_ref "$_head")
"
        fi
    done
    [ -z "$_problems" ] && return 0
    echo "pre-push: the gate tests the working tree, but it is not what is being pushed:" >&2
    printf '%s' "$_problems" >&2
    echo "  Commit or stash (git stash -u) the changes, or check out the branch you push," >&2
    echo "  then push again." >&2
    return 1
}

# $1 = the detected file list. An empty set on a branch that has commits to push
# means detection failed; reporting that as "nothing to check" is the bug this
# whole helper exists to prevent, so refuse to continue.
assert_detection() {
    [ -n "$1" ] && return 0
    if [ -n "$(git rev-list "$(main_ref)..HEAD" 2>/dev/null | head -n 1)" ]; then
        echo "pre-push: detected no changed files, but HEAD is ahead of $(main_ref)." >&2
        echo "          Change detection is broken — refusing to report a clean run." >&2
        echo "          (If this push genuinely changes no files, say so explicitly.)" >&2
        return 1
    fi
    return 0
}

# --- suite-script result helpers (Sprint 36 B1) ---
#
# Every scripts/suites/<name>.sh must emit exactly one SUITE line as the last thing it
# prints; qa-runner.sh (shared/standards/scripts/qa-runner.sh) parses it. Semicolons in
# failed-item names are not expected (test/check names don't contain them); do not pass
# names containing '"' through these helpers unescaped.
suite_pass() {
    echo "SUITE $1 PASS n_failed=0 failed=\"\" reason=\"\""
}

# $1 = suite name, $2 = human reason (no quote or semicolon characters)
suite_skip() {
    echo "SUITE $1 SKIP n_failed=0 failed=\"\" reason=\"$2\""
    # A skipped suite inside a green CI job is otherwise invisible: surface it as a job
    # annotation and in the run summary.
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo "::warning title=$1 skipped::$2"
        if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
            echo "- **$1 skipped:** $2" >> "$GITHUB_STEP_SUMMARY"
        fi
    fi
}

# $1 = suite name, $2 = failure count, $3 = ';'-joined failed item names
suite_fail() {
    echo "SUITE $1 FAIL n_failed=$2 failed=\"$3\" reason=\"\""
}

# --- GPU environment gate (Sprint 36 D1) ---
#
# $1 = suite name. Call before any suite does real GPU work. Locally, a busy GPU is a
# quiet SKIP (a human is right there and can just retry once the desktop app clears —
# low friction). In CI ($CI is GitHub Actions' own standard env var) an unattended job
# silently skipping GPU tests would be dangerous — it could mask a real regression going
# unnoticed for a week — so it reports FAIL with an unmistakably-labeled reason instead,
# forcing a human to look and rerun (`gh run rerun --failed`, already documented, C6).
# `--skip-in-ci` (second argument) makes CI skip too: for suites whose verdict can't be
# trusted on a busy card anyway (perf gates), while another suite still enforces the GPU.
# A suite that would FAIL in CI first waits longer (GPU_PREFLIGHT_CI_RETRIES samples of the
# preflight's interval, default 60 x 10 s): on a self-hosted runner shared with a desktop, a
# busy GPU is usually transient (a render window left playing, toplevel#64). 10 minutes fits
# the 20-minute GPU job timeouts. Locally and for --skip-in-ci, the preflight's own window.
# Locally the SKIP is only for an interactive push (QA_INTERACTIVE=1, set by qa-runner when
# its stdout is a terminal), where a human sees it. A non-interactive push — an agent or a
# script, which only reads the exit code — FAILs instead (review 2026-10-14,
# menger-toplevel#68), unless QA_ALLOW_ENV_SKIP=1 says the skip is knowingly accepted.
# Returns 1 (caller must stop and exit) when the GPU is unsuitable; 0 to proceed.
gpu_preflight_or_skip() {
    _suite="$1"
    _skip_in_ci="${2:-}"
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    if [ -n "${CI:-}" ] && [ "$_skip_in_ci" != "--skip-in-ci" ]; then
        _detail=$(./standards/scripts/gpu-preflight.sh \
            --retries "${GPU_PREFLIGHT_CI_RETRIES:-60}") && return 0
        suite_fail "$_suite" 0 "ENV-UNSUITABLE(GPU busy: $_detail)"
    elif [ -n "${CI:-}" ] || [ "${QA_INTERACTIVE:-}" = "1" ] \
        || [ "${QA_ALLOW_ENV_SKIP:-}" = "1" ]; then
        _detail=$(./standards/scripts/gpu-preflight.sh) && return 0
        suite_skip "$_suite" "env: GPU busy — $_detail"
    else
        _detail=$(./standards/scripts/gpu-preflight.sh) && return 0
        suite_fail "$_suite" 0 "ENV-UNSUITABLE(GPU busy, non-interactive push: $_detail; free the GPU, or set QA_ALLOW_ENV_SKIP=1 to skip knowingly)"
    fi
    return 1
}

# --- performance-gate verdict ---
#
# $1 = suite name, $2 = log of an sbt run of Perf-tagged tests (io.github.lene.qa.RelativeBenchmark
# gates; ANSI codes allowed), $3 = sbt's exit code. Emits the SUITE line. Returns 1 only for a
# FAIL: a failed gate, a build error, or no gate having run at all (a vacuous pass). A gate
# that couldn't be judged reliably is canceled by the test and makes the suite SKIP -- never a
# failure, but also never a PASS, because a PASS is cached per tree and would hide the
# unjudged gates for good.
perf_suite_verdict() {
    _suite="$1"; _log="$2"; _rc="$3"
    _clean=$(sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$_log")
    _failed=$(printf '%s\n' "$_clean" | grep -c '\*\*\* FAILED \*\*\*' || true)
    if [ "$_failed" -gt 0 ]; then
        suite_fail "$_suite" "$_failed" "$(printf '%s\n' "$_clean" | grep '\*\*\* FAILED \*\*\*' \
            | sed 's/^\[info\] *- *//; s/ \*\*\* FAILED \*\*\*.*//' | paste -sd';' -)"
        return 1
    fi
    if [ "$_rc" -ne 0 ]; then
        suite_fail "$_suite" 1 "build-or-unknown"
        return 1
    fi
    _summary=$(printf '%s\n' "$_clean" | grep 'Tests: succeeded' | tail -n 1)
    _succeeded=$(printf '%s' "$_summary" | sed -n 's/.*succeeded \([0-9]*\).*/\1/p')
    _canceled=$(printf '%s' "$_summary" | sed -n 's/.*canceled \([0-9]*\).*/\1/p')
    _total=$(( ${_succeeded:-0} + ${_canceled:-0} ))
    if [ "$_total" -eq 0 ]; then
        suite_fail "$_suite" 1 "no perf gates ran"
        return 1
    fi
    if [ "${_canceled:-0}" -gt 0 ]; then
        suite_skip "$_suite" "$_canceled/$_total gates inconclusive: $(printf '%s\n' "$_clean" \
            | grep '!!! CANCELED !!!' | sed 's/^\[info\] *- *//; s/ !!! CANCELED !!!.*//' \
            | paste -sd';' -)"
        return 0
    fi
    suite_pass "$_suite"
    return 0
}

# --- network retry (Sprint 36 D4) ---
#
# $1 = human label for logging, remaining args = the command to run. Retries up to
# RETRY_ATTEMPTS times (default 3) with exponential backoff (RETRY_BASE_DELAY seconds,
# default 5, doubling each attempt). Mirrors the shape already used ad hoc in
# menger-common's CI publish/sync steps (`for attempt in 1 2 3; do ...; sleep 30; done`)
# as a reusable POSIX function instead of a copy-pasted loop per call site.
retry_with_backoff() {
    _label="$1"; shift
    _attempts="${RETRY_ATTEMPTS:-3}"
    _delay="${RETRY_BASE_DELAY:-5}"
    _n=1
    while [ "$_n" -le "$_attempts" ]; do
        "$@" && return 0
        if [ "$_n" -lt "$_attempts" ]; then
            echo "retry_with_backoff: $_label failed (attempt $_n/$_attempts), retrying in ${_delay}s..." >&2
            sleep "$_delay"
            _delay=$((_delay * 2))
        fi
        _n=$((_n + 1))
    done
    echo "retry_with_backoff: $_label failed after $_attempts attempts" >&2
    return 1
}

# --- suite duration tracking (T2#9: pre-push progress/ETA) ---
#
# .git/qa-cache/durations/<suite> holds the last 10 wall-clock durations (whole seconds)
# of real runs of that suite, oldest first, one per line. Unlike the verdict cache
# (.git/qa-cache/<suite>-<tree-sha>, pruned at 14 days because a verdict goes stale with
# the tree), a duration doesn't depend on the tree at all, so this is never pruned.

# $1 = cache dir, $2 = suite name, $3 = elapsed seconds of a real run.
record_duration() {
    _ddir="$1/durations"
    mkdir -p "$_ddir" 2>/dev/null || true
    _dfile="$_ddir/$2"
    { [ -f "$_dfile" ] && cat "$_dfile"; echo "$3"; } | tail -n 10 > "$_dfile.tmp" && mv "$_dfile.tmp" "$_dfile"
    return 0
}

# $1 = cache dir, $2 = suite name. Prints the median of recorded durations, or nothing
# if none recorded yet.
median_duration() {
    _dfile="$1/durations/$2"
    [ -f "$_dfile" ] || return 0
    sort -n "$_dfile" | awk '{a[NR]=$1} END { if (NR==0) exit; print (NR%2==1) ? a[(NR+1)/2] : int((a[int(NR/2)]+a[int(NR/2)+1])/2) }'
}

# $1 = seconds. Prints "XmYYs" (or "Ys" under a minute). No output for an empty/missing arg.
format_duration() {
    [ -z "${1:-}" ] && return 0
    if [ "$1" -ge 60 ]; then
        printf '%dm%02ds' "$(($1 / 60))" "$(($1 % 60))"
    else
        printf '%ds' "$1"
    fi
}

# --- generic ratchet comparison (Sprint 36 F, O7) ---
#
# Quality signals (coverage, Sonar rating, sanitizer warning count, perf ratio) ratchet:
# the current value is the floor, regression fails the gate, improvement may move the
# floor. This is the one comparison all of them share; callers own reading/writing their
# own baseline file so each ratchet's own bump policy (unconditional on pass, vs.
# noise-gated) stays with the caller instead of being baked in here.
#
# $1 = label (for messages), $2 = current numeric value, $3 = baseline numeric value,
# $4 = direction ("up" = higher is better, e.g. coverage; "down" = lower is better,
# e.g. sanitizer warning count, perf ratio, Sonar rating-as-number), $5 = tolerance
# (absolute slack before FAIL; default 0 = strict).
#
# Echoes "PASS unchanged" | "PASS improved" | "FAIL regressed" to stdout; caller decides
# whether "PASS improved" warrants rewriting its baseline file.
ratchet_check() {
    _label="$1"; _current="$2"; _baseline="$3"; _direction="$4"; _tolerance="${5:-0}"

    # Fail closed on malformed input (security review finding): bc silently returns an
    # empty/garbage result for a non-numeric operand, and `[ "" -eq 1 ]` is a harmless
    # no-op rather than an error under `set -u` alone — a corrupted baseline file or a
    # bad API response would silently fall through to "PASS unchanged" instead of
    # failing the gate. Validating strictly-numeric input before it ever reaches bc also
    # closes the underlying "unsanitized value piped into an interpreter" shape.
    for _val in "$_current" "$_baseline" "$_tolerance"; do
        echo "$_val" | grep -qE '^-?[0-9]+(\.[0-9]+)?$' || {
            echo "FAIL regressed"
            echo "ratchet($_label): non-numeric input (current='$_current' baseline='$_baseline' tolerance='$_tolerance') — failing closed" >&2
            return 1
        }
    done

    if [ "$_direction" = "up" ]; then
        _delta=$(echo "$_baseline - $_current" | bc)   # positive = regression
    else
        _delta=$(echo "$_current - $_baseline" | bc)   # positive = regression
    fi
    if [ "$(echo "$_delta > $_tolerance" | bc)" -eq 1 ]; then
        echo "FAIL regressed"
        echo "ratchet($_label): $_current vs baseline $_baseline — regression (delta $_delta > tolerance $_tolerance)" >&2
        return 1
    fi
    if [ "$(echo "$_delta < 0" | bc)" -eq 1 ]; then
        echo "PASS improved"
    else
        echo "PASS unchanged"
    fi
    return 0
}

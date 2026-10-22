#!/bin/sh
# Shared QA suite runner (Sprint 36 Phase C: manifest + resume cache + fail-fast). Canon
# lives here; vendored byte-identical into menger/optix-jni/menger-common as
# standards/scripts/qa-runner.sh via scripts/sync-standards.sh (see
# shared/standards/manifest.txt, shared/standards/README.md).
#
# Usage: qa-runner.sh --tier <commit|push|pr|main|release> [--all] [<suite> ...]
#
# Reads ./qa-manifest.yaml (per-repo, not vendored — see menger/scripts/coverage-manifest.yaml
# for the established pattern of a repo-owned manifest). Two sections:
#   change_classes: <name>: [<ERE pattern>, ...]   (a pattern may be "@file:<path>" to read
#                                                     patterns from an existing pattern file)
#   suites: <name>: { tier: <t>, requires: [<class>, ...], automated: true|false }
# `requires: []` = always relevant. `automated: false` = no scripts/suites/<name>.sh exists
# (e.g. a still-manual release-tier check) — always reported SKIP, never invoked.
#
# A suite runs only if: its tier <= --tier ceiling, AND (requires is empty OR at least one
# required change-class matched $CHANGED_FILES), AND it isn't automated:false. Anything else
# is recorded straight into SKIP with a labeled reason — no scripts/suites/<name>.sh subshell
# is spawned for it. Before invoking a relevant suite, HAS_<CLASS> (uppercased) is exported
# for every change_classes entry, exactly as the calling hook used to compute and export
# HAS_SCALA/HAS_NATIVE/... itself — suite scripts need no changes.
#
# Each invoked scripts/suites/<name>.sh runs to completion (its own stdout/stderr stream
# live) and, as the LAST line it prints, emits exactly one line of the form:
#   SUITE <name> PASS|FAIL|SKIP n_failed=<k> failed="<name1;name2;...>" reason="<text>"
# A suite script that produces no such line (crash, syntax error, killed) is treated as
# FAIL, not silently ignored — same "refuse to report a clean run on missing evidence"
# principle as assert_detection in standards/hooks/lib.sh.
#
# Resume cache: a suite whose last live run on this exact tree was PASS is remembered at
# .git/qa-cache/<suite>-<tree-sha> and reported "CACHED PASS (tree unchanged)" without
# re-invoking its script. Never cached on FAIL or SKIP — only a proven-clean PASS is safe
# to trust later. Entries older than 14 days are pruned automatically.
#
# Fail-fast: on a TTY (interactive push) the run stops at the first FAIL and prints the
# summary so far. --all (or a non-TTY caller, e.g. CI) runs every relevant suite regardless.
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# shellcheck source=../hooks/lib.sh
. "$SCRIPT_DIR/../hooks/lib.sh"

usage() {
  echo "usage: qa-runner.sh --tier <commit|push|pr|main|release> [--all] [<suite> ...]" >&2
  exit 2
}

TIER=""
ALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="${2:-}"; shift 2 ;;
    --tier=*) TIER="${1#--tier=}"; shift ;;
    --all) ALL=1; shift ;;
    --) shift; break ;;
    -*) usage ;;
    *) break ;;
  esac
done
[ -n "$TIER" ] || usage
case "$TIER" in
  commit | push | pr | main | release) ;;
  *) usage ;;
esac

SUITES_DIR="./scripts/suites"
MANIFEST="./qa-manifest.yaml"
# --git-path: in a `git worktree`, .git is a file, not a directory.
CACHE_DIR=$(git rev-parse --git-path qa-cache 2>/dev/null || echo .git/qa-cache)
CHANGED_FILES="${CHANGED_FILES:-}"

[ -r "$MANIFEST" ] || { echo "qa-runner.sh: missing $MANIFEST" >&2; exit 2; }

tier_rank() {
  case "$1" in
    commit) echo 0 ;;
    push) echo 1 ;;
    pr) echo 2 ;;
    main) echo 3 ;;
    release) echo 4 ;;
    *) echo 99 ;;
  esac
}
CEILING=$(tier_rank "$TIER")

MANIFEST_FLAT=$(mktemp)
RESULTS=$(mktemp)
SUITE_LIST_FILE=$(mktemp)
DURATIONS_FILE=$(mktemp)
trap 'rm -f "$MANIFEST_FLAT" "$RESULTS" "$SUITE_LIST_FILE" "$DURATIONS_FILE"' EXIT

# --- Parse the manifest into flat, tab-separated CLASS/SUITE records --------------------
# CLASS <name> <pattern>          (one line per pattern)
# SUITE <name> <tier> <requires-csv-or-dash> <automated:true|false>
# requires-csv is "-" (not empty) when requires:[] — POSIX `read` collapses consecutive
# IFS-whitespace delimiters (tab counts as whitespace even set alone), so a truly empty
# field between two tabs would silently merge with its neighbor; "-" survives the split
# and is normalized back to "" right after each read.
awk -f - "$MANIFEST" <<'AWK_EOF' > "$MANIFEST_FLAT"
BEGIN { section = ""; class = "" }
/^change_classes:[ \t]*$/ { section = "classes"; next }
/^suites:[ \t]*$/ { section = "suites"; next }
section == "classes" {
  if ($0 ~ /^  [A-Za-z0-9_-]+:[ \t]*$/) {
    class = $0
    sub(/^  /, "", class)
    sub(/:[ \t]*$/, "", class)
    next
  }
  if ($0 ~ /^    - /) {
    pat = $0
    sub(/^    - /, "", pat)
    gsub(/^'/, "", pat); gsub(/'$/, "", pat)
    gsub(/^"/, "", pat); gsub(/"$/, "", pat)
    print "CLASS\t" class "\t" pat
  }
  next
}
section == "suites" {
  if ($0 ~ /^  [A-Za-z0-9_-]+:[ \t]*\{/) {
    line = $0
    name = line
    sub(/^  /, "", name)
    sub(/:.*/, "", name)
    tier = line
    sub(/.*tier:[ \t]*/, "", tier)
    sub(/[ \t,}].*/, "", tier)
    req = ""
    if (match(line, /requires:[ \t]*\[[^]]*\]/)) {
      r = substr(line, RSTART, RLENGTH)
      sub(/requires:[ \t]*\[/, "", r)
      sub(/\][ \t]*$/, "", r)
      gsub(/[ \t]/, "", r)
      req = r
    }
    auto = "true"
    if (match(line, /automated:[ \t]*(true|false)/)) {
      a = substr(line, RSTART, RLENGTH)
      sub(/.*automated:[ \t]*/, "", a)
      auto = a
    }
    if (req == "") req = "-"
    print "SUITE\t" name "\t" tier "\t" req "\t" auto
  } else if ($0 ~ /^  [A-Za-z0-9_-]+:/) {
    # A suite this parser cannot read (e.g. block-style YAML) used to vanish silently.
    name = $0
    sub(/^  /, "", name)
    sub(/:.*/, "", name)
    print "BAD\t" name "\tnot a one-line { tier: ..., requires: [...] } entry"
  }
  next
}
AWK_EOF

# $1 = class name; prints an ERE alternation of all its patterns, expanding any
# "@file:<path>" entry into that pattern file's non-comment/non-blank lines.
class_pattern() {
  _pats=""
  _tmp=$(mktemp)
  grep "^CLASS	$1	" "$MANIFEST_FLAT" | cut -f3 > "$_tmp"
  while IFS= read -r _p; do
    case "$_p" in
      @file:*)
        _f=${_p#@file:}
        if [ -r "$_f" ]; then
          _filepats=$(grep -v '^[[:space:]]*\(#\|$\)' "$_f" | paste -sd'|' -)
          [ -n "$_filepats" ] && _pats="${_pats:+$_pats|}$_filepats"
        fi
        ;;
      *)
        _pats="${_pats:+$_pats|}$_p"
        ;;
    esac
  done < "$_tmp"
  rm -f "$_tmp"
  printf '%s' "$_pats"
}

# Compute and export HAS_<CLASS> for every change_classes entry, replacing what each
# hook's own grep block used to do.
for _class in $(grep '^CLASS' "$MANIFEST_FLAT" | cut -f2 | sort -u); do
  _pattern=$(class_pattern "$_class")
  _upper=$(printf '%s' "$_class" | tr 'a-z-' 'A-Z_')
  _count=0
  if [ -n "$_pattern" ]; then
    _count=$(printf '%s\n' "$CHANGED_FILES" | grep -cE "$_pattern" 2>/dev/null)
    [ -z "$_count" ] && _count=0
  fi
  eval "HAS_${_upper}=${_count}; export HAS_${_upper}"
done

# $1 = comma-separated requires list (empty = always relevant).
suite_relevant() {
  [ -z "$1" ] && return 0
  for _req in $(printf '%s' "$1" | tr ',' ' '); do
    _upper=$(printf '%s' "$_req" | tr 'a-z-' 'A-Z_')
    eval "_val=\${HAS_${_upper}:-0}"
    [ "$_val" -gt 0 ] 2>/dev/null && return 0
  done
  return 1
}

# --- Validate the manifest before running anything (review 2026-10-14, menger-toplevel#67) ---
# Each of these used to turn into a SKIP (or vanish) with exit 0, so a one-character manifest
# typo or a lost exec bit removed a gate while the run stayed green.
CONFIG_ERRORS=""
config_error() { CONFIG_ERRORS="${CONFIG_ERRORS}  $1
"; }
_bad=$(grep '^BAD' "$MANIFEST_FLAT" | cut -f2,3 | tr '\t' ':')
[ -n "$_bad" ] && config_error "unparseable suite entries: $_bad"
_classes=" $(grep '^CLASS' "$MANIFEST_FLAT" | cut -f2 | sort -u | tr '\n' ' ')"
for _class in $(grep '^CLASS' "$MANIFEST_FLAT" | cut -f2 | sort -u); do
  for _p in $(grep "^CLASS	$_class	" "$MANIFEST_FLAT" | cut -f3 | grep '^@file:'); do
    [ -r "${_p#@file:}" ] || config_error "class $_class: pattern file ${_p#@file:} not readable"
  done
  _pattern=$(class_pattern "$_class")
  if [ -n "$_pattern" ]; then
    printf '' | grep -E "$_pattern" >/dev/null 2>&1
    [ $? -eq 2 ] && config_error "class $_class: invalid regular expression"
  fi
done
while IFS="$(printf '\t')" read -r _tag _name _tier _req _auto; do
  [ "$_tag" = "SUITE" ] || continue
  [ "$(tier_rank "$_tier")" -eq 99 ] && config_error "suite $_name: unknown tier '$_tier'"
  if [ "$_req" != "-" ]; then
    for _r in $(printf '%s' "$_req" | tr ',' ' '); do
      case "$_classes" in
        *" $_r "*) ;;
        *) config_error "suite $_name: requires undefined change class '$_r'" ;;
      esac
    done
  fi
  if [ "$_auto" != "false" ] && [ ! -x "$SUITES_DIR/$_name.sh" ]; then
    config_error "suite $_name: $SUITES_DIR/$_name.sh missing or not executable"
  fi
done < "$MANIFEST_FLAT"
for _want in "$@"; do
  grep -q "^SUITE	$_want	" "$MANIFEST_FLAT" || config_error "requested suite '$_want' is not in $MANIFEST"
done
if [ -n "$CONFIG_ERRORS" ]; then
  echo "qa-runner.sh: $MANIFEST has errors — refusing to run with gates silently missing:" >&2
  printf '%s' "$CONFIG_ERRORS" >&2
  exit 2
fi

grep '^SUITE' "$MANIFEST_FLAT" > "$SUITE_LIST_FILE"

if [ $# -gt 0 ]; then
  : > "$SUITE_LIST_FILE.filtered"
  for _want in "$@"; do
    grep -m1 "^SUITE	$_want	" "$MANIFEST_FLAT" >> "$SUITE_LIST_FILE.filtered" || true
  done
  mv "$SUITE_LIST_FILE.filtered" "$SUITE_LIST_FILE"
fi

# --- The tree being tested must be the tree being pushed (menger-toplevel#66) ---------
# Only for a local push: CI runs on a fresh checkout of exactly the pushed commit.
if [ "$TIER" = "push" ] && [ -z "${CI:-}" ]; then
  assert_tested_tree_is_pushed "${QA_RANGES:-}" || exit 1
fi

# Suites see whether a human is watching: a local GPU-busy SKIP is only acceptable then
# (gpu_preflight_or_skip, menger-toplevel#68).
if [ -t 1 ]; then QA_INTERACTIVE=1; else QA_INTERACTIVE=0; fi
export QA_INTERACTIVE

TOTAL=$(grep -c '^SUITE' "$SUITE_LIST_FILE" || true)
[ -z "$TOTAL" ] && TOTAL=0
N=0
OVERALL=0
TAB=$(printf '\t')

mkdir -p "$CACHE_DIR" 2>/dev/null || true
find "$CACHE_DIR" -type f -mtime +14 -delete 2>/dev/null || true
# No tree, no cache: a shared fallback key ("notree") let one PASS stand for any code.
TREE_SHA=$(git rev-parse 'HEAD^{tree}' 2>/dev/null) || TREE_SHA=""

# What a cached PASS stands for (menger-toplevel#69): the tree, and also the tier, the pushed
# range (the policy suites check commit messages, which an amend changes without changing the
# tree) and the toolchain the suites ran with. Absent commands fingerprint as "not found".
toolchain_fingerprint() {
  for _cmd in "java -version" "nvcc --version" "python3 --version" "uv --version"; do
    $_cmd 2>&1 | head -n 3
  done
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1
}
CACHE_KEY=""
if [ -n "$TREE_SHA" ]; then
  CACHE_KEY=$({ echo "$TREE_SHA"; echo "$TIER"; echo "${QA_RANGES:-}"; toolchain_fingerprint; } \
    | sha256sum | cut -c1-64)
fi

FAILFAST=0
[ -t 1 ] && [ "$ALL" -eq 0 ] && FAILFAST=1
RUN_START=$(date +%s)

# $1 = index (1-based) of the suite about to run; sums median durations of the suites
# after it that will actually execute (mirrors the skip logic below). A suite with no
# recorded duration yet makes the estimate a floor, flagged with a trailing '+?'.
remaining_eta() {
  _after="$1"
  _sum=0
  _unknown=0
  _i=0
  while IFS="$TAB" read -r _rtag _rname _rtier _rreq _rauto; do
    [ "$_rtag" = "SUITE" ] || continue
    _i=$((_i + 1))
    [ "$_i" -le "$_after" ] && continue
    [ "$_rreq" = "-" ] && _rreq=""
    [ "$(tier_rank "$_rtier")" -gt "$CEILING" ] && continue
    [ "$_rauto" = "false" ] && continue
    suite_relevant "$_rreq" || continue
    _rcache="$CACHE_DIR/$_rname-$CACHE_KEY"
    [ -n "$CACHE_KEY" ] && [ -f "$_rcache" ] && grep -q '^verdict=PASS$' "$_rcache" 2>/dev/null \
      && continue
    _rmed=$(median_duration "$CACHE_DIR" "$_rname")
    if [ -n "$_rmed" ]; then
      _sum=$((_sum + _rmed))
    else
      _unknown=1
    fi
  done < "$SUITE_LIST_FILE"
  if [ "$_sum" -gt 0 ]; then
    printf '%s' "$(format_duration "$_sum")"
    [ "$_unknown" -eq 1 ] && printf '+?'
  else
    printf '?'
  fi
  return 0
}

while IFS="$TAB" read -r _tag name tier req auto; do
  [ "$_tag" = "SUITE" ] || continue
  [ "$req" = "-" ] && req=""
  N=$((N + 1))
  _elapsed=$(( $(date +%s) - RUN_START ))
  echo "[$N/$TOTAL] $name — elapsed $(format_duration "$_elapsed"), ETA ~$(remaining_eta "$((N - 1))")"

  suite_ceiling=$(tier_rank "$tier")
  if [ "$suite_ceiling" -gt "$CEILING" ]; then
    echo "  SKIPPED (tier: $tier > $TIER)"
    printf '%s%sSKIP%stier: %s > %s\n' "$name" "$TAB" "$TAB" "$tier" "$TIER" >> "$RESULTS"
    continue
  fi

  if [ "$auto" = "false" ]; then
    echo "  SKIPPED (manual/unautomated, release-tier only)"
    printf '%s%sSKIP%smanual/unautomated, release-tier only\n' "$name" "$TAB" "$TAB" >> "$RESULTS"
    continue
  fi

  if ! suite_relevant "$req"; then
    echo "  SKIPPED (no matching changed paths)"
    printf '%s%sSKIP%sno matching changed paths\n' "$name" "$TAB" "$TAB" >> "$RESULTS"
    continue
  fi

  script="$SUITES_DIR/$name.sh"
  if [ ! -x "$script" ]; then
    # Validated up front; reaching this means the script vanished mid-run. Never a SKIP.
    echo "  ${name}: FAILED — suite script $script missing or not executable"
    printf '%s%sFAIL%smissing script\n' "$name" "$TAB" "$TAB" >> "$RESULTS"
    OVERALL=1
    [ "$FAILFAST" -eq 1 ] && break
    continue
  fi

  CACHE_FILE="$CACHE_DIR/$name-$CACHE_KEY"
  if [ -n "$CACHE_KEY" ] && [ -f "$CACHE_FILE" ] \
      && grep -q '^verdict=PASS$' "$CACHE_FILE" 2>/dev/null; then
    echo "  CACHED PASS (tree unchanged)"
    printf '%s%sCACHED%s\n' "$name" "$TAB" "$TAB" >> "$RESULTS"
    continue
  fi

  LOG=$(mktemp)
  RCFILE=$(mktemp)
  _t0=$(date +%s)
  # The suite's own exit status, which `| tee` would hide (menger-toplevel#78).
  { "$script" 2>&1; echo $? > "$RCFILE"; } | tee "$LOG"
  _t1=$(date +%s)
  SUITE_RC=$(cat "$RCFILE"); rm -f "$RCFILE"

  LINE=$(grep "^SUITE $name " "$LOG" | tail -n 1)
  # A suite that prints several verdicts for itself (output of nested runs mixed in, #78) is
  # not trusted with the last one.
  VERDICTS=$(grep "^SUITE $name " "$LOG" | awk '{print $3}' | sort -u | wc -l)
  if [ -n "$LINE" ] && [ "$VERDICTS" -gt 1 ]; then
    echo "  ${name}: FAILED — conflicting SUITE lines for $name"
    LINE="SUITE $name FAIL n_failed=1 failed=\"conflicting SUITE lines\" reason=\"\""
  elif [ -n "$LINE" ] && [ "$SUITE_RC" != "0" ] \
      && [ "$(printf '%s' "$LINE" | awk '{print $3}')" = "PASS" ]; then
    echo "  ${name}: FAILED — exit status $SUITE_RC despite a PASS line"
    LINE="SUITE $name FAIL n_failed=1 failed=\"exit status $SUITE_RC\" reason=\"\""
  fi
  if [ -z "$LINE" ]; then
    echo "  ${name}: FAILED — no SUITE line emitted (crashed or misbehaved)"
    printf '%s%sFAIL%sno SUITE line emitted\n' "$name" "$TAB" "$TAB" >> "$RESULTS"
    printf '%s\t%s\n' "$name" "$((_t1 - _t0))" >> "$DURATIONS_FILE"
    record_duration "$CACHE_DIR" "$name" "$((_t1 - _t0))"
    OVERALL=1
    rm -f "$LOG"
    [ "$FAILFAST" -eq 1 ] && break
    continue
  fi

  VERDICT=$(printf '%s' "$LINE" | awk '{print $3}')
  case "$VERDICT" in
    PASS)
      printf '%s%sPASS%s\n' "$name" "$TAB" "$TAB" >> "$RESULTS"
      printf '%s\t%s\n' "$name" "$((_t1 - _t0))" >> "$DURATIONS_FILE"
      record_duration "$CACHE_DIR" "$name" "$((_t1 - _t0))"
      if [ -n "$CACHE_KEY" ]; then
        { echo "verdict=PASS"; echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "tier=$tier"; } > "$CACHE_FILE"
      fi
      ;;
    SKIP)
      REASON=$(printf '%s' "$LINE" | sed -n 's/.*reason="\([^"]*\)".*/\1/p')
      printf '%s%sSKIP%s%s\n' "$name" "$TAB" "$TAB" "$REASON" >> "$RESULTS"
      ;;
    *)
      FAILED_NAMES=$(printf '%s' "$LINE" | sed -n 's/.*failed="\([^"]*\)".*/\1/p')
      printf '%s%sFAIL%s%s\n' "$name" "$TAB" "$TAB" "$FAILED_NAMES" >> "$RESULTS"
      printf '%s\t%s\n' "$name" "$((_t1 - _t0))" >> "$DURATIONS_FILE"
      record_duration "$CACHE_DIR" "$name" "$((_t1 - _t0))"
      OVERALL=1
      ;;
  esac
  rm -f "$LOG"
  if [ "$VERDICT" != "PASS" ] && [ "$VERDICT" != "SKIP" ] && [ "$FAILFAST" -eq 1 ]; then
    break
  fi
done < "$SUITE_LIST_FILE"

echo ""
echo "=== QA summary ==="
while IFS="$TAB" read -r name verdict detail; do
  _dur=$(grep "^$name	" "$DURATIONS_FILE" 2>/dev/null | cut -f2)
  _dur_suffix=""
  [ -n "$_dur" ] && _dur_suffix=" ($(format_duration "$_dur"))"
  case "$verdict" in
    PASS) echo "  PASS    $name$_dur_suffix" ;;
    CACHED) echo "  CACHED  $name (tree unchanged)" ;;
    SKIP) echo "  SKIPPED $name ($detail)" ;;
    FAIL) echo "  FAILED  $name${detail:+ [$detail]}$_dur_suffix" ;;
    *) echo "  UNKNOWN $name (unrecognized verdict '$verdict')" ;;
  esac
done < "$RESULTS"

_env_skips=$(grep -c "${TAB}SKIP${TAB}env:" "$RESULTS" 2>/dev/null || true)
if [ "${_env_skips:-0}" -gt 0 ]; then
  echo ""
  echo "WARNING: $_env_skips suite(s) skipped for environment reasons — this run is NOT fully gated."
fi

if [ "$OVERALL" -ne 0 ]; then
  echo ""
  echo "=== Failed suites, details ==="
  while IFS="$TAB" read -r name verdict detail; do
    [ "$verdict" = "FAIL" ] && [ -n "$detail" ] && echo "  $name: $detail"
  done < "$RESULTS"
fi

exit "$OVERALL"

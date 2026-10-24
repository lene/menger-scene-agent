#!/bin/sh
# Suite-script / manifest parity check (Sprint 36 B5 + C7). Canon lives here; vendored
# byte-identical into menger/optix-jni/menger-common as scripts/check-suite-parity.sh via
# scripts/sync-standards.sh (see shared/standards/manifest.txt). Checks:
#  1. ORPHAN: every scripts/suites/<name>.sh must be referenced by at least one of:
#     qa-manifest.yaml's suites: section, .git_hooks/pre-push, .github/workflows/ci.yml.
#  2. DANGLING (hook): if the hook still hardcodes a literal qa-runner.sh suite-name list
#     (pre-manifest style — menger, during its B3 parity period), every named suite must
#     have a matching scripts/suites/<name>.sh. Skipped once a hook moves to `--tier`
#     (C1), since suite selection there is manifest-driven, not hook-embedded.
#  3. DANGLING (manifest): every qa-manifest.yaml suites: entry must have a matching
#     scripts/suites/<name>.sh, or be marked automated: false (a documented pointer to a
#     still-manual, release-tier-only check with no script at all).
#  4. BAD TIER: every qa-manifest.yaml tier: value must be one of
#     commit|push|pr|main|release.
# Same "known gap, not silent" discipline as check-standards-drift.sh. Deliberately
# narrower than full hook-vs-CI set equality: doesn't require every suite the hook runs
# to also run in CI, or vice versa.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SUITES_DIR="$ROOT/scripts/suites"
HOOK="$ROOT/.git_hooks/pre-push"
CI="$ROOT/.github/workflows/ci.yml"
MANIFEST="$ROOT/qa-manifest.yaml"

FAIL=0

# Prints every suite name declared under qa-manifest.yaml's suites: section, one per line.
manifest_suite_names() {
  [ -f "$MANIFEST" ] || return 0
  awk '
    /^suites:[ \t]*$/ { insuites = 1; next }
    insuites && /^  [A-Za-z0-9_-]+:[ \t]*\{/ {
      name = $0
      sub(/^  /, "", name); sub(/:.*/, "", name)
      print name
    }
    insuites && $0 !~ /^  / && $0 !~ /^$/ { insuites = 0 }
  ' "$MANIFEST"
}

# $1 = suite name, $2 = field (tier|automated); prints the field's value (or, for
# automated, "true" when the field is absent — matching qa-runner.sh's own default).
manifest_field() {
  [ -f "$MANIFEST" ] || return 0
  line=$(grep -E "^  ${1}:[ \t]*\{" "$MANIFEST" | head -n 1)
  [ -z "$line" ] && return 0
  case "$2" in
    tier)
      printf '%s' "$line" | sed -n 's/.*tier:[ \t]*\([a-zA-Z]*\).*/\1/p'
      ;;
    automated)
      if printf '%s' "$line" | grep -qE 'automated:[ \t]*false'; then
        printf 'false'
      else
        printf 'true'
      fi
      ;;
    *)
      echo "manifest_field: unknown field '$2'" >&2
      return 1
      ;;
  esac
}

referenced() {
  name="$1"
  grep -qE "(^|[^A-Za-z0-9_-])${name}(\\.sh)?([^A-Za-z0-9_-]|\$)" "$HOOK" 2>/dev/null && return 0
  grep -qE "scripts/suites/${name}\\.sh" "$CI" 2>/dev/null && return 0
  manifest_suite_names | grep -qx "$name" && return 0
  return 1
}

for script in "$SUITES_DIR"/*.sh; do
  name=$(basename "$script" .sh)
  if ! referenced "$name"; then
    echo "ORPHAN: scripts/suites/${name}.sh is not referenced by .git_hooks/pre-push, .github/workflows/ci.yml, or qa-manifest.yaml"
    FAIL=1
  fi
done

if [ -f "$HOOK" ] && ! grep -q -- '--tier' "$HOOK"; then
  SUITE_LINE=$(grep -oE 'qa-runner\.sh [a-zA-Z0-9_ -]+' "$HOOK" | tail -n 1 | sed 's/^qa-runner\.sh //')
  for name in $SUITE_LINE; do
    [ -f "$SUITES_DIR/$name.sh" ] || {
      echo "DANGLING: hook invokes suite '$name' with no scripts/suites/$name.sh"
      FAIL=1
    }
  done
fi

if [ -f "$MANIFEST" ]; then
  for name in $(manifest_suite_names); do
    automated=$(manifest_field "$name" automated)
    if [ ! -f "$SUITES_DIR/$name.sh" ] && [ "$automated" != "false" ]; then
      echo "DANGLING: qa-manifest.yaml lists suite '$name' with no scripts/suites/$name.sh (and no automated: false)"
      FAIL=1
    fi
    tier=$(manifest_field "$name" tier)
    case "$tier" in
      commit | push | pr | main | release) ;;
      *)
        echo "BAD TIER: qa-manifest.yaml suite '$name' has invalid tier '$tier' (must be commit|push|pr|main|release)"
        FAIL=1
        ;;
    esac
  done
fi

if [ "$FAIL" -ne 0 ]; then
  echo "Suite parity check failed." >&2
  exit 1
fi
echo "Suite parity OK: all scripts/suites/*.sh referenced, no dangling suite names, manifest consistent."

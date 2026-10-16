"""AD-4 static allowlist check (rule 1): every *resolved identifier* -- import lines, and
fully-qualified references that need no import -- must root in an allowed package.

Syntactic/AST-adjacent name lookup only, over the scene's source text as a plain Python
string -- never a real compiler/typer pass (AD-2: the typer is exactly where Scala 3 macro
expansion happens, so a "resolve every identifier" check that invoked the real compiler
would itself execute macros in the agent domain). This module never imports or shells out
to anything in the `menger` repository.

Unqualified single-word identifiers (`Sphere(...)`, `TesseractSponge(...)`) are NOT
syntactically resolvable to a root package without a symbol table -- the vast majority of
real DSL usage is exactly this, following a scene's `import menger.dsl._`. This check does
not attempt to resolve those; it catches the realistic attack/error surface instead: an
actually-added disallowed `import` line, and a literal fully-qualified reference that
bypasses the wildcard import (e.g. `java.io.File(...)` used inline, never imported).
"""

from __future__ import annotations

import re

from gauntlet._scala_text import line_of, strip_comments_and_strings
from gauntlet.types import Finding

_STAGE = "allowlist"

# AD-4 rule 1's exact enumerated roots. This is a *literal* prefix match against the
# dotted import/reference path, not a "menger.*"/"scala.*" wildcard: `menger`'s other
# submodules (e.g. `menger.SceneCompiler`, `menger.SceneLoader`) are exactly the
# compile/execution surface AD-2 forbids reaching from the agent domain, so bare "menger"
# or "scala" cannot be treated as an allowed root, only these specific subpaths.
_ALLOWED_ROOTS = (
    "menger.dsl",
    "menger.common",
    "scala.math",
    "scala.language.implicitConversions",
)

# AD-4's one narrow exception (added after this check's first real-fixture run rejected
# every 4D scene, including the MVP flagship): every 4D scene needs `import
# menger.Projection4DSpec`, which lives outside `menger.dsl`. Allowing this one class --
# never the bare `menger` root -- keeps AD-2's execution surface (Main-like objects, etc.)
# closed while admitting the one legitimate type the DSL corpus actually needs from there.
_ALLOWED_EXACT = ("menger.Projection4DSpec",)

# One import target per line: `import a.b.c`, `import a.b._`, `import a.b.{X, Y}`. The
# DSL corpus's imports are always one-per-line (confirmed against poc-run/*.scala), so a
# single-line regex is proportionate -- no need for a general statement tokenizer.
#
# Leading/trailing whitespace is deliberately `[ \t]*`, NOT `\s*`: `\s` also matches `\n`,
# so a greedy `\s*` right after MULTILINE's `^` would swallow a preceding blank line (a
# common style, e.g. `package poc\n\nimport ...`) before starting its match one line later
# than the import actually is -- every downstream `line_of(scene_text, match.start())` call
# would then be off by one for exactly the common "blank line before an import" style.
#
# Trailing `;?` (review round 1): a semicolon-terminated import (`import evil._;`) previously
# failed this anchored-at-`$` regex outright, silently skipping the check entirely -- matched
# against comment-stripped text (see `check_allowlist`), a trailing `//comment` is already
# blanked to spaces by the time this runs, so only the semicolon needed handling here.
#
# Review 2026-10-14 (menger-scene-agent#36): the old anchored regex knew only the Scala 2
# wildcard `._`, so Scala 3's `import scala.sys.process.*` matched nothing and passed. Now every
# `import` statement is captured whole and each comma-separated clause is reduced to its dotted
# base path (dropping `._`, `.*`, `.given`, `.{...}` and ` as X`); a clause that does not reduce
# to a plain dotted path is itself a finding -- the check fails closed, not open.
_IMPORT_RE = re.compile(r"^[ \t]*import\s+([^;\n]+?)[ \t]*;?[ \t]*$", re.MULTILINE)
_PACKAGE_RE = re.compile(r"^[ \t]*package\s+[\w.]+[ \t]*;?[ \t]*$", re.MULTILINE)
_IMPORT_SELECTOR_RE =re.compile(r"(?:\.\{[^}]*\}|\._|\.\*|\.given\b.*|\s+as\s+\w+)$")
_DOTTED_PATH_RE = re.compile(r"^(?:_root_\.)?[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*$")

# A qualified reference used inline without an import, rooted in a package namespace (not a
# local value's member access like `scene.objects`): `java.io.File`, `scala.sys.process.Process`,
# `_root_.java.lang.Runtime`. The old pattern required a capitalised last segment and missed
# `scala.sys.process.stringToProcess` (#36).
_PACKAGE_NAMESPACES = ("java", "javax", "jdk", "sun", "scala", "dotty", "menger", "com", "org",
                       "io", "net", "examples")
_QUALIFIED_REF_RE = re.compile(
    r"(?<![\w.])((?:_root_\.)?(?:" + "|".join(_PACKAGE_NAMESPACES) + r")(?:\.[A-Za-z_]\w*)+)"
)

# Names every Scala file sees without an import (`java.lang._`, `scala._`, `Predef`) that reach
# the process, the environment, threads or reflection. Unqualified, they are invisible to the
# two checks above. This list is advisory -- the sandbox is the barrier, not this check -- but
# it turns the obvious attempts into a repairable finding before any compile (#36).
_IMPLICIT_DANGEROUS_RE = re.compile(
    r"(?<![\w.])(System|Runtime|ProcessBuilder|Process|ProcessHandle|Thread|ClassLoader|Class|"
    r"SecurityManager|sys|classOf|getClass)\b"
)


def _root_of(raw_clause: str) -> str | None:
    """The dotted base path of one import clause, or None if it is not one."""
    stripped = _IMPORT_SELECTOR_RE.sub("", raw_clause.strip())
    if not _DOTTED_PATH_RE.match(stripped):
        return None
    return stripped.removeprefix("_root_.")


def _import_clauses(statement: str) -> list[str]:
    """`a.b.{X, Y}, c.d.*` -> [`a.b.{X, Y}`, `c.d.*`]: commas inside braces don't split."""
    clauses, depth, current = [], 0, ""
    for char in statement:
        depth += {"{": 1, "}": -1}.get(char, 0)
        if char == "," and depth == 0:
            clauses.append(current)
            current = ""
        else:
            current += char
    return clauses + [current]


def _is_allowed(path: str) -> bool:
    # `math` (e.g. `math.Pi`, confirmed in reference/dsl-corpus.json's RotatingSilverSponge,
    # SierpinskiHDRRotation, TrefoilKnot) is Scala's own implicit top-level alias for
    # `scala.math` -- every Scala file has an implicit `import scala._` (the "language's own
    # literal/operator core" the frozen allowlist text names), through which the `scala`
    # package's `math` member is reachable unqualified. It needs no `import scala.math._`
    # line at all, so a reference rooted in bare `math` is equivalent to one rooted in the
    # already-allowed `scala.math`, not a bypass of it.
    if path == "math" or path.startswith("math."):
        path = "scala." + path
    if path in _ALLOWED_EXACT:
        return True
    return any(path == root or path.startswith(root + ".") for root in _ALLOWED_ROOTS)


def check_allowlist(scene_text: str) -> list[Finding]:
    """AD-4 rule 1: flags any import whose target doesn't root in an allowed package, and
    any literal fully-qualified reference outside an import line that does the same.

    Runs against comment/string-stripped text (review round 1): a disallowed import or
    reference hidden inside a `//`/`/* */` comment or a string literal must not evade this
    check, and a string that merely *contains* lint-shaped text must not false-positive.
    Line numbers stay correct against the original `scene_text` since stripping only blanks
    content, never removes a newline (`strip_comments_and_strings`)."""
    text = strip_comments_and_strings(scene_text)
    findings: list[Finding] = []
    import_spans: list[tuple[int, int]] = []

    for match in _IMPORT_RE.finditer(text):
        import_spans.append((match.start(), match.end()))
        for raw_target in (c.strip() for c in _import_clauses(match.group(1))):
            root_path = _root_of(raw_target)
            if root_path is not None and _is_allowed(root_path):
                continue
            findings.append(
                Finding(
                    stage=_STAGE,
                    message=(
                        f"Import '{raw_target}' does not root in an allowed package "
                        f"(menger.dsl, menger.common, scala.math, "
                        f"scala.language.implicitConversions, menger.Projection4DSpec)"
                    ),
                    identifier=raw_target,
                    line=line_of(scene_text, match.start()),
                )
            )

    # A `package` clause names where the scene lives, it reaches nothing.
    import_spans += [(m.start(), m.end()) for m in _PACKAGE_RE.finditer(text)]

    def outside_imports(start: int) -> bool:
        return not any(span_start <= start < span_end for span_start, span_end in import_spans)

    seen_outside_imports: set[str] = set()
    for match in _QUALIFIED_REF_RE.finditer(text):
        candidate = match.group(1).removeprefix("_root_.")
        if (not outside_imports(match.start()) or candidate in seen_outside_imports
                or _is_allowed(candidate)):
            continue
        seen_outside_imports.add(candidate)
        findings.append(
            Finding(
                stage=_STAGE,
                message=(
                    f"Fully-qualified reference '{candidate}' does not root in an allowed "
                    f"package, and bypasses any `import menger.dsl._` wildcard"
                ),
                identifier=candidate,
                line=line_of(scene_text, match.start()),
            )
        )

    for match in _IMPLICIT_DANGEROUS_RE.finditer(text):
        name = match.group(1)
        if not outside_imports(match.start()) or name in seen_outside_imports:
            continue
        seen_outside_imports.add(name)
        findings.append(
            Finding(
                stage=_STAGE,
                message=(
                    f"'{name}' reaches the process, environment, threads or reflection; a "
                    "scene uses only the DSL"
                ),
                identifier=name,
                line=line_of(scene_text, match.start()),
            )
        )

    return findings

"""Stage 3 clean-code check: "the file *is* the deliverable" (validation-gauntlet.md).
Three independent checks over the scene's source text:

  1. File length within a sane bound -- the 32-scene reference corpus (dsl-corpus.json)
     runs 26-248 lines (one deliberate multi-object outlier, `ParametricScenes.scala`) and
     the two poc-run fixtures sit at 53; a generated or hand-edited scene that balloons past
     the corpus's own outlier is itself a code-quality signal, not something a lint/
     allowlist/resource-bound check would catch.
  2. No leftover placeholder text (`TODO`, `FIXME`, `XXX`) -- a model or human leaving one
     behind means the file isn't actually finished.
  3. An independent re-check that the top-level `object` declaration is the first one and
     appears within the first 60 lines. `adapters/model.py`'s `extract_scene_text` already
     enforces this at generation time (`_OBJECT_DECLARATION`/`_FIRST_OBJECT_LINE_LIMIT`), but
     a hand-edited or CAP-2-revised file bypasses that path entirely -- this module
     reimplements the same regex/threshold locally rather than importing `adapters.model`,
     since `gauntlet/` must have no dependency on `adapters/`'s model-specific code (Code Map).
"""

from __future__ import annotations

import re

from gauntlet._scala_text import line_of, strip_comments_and_strings, strip_strings_only
from gauntlet.types import Finding

_STAGE = "clean_code"

# The corpus's own scenes run 26-248 lines (dsl-corpus.json's ParametricScenes.scala is the
# outlier at 248; poc-run/*.scala sit at 53); 150 catches a runaway file while allowing the
# corpus's own legitimate multi-object outlier as a documented exception -- see review round
# 1's note: the original "150, generous headroom above 34-92" rationale was itself based on
# a stale corpus figure and would have flagged a real corpus example.
_LINE_COUNT_CEILING = 150

# Matches TODO/FIXME/XXX as a whole word, case-insensitively (review round 1: a lowercase
# "todo" is exactly the same leftover-placeholder signal as "TODO" and must not evade this
# by casing alone) -- but still requires a real word boundary, so a legitimate identifier
# that merely contains one of these as a substring (e.g. "TodoList") is never a false
# positive purely from the substring match; case-insensitivity doesn't change that, since
# the boundary requirement is independent of case folding.
_PLACEHOLDER_RE = re.compile(r"\b(TODO|FIXME|XXX)\b", re.IGNORECASE)

# Mirrors adapters/model.py's `_OBJECT_DECLARATION`/`_FIRST_OBJECT_LINE_LIMIT` exactly
# (SceneLoader.detectObjectName's convention) -- reimplemented locally, not imported, per
# this story's Code Map/Boundaries: gauntlet/ has no dependency on adapters/. Accepts
# `case object`/`package object` too (review round 1) -- both are real Scala object-
# declaration forms `SceneLoader.detectObjectName`'s own line-based heuristic would match
# just as readily as a bare `object`.
# `[ \t]*`, not `\s*`, for the same reason `allowlist.py`'s `_IMPORT_RE` uses it (review
# round 1 bug, caught by this module's own test suite once comment-stripping turned
# padding lines into pure whitespace): `\s` matches `\n`, so a greedy `\s*` right after
# `^` could span many blank/whitespace-only lines from the very start of the text and
# match "object" wherever it first appears, regardless of which actual line it's on --
# defeating the whole point of this check (whether `object` is within the first N lines).
_OBJECT_DECLARATION = re.compile(
    r"^[ \t]*(?:private[ \t]+|case[ \t]+|package[ \t]+)*object[ \t]+\w", re.MULTILINE
)
_FIRST_OBJECT_LINE_LIMIT = 60


def _check_line_count(scene_text: str) -> list[Finding]:
    line_count = len(scene_text.splitlines())
    if line_count <= _LINE_COUNT_CEILING:
        return []
    return [
        Finding(
            stage=_STAGE,
            message=(
                f"Scene file is {line_count} lines, above the {_LINE_COUNT_CEILING}-line "
                f"sane bound (the reference corpus's own scenes run 26-248 lines)"
            ),
        )
    ]


def _check_placeholders(scene_text: str) -> list[Finding]:
    # Strings only, not comments -- the signal legitimately lives in a `// TODO` comment,
    # so stripping comments would defeat this check's purpose; a string that merely
    # contains one of these words (a filename, a description) should still not flag.
    text = strip_strings_only(scene_text)
    findings = []
    for match in _PLACEHOLDER_RE.finditer(text):
        token = match.group(1)
        findings.append(
            Finding(
                stage=_STAGE,
                message=f"Leftover placeholder text '{token}' found -- the file isn't finished",
                identifier=token.upper(),
                line=line_of(scene_text, match.start()),
            )
        )
    return findings


def _check_object_declaration_position(scene_text: str) -> list[Finding]:
    text = strip_comments_and_strings(scene_text)
    match = _OBJECT_DECLARATION.search(text)
    if match is None:
        return [
            Finding(
                stage=_STAGE,
                message="No top-level 'object' declaration found -- not a valid scene file",
            )
        ]
    line_number = text.count("\n", 0, match.start())  # 0-based, matches model.py's check
    if line_number >= _FIRST_OBJECT_LINE_LIMIT:
        return [
            Finding(
                stage=_STAGE,
                message=(
                    f"Top-level 'object' declaration is past line {_FIRST_OBJECT_LINE_LIMIT} "
                    f"-- SceneLoader.detectObjectName would miss it"
                ),
                line=line_of(scene_text, match.start()),
            )
        ]
    return []


def check_clean_code(scene_text: str) -> list[Finding]:
    """Stage 3: file-length bound, placeholder-text scan, and an independent re-check that
    the top-level `object` declaration is the first one, within the first 60 lines."""
    findings: list[Finding] = []
    findings.extend(_check_line_count(scene_text))
    findings.extend(_check_placeholders(scene_text))
    findings.extend(_check_object_declaration_position(scene_text))
    return findings

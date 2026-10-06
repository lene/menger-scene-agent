"""Shared parsing utilities for gauntlet/'s static-analysis checks. Pure string processing
over Scala source text -- never a real parser/compiler (AD-2). Proportionate to the DSL's
templated, named-parameter-constructor style, not a general Scala tokenizer.

Factored out of lint.py during review (story 4, round 1): resource_bounds.py needs the same
constructor-scoped extraction lint.py already had, after a real corpus scene
(`CausticsCanonical.scala`, `TwoSpheres.scala`) showed a bare `field = literal` regex with no
constructor scoping false-positives on `Caustics.iterations`, a same-named but unrelated field
to `LSystem.iterations`.
"""

from __future__ import annotations

import re
from typing import Optional

# Blanks out (replaces with same-length whitespace, preserving line/column offsets for every
# other module's line-number reporting) block comments, line comments, and string-literal
# contents -- so a check can't be evaded by hiding a violation inside a comment, and can't
# false-positive on a string that merely contains lint-shaped text (a texture path with a
# paren, a "TODO" inside a description string, etc.). Order matters: triple-quoted strings
# first (so an embedded `"` or `//` inside one doesn't get mis-terminated by the simpler
# single-line-string/comment patterns), then block comments, then line comments, then
# ordinary strings.
_TRIPLE_QUOTED_STRING = re.compile(r'"""(?:(?!""").)*"""', re.DOTALL)
_BLOCK_COMMENT = re.compile(r"/\*(?:(?!\*/).)*\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
# Ordinary double-quoted string, `\"`-escape aware, does not span a newline (a real Scala
# single-quoted string can't either -- an unescaped newline inside one is a syntax error).
_STRING_LITERAL = re.compile(r'"(?:\\.|[^"\\\n])*"')


def _blank(match: re.Match) -> str:
    """Replaces a match with same-length whitespace, keeping embedded newlines so every
    other line-number computation in this package (`text.count("\\n", 0, offset)`) stays
    correct against the blanked text."""
    return "".join(ch if ch == "\n" else " " for ch in match.group(0))


def strip_comments_and_strings(text: str) -> str:
    """Blanks block/line comments and string-literal contents, preserving length and line
    breaks. Findings' `line` numbers computed against the *original* text remain correct
    when computed against this function's output too, since only content, never newlines,
    is removed."""
    text = _TRIPLE_QUOTED_STRING.sub(_blank, text)
    text = _BLOCK_COMMENT.sub(_blank, text)
    text = _LINE_COMMENT.sub(_blank, text)
    text = _STRING_LITERAL.sub(_blank, text)
    return text


def strip_strings_only(text: str) -> str:
    """Blanks only string-literal contents, leaving comments intact -- for a check whose
    signal legitimately lives inside a comment (e.g. a leftover `// TODO` note), where
    stripping comments would defeat the check's own purpose, but a string that merely
    contains lint-shaped text (a texture filename, a description) should still not
    false-positive."""
    text = _TRIPLE_QUOTED_STRING.sub(_blank, text)
    text = _STRING_LITERAL.sub(_blank, text)
    return text


def scan_balanced(text: str, open_paren_index: int) -> int:
    """From the index of an already-located `(`, returns the index of its matching `)` via
    a simple bracket-depth scan -- proportionate to the DSL's constructor-call style, not a
    general expression parser. Callers should scan comment/string-stripped text (see
    `strip_comments_and_strings`) so a stray `(`/`)` inside a string doesn't desync this."""
    depth = 1
    i = open_paren_index + 1
    n = len(text)
    while i < n and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    return i - 1


def find_call_bodies(text: str, name: str) -> list[tuple[int, str]]:
    """Finds each `name(...)` call, returning `(start_offset, inner_text)` for each
    balanced-paren body."""
    results = []
    for match in re.finditer(rf"\b{re.escape(name)}\s*\(", text):
        open_idx = match.end() - 1
        close_idx = scan_balanced(text, open_idx)
        results.append((match.start(), text[open_idx + 1 : close_idx]))
    return results


def split_top_level(s: str) -> list[str]:
    """Splits a comma-separated argument list at depth-0 commas only, so a nested call's
    own commas (`Vec3(0f, 2f, 6f)`, `Some(Color(0f, 0f, 0f))`) are not split."""
    parts = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def named_args(inner: str) -> dict[str, str]:
    args: dict[str, str] = {}
    for part in split_top_level(inner):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        args[key.strip()] = value.strip()
    return args


# Accepts Scala's `_`-digit-separator syntax (`100_000`) and `f`/`F`/`d`/`D` suffixes --
# both real, common Scala numeric-literal forms this module must not silently fail to
# match (a match failure means the check is skipped, not that the value is in-bounds).
_NUMERIC_LITERAL_RE = re.compile(r"^-?\d[\d_]*(?:\.[\d_]+)?[fFdD]?$")


def as_float(value: str) -> Optional[float]:
    value = value.strip()
    if _NUMERIC_LITERAL_RE.match(value):
        return float(value.rstrip("fFdD").replace("_", ""))
    return None


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1

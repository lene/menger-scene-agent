"""Ticket sink adapter: writes a ticket draft file for a missing capability, mirroring
`adapters/scene_store.py`'s own shape.

AD-3/AD-1: writing a draft is this adapter's only effect -- no network call, no credential,
no shelling out (Boundaries & Constraints). This mirrors `adapters/artifacts.py`'s already-
established "no I/O beyond disk" precedent for a non-model adapter; the ticket-filing token
that turns a draft into a real issue lives with the (out-of-scope) external filing step, not
here.

AD-17: ticket drafts are written to an injected `drafts_dir` (AD-10: paths are injected,
never discovered or hardcoded), one file per draft, named *deterministically* from
`missing_capability` alone -- a repeat request for the same missing capability overwrites the
prior draft rather than creating a second one. This is the one place this module
deliberately diverges from `scene_store.py`'s durability convention: the write is
temp-file-plus-`fsync`-plus-`os.replace()`, not the exclusive `os.link()` `scene_store.py`
uses for ordinal files -- overwrite-on-repeat is this adapter's whole point, not a race to
prevent.

This story builds only the sink adapter itself. Deciding *when* a request needs a missing
capability (wiring this into `core/generation.py`'s `generate()`/`revise()`) and the external
filing step that turns a draft into a real tracker issue are both out of scope -- see
`_bmad-output/specs/spec-ai-scene-agent/stories/9-ticket-escalation.md`'s Boundaries &
Constraints and Design Notes.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from core.types import TicketError, TicketResult

# A draft's filename stem is derived from `missing_capability` by replacing every character
# outside this safe set with '-' -- the same sanitize-then-collapse shape
# `scene_store._new_session_id` uses, chosen so no character from an arbitrary, possibly
# path-hostile `missing_capability` string (e.g. "../../etc/passwd", "a/b") can ever produce
# a path separator or a ".." segment in the resulting filename.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")

# Bounds the human-readable portion of the filename so an unusually long
# `missing_capability` string doesn't run into filesystem filename-length limits.
_MAX_STEM_LENGTH = 60

# The hex digest of the full, un-truncated (but stripped -- see below) `missing_capability`
# text is appended to the sanitized stem. This is what actually makes the filename
# deterministic-and-collision-free per the I/O & Edge-Case Matrix ("two distinct
# capabilities... no collision"): the sanitize step alone can map two different strings to
# the same stem (e.g. "a/b" and "a b" both collapse to "a-b"), but their digests differ, so
# the final filenames never collide. Being a plain hash of the input (no salt, no timestamp),
# it is also exactly what "deterministic ... so a repeat request overwrites rather than
# duplicates" (AD-17) requires: the same `missing_capability` text always hashes to the same
# digest.
_DIGEST_LENGTH = 10


def _draft_filename(missing_capability: str) -> str:
    # Review round: the digest MUST be computed from the same normalized (stripped) text the
    # stem uses, not the raw input -- otherwise "torus knot geometry" and
    # " torus knot geometry " share a stem but hash to different digests, silently breaking
    # AD-17's overwrite guarantee whenever a caller's text differs only by surrounding
    # whitespace.
    normalized = missing_capability.strip()
    stem = _UNSAFE_CHARS.sub("-", normalized).strip("-")
    stem = stem[:_MAX_STEM_LENGTH].strip("-") or "capability"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    return f"{stem}-{digest}.md"


def _title(missing_capability: str) -> str:
    """The capability as a request (session 3, F79: titles were the decline sentence): the
    first clause, without the negation around it -- "there is no glow or halo effect in this
    DSL" -> "Support glow or halo effect"."""
    text = re.split(r",? and no |; | -- ", missing_capability.strip())[0]
    text = re.sub(r"^(?:there is |there are |the DSL has |this DSL has )?no\s+", "", text, flags=re.I)
    text = re.sub(r"\s+(?:exists?|is possible|in (?:this|the) DSL)\b.*$", "", text, flags=re.I)
    return f"Support {text.strip()}"


# Words that say nothing about which capability a title names (de-dupe, F79).
_GENERIC_WORDS = frozenset({
    "support", "the", "and", "around", "with", "for", "from", "that", "this", "object",
    "objects", "effect", "effects", "scene", "any", "into", "onto",
})


def _content_words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 1} - _GENERIC_WORDS


def _same_capability(a: str, b: str) -> bool:
    """Two titles name one capability when their content words mostly overlap: session 3's
    three glow declines ({glow, halo}, {glow, halo}, {glow, halo}) but not "camera shake" and
    "orbiting camera path"."""
    words_a, words_b = _content_words(a), _content_words(b)
    if not words_a or not words_b:
        return a == b
    shared = len(words_a & words_b)
    smaller = min(len(words_a), len(words_b))
    return shared >= min(2, smaller) and shared / smaller >= 2 / 3


def _existing_draft(drafts_dir: Path, title: str) -> Optional[Path]:
    for draft in sorted(drafts_dir.glob("*.md")):
        with suppress(OSError, UnicodeError):
            first_line = draft.read_text(encoding="utf-8").splitlines()[0]
            if _same_capability(first_line.removeprefix("# "), title):
                return draft
    return None


def _use_cases(draft: Optional[Path]) -> list[str]:
    if draft is None:
        return []
    with suppress(OSError, UnicodeError):
        text = draft.read_text(encoding="utf-8")
        section = text.split("## Use cases\n", 1)[1].split("\n## ", 1)[0]
        return [line for line in section.splitlines() if line.startswith("- ")]
    return []


def _draft_text(
    missing_capability: str, use_cases: list[str], nearest: str = ""
) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    text = (
        f"# {_title(missing_capability)}\n\n"
        f"**Declined as:** {missing_capability.strip()}\n\n"
        f"**Last requested:** {timestamp}\n\n"
        "## Use cases\n\n"
        + "".join(f"{case}\n" for case in use_cases)
    )
    if nearest.strip():
        text += f"\n## Nearest possible today\n\n{nearest.strip()}\n"
    text += (
        "\n## Done when\n\n"
        "A scene can express the use cases above as asked, not only through the nearest "
        "workaround.\n"
    )
    return text


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_draft(
    missing_capability: str,
    prompt: str,
    drafts_dir: Union[str, Path],
    nearest: str = "",
    session_id: str = "",
) -> TicketResult:
    """Writes a ticket draft naming `missing_capability` and the `prompt` that surfaced it,
    under `drafts_dir` (always an injected parameter -- AD-10). Returns the written file's
    path as `str` on success, or a `TicketError` -- never raises for an expected outcome
    (Boundaries & Constraints).

    The draft filename is deterministic from `missing_capability` alone (AD-17): a second
    call with the same `missing_capability` text rewrites the first draft's file rather
    than creating a second one, regardless of `prompt`. Session 3 (F79): so does a differently
    worded decline of the same capability (`_same_capability`), and the rewrite keeps every
    earlier use case, adding this `prompt` (with `session_id`, when given).
    """
    if not missing_capability.strip():
        return TicketError(
            kind="invalid_capability_text",
            message="missing_capability must not be empty or whitespace-only",
        )

    resolved_dir = Path(drafts_dir)
    if not resolved_dir.is_dir():
        return TicketError(
            kind="drafts_dir_unavailable",
            message=f"drafts_dir '{resolved_dir}' does not exist or is not a directory",
        )

    # Review round: `mkstemp` itself can raise (permission-denied drafts_dir, drafts_dir
    # removed between the is_dir() check above and here, disk full) -- it must live inside
    # the try/except below like every other I/O step here, so such a failure returns a
    # TicketError instead of escaping write_draft()'s "never raises for an expected outcome"
    # contract.
    tmp_name = None
    try:
        # F79: one draft per capability -- a differently worded decline of the same one adds
        # its use case to the existing draft instead of starting a second.
        existing = _existing_draft(resolved_dir, _title(missing_capability))
        target = existing or resolved_dir / _draft_filename(missing_capability)
        case = " ".join(prompt.split()) + (f" (session {session_id})" if session_id else "")
        use_cases = [c for c in _use_cases(existing) if c != f"- {case}"] + [f"- {case}"]
        text = _draft_text(missing_capability, use_cases, nearest)

        fd, tmp_name = tempfile.mkstemp(
            dir=resolved_dir, prefix=f".{target.name}.", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # AD-17's overwrite-on-repeat semantics: os.replace(), not scene_store.py's exclusive
        # os.link() -- a pre-existing draft for this exact missing_capability is meant to be
        # replaced, not raced against.
        os.replace(tmp_name, target)
        _fsync_dir(resolved_dir)
    except (OSError, UnicodeError) as e:
        # Review round: UnicodeError (e.g. an unpaired UTF-16 surrogate in
        # missing_capability/prompt defeating the digest's/write's utf-8 encode) is caught
        # alongside OSError for the same "never raises" reason. A distinct kind from the
        # upfront drafts_dir_unavailable check (mismatched/missing directory) -- this branch
        # means the directory existed but the write itself failed (permission, disk full,
        # TOCTOU removal, bad text), which a caller needs to be able to tell apart.
        return TicketError(
            kind="write_failed",
            message=f"Could not write draft to '{resolved_dir}': {e}",
            cause=e,
        )
    finally:
        if tmp_name is not None:
            with suppress(OSError):
                os.unlink(tmp_name)

    return str(target)

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
from typing import Union

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


def _draft_text(missing_capability: str, prompt: str) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    return (
        f"# Missing capability: {missing_capability}\n\n"
        f"**Requested:** {timestamp}\n\n"
        "## Triggering prompt\n\n"
        f"{prompt}\n"
    )


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
) -> TicketResult:
    """Writes a ticket draft naming `missing_capability` and the `prompt` that surfaced it,
    under `drafts_dir` (always an injected parameter -- AD-10). Returns the written file's
    path as `str` on success, or a `TicketError` -- never raises for an expected outcome
    (Boundaries & Constraints).

    The draft filename is deterministic from `missing_capability` alone (AD-17): a second
    call with the same `missing_capability` text overwrites the first draft's file rather
    than creating a second one, regardless of `prompt`.
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
        target = resolved_dir / _draft_filename(missing_capability)
        text = _draft_text(missing_capability, prompt)

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

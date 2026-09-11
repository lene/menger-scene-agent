"""Session/scene-store adapter: persists a session's turns to disk as an ordinal-numbered
scene history plus an append-only `history.jsonl`.

AD-15: session creation is exclusive -- a session ID is never a human-chosen slug alone
("a slug alone is not an identity"), and the directory is made with an exclusive-create
primitive (`Path.mkdir(exist_ok=False)`), never a check-then-create race.

AD-12/AD-13: a turn is one user request. Only an *accepted* attempt consumes the next
zero-padded ordinal (`001.scala`, `002.scala`, ...); a *rejected* attempt writes no scene
file and consumes no ordinal, though it still gets a `history.jsonl` entry. "The current
scene" always means the highest ordinal file that actually exists on disk -- this module
never caches that value in memory, it re-scans the session directory every time.

AD-8: every accepted attempt writes a brand-new file. Nothing here ever opens an existing
ordinal file for writing, or deletes one -- "edit" is a derivation performed by
`core/generation.py`'s `revise()` before this module ever sees the result, not an in-place
mutation performed by this module.

AD-7: `history.jsonl` is append-only JSON-lines, one JSON object per line, never rewritten
or reordered. Every entry records at least the ordinal (when accepted; `None` when
rejected), the turn's prompt, and the outcome.

AD-14: every write that crosses into the session directory is atomic AND exclusive. A new
ordinal file is written to a temp name in the same directory, flushed, `fsync`'d, then
`os.link()`'d into place under its final name and the temp name unlinked -- a reader can
never observe a partial scene file, and (review round: an `os.replace()`-based version was
crash-safe but not race-safe) `os.link()` raises `FileExistsError` rather than silently
overwriting when two concurrent callers compute the same next ordinal, so `accept()` can
detect the collision and retry instead of one writer's content silently vanishing. The
session directory itself is also `fsync`'d after a new directory entry is created, so the
entry's durability doesn't depend solely on the file's own content being flushed. The
`history.jsonl` append uses a single buffered `write()` call of one complete JSON line plus
newline instead of temp-then-rename: temp-then-rename doesn't compose with "append" (a
rename would replace the whole file, not add a line), and on POSIX a single `write()` of a
chunk this small (well under `PIPE_BUF`, typically >= 4096 bytes) is atomic with respect to
other writers/readers of the same file -- no reader can observe a torn line. (This module
assumes a single writer *per turn* -- two callers racing to accept the *same* turn is not a
scenario AD-12 describes -- but never assumes a single writer *per process lifetime*, hence
the exclusive-link retry above rather than a documented single-writer restriction.)

This module never opens the filesystem for anything the caller didn't ask for -- no
manifest, no corpus, no model adapter, no gauntlet. Composing those with this store is a
future pipeline-wiring story's job (Design Notes).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

# Zero-padded to 3 digits per the Consistency Conventions table ("scene files zero-padded
# ordinal (`001.scala`)"). The regex accepts 3-or-more digits so a session that somehow grows
# past 999 turns still round-trips its own ordinals correctly instead of silently truncating.
_ORDINAL_FILENAME = re.compile(r"^(\d{3,})\.scala$")
_ORDINAL_WIDTH = 3

_HISTORY_FILENAME = "history.jsonl"


class SceneStoreError(RuntimeError):
    """Raised for a real filesystem/session-identity failure -- an already-existing session
    directory (AD-15), a missing ordinal on restore, or an underlying OSError this module
    cannot recover from. Mirrors `adapters/artifacts.py`'s `ArtifactError`: infrastructure
    failures here are not the kind of typed business-logic outcome `GenerationResult`/
    `ReadbackResult` exist for (those model a real derivation choice; a session directory
    that already exists is an environment/caller error), so this module lets them raise
    rather than threading a `Union[..., SceneStoreError]` return type through every method."""


def _new_session_id(slug: str) -> str:
    """A session ID is not a human-friendly slug alone (AD-15: "A slug alone is not an
    identity") -- combine a sanitized slug with a UTC timestamp and a short UUID4 suffix so
    two sessions started from the same slug, even in the same second, cannot collide."""
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", slug).strip("-") or "session"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    suffix = uuid.uuid4().hex[:8]
    return f"{sanitized}-{timestamp}-{suffix}"


@dataclass(frozen=True)
class SceneStore:
    """A session's on-disk scene history: ordinal-numbered scene files plus an append-only
    `history.jsonl`, both living under `session_dir`. Two ways to get one: `create_session()`
    (a brand-new session, created exclusively) or direct construction, `SceneStore(session_dir
    =existing_path)`, to re-attach to a session an earlier process already created -- there is
    deliberately no separate "open existing session" factory, since a `SceneStore` carries no
    state beyond the path itself (every read re-scans the directory, per AD-13). Direct
    construction validates `session_dir` actually exists and is a directory (review round:
    it used to accept any path silently, so every method would fail with a raw filesystem
    error instead of a clear one)."""

    session_dir: Path

    def __post_init__(self) -> None:
        if not self.session_dir.is_dir():
            raise SceneStoreError(
                f"Session directory '{self.session_dir}' does not exist or is not a "
                "directory -- use SceneStore.create_session() for a new session, or pass "
                "the path of a session an earlier process already created"
            )

    @property
    def history_path(self) -> Path:
        return self.session_dir / _HISTORY_FILENAME

    @classmethod
    def create_session(
        cls,
        base_path: Union[str, Path],
        slug: str = "session",
        session_id: Optional[str] = None,
    ) -> "SceneStore":
        """Creates a new session directory exclusively under `base_path` (AD-10: the base
        path is always an injected parameter, never discovered or hardcoded) and returns a
        `SceneStore` bound to it.

        `session_id` is normally left unset -- production callers get a collision-resistant
        ID derived from `slug` (AD-15). It exists as an explicit parameter so a caller (e.g.
        a unit test proving AD-15's exclusivity) can force two `create_session()` calls to
        target the same directory and observe the second one fail, rather than relying on
        astronomically unlikely UUID collision. An explicit `session_id` is validated against
        path traversal (review round) -- it names a single path segment under `base_path`,
        never `..` or an absolute path that could escape it.
        """
        sid = session_id if session_id is not None else _new_session_id(slug)
        if session_id is not None and (
            not sid
            or Path(sid).is_absolute()
            or "/" in sid
            or "\\" in sid
            or sid in (".", "..")
        ):
            raise SceneStoreError(
                f"Invalid session_id {sid!r}: must be a single non-empty path segment, "
                "not an absolute path or one containing '..', '/', or '\\\\'"
            )
        session_dir = Path(base_path) / sid
        try:
            # AD-15: exclusive create, never check-then-create. `parents=True` lets a fresh
            # `base_path` be created as a side effect, but `exist_ok=False` still governs the
            # session directory itself -- a pre-existing leaf directory always raises.
            session_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            raise SceneStoreError(
                f"Session directory '{session_dir}' already exists -- refusing to reuse or "
                "overwrite an existing session (AD-15: session identity is exclusive)"
            ) from e
        except OSError as e:
            raise SceneStoreError(
                f"Could not create session directory '{session_dir}': {e}"
            ) from e
        return cls(session_dir=session_dir)

    def _existing_ordinals(self) -> List[int]:
        """Always re-reads the directory -- AD-13: "the current scene" is the last file that
        actually exists on disk, never an in-memory cached value."""
        try:
            entries = list(self.session_dir.iterdir())
        except OSError as e:
            raise SceneStoreError(f"Could not list session directory '{self.session_dir}': {e}") from e
        ordinals = []
        for entry in entries:
            if not entry.is_file():
                continue
            match = _ORDINAL_FILENAME.match(entry.name)
            if match is not None:
                ordinals.append(int(match.group(1)))
        return sorted(ordinals)

    def _last_ordinal(self) -> Optional[int]:
        ordinals = self._existing_ordinals()
        return ordinals[-1] if ordinals else None

    def _ordinal_path(self, ordinal: int) -> Path:
        return self.session_dir / f"{ordinal:0{_ORDINAL_WIDTH}d}.scala"

    def current_scene(self) -> Optional[str]:
        """The last written file's content, per AD-13 -- or `None` if no turn has been
        accepted yet in this session (a defined "no scene yet" result, never a crash or a
        fabricated file)."""
        ordinal = self._last_ordinal()
        if ordinal is None:
            return None
        return self.read_ordinal(ordinal)

    def read_ordinal(self, n: int) -> str:
        """Reads one prior ordinal's real content back from disk -- used both by
        `current_scene()` and to restore an earlier turn (Boundaries & Constraints: restoring
        is reading an old ordinal and handing it to the next turn as `prior_scene`, never a
        mutating "roll back" -- later ordinals are never touched)."""
        path = self._ordinal_path(n)
        if not path.is_file():
            raise SceneStoreError(
                f"Ordinal {n} does not exist in session '{self.session_dir}'"
            )
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            raise SceneStoreError(f"Could not read '{path}': {e}") from e

    def _fsync_dir(self) -> None:
        """Fsyncs the session directory's own fd -- content fsync (inside
        `_write_new_file_exclusive`/`_append_history`) guarantees a file's *bytes* survive a
        crash; it says nothing about the *directory entry* (a new link, a rename) being
        durable too. Review round: the original version fsynced file contents but never the
        directory, so a crash right after a successful write could still lose the directory
        entry pointing at it."""
        fd = os.open(self.session_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _write_new_file_exclusive(self, target: Path, text: str) -> None:
        """AD-14 + AD-8, made exclusive (review round): write to a temp name, flush, fsync,
        then `os.link()` the temp name to `target` -- unlike `os.replace()`, `os.link()`
        raises `FileExistsError` if `target` already exists rather than silently overwriting
        it. This is what actually enforces "never touches an existing ordinal file": the
        original `os.replace()`-based version was crash-safe (a reader never sees a *partial*
        file) but not race-safe (two concurrent writers computing the same next ordinal could
        both "succeed," the second silently discarding the first's content with no error).
        Raises `FileExistsError` on collision -- `accept()` catches it and retries against a
        freshly-recomputed ordinal."""
        fd, tmp_name = tempfile.mkstemp(
            dir=self.session_dir, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.link(tmp_name, target)
            self._fsync_dir()
        except FileExistsError:
            raise
        except OSError as e:
            raise SceneStoreError(f"Could not write '{target}': {e}") from e
        finally:
            with suppress(OSError):
                os.unlink(tmp_name)

    def _append_history(self, entry: dict) -> None:
        """AD-7: one JSON object per line, append-only, never rewritten or reordered. A
        single buffered `write()` of one complete line is the atomic primitive here (see
        module docstring) rather than temp-then-rename, which does not compose with
        "append"."""
        try:
            line = json.dumps(entry, sort_keys=True) + "\n"
            is_new_file = not self.history_path.exists()
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            if is_new_file:
                self._fsync_dir()
        except OSError as e:
            raise SceneStoreError(f"Could not append to '{self.history_path}': {e}") from e

    # Bounded, not infinite: real contention resolves in one or two retries: an unbounded
    # retry loop would turn a bug elsewhere (e.g. two callers permanently racing) into a
    # silent hang instead of a clear failure.
    _MAX_ACCEPT_RETRIES = 10

    def accept(self, scene_text: str, prompt: str, readback_summary: Optional[str] = None) -> int:
        """Records an accepted attempt: writes the next sequential zero-padded ordinal file
        (AD-12) atomically and exclusively (AD-14, AD-8 -- see `_write_new_file_exclusive`),
        appends an accepted `history.jsonl` entry (AD-7), and returns the new ordinal. Never
        touches any earlier ordinal file.

        `readback_summary` (story 20, `core/turn.py`'s `run_turn()`) is optional and, when
        given, folded into the same `history.jsonl` entry this method already appends --
        this is the "history.jsonl's existing append call" that story's Boundaries &
        Constraints names as the only place a turn's readback summary is ever persisted, not
        a second write elsewhere. Omitted (`None`) for any caller with no readback summary to
        record -- the key is left out of the entry entirely rather than written as `null`.

        Safe under concurrent callers targeting the same session (review round): if another
        writer claims the computed ordinal first, `_write_new_file_exclusive` raises
        `FileExistsError` instead of silently overwriting it, and this method re-reads the
        (now-updated) last ordinal and retries -- no accepted content is ever silently lost,
        unlike the original `os.replace()`-based version."""
        for _ in range(self._MAX_ACCEPT_RETRIES):
            ordinal = (self._last_ordinal() or 0) + 1
            target = self._ordinal_path(ordinal)
            try:
                self._write_new_file_exclusive(target, scene_text)
            except FileExistsError:
                continue  # another writer claimed this ordinal first -- retry with a fresh read
            entry = {
                "ordinal": ordinal,
                "prompt": prompt,
                "file": target.name,
                "outcome": "accepted",
            }
            if readback_summary is not None:
                entry["readback_summary"] = readback_summary
            self._append_history(entry)
            return ordinal
        raise SceneStoreError(
            f"Could not claim a new ordinal in '{self.session_dir}' after "
            f"{self._MAX_ACCEPT_RETRIES} attempts -- persistent concurrent writers"
        )

    def record_rejected(self, prompt: str, reason: str) -> None:
        """Records a rejected/failed attempt: no scene file is written, no ordinal is
        consumed (AD-12), but `history.jsonl` still grows by one entry (AD-7)."""
        self._append_history(
            {
                "ordinal": None,
                "prompt": prompt,
                "file": None,
                "outcome": "rejected",
                "reason": reason,
            }
        )

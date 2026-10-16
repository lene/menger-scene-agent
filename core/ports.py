"""The ports `core/` depends on (review 2026-10-14, msa#34): the model adapter protocol with its
typed request and result, scene extraction from a model response, and the scene store's
error type. Adapters implement or raise these; `core/` never imports `adapters/` (an
import-linter contract in the lint suite holds it to that)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Protocol, Union

ModelErrorKind = Literal[
    "call_failed", "invalid_output", "timeout", "needs_clarification", "unsupported"
]


@dataclass(frozen=True)
class ModelError:
    """A typed failure from a model call, or from extracting a scene out of its response --
    never a bare exception escaping the adapter (I/O & Edge-Case Matrix)."""

    kind: ModelErrorKind
    message: str
    cause: Optional[BaseException] = None


@dataclass(frozen=True)
class ModelRequest:
    """Everything the core has already composed for one model turn. `system_prompt` carries
    the manifest + corpus context and generation rules; `user_prompt` carries the plain-
    language request and, for a revision, the prior scene's full text -- string composition
    is entirely core's job, so the adapter never needs to know about manifests or corpora."""

    system_prompt: str
    user_prompt: str


ModelResult = Union[str, ModelError]


class ModelAdapter(Protocol):
    """The port the agent core depends on. No concrete adapter's SDK types appear on this
    boundary -- only plain strings and the typed result above."""

    def complete(self, request: ModelRequest) -> ModelResult: ...


# Matches a fenced code block, tagged with any word (scala, scala3, Scala, ...) or untagged.
# `re.DOTALL` so `.` spans newlines. The exact tag doesn't matter -- content is what's checked.
_CODE_FENCE = re.compile(r"```[ \t]*\w*[ \t]*\n(.*?)```", re.DOTALL)

# A candidate scene's top-level `object` must appear near the top of the file (mirrors
# SceneLoader.detectObjectName's first-60-lines heuristic) -- a bare "object " substring
# anywhere in a prose response is not enough to call it scene text.
_OBJECT_DECLARATION = re.compile(r"^\s*(?:private\s+)?object\s+\w", re.MULTILINE)
_FIRST_OBJECT_LINE_LIMIT = 60

# spec-ai-scene-agent story 18: the model's own sentinel for "too ambiguous or
# self-contradictory to compose" (PRD FR5), added to `core/generation.py`'s `_RULES` system
# prompt. Matched against the *entire* stripped response -- a single line, case-sensitive
# keyword -- so a sentinel merely mentioned in passing, or wrapped in prose/a code fence,
# does not match and falls through to the existing fence/`object`-declaration checks instead
# (I/O & Edge-Case Matrix: malformed sentinel emission is not specially handled). `[ \t]*`,
# not `\s*`, after the colon (review round, patch-level fix): `\s*` would let a newline
# right after the colon absorb into the gap, letting a two-line response (keyword on its own
# line, reason on the next) match despite this being documented as single-line-only -- `.`
# without `re.DOTALL` already can't cross a newline, but the gap before the capture group
# could.
_NEEDS_CLARIFICATION = re.compile(r"^NEEDS_CLARIFICATION:[ \t]*(.+)$")

# Usability review 2026-09 (F16, msa#3): a second, distinct sentinel for a request that is
# perfectly clear but asks for an effect this DSL has no way to produce at all -- previously
# folded into NEEDS_CLARIFICATION, which wrongly implied more information from the user could
# resolve it (there's nothing to clarify; the DSL simply can't do it). Same single-line,
# case-sensitive, whole-response-only matching discipline as NEEDS_CLARIFICATION above.
_UNSUPPORTED = re.compile(r"^UNSUPPORTED:[ \t]*(.+)$")

# Per the claude-api skill's current defaults: claude-opus-5 unless a caller names a
# different model (constructor argument), and a non-streaming max_tokens of 16000 --
# generated scene files are 30-250 lines, well under that ceiling.


def extract_scene_text(raw_text: str) -> ModelResult:
    """Extract a single scene file's text from a raw model response. Fails typed (never a
    best-effort guess, per the Ask First / I/O & Edge-Case Matrix) on an empty response, a
    response with no code and no `object` declaration, or one containing more than one
    candidate code block -- exactly the "single, straightforward extraction pass" the
    story's Ask First boundary allows."""
    if not raw_text:
        return ModelError(kind="invalid_output", message="Model returned an empty response")

    sentinel_match = _NEEDS_CLARIFICATION.match(raw_text.strip())
    if sentinel_match is not None:
        return ModelError(kind="needs_clarification", message=sentinel_match.group(1).strip())

    unsupported_match = _UNSUPPORTED.match(raw_text.strip())
    if unsupported_match is not None:
        return ModelError(kind="unsupported", message=unsupported_match.group(1).strip())

    fences = _CODE_FENCE.findall(raw_text)
    if len(fences) > 1:
        return ModelError(
            kind="invalid_output",
            message=f"Model returned {len(fences)} candidate code blocks, expected exactly one",
        )
    if len(fences) == 1:
        candidate = fences[0].strip()
    elif "```" in raw_text:
        # A fence marker is present but didn't match _CODE_FENCE (malformed fencing) --
        # never fall back to treating raw markdown syntax as scene text.
        return ModelError(
            kind="invalid_output",
            message="Model response contains malformed code fencing that could not be parsed",
        )
    else:
        candidate = raw_text

    match = _OBJECT_DECLARATION.search(candidate)
    if match is None:
        return ModelError(
            kind="invalid_output",
            message="Model response has no top-level 'object' declaration -- not scene text",
        )
    line_number = candidate.count("\n", 0, match.start())
    if line_number >= _FIRST_OBJECT_LINE_LIMIT:
        return ModelError(
            kind="invalid_output",
            message=f"Model response's 'object' declaration is past line "
            f"{_FIRST_OBJECT_LINE_LIMIT} -- SceneLoader.detectObjectName would miss it",
        )
    return candidate


class SceneStoreError(RuntimeError):
    """Raised for a real filesystem or session-identity failure in a session's scene store
    (adapters/scene_store.py) -- an infrastructure failure, not a typed turn outcome."""


class SceneStore(Protocol):
    """What a turn needs of a session's scene history (adapters/scene_store.py implements it):
    the session directory for staging files, the current scene and its evaluated snapshots,
    and the two ways a turn ends -- accepted (a new ordinal) or rejected (a history entry)."""

    @property
    def session_dir(self) -> Path: ...

    def current_scene(self) -> Optional[str]: ...

    def current_snapshots(self) -> List[dict]: ...

    def write_snapshots(self, ordinal: int, snapshots: List[dict]) -> None: ...

    def accept(self, scene_text: str, prompt: str, readback_summary: Optional[str] = None) -> int: ...

    def record_rejected(self, prompt: str, reason: str, scene_text: Optional[str] = None) -> None: ...

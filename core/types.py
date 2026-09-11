"""Pure data types shared by the agent core.

AD-1/AD-3: no I/O, no credentials, no import of the `anthropic` SDK or anything else that
crosses a privilege boundary -- only plain dataclasses and string constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

# The manifest and corpus artifacts (AD-9) are versioned by the renderer-domain tools that
# produce them (`ManifestGenerator`, `CorpusExporter`). This is the version this core knows
# how to consume; a mismatch is a real failure mode (a stale artifact silently missing new
# DSL vocabulary), not a style nit -- see the I/O & Edge-Case Matrix.
EXPECTED_MANIFEST_SCHEMA_VERSION = "1.0.0"
EXPECTED_CORPUS_SCHEMA_VERSION = "1.0.0"

ErrorKind = Literal[
    "model_call_failed",
    "invalid_model_output",
    "stale_manifest",
    "stale_corpus",
]


@dataclass(frozen=True)
class GenerationError:
    """A typed failure result from `generate()`/`revise()` -- never an exception escaping
    to the caller (I/O & Edge-Case Matrix: model failures, non-scene output, and stale
    artifacts are all reported this way, never a silent fallback or a best-effort guess)."""

    kind: ErrorKind
    message: str
    cause: Optional[BaseException] = None


# generate()/revise() return either the derived scene text, or a typed error -- never raise.
GenerationResult = Union[str, GenerationError]

ReadbackErrorKind = Literal[
    "model_call_failed",
    "invalid_model_output",
]


@dataclass(frozen=True)
class ReadbackError:
    """A typed failure result from `semantic_readback()` -- never an exception escaping to
    the caller (I/O & Edge-Case Matrix: a model-call failure and an empty/unusable model
    response are both reported this way, never a silent fallback or a best-effort guess).
    Mirrors `GenerationError`'s shape exactly (story 6)."""

    kind: ReadbackErrorKind
    message: str
    cause: Optional[BaseException] = None


# semantic_readback() returns either the plain-language readback text, or a typed error --
# never raise. Mirrors GenerationResult's exact convention.
ReadbackResult = Union[str, ReadbackError]

TicketErrorKind = Literal[
    "invalid_capability_text",
    "drafts_dir_unavailable",
    "write_failed",
]


@dataclass(frozen=True)
class TicketError:
    """A typed failure result from `adapters.tickets.write_draft()` -- never an exception
    escaping to the caller (Boundaries & Constraints: "Returns a typed result, never raises
    for an expected outcome"). Mirrors `GenerationError`/`ReadbackError`'s exact shape."""

    kind: TicketErrorKind
    message: str
    cause: Optional[BaseException] = None


# write_draft() returns either the written draft file's path as str, or a typed error --
# never raise. Mirrors GenerationResult/ReadbackResult's exact convention.
TicketResult = Union[str, TicketError]

# AD-5's tagged renderer-domain result, in this module's own snake_case form. Translated from
# `SceneValidator.Tag`'s hyphenated wire strings (`ok` | `compile-errors` | `lint-findings` |
# `refused`) by `adapters.scene_validator`'s `_WIRE_TAG_TO_VALIDATION_TAG` -- nothing outside
# that adapter ever sees the wire string itself.
ValidationTag = Literal["ok", "compile_errors", "lint_findings", "refused"]

ValidationErrorKind = Literal["timeout", "malformed_output", "subprocess_failed"]


@dataclass(frozen=True)
class ValidationError:
    """A typed failure result from `adapters.scene_validator.validate_scene()` -- never an
    exception escaping to the caller (Boundaries & Constraints: "Returns a typed result,
    never raises for an expected outcome"). Mirrors `GenerationError`/`ReadbackError`/
    `TicketError`'s exact shape (story 10)."""

    kind: ValidationErrorKind
    message: str
    cause: Optional[BaseException] = None


@dataclass(frozen=True)
class ValidationFinding:
    """One geometric-invariant violation, mirroring `SceneValidator.Finding`
    (`menger/menger-app/src/main/scala/menger/tools/SceneValidator.scala`, story 5)
    field-for-field: a short, stable machine-readable `invariant` tag plus a human-readable
    `message` -- carried structurally rather than flattened into `messages`, same reasoning
    as the Scala side (Code Map: "don't invent fields it doesn't have")."""

    invariant: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """AD-5's tagged renderer-domain result, parsed from `SceneValidator.ValidationResult`'s
    JSON (`menger`, story 5) field-for-field -- `tag`, `messages`, `findings`, `scene`,
    `schema_version` -- no invented fields (Boundaries & Constraints). `tag` is this module's
    own snake_case `ValidationTag`, never the wire format's hyphenated string."""

    tag: ValidationTag
    messages: List[str]
    findings: List[ValidationFinding] = field(default_factory=list)
    scene: Optional[str] = None
    schema_version: str = ""


# validate_scene() returns either the renderer's typed tagged result, or a typed error --
# never raise. Mirrors GenerationResult/ReadbackResult/TicketResult's exact convention.
ValidationOutcome = Union[ValidationResult, ValidationError]

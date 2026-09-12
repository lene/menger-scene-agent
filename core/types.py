"""Pure data types shared by the agent core.

AD-1/AD-3: no I/O, no credentials, no import of the `anthropic` SDK or anything else that
crosses a privilege boundary -- only plain dataclasses and string constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional, Union

# `subprocess.Popen` is referenced only as a type annotation on `RenderWindowResult.process`
# (story 13) -- guarded behind `TYPE_CHECKING` (this module's own `from __future__ import
# annotations` makes every annotation a lazily-evaluated string, so this import never runs at
# module load time) so this module still imports nothing at runtime beyond plain dataclasses
# and string constants (AD-1/AD-3's own header comment above).
if TYPE_CHECKING:
    import subprocess

# `gauntlet/types.py`'s `Finding` is, like this module, a plain dataclass with no I/O of its
# own -- importing it here (story 20, `TurnResult.findings`) doesn't cross AD-1/AD-3's
# privilege boundary, and `gauntlet/` has no dependency on `core/` (no import cycle).
from gauntlet.types import Finding

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
    "model_call_timeout",
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

# `on_stage`'s three-value contract (spec-ai-scene-agent story 12, "live status line") --
# enforced at the type level via `Optional[Callable[[StageName], None]]` on `run_turn()`'s
# signature, not just documented in its docstring.
StageName = Literal["generating", "validating", "reading back"]

# `run_turn()`'s own tagged outcome (story 20, core/turn.py), composed from every stage it
# wires together rather than inventing a parallel vocabulary for the same failures:
#   - "accepted" -- the only success tag; unique to this module.
#   - "local_finding" -- one or more agent-side gauntlet.check_*() findings; unique to this
#     module (there is no upstream type for "a local check found something").
#   - "compile_errors" / "lint_findings" / "refused" -- reused verbatim from ValidationTag:
#     the renderer's own non-"ok" tags, unchanged by passing through run_turn().
#   - "timeout" / "malformed_output" / "subprocess_failed" -- reused verbatim from
#     ValidationErrorKind: validate_scene() itself failed to produce a renderer verdict.
#   - "generation_failed" -- generate()/revise() returned a GenerationError before any
#     gauntlet check could run (check_*() requires a str; not one of the frozen I/O & Edge
#     Case Matrix's five rows, but a real precondition for all of them).
#   - "generation_timeout" -- spec-ai-scene-agent story 15: a GenerationError whose kind is
#     specifically "model_call_timeout" (the model-provider call itself hung past its
#     configured timeout), distinguished from every other "generation_failed" cause so a
#     user can tell "the model hung" apart from a generic generation failure. Distinct from
#     the existing "timeout" tag above, which is validate_scene()'s renderer-side subprocess
#     timeout (story 20) -- conflating the two would hide *which* stage timed out.
#   - "readback_failed" -- the renderer said "ok" but semantic_readback() itself failed;
#     the I/O & Edge-Case Matrix's explicit "not accepted without its summary" row.
#   - "storage_failed" -- a `SceneStore` write failed for real (`store.accept()` raised
#     `SceneStoreError` after exhausting its ordinal-claim retries, or writing the staging
#     file itself raised `OSError`) -- not one of the frozen I/O & Edge-Case Matrix's five
#     rows either, but the same kind of real precondition failure `generation_failed`
#     already covers on the generation side (review round, patch-level fix).
TurnTag = Literal[
    "accepted",
    "generation_failed",
    "generation_timeout",
    "local_finding",
    "compile_errors",
    "lint_findings",
    "refused",
    "timeout",
    "malformed_output",
    "subprocess_failed",
    "readback_failed",
    "storage_failed",
]


@dataclass(frozen=True)
class TurnResult:
    """The typed, never-raising outcome of one `core.turn.run_turn()` call -- covers every
    row of that story's I/O & Edge-Case Matrix plus the `generation_failed` precondition
    (see `TurnTag`). Mirrors `GenerationError`/`ReadbackError`/`TicketError`/`ValidationError`
    's dataclass shape, extended with the fields a turn's outcome actually needs to carry:
    `messages` (always populated with at least one human-readable string on any non-accepted
    tag), `findings` (populated only for `local_finding`, the one outcome with structured
    `gauntlet.types.Finding` data of its own), `readback_summary` (populated only on
    `accepted` -- the plain-language sentence `semantic_readback()` produced), and `ordinal`
    (populated only on `accepted` -- the new `SceneStore.accept()`-assigned ordinal)."""

    tag: TurnTag
    messages: List[str]
    findings: List[Finding] = field(default_factory=list)
    readback_summary: Optional[str] = None
    ordinal: Optional[int] = None


# `adapters.render_window.refresh_render_window()`'s own typed failure vocabulary (story 13):
#   - "refused" -- AD-16's own lock-conflict tag, reused verbatim from `menger`'s
#     `Main.refusedResultJson` (story 8, itself reusing `SceneValidator`'s AD-5 `Tag` shape).
#     A real conflict with a *different* process's render session -- nothing is terminated.
#   - "malformed_output" -- the launcher exited within the grace period but its stdout wasn't
#     the expected tagged JSON (mirrors `ValidationErrorKind`'s own tag of the same name).
#   - "launch_failed" -- `launcher_path` doesn't exist or can't be executed at all (the
#     `subprocess.Popen(...)` call itself raised `OSError`), never a raised exception.
RenderWindowErrorKind = Literal["refused", "malformed_output", "launch_failed"]


@dataclass(frozen=True)
class RenderWindowError:
    """A typed failure result from `adapters.render_window.refresh_render_window()` -- never
    an exception escaping to the caller (Boundaries & Constraints: "typed error ... never an
    unhandled exception"). Mirrors `GenerationError`/`ReadbackError`/`TicketError`/
    `ValidationError`'s exact shape (story 13)."""

    kind: RenderWindowErrorKind
    message: str
    cause: Optional[BaseException] = None


@dataclass(frozen=True)
class RenderWindowResult:
    """A successful render-window launch (story 13): still running past `grace_period`, per
    the Design Notes' own "no ready signal beyond staying alive" reasoning. Carries the live
    `subprocess.Popen` handle itself -- unlike `ValidationResult`'s `messages`/`findings`
    payload, there is nothing else to report on success, and the caller needs the handle to
    track/replace this window on a later call (`previous_process`).

    Note: this module's own "no I/O at runtime" guarantee (AD-1/AD-3, module header) is about
    this module's *own code* never performing I/O itself -- it says nothing about whether the
    dataclasses it defines can *reference* objects that do. `process` is exactly that case: a
    live `subprocess.Popen` that performs real I/O, just not I/O this module initiates.

    Also note: `frozen=True` only prevents reassigning the `process` attribute itself -- it has
    no effect on the wrapped `Popen`'s own mutable OS-level state. The underlying process can
    still exit, be waited on, terminated, etc. independently of this wrapper; a reader should
    not assume stronger immutability than that.
    """

    process: "subprocess.Popen[str]"


# refresh_render_window() returns either the running launch's handle, or a typed error --
# never raise. Mirrors GenerationResult/ReadbackResult/TicketResult/ValidationOutcome's exact
# convention.
RenderWindowOutcome = Union[RenderWindowResult, RenderWindowError]

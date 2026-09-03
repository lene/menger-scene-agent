"""Pure data types shared by the agent core.

AD-1/AD-3: no I/O, no credentials, no import of the `anthropic` SDK or anything else that
crosses a privilege boundary -- only plain dataclasses and string constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

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

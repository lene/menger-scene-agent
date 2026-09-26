"""Renderer-side gauntlet adapter (spec-ai-scene-agent story 10): wraps
`menger/docker/scene-validator/run-sandboxed.sh`, the AD-18 sandboxed subprocess entry point
to `SceneValidator` (Scala, story 5) -- the renderer-side compile (gauntlet stage 1) and
geometric-invariant (stage 4) checks that produce AD-5's tagged result.

AD-2/AD-18: this adapter never calls `SceneValidator`'s Scala class in-process and never
bypasses or reimplements the sandbox (Boundaries & Constraints) -- `run-sandboxed.sh`'s own
OS-level containment IS the boundary, not an implementation convenience to optimize away (see
this story's Design Notes). This mirrors `adapters/tickets.py`'s shape: a typed result, never
an exception escaping to the caller, for every outcome the subprocess boundary can produce.

No pipeline wiring here -- calling this from `core/generation.py`'s `generate()`/`revise()` is
a separate, deferred story (see `_bmad-output/implementation-artifacts/deferred-work.md` in
`menger-toplevel`, the workspace root).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Union

from core.types import (
    ValidationError,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
)

# run-sandboxed.sh's own documented exit-code contract (its usage text, and
# menger/docker/scene-validator/run-sandboxed.sh's header comment): 0 `ok`, 1 a scene defect
# (`compile-errors`/`lint-findings` -- the JSON `tag` disambiguates which), 2 `refused`, 124 a
# wall-clock timeout with **no JSON at all** on stdout.
_TIMEOUT_EXIT_CODE = 124

# Wire-format tags SceneValidator.scala's `Tag.toWire` emits (hyphenated, per AD-5) mapped to
# this module's own snake_case `ValidationTag` (`core/types.py`). Nothing outside this module
# ever sees the wire string itself.
_WIRE_TAG_TO_VALIDATION_TAG = {
    "ok": "ok",
    "compile-errors": "compile_errors",
    "lint-findings": "lint_findings",
    "refused": "refused",
}

# Bounds how much of a malformed/unexpected stdout payload is echoed back in a
# ValidationError's message, so a pathological or adversarial response can't blow up the
# error message itself.
_STDOUT_PREVIEW_LENGTH = 500


def validate_scene(
    scene_file: Union[str, Path],
    script_path: Union[str, Path],
    image: Optional[str] = None,
    timeout: Optional[float] = None,
) -> ValidationOutcome:
    """Runs `<script_path> <scene_file> [--image <image>]` and returns AD-5's tagged
    renderer-domain result, or a typed `ValidationError` -- never raises for an expected
    outcome (Boundaries & Constraints). `script_path` names `run-sandboxed.sh`'s location: an
    injected parameter, never discovered or hardcoded (Consistency Conventions), mirroring
    `adapters.tickets.write_draft`'s `drafts_dir`.

    `timeout` (seconds) is an optional defense-in-depth bound on top of `run-sandboxed.sh`'s
    own internal wall-clock timeout (`SCENE_VALIDATOR_TIMEOUT`, default 120s) -- the exit-124
    row below is the documented way the subprocess itself reports a timeout; a
    `subprocess.TimeoutExpired` from this process's own `timeout=` is handled identically in
    case the subprocess doesn't return control even after its own bound.
    """
    if image is not None and not isinstance(image, str):
        return ValidationError(
            kind="subprocess_failed",
            message=f"'image' must be a str or None, got {type(image).__name__}",
        )

    cmd = [str(script_path), str(scene_file)]
    if image is not None:
        cmd += ["--image", image]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return ValidationError(
            kind="timeout",
            message=f"scene validator subprocess '{script_path}' exceeded its timeout",
            cause=e,
        )
    except OSError as e:
        return ValidationError(
            kind="subprocess_failed",
            message=f"could not run scene validator subprocess '{script_path}': {e}",
            cause=e,
        )
    except UnicodeDecodeError as e:
        # `text=True` decodes stdout/stderr as text with no error handling of its own -- if
        # the subprocess ever emits non-UTF-8 bytes, subprocess.run itself raises this. Caught
        # here so it never escapes validate_scene unhandled (Boundaries & Constraints).
        return ValidationError(
            kind="subprocess_failed",
            message=f"scene validator subprocess '{script_path}' produced output that could "
            f"not be decoded as text: {e}",
            cause=e,
        )

    if completed.returncode == _TIMEOUT_EXIT_CODE:
        # run-sandboxed.sh's own wall-clock timeout: exit 124, no JSON on stdout at all.
        # Never attempt to parse stdout as JSON for this exit code (I/O & Edge-Case Matrix:
        # "never attempting to parse stdout as JSON").
        return ValidationError(
            kind="timeout",
            message=f"scene validator subprocess '{script_path}' timed out (exit 124)",
        )

    return _parse_stdout(completed.stdout, completed.stderr, completed.returncode)


def _parse_stdout(stdout: str, stderr: str, returncode: int) -> ValidationOutcome:
    """Parses `SceneValidator.ValidationResult`'s JSON off stdout, or returns a typed
    `malformed_output` error for corrupted JSON or an unexpected shape -- never an unhandled
    exception (Acceptance Criteria). Parses exactly the real fields (`tag`, `messages`,
    `findings`, `scene`, `schemaVersion`) -- no invented fields (Code Map).

    `stderr` is not parsed as data -- it's only folded into a `malformed_output` error's
    message when stdout itself didn't yield a usable payload, since `run-sandboxed.sh`'s
    usage-error path (e.g. "Scene file not found") puts its real diagnostic there while
    stdout is empty."""
    try:
        payload = json.loads(stdout)
        if not isinstance(payload, dict):
            raise TypeError(f"expected a JSON object, got {type(payload).__name__}")

        wire_tag = payload["tag"]
        tag = _WIRE_TAG_TO_VALIDATION_TAG[wire_tag]

        # An explicit JSON `null` is treated the same as the field being omitted entirely,
        # same as `scene` below -- not a type violation.
        messages = payload.get("messages")
        if messages is None:
            messages = []
        if not isinstance(messages, list) or not all(isinstance(m, str) for m in messages):
            raise TypeError("'messages' was not a list of strings")

        raw_findings = payload.get("findings")
        if raw_findings is None:
            raw_findings = []
        if not isinstance(raw_findings, list):
            raise TypeError("'findings' was not a list")
        findings = []
        for f in raw_findings:
            invariant = f["invariant"]
            finding_message = f["message"]
            if not isinstance(invariant, str) or not isinstance(finding_message, str):
                raise TypeError("finding 'invariant'/'message' were not strings")
            findings.append(ValidationFinding(invariant=invariant, message=finding_message))

        scene = payload.get("scene")
        if scene is not None and not isinstance(scene, str):
            raise TypeError("'scene' was not a string")

        schema_version = payload.get("schemaVersion", "")
        if not isinstance(schema_version, str):
            raise TypeError("'schemaVersion' was not a string")
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        preview = stdout[:_STDOUT_PREVIEW_LENGTH]
        detail = (
            f"could not parse renderer response (exit {returncode}): {e} "
            f"-- stdout was {preview!r}"
        )
        stderr_preview = stderr[:_STDOUT_PREVIEW_LENGTH]
        if stderr_preview:
            detail += f" -- stderr was {stderr_preview!r}"
        return ValidationError(
            kind="malformed_output",
            message=detail,
            cause=e,
        )

    return ValidationResult(
        tag=tag, messages=messages, findings=findings, scene=scene, schema_version=schema_version
    )

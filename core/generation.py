"""Pure derivation: prompt (+ optional prior scene) + manifest + corpus -> scene text.

AD-1/AD-2/AD-3: no file I/O, no credentials, no import of the `anthropic` SDK or any other
concrete adapter -- only the `ModelAdapter` port and typed request/result shapes from
`adapters.model`, plus pure string/dict composition. Never compiles or loads the generated
scene (AD-2, story 5's job) -- the only check here is the shape `generate()`'s caller can
observe for free: the returned text, verbatim from the adapter.
"""

from __future__ import annotations

import json
from typing import Optional

from adapters.model import ModelAdapter, ModelError, ModelRequest, extract_scene_text
from core.types import (
    EXPECTED_CORPUS_SCHEMA_VERSION,
    EXPECTED_MANIFEST_SCHEMA_VERSION,
    GenerationError,
    GenerationResult,
)

_RULES = (
    "You compose Menger DSL scene files (Scala 3). Rules:\n"
    "- Use only vocabulary present in the capability manifest below -- never invent a DSL "
    "type, field, method or preset name that isn't listed there.\n"
    "- The entire example corpus is provided below. Prefer composing from the example(s) "
    "nearest the request over writing from scratch, but never copy an unrelated example's "
    "specifics (camera, materials) into an unrelated request.\n"
    "- Never import from `examples.dsl.*` (e.g. `examples.dsl.common.Lighting`). Those "
    "cross-file imports only resolve inside the renderer's own example-source tree, not "
    "for a standalone generated scene -- if an example composes reusable materials or "
    "lighting via such an import, inline the pattern into your own file instead.\n"
    "- The scene's top-level `object` declaration must be the FIRST `object` in the file, "
    "within the first 60 lines (SceneLoader's detectObjectName heuristic).\n"
    "- Respond with exactly one Scala code block containing the complete scene file's "
    "source text -- no prose before or after, no multiple candidates.\n"
)


def validate_artifacts(manifest: dict, corpus: dict) -> Optional[GenerationError]:
    if not isinstance(manifest, dict):
        return GenerationError(
            kind="stale_manifest",
            message=f"Manifest artifact must be an object, got {type(manifest).__name__}",
        )
    manifest_version = manifest.get("schemaVersion")
    if manifest_version != EXPECTED_MANIFEST_SCHEMA_VERSION:
        return GenerationError(
            kind="stale_manifest",
            message=(
                f"Manifest artifact schemaVersion is {manifest_version!r}, expected "
                f"{EXPECTED_MANIFEST_SCHEMA_VERSION!r}"
            ),
        )
    if not isinstance(corpus, dict):
        return GenerationError(
            kind="stale_corpus",
            message=f"Corpus artifact must be an object, got {type(corpus).__name__}",
        )
    corpus_version = corpus.get("schemaVersion")
    if corpus_version != EXPECTED_CORPUS_SCHEMA_VERSION:
        return GenerationError(
            kind="stale_corpus",
            message=(
                f"Corpus artifact schemaVersion is {corpus_version!r}, expected "
                f"{EXPECTED_CORPUS_SCHEMA_VERSION!r}"
            ),
        )
    if not isinstance(corpus.get("scenes", []), list):
        return GenerationError(
            kind="stale_corpus",
            message="Corpus artifact's 'scenes' field must be a list",
        )
    return None


def _build_system_prompt(manifest: dict, corpus: dict) -> str:
    manifest_json = json.dumps(manifest, indent=2)
    scenes = corpus.get("scenes", [])
    examples = "\n\n".join(
        f"# {scene.get('path', scene.get('name', 'example'))}\n{scene.get('source', '')}"
        for scene in scenes
    )
    return (
        f"{_RULES}\n"
        f"## DSL capability manifest\n```json\n{manifest_json}\n```\n\n"
        f"## Example scene corpus\n{examples}\n"
    )


def _model_error_to_generation_error(error: ModelError) -> GenerationError:
    # spec-ai-scene-agent story 15: a timeout is mapped to its own distinct kind, never
    # folded into "model_call_failed" -- otherwise a hung model-provider request would be
    # indistinguishable from a network error or a rate limit all the way up through
    # `run_turn()`'s eventual `TurnTag`.
    if error.kind == "timeout":
        kind = "model_call_timeout"
    elif error.kind == "call_failed":
        kind = "model_call_failed"
    else:
        kind = "invalid_model_output"
    return GenerationError(kind=kind, message=error.message, cause=error.cause)


def _complete(system_prompt: str, user_prompt: str, adapter: ModelAdapter) -> GenerationResult:
    # generate()/revise() must never raise (their own contract, see core/types.py's
    # GenerationResult docstring) -- this holds regardless of whether a given ModelAdapter
    # implementation honors its own "never raise" contract, so the core defends its own
    # guarantee rather than trusting every adapter to defend it independently.
    try:
        result = adapter.complete(ModelRequest(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception as e:  # noqa: BLE001 -- adapter misbehavior must not escape core
        return GenerationError(
            kind="model_call_failed", message=f"Model adapter raised: {e}", cause=e
        )
    if isinstance(result, ModelError):
        return _model_error_to_generation_error(result)
    # `extract_scene_text` (review round, story 6): the adapter returns the model's raw
    # response text -- extracting a single scene file's text out of it (rejecting prose, a
    # missing/misplaced `object` declaration, multiple candidate code blocks) is this
    # module's own concern, not every `ModelAdapter` caller's. It used to run unconditionally
    # inside `AnthropicModelAdapter.complete()`, which broke `core/readback.py`'s
    # `semantic_readback()` (a deliberately non-scene-shaped response) in production.
    extracted = extract_scene_text(result)
    if isinstance(extracted, ModelError):
        return _model_error_to_generation_error(extracted)
    return extracted


def generate(prompt: str, manifest: dict, corpus: dict, adapter: ModelAdapter) -> GenerationResult:
    """CAP-1: compose a brand-new scene from a plain-language prompt. No prior scene."""
    invalid = validate_artifacts(manifest, corpus)
    if invalid is not None:
        return invalid

    system_prompt = _build_system_prompt(manifest, corpus)
    user_prompt = f"Request: {prompt}\n\nCompose a new scene satisfying this request."
    return _complete(system_prompt, user_prompt, adapter)


def revise(
    prompt: str, prior_scene: str, manifest: dict, corpus: dict, adapter: ModelAdapter
) -> GenerationResult:
    """CAP-2: derive a modified scene from an existing file's text and a change request.

    The prior scene is passed through to the model verbatim, inside the user prompt --
    this is a one-shot edit derived from what already exists, not a from-scratch
    regeneration (Design Notes: CAP-2 is not held to CAP-3's full diff-minimality rigor
    here, but the model is explicitly instructed to carry over anything the request does
    not implicate)."""
    invalid = validate_artifacts(manifest, corpus)
    if invalid is not None:
        return invalid

    system_prompt = _build_system_prompt(manifest, corpus)
    user_prompt = (
        "Here is the current scene file, verbatim:\n\n"
        f"```scala\n{prior_scene}\n```\n\n"
        f"Change request: {prompt}\n\n"
        "Modify the scene above to satisfy the change request. Keep everything the request "
        "does not implicate unchanged -- camera, materials, lights, background and any "
        "other content not mentioned by the request should carry over as-is."
    )
    return _complete(system_prompt, user_prompt, adapter)

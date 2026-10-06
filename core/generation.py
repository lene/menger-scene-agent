"""Pure derivation: prompt (+ optional prior scene) + manifest + corpus -> scene text.

AD-1/AD-2/AD-3: no file I/O, no credentials, no import of the `anthropic` SDK or any other
concrete adapter -- only the `ModelAdapter` port and typed request/result shapes from
`adapters.model`, plus pure string/dict composition. Never compiles or loads the generated
scene (AD-2, story 5's job) -- the only check here is the shape `generate()`'s caller can
observe for free: the returned text, verbatim from the adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from adapters.model import ModelAdapter, ModelError, ModelRequest, extract_scene_text
from core.scene_facts import extract_scene_facts
from core.types import (
    EXPECTED_CORPUS_SCHEMA_VERSION,
    EXPECTED_MANIFEST_SCHEMA_VERSION,
    GenerationError,
    GenerationResult,
)
from gauntlet._scala_text import strip_comments_and_strings

# The generation system prompt's rules, kept as prose in its own file so they can be edited
# without touching this module's code (usability review 2026-09 inbox item).
_RULES = (Path(__file__).resolve().parent / "generation_rules.md").read_text()


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


def _extent_fact(prior_scene: str) -> str:
    """F20: the current scene's bounding-sphere extent as ready numbers, so a "zoom out"/
    "frame everything" request doesn't require the model to sum up every object's `pos`/
    `size` itself from the raw scene text -- error-prone arithmetic a static extraction
    (`core.scene_facts`, AD-2) already does exactly once. Empty when there's nothing
    positioned to frame (`extent_center` is `None`), so callers can splice it in
    unconditionally."""
    facts = extract_scene_facts(strip_comments_and_strings(prior_scene))
    center = facts.extent_center
    radius = facts.extent_radius
    if center is None or radius is None:
        return ""
    return (
        f"\nCurrent scene extent: center = {center}, bounding radius = {radius}. If the "
        "request asks to zoom out, frame, or fit everything in view, aim `lookAt` at this "
        "center and set the camera's distance from it to at least "
        "radius / sin(22.5 degrees) (the manifest conventions give the exact formula), with "
        "a margin for comfortable framing -- keep the current viewing direction unless the "
        "request says otherwise.\n"
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
    elif error.kind == "needs_clarification":
        # spec-ai-scene-agent story 18: the model's own sentinel response (detected by
        # `extract_scene_text`) is its own distinct outcome end to end -- never folded into
        # "invalid_model_output" alongside every other malformed/rejected response shape.
        kind = "needs_clarification"
    elif error.kind == "unsupported":
        # usability review 2026-09 (F16, msa#3): the model's own second sentinel, for a
        # request that's perfectly clear but asks for an effect this DSL cannot produce at
        # all -- distinct from "needs_clarification" the same way, never folded into it.
        kind = "unsupported"
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
        f"```scala\n{prior_scene}\n```\n"
        f"{_extent_fact(prior_scene)}\n"
        f"Change request: {prompt}\n\n"
        "Modify the scene above to satisfy the change request. Keep everything the request "
        "does not implicate unchanged -- camera, materials, lights, background and any "
        "other content not mentioned by the request should carry over as-is. Never remove a "
        "property the request does not ask to remove (a procedural texture, rotation, edges, "
        "projection, colour, an object): if satisfying the request seems to require removing "
        "one, keep it and find another way, or respond with NEEDS_CLARIFICATION explaining "
        "the conflict. The same holds when satisfying the request would also change something "
        "the request does not name (moving or resizing an object, changing its rotation or "
        "material): ask with NEEDS_CLARIFICATION instead of doing it silently. One exception: "
        "a new material replaces the look the old one imitated -- when the request swaps an "
        "object's material (\"make it aluminium\" on a wood-grained object), drop a "
        "`proceduralType` that only imitated the old material (wood grain, marble veins) "
        "unless the request keeps it. The doc comment, the `object` name and the "
        "`SceneRegistry.register` id are not content to carry over: when the change makes any "
        "of them false (a level, type or material they name), update them to match the "
        "modified scene, the name and the id consistently."
    )
    return _complete(system_prompt, user_prompt, adapter)

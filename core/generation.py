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
    "- Any field the manifest types as `scala.Option<...>` (e.g. `material`, `color`, "
    "`texture`) must be given as `Some(...)` when you provide a value, never as a bare "
    "unwrapped value -- e.g. `material = Some(Material(...))`, not `material = "
    "Material(...)`. Some example scenes in the corpus omit `Some(...)` around `material`; "
    "that shorthand only compiles through a small family of convenience constructor "
    "overloads covering a few fixed named-argument combinations, and silently breaks with "
    "an unrelated-looking type error the moment any other field (e.g. `rotation`) is also "
    "named -- do not imitate that shorthand, always wrap explicitly.\n"
    "- `proceduralType`/`proceduralScale` (present on Sponge, TesseractSponge, and other "
    "solid object types) select a built-in procedural texture -- the manifest lists these "
    "as a bare `int` with no further explanation, but the value is not free-form: it "
    "selects one of exactly these fixed presets, none else:\n"
    "    0 none (default) -- no procedural effect, material color used as-is\n"
    "    1 value_noise, 2 fbm, 3 worley, 4 gradient, 5 wood, 6 marble, 7 layered_noise -- "
    "each MODULATES the material's own base color by a scalar noise/pattern value (same "
    "hue, varying brightness/pattern); pick whichever name matches the requested texture "
    "(e.g. a request for a marbled/veined look is proceduralType 6, not a texture file)\n"
    "    8 xyz_rgb -- REPLACES the material's color entirely with a color derived from the "
    "object's position (`|x|,|y|,|z| mod 1` -> R,G,B, in WORLD space, not object-local). The "
    "`abs` folds each axis at zero: for an object centered at/near the world origin (`pos` "
    "close to `(0,0,0)`, the usual default) this MIRRORS the color pattern on each axis -- "
    "the request \"every color value should appear only once\"/\"one white corner\" is NOT "
    "satisfied by simply adding `proceduralType = 8` to a centered object; verified by "
    "rendering it (a centered sponge showed multiple light/dark corners, not one).\n"
    "  For a genuinely one-shot, non-repeating position->color gradient (each surface point "
    "gets a distinct color, one dark corner and one light corner, no mirroring) -- verified "
    "by rendering both the broken and fixed version -- the object's local extent must never "
    "cross zero on any axis, since a negative-to-positive local range is exactly what the "
    "`abs` fold mirrors. Since `pos` is this object type's CENTER (spans roughly "
    "`[pos - size/2, pos + size/2]` per axis, unrotated), shift it into the positive octant: "
    "set `pos = (size/2, size/2, size/2)` (adjust the camera's `lookAt` to the same point) "
    "and `proceduralScale = 1 / size` exactly -- this keeps every point's scaled coordinate "
    "in `[0, 1)`, so the mod never wraps and the abs never flips sign. Do not apply this "
    "offset trick unless the request specifically needs a unique, non-repeating gradient; a "
    "generic \"color by position\" request is satisfied by the simpler centered form above, "
    "mirroring and all. Because xyz_rgb colors by WORLD position, the offset only gives a "
    "local-coordinate gradient for an unrotated object. NEVER remove or change an existing "
    "`rotation` (or any other property the request does not mention) to make this recipe "
    "work: if the object currently has a non-zero rotation and the request needs a "
    "local-coordinate gradient, respond with NEEDS_CLARIFICATION explaining that the colors "
    "would follow the world axes rather than the rotated object's own axes, and ask whether "
    "to drop the rotation or accept world-axis coloring.\n"
    "    9 heatmap -- REPLACES color with a blue->cyan->green->yellow->red gradient driven "
    "by a noise value (not position directly)\n"
    "    10 triplanar -- MODULATES base color using a noise pattern projected along the "
    "surface normal (avoids stretching on near-axis-aligned faces)\n"
    "  `proceduralScale` uniformly rescales world position before any of the above are "
    "evaluated (higher = smaller/more frequent pattern); to make one full pattern period "
    "span an object of a given `size`, set `proceduralScale` to roughly `1 / size`.\n"
    "- This procedural-texture preset list is fixed and closed -- there is no mechanism in "
    "this DSL to define a NEW procedural texture pattern from a scene file (that would mean "
    "writing native CUDA shader code in a different repository entirely, well outside a "
    "scene file's reach). If a request needs a visual pattern none of the 10 presets above "
    "reasonably approximates, do not invent DSL syntax or silently substitute an unrelated "
    "preset -- respond with NEEDS_CLARIFICATION naming the specific pattern that has no "
    "built-in equivalent.\n"
    "- Respond with exactly one Scala code block containing the complete scene file's "
    "source text -- no prose before or after, no multiple candidates.\n"
    "- If the request is too ambiguous or self-contradictory to compose (an undefined "
    "domain term, or a direct contradiction like \"redder and greener\"), do not guess -- "
    "respond with exactly one plain line, NEEDS_CLARIFICATION: <reason naming the specific "
    "term or contradiction>, instead of a scene code block. Do not wrap this line in "
    "backticks or a code fence -- it must be plain text, not formatted as code.\n"
    "- The same applies if the request reads as a question or observation about the "
    "CURRENT scene (\"are you sure the texture is silver?\", \"why does it look like "
    "chrome?\", \"I thought it was brighter\") rather than an explicit instruction for a "
    "new state -- do not silently guess what change, if any, is wanted. Respond with "
    "NEEDS_CLARIFICATION: this reads as a question about the current scene, not a change "
    "request -- use /ask or /question for an explanation, or say what to change. This "
    "category is narrow: it applies ONLY when the request contains no directive at all -- "
    "never to an ordinary change request just because more than one DSL field could satisfy "
    "it. \"make it darker\", \"make it slower\", \"make it bigger\" ARE directives (note the "
    "imperative \"make it...\" phrasing), not ambiguous requests -- even though \"darker\" "
    "could mean light intensity, background, or material color, pick the single most direct, "
    "conventional lever for the term used and compose the scene; do not ask which one was "
    "meant. Reserve NEEDS_CLARIFICATION for a genuinely undefined term, a direct "
    "contradiction, or input with no actionable instruction in it at all.\n"
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
    elif error.kind == "needs_clarification":
        # spec-ai-scene-agent story 18: the model's own sentinel response (detected by
        # `extract_scene_text`) is its own distinct outcome end to end -- never folded into
        # "invalid_model_output" alongside every other malformed/rejected response shape.
        kind = "needs_clarification"
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

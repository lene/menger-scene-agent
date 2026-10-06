"""Pure derivation: a scene's source text -> one plain-language sentence restating it.

AD-1/AD-2/AD-3: no file I/O, no credentials, no import of the `anthropic` SDK or any other
concrete adapter -- only the `ModelAdapter` port and typed request/result shapes from
`adapters.model`, plus pure string composition. Reads the scene's *source text* only --
never compiles, loads, or constructs it (AD-2, story 5's job instead); this is a restatement
of what the DSL text says, not a report on what actually rendered.

Story 6 (`validation-gauntlet.md` stage 5, "Semantic readback"): stages 1-4 all verify a
generated scene is well-formed; none catch a well-formed scene that is simply a misreading
of the request. This function mirrors `core/generation.py`'s `generate`/`revise` architecture
exactly -- same `_complete`-style defensive wrapping, same typed-result-never-raises
discipline -- applied to a different task.
"""

from __future__ import annotations

import math
from typing import Optional

from adapters.model import ModelAdapter, ModelError, ModelRequest
from core.scene_facts import SceneFacts, extract_scene_facts, scene_changes
from core.types import ReadbackError, ReadbackResult
from gauntlet._scala_text import strip_comments_and_strings

# The exact worked example validation-gauntlet.md:13 demonstrates -- the style the model's
# output must match: object + key parameters, material, light color/rough position, camera
# distance/angle, as one sentence (or a short handful of clauses), never raw DSL syntax and
# never a paragraph.
_STYLE_EXAMPLE = (
    "level-3 sponge, 2 units, glass, warm area light upper-left, camera 5 units out at 30°"
)

_SYSTEM_PROMPT = (
    "You restate a Menger DSL scene file's source text in plain language, for a human to "
    "sanity-check against what they actually asked for.\n"
    "Rules:\n"
    "- Output exactly one plain-language sentence, or at most a short handful of "
    "comma-separated clauses -- never a paragraph, never bullet points.\n"
    "- Name: the object type and its key parameters (e.g. level/size), the material, each "
    "light's color and rough position, and the camera's distance and angle.\n"
    f'- Match this exact style: "{_STYLE_EXAMPLE}"\n'
    "- Base the restatement only on what the scene text actually says -- never invent a "
    "detail, parameter, or interpretation the text itself doesn't support.\n"
    "- The <facts> block is ground truth, already extracted from the scene: it is "
    "authoritative over your own reading of the raw text. Never state a light direction, "
    "material, or transparency that contradicts it -- in particular, an object's material "
    "(e.g. Glass) keeps its own transparency regardless of any color also set on it.\n"
    "- Describe a Directional light by where it appears to shine FROM (the facts block "
    "already gives this as a phrase, e.g. \"from above\") -- never restate a raw direction "
    "vector, and never invert it into where the light travels TO.\n"
    "- Never restate the raw DSL syntax (field names, method calls, code structure) -- "
    "describe what it means, not how it's written.\n"
    "- If the scene text has no recognizable object, say so plainly in one sentence instead "
    "of guessing.\n"
    "- If <request> and <changes> blocks are present, lead with what this turn changed for "
    "the request, using the values in <changes> exactly (e.g. \"now level 3 and glass; "
    "...\"), then the rest of the scene. If <changes> says nothing changed, say so first.\n"
    "- State a camera distance only as the <facts> block gives it; never estimate one.\n"
    "- Do not mention warnings about unrequested changes; they are shown separately.\n"
    "- Respond with only the sentence -- no preamble, no code block, no markdown.\n"
)


_MAX_OBJECTS_FOR_PLACEMENTS = 6
_AXES = ("x", "y", "z")


def _relative_placements(facts: SceneFacts) -> list[str]:
    """Where each object sits relative to the ones before it in the source, along the dominant
    axis of their offset, +y being up (F45: "a 24-cell above the sponge" was placed along +x
    and read back as "above"; positions alone didn't stop that). Skipped for larger scenes,
    where the pairs would drown the other facts."""
    objects = sorted((o for o in facts.objects if o.pos is not None), key=lambda o: o.offset)
    if len(objects) > _MAX_OBJECTS_FOR_PLACEMENTS:
        return []
    lines = []
    for i, later in enumerate(objects):
        for earlier in objects[:i]:
            delta = [a - b for a, b in zip(later.pos, earlier.pos)]
            axis = max(range(3), key=lambda k: abs(delta[k]))
            d = delta[axis]
            if abs(d) < 1e-6:
                continue
            if axis == 1:
                relation = "above" if d > 0 else "below"
            elif axis == 0:
                relation = "beside"
            else:
                relation = "in front of" if d > 0 else "behind"
            note = "" if axis == 1 else ", not above it"
            lines.append(
                f"- the {later.type_name} is {relation} the {earlier.type_name} "
                f"({d:+.1f} along {_AXES[axis]}){note}"
            )
    return lines


def _format_facts(facts: SceneFacts) -> str:
    """Ground-truth facts as short bullet lines for the prompt -- not prose (that's the
    model's job), just the values it must not contradict."""
    lines: list[str] = []
    for obj in facts.objects:
        parts = [obj.type_name]
        if obj.pos is not None:
            parts.append(f"pos={obj.pos}")
        if obj.size is not None:
            parts.append(f"size={obj.size}")
        if obj.material is not None:
            parts.append(f"material={obj.material}")
        if obj.color is not None:
            parts.append(f"color={obj.color}")
        if obj.rotation is not None:
            parts.append(f"rotation={obj.rotation}")
        if obj.projection_text is not None:
            parts.append(f"4D projection={obj.projection_text}")
        parts.append("opaque" if obj.is_opaque else "transparent")
        lines.append("- " + ", ".join(parts))
    lines += _relative_placements(facts)
    for light in facts.lights:
        if light.type_name == "Directional":
            phrase = light.direction_phrase
            lines.append(f"- Directional light, shines {phrase}" if phrase else "- Directional light")
        elif light.position is not None:
            lines.append(f"- {light.type_name} light at pos={light.position}")
    if facts.camera is not None:
        position, look_at = facts.camera
        distance = math.dist(position, look_at)
        lines.append(
            f"- Camera at {position}, looking at {look_at}, {distance:.1f} units from its target"
        )
    return "\n".join(lines) if lines else "(no facts could be extracted from this scene)"


def _model_error_to_readback_error(error: ModelError) -> ReadbackError:
    kind = "model_call_failed" if error.kind == "call_failed" else "invalid_model_output"
    return ReadbackError(kind=kind, message=error.message, cause=error.cause)


def _complete(system_prompt: str, user_prompt: str, adapter: ModelAdapter) -> ReadbackResult:
    # semantic_readback() must never raise (its own contract, mirroring GenerationResult's
    # docstring in core/types.py) -- this holds regardless of whether a given ModelAdapter
    # implementation honors its own "never raise" contract, so the core defends its own
    # guarantee rather than trusting every adapter to defend it independently. That includes
    # an adapter that doesn't honor `ModelResult`'s own `Union[str, ModelError]` contract and
    # returns some third thing -- `.strip()` on a non-`str` would raise, so the type is
    # checked before it's used, not assumed (review round).
    try:
        result = adapter.complete(ModelRequest(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception as e:  # noqa: BLE001 -- adapter misbehavior must not escape core
        return ReadbackError(kind="model_call_failed", message=f"Model adapter raised: {e}", cause=e)
    if isinstance(result, ModelError):
        return _model_error_to_readback_error(result)
    if not isinstance(result, str):
        return ReadbackError(
            kind="invalid_model_output",
            message=f"Model adapter returned {type(result).__name__}, expected str or ModelError",
        )
    text = result.strip()
    if not text:
        return ReadbackError(
            kind="invalid_model_output", message="Model returned an empty response"
        )
    return text


def semantic_readback(
    scene_text: str,
    adapter: ModelAdapter,
    request: Optional[str] = None,
    prior_scene: Optional[str] = None,
) -> ReadbackResult:
    """Stage 5 of the validation gauntlet: restate what `scene_text` describes in one
    plain-language sentence, so a misreading of the original request surfaces here rather
    than after a render (`validation-gauntlet.md`'s own framing). `scene_text` goes verbatim
    into the user turn, delimited by an XML-style tag rather than a markdown code fence
    (review round): a triple-backtick sequence *inside* `scene_text` itself would otherwise
    prematurely close a ` ```scala ` fence, splicing the remainder of the scene text out of
    the quoted block and into free-standing prompt content. `</scene>` is exceedingly
    unlikely to appear inside real DSL source and, unlike a fence, has no shorter prefix that
    could partially collide.

    Usability review 2026-09 (F7): the model's own reading of raw DSL text had misread light
    direction and material transparency (F14's opaque-Glass bug) even when the text was
    right there -- a `<facts>` block, extracted the same static way `gauntlet/lint.py`'s
    checks are (AD-2: no compiler), gives it ground truth to check its restatement against
    instead of re-deriving it from syntax every time.

    Usability review 2026-10, session 3 (F62): readbacks omitted the change just made, so
    `request` and the computed `scene_changes` against `prior_scene` go in as `<request>` and
    `<changes>` blocks for the summary to lead with. The turn's warnings no longer go in
    (F80a: the CLI prints them as their own lines, so they appeared twice)."""
    facts = extract_scene_facts(strip_comments_and_strings(scene_text))
    turn_block = ""
    if request is not None and prior_scene is not None:
        prior_facts = extract_scene_facts(strip_comments_and_strings(prior_scene))
        changes = scene_changes(prior_facts, facts)
        listed = "\n".join(f"- {c}" for c in changes) if changes else "- nothing the facts cover changed"
        turn_block = f"<request>\n{request}\n</request>\n\n<changes>\n{listed}\n</changes>\n\n"
    user_prompt = (
        "Here is the scene file's source text, verbatim, between <scene> tags, and "
        "ground-truth facts already extracted from it between <facts> tags (see the system "
        "prompt's rule on how to use them):\n\n"
        f"<scene>\n{scene_text}\n</scene>\n\n"
        f"<facts>\n{_format_facts(facts)}\n</facts>\n\n"
        f"{turn_block}"
        "Restate what this scene describes, following the system prompt's rules exactly."
    )
    return _complete(_SYSTEM_PROMPT, user_prompt, adapter)

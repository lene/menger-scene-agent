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

from adapters.model import ModelAdapter, ModelError, ModelRequest
from core.types import ReadbackError, ReadbackResult

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
    "- Never restate the raw DSL syntax (field names, method calls, code structure) -- "
    "describe what it means, not how it's written.\n"
    "- If the scene text has no recognizable object, say so plainly in one sentence instead "
    "of guessing.\n"
    "- Respond with only the sentence -- no preamble, no code block, no markdown.\n"
)


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


def semantic_readback(scene_text: str, adapter: ModelAdapter) -> ReadbackResult:
    """Stage 5 of the validation gauntlet: restate what `scene_text` describes in one
    plain-language sentence, so a misreading of the original request surfaces here rather
    than after a render (`validation-gauntlet.md`'s own framing). `scene_text` goes verbatim
    into the user turn, delimited by an XML-style tag rather than a markdown code fence
    (review round): a triple-backtick sequence *inside* `scene_text` itself would otherwise
    prematurely close a ` ```scala ` fence, splicing the remainder of the scene text out of
    the quoted block and into free-standing prompt content. `</scene>` is exceedingly
    unlikely to appear inside real DSL source and, unlike a fence, has no shorter prefix that
    could partially collide."""
    user_prompt = (
        "Here is the scene file's source text, verbatim, between <scene> tags:\n\n"
        f"<scene>\n{scene_text}\n</scene>\n\n"
        "Restate what this scene describes, following the system prompt's rules exactly."
    )
    return _complete(_SYSTEM_PROMPT, user_prompt, adapter)

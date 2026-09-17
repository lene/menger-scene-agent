"""Pure derivation: a domain question (+ optional current scene) + manifest + corpus ->
prose answer.

AD-1/AD-2/AD-3: no file I/O, no credentials, no import of the `anthropic` SDK or any other
concrete adapter -- only the `ModelAdapter` port and typed request/result shapes from
`adapters.model`, plus pure string/dict composition. Never writes a scene file, never stages
a candidate, never invokes any `gauntlet.check_*()` stage (Boundaries & Constraints) -- a
consult turn answers in prose only, it never touches the scene.

spec-ai-scene-agent story 19 (PRD FR10): `/ask`/`/question`-prefixed input must be answered
in prose, using the agent's domain knowledge, without ever touching the scene file. Story 11
already built the recognition and dispatch (`cli.py`'s `_is_consult_input()`) but only
printed a stub -- this module is the real Q&A logic the stub named. Mirrors
`core/readback.py`'s exact shape (same typed-result-never-raises `_complete()` pattern), and
`core/generation.py`'s manifest/corpus-grounded system-prompt style (`_build_system_prompt`)
-- domain knowledge grounded the same way `generate()`/`revise()` already ground composition.
"""

from __future__ import annotations

import json
from typing import Optional

from adapters.model import ModelAdapter, ModelError, ModelRequest
from core.types import ConsultError, ConsultResult

_RULES = (
    "You are the Menger AI scene agent's domain-knowledge assistant. A user has prefixed "
    "their input with /ask or /question, expecting a prose answer to a question -- not a "
    "scene edit or a new scene.\n"
    "Rules:\n"
    "- Answer using only the domain knowledge in the DSL capability manifest and example "
    "scene corpus below, plus the current scene's source text when one is provided below "
    "-- never invent a DSL type, field, method, or preset name that isn't listed there.\n"
    "- When a current scene is provided, you may reference its content in your answer (e.g. "
    "\"where should I position the light\" implies awareness of what's already there); when "
    "none is provided, answer from the manifest/corpus alone.\n"
    "- Respond in prose only -- no code block, no scene file, no raw DSL syntax as the "
    "answer itself.\n"
    "- Keep the answer focused -- a few sentences, not an essay.\n"
)


def _build_system_prompt(manifest: dict, corpus: dict) -> str:
    # Mirrors `core/generation.py`'s `_build_system_prompt` exactly (Code Map): the same
    # manifest/corpus-grounded system prompt style `generate()`/`revise()` already use,
    # applied to Q&A instead of composition.
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


def _model_error_to_consult_error(error: ModelError) -> ConsultError:
    # Review round, patch-level fix: a timeout is its own distinct kind, mirroring
    # core/generation.py's story-15 precedent (`_model_error_to_generation_error`) --
    # without this, a hung model-provider request was indistinguishable from empty/garbage
    # model output, both on screen and in history.jsonl.
    if error.kind == "timeout":
        kind = "model_call_timeout"
    elif error.kind == "call_failed":
        kind = "model_call_failed"
    else:
        kind = "invalid_model_output"
    return ConsultError(kind=kind, message=error.message, cause=error.cause)


def _complete(system_prompt: str, user_prompt: str, adapter: ModelAdapter) -> ConsultResult:
    # answer_consult() must never raise (its own contract, mirroring GenerationResult's/
    # ReadbackResult's docstrings in core/types.py) -- this holds regardless of whether a
    # given ModelAdapter implementation honors its own "never raise" contract, so the core
    # defends its own guarantee rather than trusting every adapter to defend it
    # independently. That includes an adapter that doesn't honor `ModelResult`'s own
    # `Union[str, ModelError]` contract and returns some third thing -- `.strip()` on a
    # non-`str` would raise, so the type is checked before it's used, not assumed (copied
    # verbatim from `core/readback.py:53`'s defensive wrapping, per Code Map).
    try:
        result = adapter.complete(ModelRequest(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception as e:  # noqa: BLE001 -- adapter misbehavior must not escape core
        return ConsultError(kind="model_call_failed", message=f"Model adapter raised: {e}", cause=e)
    if isinstance(result, ModelError):
        return _model_error_to_consult_error(result)
    if not isinstance(result, str):
        return ConsultError(
            kind="invalid_model_output",
            message=f"Model adapter returned {type(result).__name__}, expected str or ModelError",
        )
    text = result.strip()
    if not text:
        return ConsultError(
            kind="invalid_model_output", message="Model returned an empty response"
        )
    return text


def answer_consult(
    question: str,
    manifest: dict,
    corpus: dict,
    adapter: ModelAdapter,
    prior_scene: Optional[str] = None,
) -> ConsultResult:
    """A consult turn (spec-ai-scene-agent story 19, PRD FR10): answers `question` in prose,
    grounded in the DSL capability manifest/example corpus and, when `prior_scene` is not
    `None`, the current scene's source text (UJ-3's "where should I position the light"
    implies scene awareness). Makes exactly one model call, prose-in/prose-out -- never
    calls `generate()`/`revise()`, never stages a candidate, never runs any
    `gauntlet.check_*()` stage. Never raises for an expected outcome.

    `prior_scene`, when given, is embedded verbatim in the user prompt between `<scene>`
    tags rather than a markdown code fence -- the same choice `core/readback.py:85` makes
    and for the same reason: a triple-backtick sequence *inside* the scene text would
    otherwise prematurely close a fence, splicing the remainder out of the quoted block."""
    system_prompt = _build_system_prompt(manifest, corpus)
    if prior_scene is not None:
        user_prompt = (
            "Here is the current scene file's source text, verbatim, between <scene> "
            f"tags:\n\n<scene>\n{prior_scene}\n</scene>\n\n"
            f"Question: {question}\n\n"
            "Answer the question, following the system prompt's rules exactly. You may "
            "reference the current scene above where relevant."
        )
    else:
        user_prompt = (
            f"Question: {question}\n\n"
            "Answer the question, following the system prompt's rules exactly. There is no "
            "current scene in this session yet -- answer from the manifest/corpus alone."
        )
    return _complete(system_prompt, user_prompt, adapter)

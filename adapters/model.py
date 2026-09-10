"""Model adapter port, shared error vocabulary, and the Anthropic implementation.

AD-1: each concrete adapter's API credential lives ONLY in that adapter (this file's
`AnthropicModelAdapter`, and its siblings `adapters/gemini_model.py`,
`adapters/openai_compatible_model.py`), read from that provider's own environment variable,
never hardcoded, never logged. `ModelAdapter` is the port `core/` depends on (a `Protocol`)
-- `core/` imports this module for the port and the typed request/result shapes, but never
imports any concrete adapter or vendor SDK, so swapping or adding providers is an adapter
swap, not a core rewrite (spike criterion 4, "bring your own model" -- see
`adapters/model_factory.py` for the first place that promise is actually exercised).

`MissingAPIKeyError` and `UnknownProviderError` live here, not in each adapter file, so
every adapter across every provider raises the same two exception types for the same two
failure shapes -- a caller (the factory, or anything else) can catch one type regardless of
which provider is in play.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Union

ModelErrorKind = Literal["call_failed", "invalid_output"]


@dataclass(frozen=True)
class ModelError:
    """A typed failure from a model call, or from extracting a scene out of its response --
    never a bare exception escaping the adapter (I/O & Edge-Case Matrix)."""

    kind: ModelErrorKind
    message: str
    cause: Optional[BaseException] = None


@dataclass(frozen=True)
class ModelRequest:
    """Everything the core has already composed for one model turn. `system_prompt` carries
    the manifest + corpus context and generation rules; `user_prompt` carries the plain-
    language request and, for a revision, the prior scene's full text -- string composition
    is entirely core's job, so the adapter never needs to know about manifests or corpora."""

    system_prompt: str
    user_prompt: str


ModelResult = Union[str, ModelError]


class ModelAdapter(Protocol):
    """The port the agent core depends on. No concrete adapter's SDK types appear on this
    boundary -- only plain strings and the typed result above."""

    def complete(self, request: ModelRequest) -> ModelResult: ...


class MissingAPIKeyError(RuntimeError):
    """Raised at any concrete `ModelAdapter`'s construction when its required API-key
    environment variable is unset -- fails fast, before any network call (I/O & Edge-Case
    Matrix: missing API key). Shared across every adapter (`AnthropicModelAdapter` below,
    `GeminiModelAdapter`, `OpenAICompatibleModelAdapter`) rather than one class per provider,
    so a caller that only cares "was a key missing" can catch this one type."""


class UnknownProviderError(RuntimeError):
    """Raised for a bad/unrecognized provider name -- a programmer error (wrong string
    literal at a call site), distinct from `MissingAPIKeyError`'s environment error, even
    though both are construction-time `RuntimeError`s. Raised by `adapters.model_factory`
    for an unrecognized top-level provider selection, and by `OpenAICompatibleModelAdapter`
    for an unrecognized `provider=` argument (it serves three vendors from one class)."""


# Matches a fenced code block, tagged with any word (scala, scala3, Scala, ...) or untagged.
# `re.DOTALL` so `.` spans newlines. The exact tag doesn't matter -- content is what's checked.
_CODE_FENCE = re.compile(r"```[ \t]*\w*[ \t]*\n(.*?)```", re.DOTALL)

# A candidate scene's top-level `object` must appear near the top of the file (mirrors
# SceneLoader.detectObjectName's first-60-lines heuristic) -- a bare "object " substring
# anywhere in a prose response is not enough to call it scene text.
_OBJECT_DECLARATION = re.compile(r"^\s*(?:private\s+)?object\s+\w", re.MULTILINE)
_FIRST_OBJECT_LINE_LIMIT = 60

# Per the claude-api skill's current defaults: claude-opus-5 unless a caller names a
# different model (constructor argument), and a non-streaming max_tokens of 16000 --
# generated scene files are 30-250 lines, well under that ceiling.
DEFAULT_MODEL = "claude-opus-5"
_MAX_TOKENS = 16000
# Server-side fallback: recommended by default for claude-opus-5 (claude-api skill) so a
# transient refusal/overload on the primary model doesn't surface as a bare call failure.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicModelAdapter:
    """The one concrete `ModelAdapter`, backed by the real Anthropic API (`anthropic` SDK).
    This is the only place in the repo that reads the API key or imports the SDK (AD-1).

    `complete()` returns the model's raw response text verbatim (stripped, non-empty), or a
    typed `ModelError` for an actual call/refusal/truncation/parse failure -- it does NOT
    apply `extract_scene_text` (review round, story 6): this adapter is a generic
    `ModelAdapter` port implementation, shared by every caller (`core/generation.py`'s
    scene-file generation, `core/readback.py`'s plain-language summary, and any future
    caller), and scene-specific extraction is meaningful only to the caller that actually
    wants scene text. An earlier version called `extract_scene_text` unconditionally here,
    which made every successful, correctly-behaving `semantic_readback()` call fail with
    `invalid_output` in production -- a real plain-language sentence has no code fence and no
    top-level `object` declaration, so it always failed `extract_scene_text`'s own checks.
    `core/generation.py`'s `generate()`/`revise()` now call `extract_scene_text` themselves,
    on the raw text this method returns."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> None:
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not set -- the Anthropic model adapter refuses to "
                "construct without it, before making any network call."
            )
        import anthropic  # imported lazily: keeps the SDK out of every module that merely

        # imports this file's type annotations without ever constructing this adapter.
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def complete(self, request: ModelRequest) -> ModelResult:
        import anthropic

        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=request.system_prompt,
                messages=[{"role": "user", "content": request.user_prompt}],
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIConnectionError as e:
            return ModelError(
                kind="call_failed", message=f"Network error calling Anthropic API: {e}", cause=e
            )
        except anthropic.RateLimitError as e:
            return ModelError(
                kind="call_failed", message=f"Anthropic API rate limited: {e}", cause=e
            )
        except anthropic.APIStatusError as e:
            return ModelError(
                kind="call_failed", message=f"Anthropic API returned an error: {e}", cause=e
            )
        except Exception as e:  # noqa: BLE001 -- never let a raw exception escape the adapter
            return ModelError(kind="call_failed", message=f"Anthropic API call failed: {e}", cause=e)

        try:
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "refusal":
                return ModelError(
                    kind="invalid_output",
                    message=f"Model refused the request: {getattr(response, 'stop_details', None)}",
                )
            if stop_reason == "max_tokens":
                return ModelError(
                    kind="invalid_output",
                    message=f"Model response was truncated at {_MAX_TOKENS} tokens "
                    "before completing -- not usable scene text",
                )

            text_blocks = [
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ]
            raw_text = "\n".join(text_blocks).strip()
        except Exception as e:  # noqa: BLE001 -- malformed/unexpected response shape
            return ModelError(
                kind="invalid_output", message=f"Could not parse the model response: {e}", cause=e
            )

        if not raw_text:
            return ModelError(kind="invalid_output", message="Model returned an empty response")
        return raw_text


def extract_scene_text(raw_text: str) -> ModelResult:
    """Extract a single scene file's text from a raw model response. Fails typed (never a
    best-effort guess, per the Ask First / I/O & Edge-Case Matrix) on an empty response, a
    response with no code and no `object` declaration, or one containing more than one
    candidate code block -- exactly the "single, straightforward extraction pass" the
    story's Ask First boundary allows."""
    if not raw_text:
        return ModelError(kind="invalid_output", message="Model returned an empty response")

    fences = _CODE_FENCE.findall(raw_text)
    if len(fences) > 1:
        return ModelError(
            kind="invalid_output",
            message=f"Model returned {len(fences)} candidate code blocks, expected exactly one",
        )
    if len(fences) == 1:
        candidate = fences[0].strip()
    elif "```" in raw_text:
        # A fence marker is present but didn't match _CODE_FENCE (malformed fencing) --
        # never fall back to treating raw markdown syntax as scene text.
        return ModelError(
            kind="invalid_output",
            message="Model response contains malformed code fencing that could not be parsed",
        )
    else:
        candidate = raw_text

    match = _OBJECT_DECLARATION.search(candidate)
    if match is None:
        return ModelError(
            kind="invalid_output",
            message="Model response has no top-level 'object' declaration -- not scene text",
        )
    line_number = candidate.count("\n", 0, match.start())
    if line_number >= _FIRST_OBJECT_LINE_LIMIT:
        return ModelError(
            kind="invalid_output",
            message=f"Model response's 'object' declaration is past line "
            f"{_FIRST_OBJECT_LINE_LIMIT} -- SceneLoader.detectObjectName would miss it",
        )
    return candidate

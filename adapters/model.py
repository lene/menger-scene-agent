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

import math
import os
from typing import Optional

# The port, its request/result types and scene extraction live in core.ports (msa#34); they are
# imported here so `from adapters.model import ...` keeps working for the adapters and tests.
from core.ports import (  # noqa: F401 -- re-exported
    ModelAdapter as ModelAdapter,
)
from core.ports import (
    ModelError as ModelError,
)
from core.ports import (
    ModelErrorKind as ModelErrorKind,
)
from core.ports import (
    ModelRequest as ModelRequest,
)
from core.ports import (
    ModelResult as ModelResult,
)
from core.ports import (
    extract_scene_text as extract_scene_text,
)


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

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: Optional[float] = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not set -- the Anthropic model adapter refuses to "
                "construct without it, before making any network call."
            )
        # A programmer-error precondition, not a modeled runtime outcome (mirrors
        # `adapters/render_window.py`'s `grace_period <= 0` check): a zero, negative, NaN, or
        # infinite timeout would otherwise be forwarded straight to the SDK client unchecked.
        if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
            raise ValueError(f"timeout must be a finite, positive number, got {timeout!r}")
        import anthropic  # imported lazily: keeps the SDK out of every module that merely

        # imports this file's type annotations without ever constructing this adapter.
        # `timeout` is only forwarded when explicitly set -- passing `timeout=None` to the
        # SDK client disables its own default timeout rather than leaving it alone, which
        # would change today's behavior for every caller that doesn't set it (Boundaries &
        # Constraints: "No adapter's default behavior changes when timeout is left unset").
        client_kwargs: dict[str, object] = {"api_key": key}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = anthropic.Anthropic(**client_kwargs)
        self._model = model

    @property
    def model_name(self) -> str:
        """The model this adapter calls, for the CLI's startup line."""
        return self._model

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
        except anthropic.APITimeoutError as e:
            # Must come before `APIConnectionError` below -- `APITimeoutError` is a subclass
            # of it, so the broader except would otherwise shadow this one and every timeout
            # would be misreported as a generic network error.
            return ModelError(
                kind="timeout", message=f"Anthropic API call timed out: {e}", cause=e
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

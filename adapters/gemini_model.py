"""The Gemini `ModelAdapter` implementation -- provider #2 behind the same port
`adapters/model.py` defines.

AD-1 (generalized, see `adapters/model.py`'s `MissingAPIKeyError`): the Gemini API credential
lives ONLY here, read from the `GEMINI_API_KEY` environment variable, never hardcoded, never
logged. `core/` never imports this module -- it depends only on `adapters.model.ModelAdapter`
(the `Protocol`), so this file existing at all doesn't change anything about `core/`.

Native adapter, not a shim over the OpenAI-compatible family in
`adapters/openai_compatible_model.py`, even though Google also exposes an OpenAI-compatible
endpoint for Gemini: that compatibility layer flattens Gemini's own finish-reason and
safety-block vocabulary (`STOP`/`MAX_TOKENS`/`SAFETY`/`RECITATION`/... , plus a separate
prompt-level `prompt_feedback.block_reason`) down into OpenAI's narrower `finish_reason` set,
which would lose the distinction between "the model refused mid-generation" and "the prompt
itself was blocked before generation started." Mapping the native shapes directly keeps that
distinction in the `ModelError` message.

Uses the `google-genai` SDK (`from google import genai`) -- the current unified Gemini/Vertex
SDK, not the deprecated `google-generativeai` package.
"""

from __future__ import annotations

import math
import os
from typing import Optional

from adapters.model import MissingAPIKeyError, ModelRequest, ModelResult, ModelError

# Current-generation default; a caller can override via the `model` constructor argument.
# gemini-2.5-pro was retired for new API keys (404 NOT_FOUND, "no longer available to new
# users") -- the error response itself named gemini-3.1-pro-preview as the replacement.
DEFAULT_MODEL = "gemini-3.1-pro-preview"
_MAX_TOKENS = 16000

# The `FinishReason` values that mean "the model did not produce usable output," checked
# explicitly below -- deliberately as narrow a set as Anthropic's own adapter checks
# (`stop_reason == "refusal"` / `"max_tokens"`, nothing else): any other/future value falls
# through to text extraction, matching Anthropic's own permissiveness rather than inventing a
# stricter policy for this one provider. `MAX_TOKENS` gets its own branch (truncation, not
# refusal) so its message matches Anthropic's truncation message shape.
_REFUSAL_FINISH_REASONS = frozenset({"SAFETY", "RECITATION"})


class GeminiModelAdapter:
    """The Gemini `ModelAdapter`, backed by the real Gemini API (`google-genai` SDK). This is
    the only place in the repo that reads `GEMINI_API_KEY` or imports `google.genai` (AD-1).

    `complete()` returns the model's raw response text verbatim (stripped, non-empty), or a
    typed `ModelError` -- it does NOT apply `extract_scene_text` (see `adapters/model.py`'s
    `AnthropicModelAdapter` docstring for the story-6 bug this would reintroduce: an earlier
    version of that adapter applied scene-extraction inside `complete()`, which made every
    correctly-behaving `semantic_readback()` call fail, since a plain-language summary has no
    code fence and no top-level `object` declaration). `core/generation.py`'s `generate()`/
    `revise()` apply `extract_scene_text` themselves, on the raw text this method returns.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: Optional[float] = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not set -- the Gemini model adapter refuses to construct "
                "without it, before making any network call."
            )
        # A programmer-error precondition, not a modeled runtime outcome (mirrors
        # `adapters/render_window.py`'s `grace_period <= 0` check): a zero, negative, NaN, or
        # infinite timeout would otherwise be forwarded straight to the SDK client unchecked.
        if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
            raise ValueError(f"timeout must be a finite, positive number, got {timeout!r}")
        from google import genai  # imported lazily: keeps the SDK out of every module that
        from google.genai import types

        # merely imports this file's type annotations without ever constructing this adapter.
        # `timeout` is only forwarded when explicitly set -- leaving `http_options` unset
        # preserves the SDK's own default timeout exactly (see
        # `adapters/model.py`'s `AnthropicModelAdapter.__init__` for the same reasoning).
        # `HttpOptions.timeout` is in milliseconds per the SDK's own convention (verified
        # against the installed `google-genai` package), while this adapter's own `timeout`
        # parameter is in seconds, matching the other two adapters' convention. Rounded
        # (not truncated) to the nearest millisecond, with a 1ms floor -- `int(timeout * 1000)`
        # would truncate (0.1005s -> 100ms instead of 101ms) and could round a tiny-but-positive
        # timeout (e.g. 0.0003s) down to 0ms, a degenerate timeout the caller never asked for.
        client_kwargs: dict[str, object] = {"api_key": key}
        if timeout is not None:
            client_kwargs["http_options"] = types.HttpOptions(
                timeout=max(1, round(timeout * 1000))
            )
        self._client = genai.Client(**client_kwargs)
        self._model = model

    def complete(self, request: ModelRequest) -> ModelResult:
        import httpx  # the google-genai SDK's default sync HTTP client is an httpx.Client
        from google.genai import errors, types

        # subclass (confirmed against the installed package: `SyncHttpxClient(httpx.Client)`)
        # and re-raises the underlying transport exception verbatim after exhausting its own
        # retries on a timeout -- `google.genai.errors` has no dedicated timeout exception
        # type (confirmed: no `Timeout`/`TimeoutError` member), so the raw `httpx` exception
        # (plus the builtin `TimeoutError`, for any code path that raises that instead) is
        # what must be caught here, before the generic `except Exception` below would
        # otherwise swallow it into an indistinguishable `call_failed`.
        #
        # Caveat worth keeping visible in the code, not just in a chat transcript: this
        # detection depends on google-genai's internal transport choice (an undocumented
        # implementation detail, not a public contract), confirmed only against the version
        # of the SDK installed when this was written. A future SDK upgrade that swaps its
        # default sync transport away from httpx would silently break this except clause --
        # a genuine timeout would then fall through to the generic `except Exception` below
        # and be misreported as `call_failed` instead of `timeout`, with nothing catching the
        # regression until that upgrade actually happens.
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=request.user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=request.system_prompt,
                    max_output_tokens=_MAX_TOKENS,
                ),
            )
        except (TimeoutError, httpx.TimeoutException) as e:
            return ModelError(kind="timeout", message=f"Gemini API call timed out: {e}", cause=e)
        except errors.APIError as e:
            return ModelError(kind="call_failed", message=f"Gemini API returned an error: {e}", cause=e)
        except Exception as e:  # noqa: BLE001 -- never let a raw exception escape the adapter
            return ModelError(kind="call_failed", message=f"Gemini API call failed: {e}", cause=e)

        try:
            # Prompt-level block: no candidate was ever generated at all. Has no Anthropic
            # analogue -- Gemini's own shape, checked before looking at any candidate.
            prompt_feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason is not None and str(block_reason) != "BLOCKED_REASON_UNSPECIFIED":
                return ModelError(
                    kind="invalid_output", message=f"Prompt was blocked before generation: {block_reason}"
                )

            candidates = getattr(response, "candidates", None) or []
            finish_reason = (
                getattr(candidates[0], "finish_reason", None) if candidates else None
            )
            finish_reason_name = str(finish_reason) if finish_reason is not None else None

            if finish_reason_name == "MAX_TOKENS":
                return ModelError(
                    kind="invalid_output",
                    message=f"Model response was truncated at {_MAX_TOKENS} tokens "
                    "before completing -- not usable scene text",
                )
            if finish_reason_name in _REFUSAL_FINISH_REASONS:
                return ModelError(
                    kind="invalid_output",
                    message=f"Model refused the request: finish_reason={finish_reason_name}",
                )

            # `.text` concatenates all text parts of the first candidate; returns None rather
            # than raising for an empty/malformed response (confirmed against the installed
            # google-genai SDK), so no exception is expected here for that case specifically --
            # the try/except below is still the same defensive "malformed shape" guard the
            # Anthropic adapter uses, for any other unexpected attribute-access failure.
            raw_text = (response.text or "").strip()
        except Exception as e:  # noqa: BLE001 -- malformed/unexpected response shape
            return ModelError(
                kind="invalid_output", message=f"Could not parse the model response: {e}", cause=e
            )

        if not raw_text:
            return ModelError(kind="invalid_output", message="Model returned an empty response")
        return raw_text

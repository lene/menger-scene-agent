"""One `ModelAdapter` implementation shared by every OpenAI-chat-completions-compatible
vendor -- DeepSeek, OpenAI/ChatGPT itself, and Kimi (Moonshot AI) all document themselves as
compatible with the OpenAI request/response shape and `finish_reason` vocabulary (`stop`/
`length`/`content_filter`/`tool_calls`), so one class parametrized by `base_url` + an
env-var name + a default model id serves all three -- a fourth OpenAI-compatible vendor is a
new `_ProviderSpec` table entry, not a new class (see `adapters/model_factory.py` for where
that table gets wired into provider selection).

AD-1 (generalized, see `adapters/model.py`'s `MissingAPIKeyError`): each vendor's credential
lives ONLY here, read from that vendor's own environment variable, never hardcoded, never
logged. `core/` never imports this module -- it depends only on `adapters.model.ModelAdapter`.

Uses the official `openai` Python SDK pointed at each vendor's own `base_url`, which is the
integration path each of these vendors documents for third-party OpenAI-SDK compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from adapters.model import MissingAPIKeyError, ModelRequest, ModelResult, ModelError, UnknownProviderError

_MAX_TOKENS = 16000


@dataclass(frozen=True)
class _ProviderSpec:
    name: str
    base_url: str
    api_key_env: str
    default_model: str


# Default model ids are each vendor's current flagship chat model as of this writing --
# confirm against that vendor's own docs before relying on them long-term; a caller can
# always override via the `model` constructor argument regardless.
_PROVIDERS: dict[str, _ProviderSpec] = {
    "deepseek": _ProviderSpec("DeepSeek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "openai": _ProviderSpec("OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5"),
    "kimi": _ProviderSpec("Kimi/Moonshot", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "kimi-k2"),
}


class OpenAICompatibleModelAdapter:
    """The `ModelAdapter` shared by every OpenAI-compatible vendor in `_PROVIDERS`. This is
    the only place in the repo that reads any of these vendors' API keys or imports `openai`
    (AD-1).

    `complete()` returns the model's raw response text verbatim (stripped, non-empty), or a
    typed `ModelError` -- it does NOT apply `extract_scene_text` (see
    `adapters/model.py`'s `AnthropicModelAdapter` docstring for the story-6 bug this would
    reintroduce). `core/generation.py`'s `generate()`/`revise()` apply `extract_scene_text`
    themselves, on the raw text this method returns.
    """

    def __init__(
        self, provider: str, api_key: Optional[str] = None, model: Optional[str] = None
    ) -> None:
        spec = _PROVIDERS.get(provider)
        if spec is None:
            raise UnknownProviderError(
                f"Unknown OpenAI-compatible provider '{provider}' -- valid values: "
                f"{', '.join(sorted(_PROVIDERS))}"
            )
        key = api_key if api_key is not None else os.environ.get(spec.api_key_env)
        if not key:
            raise MissingAPIKeyError(
                f"{spec.api_key_env} is not set -- the {spec.name} model adapter refuses to "
                "construct without it, before making any network call."
            )
        import openai  # imported lazily: keeps the SDK out of every module that merely

        # imports this file's type annotations without ever constructing this adapter.
        self._client = openai.OpenAI(api_key=key, base_url=spec.base_url)
        self._model = model or spec.default_model
        self._provider_name = spec.name

    def complete(self, request: ModelRequest) -> ModelResult:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
            )
        except openai.APIConnectionError as e:
            return ModelError(
                kind="call_failed",
                message=f"Network error calling {self._provider_name} API: {e}",
                cause=e,
            )
        except openai.RateLimitError as e:
            return ModelError(
                kind="call_failed", message=f"{self._provider_name} API rate limited: {e}", cause=e
            )
        except openai.APIStatusError as e:
            return ModelError(
                kind="call_failed",
                message=f"{self._provider_name} API returned an error: {e}",
                cause=e,
            )
        except Exception as e:  # noqa: BLE001 -- never let a raw exception escape the adapter
            return ModelError(
                kind="call_failed", message=f"{self._provider_name} API call failed: {e}", cause=e
            )

        try:
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            if finish_reason == "content_filter":
                return ModelError(
                    kind="invalid_output",
                    message=f"Model refused the request: finish_reason={finish_reason}",
                )
            if finish_reason == "length":
                return ModelError(
                    kind="invalid_output",
                    message=f"Model response was truncated at {_MAX_TOKENS} tokens "
                    "before completing -- not usable scene text",
                )
            if finish_reason == "tool_calls":
                # This adapter never sends `tools=`, so a tool-call finish reason is an
                # unexpected response shape, not a case to silently accept.
                return ModelError(
                    kind="invalid_output",
                    message="Model returned a tool call, but no tools were offered",
                )

            raw_text = (choice.message.content or "").strip()
        except Exception as e:  # noqa: BLE001 -- malformed/unexpected response shape
            return ModelError(
                kind="invalid_output", message=f"Could not parse the model response: {e}", cause=e
            )

        if not raw_text:
            return ModelError(kind="invalid_output", message="Model returned an empty response")
        return raw_text

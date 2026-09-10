"""The first place provider selection has ever existed in this repo (AD-1, spike criterion 4
"bring your own model"). Until now `AnthropicModelAdapter` was simply the only option -- every
call site constructed it directly. `get_model_adapter()` is what that promise has actually
meant since story 3, made concrete: pick a provider by explicit argument or the
`MENGER_AGENT_MODEL_PROVIDER` environment variable, defaulting to Anthropic so every existing
`AnthropicModelAdapter()` call site keeps working unchanged.

Each provider's submodule is imported *inside* its builder function below, not at this
module's top level -- importing `adapters.model_factory` alone must never import
`google.genai` or `openai`, or the whole point of those SDKs being optional
(`pyproject.toml`'s `gemini`/`openai-compatible` extras) breaks transitively the moment
anything imports this factory module, even without ever selecting that provider.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

from adapters.model import ModelAdapter, UnknownProviderError


def _build_anthropic() -> ModelAdapter:
    from adapters.model import AnthropicModelAdapter

    return AnthropicModelAdapter()


def _build_gemini() -> ModelAdapter:
    from adapters.gemini_model import GeminiModelAdapter

    return GeminiModelAdapter()


def _build_deepseek() -> ModelAdapter:
    from adapters.openai_compatible_model import OpenAICompatibleModelAdapter

    return OpenAICompatibleModelAdapter(provider="deepseek")


def _build_openai() -> ModelAdapter:
    from adapters.openai_compatible_model import OpenAICompatibleModelAdapter

    return OpenAICompatibleModelAdapter(provider="openai")


def _build_kimi() -> ModelAdapter:
    from adapters.openai_compatible_model import OpenAICompatibleModelAdapter

    return OpenAICompatibleModelAdapter(provider="kimi")


_FACTORIES: Dict[str, Callable[[], ModelAdapter]] = {
    "anthropic": _build_anthropic,
    "gemini": _build_gemini,
    "deepseek": _build_deepseek,
    "openai": _build_openai,
    "kimi": _build_kimi,
}

# Preserves today's zero-config behavior exactly: a caller with no argument and no env var
# set gets Anthropic, same as every existing `AnthropicModelAdapter()` call site.
DEFAULT_PROVIDER = "anthropic"
_ENV_VAR = "MENGER_AGENT_MODEL_PROVIDER"


def get_model_adapter(provider: Optional[str] = None) -> ModelAdapter:
    """Resolves and constructs a `ModelAdapter` for the selected provider.

    Precedence (mirrors `AnthropicModelAdapter.__init__`'s own arg-beats-env-var shape, so
    this doesn't invent a new precedence rule): an explicit `provider` argument wins outright
    if given, regardless of the environment variable. Otherwise `MENGER_AGENT_MODEL_PROVIDER`
    is used if set to a non-empty value (case-insensitive, whitespace-stripped). Otherwise
    `DEFAULT_PROVIDER`.

    Raises `UnknownProviderError` immediately -- before any SDK import or network call -- if
    the resolved provider name isn't recognized; never silently falls back to the default for
    a bad explicit selection. Raises `MissingAPIKeyError` (unchanged, propagated from the
    provider's own builder) if the resolved provider's API key isn't set; not caught or
    wrapped here, so callers can catch one exception type regardless of provider.
    """
    resolved = provider if provider is not None else os.environ.get(_ENV_VAR)
    resolved = (resolved or DEFAULT_PROVIDER).strip().lower()

    build = _FACTORIES.get(resolved)
    if build is None:
        raise UnknownProviderError(
            f"Unknown model provider '{resolved}' -- valid values: "
            f"{', '.join(sorted(_FACTORIES))}"
        )
    return build()

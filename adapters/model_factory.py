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


# Each builder passes `model` on only when one was chosen, so every adapter keeps its own
# default otherwise.
def _model_kwargs(model: Optional[str]) -> Dict[str, str]:
    return {"model": model} if model else {}


def _build_anthropic(timeout: Optional[float], model: Optional[str]) -> ModelAdapter:
    from adapters.model import AnthropicModelAdapter

    return AnthropicModelAdapter(timeout=timeout, **_model_kwargs(model))


def _build_gemini(timeout: Optional[float], model: Optional[str]) -> ModelAdapter:
    from adapters.gemini_model import GeminiModelAdapter

    return GeminiModelAdapter(timeout=timeout, **_model_kwargs(model))


def _openai_compatible(provider: str) -> Callable[[Optional[float], Optional[str]], ModelAdapter]:
    def build(timeout: Optional[float], model: Optional[str]) -> ModelAdapter:
        from adapters.openai_compatible_model import OpenAICompatibleModelAdapter

        return OpenAICompatibleModelAdapter(provider=provider, timeout=timeout, model=model)

    return build


_FACTORIES: Dict[str, Callable[[Optional[float], Optional[str]], ModelAdapter]] = {
    "anthropic": _build_anthropic,
    "gemini": _build_gemini,
    "deepseek": _openai_compatible("deepseek"),
    "openai": _openai_compatible("openai"),
    "kimi": _openai_compatible("kimi"),
    # Any OpenAI-compatible LiteLLM proxy: its URL and key come from the environment, its model
    # names are the proxy's own (decided by user 2026-10-14).
    "litellm": _openai_compatible("litellm"),
}

# Preserves today's zero-config behavior exactly: a caller with no argument and no env var
# set gets Anthropic, same as every existing `AnthropicModelAdapter()` call site.
DEFAULT_PROVIDER = "anthropic"
_ENV_VAR = "MENGER_AGENT_MODEL_PROVIDER"
# The model to use with the selected provider (e.g. one of a LiteLLM proxy's model names);
# unset, each provider uses its own default.
_MODEL_ENV_VAR = "MENGER_AGENT_MODEL"


def selected_provider(provider: Optional[str] = None) -> str:
    """The provider `get_model_adapter(provider)` resolves to: the argument, else
    `MENGER_AGENT_MODEL_PROVIDER`, else `DEFAULT_PROVIDER` (lower-case, stripped)."""
    resolved = provider if provider is not None else os.environ.get(_ENV_VAR)
    return (resolved or DEFAULT_PROVIDER).strip().lower()


def get_model_adapter(
    provider: Optional[str] = None,
    timeout: Optional[float] = None,
    model: Optional[str] = None,
) -> ModelAdapter:
    """Resolves and constructs a `ModelAdapter` for the selected provider.

    Precedence (mirrors `AnthropicModelAdapter.__init__`'s own arg-beats-env-var shape, so
    this doesn't invent a new precedence rule): an explicit `provider` argument wins outright
    if given, regardless of the environment variable. Otherwise `MENGER_AGENT_MODEL_PROVIDER`
    is used if set to a non-empty value (case-insensitive, whitespace-stripped). Otherwise
    `DEFAULT_PROVIDER`.

    `timeout` (story 15, "model-call timeout, typed and distinct from other failures") is
    forwarded verbatim to whichever provider builder is selected -- `None` (the default)
    leaves every adapter's own SDK-default timeout behavior unchanged, exactly as before this
    parameter existed.

    Raises `UnknownProviderError` immediately -- before any SDK import or network call -- if
    the resolved provider name isn't recognized; never silently falls back to the default for
    a bad explicit selection. Raises `MissingAPIKeyError` (unchanged, propagated from the
    provider's own builder) if the resolved provider's API key isn't set; not caught or
    wrapped here, so callers can catch one exception type regardless of provider. Also raises
    `ValueError` (propagated from the constructed adapter's own `__init__`, unchanged) if
    `timeout` is not `None` and isn't a finite, positive number -- a zero, negative, NaN, or
    infinite `timeout` is a programmer error rejected before any network call, the same
    precondition every adapter (`AnthropicModelAdapter`, `GeminiModelAdapter`,
    `OpenAICompatibleModelAdapter`) enforces itself.

    `model` picks the model the same way: the argument, else `MENGER_AGENT_MODEL` if set to a
    non-empty value (whitespace-stripped, case kept -- model names are the vendor's), else the
    provider's own default.
    """
    resolved = selected_provider(provider)
    chosen_model = model if model is not None else (os.environ.get(_MODEL_ENV_VAR) or "").strip()

    build = _FACTORIES.get(resolved)
    if build is None:
        raise UnknownProviderError(
            f"Unknown model provider '{resolved}' -- valid values: "
            f"{', '.join(sorted(_FACTORIES))}"
        )
    return build(timeout, chosen_model or None)

"""Unit tests for adapters/openai_compatible_model.py -- construction (per provider, since
one class serves three vendors) and (via a fake `_client`) `complete()`'s response-parsing
logic. No real network call anywhere. `complete()`'s parsing logic is identical across the
three vendors (same OpenAI-compatible request/response shape), so it's exercised against one
provider (`deepseek`) rather than tripled; construction is tested per provider since that's
exactly where the parametrization lives."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from adapters.model import ModelRequest, MissingAPIKeyError, ModelError, UnknownProviderError
from adapters.openai_compatible_model import OpenAICompatibleModelAdapter, _PROVIDERS


def _adapter_with_fake_client(provider: str, create_fn) -> OpenAICompatibleModelAdapter:
    """Builds a real OpenAICompatibleModelAdapter (so `complete()`'s own logic runs
    unmodified) but swaps `_client` for a fake whose `chat.completions.create` is
    `create_fn` -- no network."""
    adapter = OpenAICompatibleModelAdapter(provider=provider, api_key="test-key-not-a-real-credential")
    adapter._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn))
    )
    return adapter


def _chat_response(text: str, finish_reason: str = "stop") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=text))
        ]
    )


_A_REQUEST = ModelRequest(system_prompt="system", user_prompt="user")


# --- Edge-Case Matrix: missing API key, per provider --------------------------------------


@pytest.mark.parametrize(
    "provider,env_var",
    [("deepseek", "DEEPSEEK_API_KEY"), ("openai", "OPENAI_API_KEY"), ("kimi", "MOONSHOT_API_KEY")],
)
def test_construction_fails_fast_without_api_key(provider, env_var, monkeypatch):
    monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(MissingAPIKeyError):
        OpenAICompatibleModelAdapter(provider=provider)


@pytest.mark.parametrize(
    "provider,env_var",
    [("deepseek", "DEEPSEEK_API_KEY"), ("openai", "OPENAI_API_KEY"), ("kimi", "MOONSHOT_API_KEY")],
)
def test_construction_succeeds_with_an_explicit_api_key(provider, env_var, monkeypatch):
    monkeypatch.delenv(env_var, raising=False)

    adapter = OpenAICompatibleModelAdapter(provider=provider, api_key="test-key-not-a-real-credential")

    assert adapter is not None


@pytest.mark.parametrize(
    "provider,env_var",
    [("deepseek", "DEEPSEEK_API_KEY"), ("openai", "OPENAI_API_KEY"), ("kimi", "MOONSHOT_API_KEY")],
)
def test_construction_succeeds_from_the_environment_variable(provider, env_var, monkeypatch):
    monkeypatch.setenv(env_var, "test-key-not-a-real-credential")

    adapter = OpenAICompatibleModelAdapter(provider=provider)

    assert adapter is not None


@pytest.mark.parametrize("provider", ["deepseek", "openai", "kimi"])
def test_construction_uses_the_correct_base_url_for_each_provider(provider):
    adapter = OpenAICompatibleModelAdapter(provider=provider, api_key="test-key-not-a-real-credential")

    expected = _PROVIDERS[provider].base_url
    assert str(adapter._client.base_url).rstrip("/") == expected.rstrip("/")


def test_unknown_provider_raises_a_typed_error_not_a_bare_key_error():
    with pytest.raises(UnknownProviderError):
        OpenAICompatibleModelAdapter(provider="not-a-real-provider", api_key="x")


# --- complete(): response parsing, against a fake `_client`, no network -------------------
#
# Mirrors adapters/model.py's AnthropicModelAdapter docstring (story 6): complete() never
# applies extract_scene_text -- it returns the model's raw text verbatim.


def test_complete_returns_the_raw_response_text_verbatim_not_extracted():
    adapter = _adapter_with_fake_client(
        "deepseek",
        lambda **kwargs: _chat_response("```scala\nobject Foo:\n  val scene = Scene()\n```"),
    )

    result = adapter.complete(_A_REQUEST)

    assert result == "```scala\nobject Foo:\n  val scene = Scene()\n```"


def test_complete_returns_plain_prose_verbatim_not_rejected_as_non_scene_text():
    prose = "A level-3 sponge, glass, camera 5 units out at 30 degrees."
    adapter = _adapter_with_fake_client("deepseek", lambda **kwargs: _chat_response(prose))

    result = adapter.complete(_A_REQUEST)

    assert result == prose


def test_complete_maps_an_empty_response_to_a_typed_error():
    adapter = _adapter_with_fake_client("deepseek", lambda **kwargs: _chat_response(""))

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_content_filter_finish_reason_to_a_typed_error():
    response = _chat_response("", finish_reason="content_filter")
    adapter = _adapter_with_fake_client("deepseek", lambda **kwargs: response)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_length_finish_reason_to_a_typed_error_not_a_silent_success():
    response = _chat_response("object Foo:\n  val scene = Scene(", finish_reason="length")
    adapter = _adapter_with_fake_client("deepseek", lambda **kwargs: response)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_an_unexpected_tool_calls_finish_reason_to_a_typed_error():
    # This adapter never sends `tools=`, so this shape is unexpected, not silently accepted.
    response = _chat_response("", finish_reason="tool_calls")
    adapter = _adapter_with_fake_client("deepseek", lambda **kwargs: response)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_malformed_response_shape_to_a_typed_error_not_a_raise():
    adapter = _adapter_with_fake_client("deepseek", lambda **kwargs: SimpleNamespace())

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda openai: openai.APIConnectionError(request=SimpleNamespace()),
        lambda openai: openai.RateLimitError(
            message="rate limited", response=SimpleNamespace(headers={}, status_code=429), body=None
        ),
        lambda openai: RuntimeError("simulated SDK failure"),
    ],
)
def test_complete_maps_a_raised_sdk_exception_from_the_call_to_a_typed_error(exc_factory):
    import openai as openai_module

    def raise_it(**kwargs):
        raise exc_factory(openai_module)

    adapter = _adapter_with_fake_client("deepseek", raise_it)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "call_failed"


def test_complete_error_message_names_the_failing_provider():
    # One class serves three vendors -- an undifferentiated error message would be a real
    # debugging regression versus today's single-adapter-per-file shape.
    def raise_it(**kwargs):
        raise RuntimeError("boom")

    adapter = _adapter_with_fake_client("deepseek", raise_it)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert "DeepSeek" in result.message


# --- spec-ai-scene-agent story 15: model-call timeout, typed and distinct -----------------


def test_complete_maps_an_api_timeout_error_to_a_distinct_timeout_kind_not_call_failed():
    import openai
    import httpx

    exc = openai.APITimeoutError(request=httpx.Request("POST", "https://example.com"))

    def raise_it(**kwargs):
        raise exc

    adapter = _adapter_with_fake_client("deepseek", raise_it)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "timeout"
    # The original exception must be preserved as `cause`, not swallowed.
    assert result.cause is exc


@pytest.mark.parametrize("provider", ["deepseek", "openai", "kimi"])
def test_construction_forwards_an_explicit_timeout_to_the_sdk_client(provider):
    adapter = OpenAICompatibleModelAdapter(
        provider=provider, api_key="test-key-not-a-real-credential", timeout=5.0
    )

    assert adapter._client.timeout == 5.0


@pytest.mark.parametrize("provider", ["deepseek", "openai", "kimi"])
def test_construction_leaves_the_sdk_default_timeout_alone_when_unset(provider):
    # Boundaries & Constraints: "No adapter's default behavior changes when timeout is left
    # unset" -- passing `timeout=None` to the SDK client would override its own internal
    # default with "no timeout", not leave it alone. Compared against a bare
    # `openai.OpenAI(api_key=..., base_url=...)` construction (no `timeout` kwarg at all).
    import openai

    spec = _PROVIDERS[provider]
    adapter = OpenAICompatibleModelAdapter(provider=provider, api_key="test-key-not-a-real-credential")
    reference_client = openai.OpenAI(api_key="test-key-not-a-real-credential", base_url=spec.base_url)

    assert adapter._client.timeout == reference_client.timeout


# --- Patch-level fix (review round): reject a non-finite/non-positive timeout ------------


@pytest.mark.parametrize("bad_timeout", [0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_construction_rejects_a_non_finite_or_non_positive_timeout(bad_timeout):
    with pytest.raises(ValueError):
        OpenAICompatibleModelAdapter(
            provider="deepseek", api_key="test-key-not-a-real-credential", timeout=bad_timeout
        )

"""Unit tests for adapters/gemini_model.py -- construction and (via a fake `_client`)
`complete()`'s response-parsing logic. No real network call anywhere:
`GeminiModelAdapter.__init__` never talks to the network, and `complete()`'s parsing is
exercised against a fake `_client` whose `models.generate_content` returns a scripted stub
response or raises a scripted exception -- never a real SDK call. `extract_scene_text` itself
is already covered by `tests/test_model_adapter.py` (it's a pure function in `adapters/model.py`,
shared by every adapter -- not re-tested per provider)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from adapters.model import ModelRequest, MissingAPIKeyError, ModelError
from adapters.gemini_model import GeminiModelAdapter


def _adapter_with_fake_client(create_fn) -> GeminiModelAdapter:
    """Builds a real GeminiModelAdapter (so `complete()`'s own logic runs unmodified) but
    swaps `_client` for a fake whose `models.generate_content` is `create_fn` -- no network."""
    adapter = GeminiModelAdapter(api_key="test-key-not-a-real-credential")
    adapter._client = SimpleNamespace(models=SimpleNamespace(generate_content=create_fn))
    return adapter


def _candidate_response(text: str, finish_reason: str = "STOP", block_reason: Any = None) -> Any:
    """Mimics `google.genai.types.GenerateContentResponse`'s shape closely enough for
    `complete()`'s attribute access: `.text` (the SDK's own convenience property, which
    `complete()` calls directly rather than walking `.candidates[0].content.parts` itself --
    faked here as a plain attribute since the fake response isn't a real `GenerateContentResponse`
    instance and so has no real `.text` property to inherit), `.candidates[0].finish_reason`,
    `.prompt_feedback.block_reason`."""
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        prompt_feedback=SimpleNamespace(block_reason=block_reason),
    )


_A_REQUEST = ModelRequest(system_prompt="system", user_prompt="user")


# --- Edge-Case Matrix: missing API key ----------------------------------------------------


def test_construction_fails_fast_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError):
        GeminiModelAdapter()


def test_construction_fails_fast_with_empty_api_key_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")

    with pytest.raises(MissingAPIKeyError):
        GeminiModelAdapter()


def test_construction_succeeds_with_an_explicit_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    adapter = GeminiModelAdapter(api_key="test-key-not-a-real-credential")

    assert adapter is not None


def test_construction_succeeds_from_the_environment_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")

    adapter = GeminiModelAdapter()

    assert adapter is not None


# --- complete(): response parsing, against a fake `_client`, no network -------------------
#
# Mirrors adapters/model.py's AnthropicModelAdapter docstring (story 6): complete() never
# applies extract_scene_text -- it returns the model's raw text verbatim, scene-specific or
# not, so core/generation.py and core/readback.py can each apply their own interpretation.


def test_complete_returns_the_raw_response_text_verbatim_not_extracted():
    adapter = _adapter_with_fake_client(
        lambda **kwargs: _candidate_response("```scala\nobject Foo:\n  val scene = Scene()\n```")
    )

    result = adapter.complete(_A_REQUEST)

    assert result == "```scala\nobject Foo:\n  val scene = Scene()\n```"


def test_complete_returns_plain_prose_verbatim_not_rejected_as_non_scene_text():
    # The same production case adapters/model.py's own tests guard against: a plain-language
    # response is a valid complete() result for a non-generation caller like
    # semantic_readback() -- complete() itself has no opinion on shape.
    prose = "A level-3 sponge, glass, camera 5 units out at 30 degrees."
    adapter = _adapter_with_fake_client(lambda **kwargs: _candidate_response(prose))

    result = adapter.complete(_A_REQUEST)

    assert result == prose


def test_complete_maps_an_empty_response_to_a_typed_error():
    adapter = _adapter_with_fake_client(lambda **kwargs: _candidate_response(""))

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_safety_finish_reason_to_a_typed_error():
    response = _candidate_response("", finish_reason="SAFETY")
    adapter = _adapter_with_fake_client(lambda **kwargs: response)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_recitation_finish_reason_to_a_typed_error():
    response = _candidate_response("", finish_reason="RECITATION")
    adapter = _adapter_with_fake_client(lambda **kwargs: response)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_max_tokens_finish_reason_to_a_typed_error_not_a_silent_success():
    response = _candidate_response("object Foo:\n  val scene = Scene(", finish_reason="MAX_TOKENS")
    adapter = _adapter_with_fake_client(lambda **kwargs: response)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_blocked_prompt_to_a_typed_error():
    # Gemini-specific: the prompt itself was blocked before any candidate was generated at
    # all -- a failure shape Anthropic's stop_reason vocabulary has no analogue for.
    response = _candidate_response("", block_reason="SAFETY")
    adapter = _adapter_with_fake_client(lambda **kwargs: response)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_malformed_response_shape_to_a_typed_error_not_a_raise():
    # No `.candidates`/`.text` attributes at all -- simulates an unexpected/future SDK
    # response shape.
    adapter = _adapter_with_fake_client(lambda **kwargs: SimpleNamespace())

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_an_sdk_api_error_from_the_call_to_a_typed_error():
    from google.genai import errors

    def raise_it(**kwargs):
        raise errors.ClientError(code=429, response_json={"error": {"message": "rate limited"}})

    adapter = _adapter_with_fake_client(raise_it)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "call_failed"


def test_complete_maps_a_raised_bare_exception_from_the_sdk_call_to_a_typed_error():
    def raise_it(**kwargs):
        raise RuntimeError("simulated SDK failure")

    adapter = _adapter_with_fake_client(raise_it)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "call_failed"

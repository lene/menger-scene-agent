"""Unit tests for adapters/model.py -- construction, response-extraction, and (via a fake
`_client`) `complete()`'s response-parsing logic. No real network call anywhere:
`AnthropicModelAdapter.__init__` never talks to the network, `extract_scene_text` is pure
string processing, and `complete()`'s parsing is exercised against a fake `_client` whose
`beta.messages.create` returns a scripted stub response or raises a scripted exception --
never a real SDK call."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from adapters.model import (
    AnthropicModelAdapter,
    ModelRequest,
    MissingAPIKeyError,
    ModelError,
    extract_scene_text,
)


def _adapter_with_fake_client(create_fn) -> AnthropicModelAdapter:
    """Builds a real AnthropicModelAdapter (so `complete()`'s own logic runs unmodified) but
    swaps `_client` for a fake whose `beta.messages.create` is `create_fn` -- no network."""
    adapter = AnthropicModelAdapter(api_key="test-key-not-a-real-credential")
    adapter._client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create_fn))
    )
    return adapter


def _text_response(text: str, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        content=[SimpleNamespace(type="text", text=text)],
    )


_A_REQUEST = ModelRequest(system_prompt="system", user_prompt="user")


# --- Edge-Case Matrix: missing API key ----------------------------------------------------


def test_construction_fails_fast_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError):
        AnthropicModelAdapter()


def test_construction_fails_fast_with_empty_api_key_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(MissingAPIKeyError):
        AnthropicModelAdapter()


def test_construction_succeeds_with_an_explicit_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    adapter = AnthropicModelAdapter(api_key="test-key-not-a-real-credential")

    assert adapter is not None


def test_construction_succeeds_from_the_environment_variable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-real-credential")

    adapter = AnthropicModelAdapter()

    assert adapter is not None


# --- extract_scene_text: single code fence ------------------------------------------------


def test_extract_scene_text_from_a_single_scala_code_fence():
    raw = "Here you go:\n```scala\nobject Foo:\n  val scene = Scene()\n```"

    result = extract_scene_text(raw)

    assert result == "object Foo:\n  val scene = Scene()"


def test_extract_scene_text_from_a_single_untagged_code_fence():
    raw = "```\nobject Foo:\n  val scene = Scene()\n```"

    result = extract_scene_text(raw)

    assert result == "object Foo:\n  val scene = Scene()"


def test_extract_scene_text_accepts_a_bare_response_with_no_fence_but_an_object():
    raw = "object Foo:\n  val scene = Scene()\n"

    result = extract_scene_text(raw)

    assert result == raw


# --- Edge-Case Matrix: model returns non-scene text ---------------------------------------


def test_extract_scene_text_rejects_an_empty_response():
    result = extract_scene_text("")

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_extract_scene_text_rejects_prose_with_no_object_declaration():
    result = extract_scene_text("Sure, I can help you with that -- here's my plan.")

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_extract_scene_text_rejects_multiple_candidate_code_blocks():
    raw = "```scala\nobject A:\n  val scene = Scene()\n```\n\nOr maybe:\n```scala\nobject B:\n  val scene = Scene()\n```"

    result = extract_scene_text(raw)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_extract_scene_text_accepts_a_non_scala_language_tag():
    raw = "```scala3\nobject Foo:\n  val scene = Scene()\n```"

    result = extract_scene_text(raw)

    assert result == "object Foo:\n  val scene = Scene()"


def test_extract_scene_text_rejects_object_declaration_past_the_first_60_lines():
    padding = "\n".join(f"// line {i}" for i in range(65))
    raw = f"```scala\n{padding}\nobject Foo:\n  val scene = Scene()\n```"

    result = extract_scene_text(raw)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_extract_scene_text_rejects_stray_fence_syntax_with_no_clean_match():
    raw = "``` object Foo mentioned in passing, not actually fenced"

    result = extract_scene_text(raw)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


# --- complete(): response parsing, against a fake `_client`, no network -------------------


def test_complete_extracts_text_from_a_successful_response():
    adapter = _adapter_with_fake_client(
        lambda **kwargs: _text_response("```scala\nobject Foo:\n  val scene = Scene()\n```")
    )

    result = adapter.complete(_A_REQUEST)

    assert result == "object Foo:\n  val scene = Scene()"


def test_complete_maps_a_refusal_stop_reason_to_a_typed_error():
    response = _text_response("", stop_reason="refusal")
    adapter = _adapter_with_fake_client(lambda **kwargs: response)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_max_tokens_stop_reason_to_a_typed_error_not_a_silent_success():
    response = _text_response(
        "object Foo:\n  val scene = Scene(", stop_reason="max_tokens"
    )
    adapter = _adapter_with_fake_client(lambda **kwargs: response)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_malformed_response_shape_to_a_typed_error_not_a_raise():
    # No `.content` attribute at all -- simulates an unexpected/future SDK response shape.
    adapter = _adapter_with_fake_client(
        lambda **kwargs: SimpleNamespace(stop_reason="end_turn")
    )

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_complete_maps_a_raised_exception_from_the_sdk_call_to_a_typed_error():
    def raise_it(**kwargs):
        raise RuntimeError("simulated SDK failure")

    adapter = _adapter_with_fake_client(raise_it)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "call_failed"

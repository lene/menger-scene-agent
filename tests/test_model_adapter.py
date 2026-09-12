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
#
# Review round (story 6): `complete()` used to call `extract_scene_text` on every successful
# response before returning -- which broke `core/readback.py`'s `semantic_readback()` in
# production, since a real plain-language sentence has no code fence and no `object`
# declaration and always failed `extract_scene_text`'s own checks. `complete()` now returns
# the model's raw text verbatim; `core/generation.py`'s `generate()`/`revise()` apply
# `extract_scene_text` themselves, since scene extraction is their concern, not every
# `ModelAdapter` caller's.


def test_complete_returns_the_raw_response_text_verbatim_not_extracted():
    adapter = _adapter_with_fake_client(
        lambda **kwargs: _text_response("```scala\nobject Foo:\n  val scene = Scene()\n```")
    )

    result = adapter.complete(_A_REQUEST)

    assert result == "```scala\nobject Foo:\n  val scene = Scene()\n```"


def test_complete_returns_plain_prose_verbatim_not_rejected_as_non_scene_text():
    # The exact case that broke in production: a plain-language response (no fence, no
    # `object` declaration) is a perfectly valid `complete()` result for a non-generation
    # caller like `semantic_readback()` -- `complete()` itself has no opinion on shape.
    prose = "A level-3 sponge, glass, camera 5 units out at 30 degrees."
    adapter = _adapter_with_fake_client(lambda **kwargs: _text_response(prose))

    result = adapter.complete(_A_REQUEST)

    assert result == prose


def test_complete_maps_an_empty_response_to_a_typed_error():
    adapter = _adapter_with_fake_client(lambda **kwargs: _text_response(""))

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


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


# --- spec-ai-scene-agent story 15: model-call timeout, typed and distinct -----------------


def test_complete_maps_an_api_timeout_error_to_a_distinct_timeout_kind_not_call_failed():
    import anthropic
    import httpx

    exc = anthropic.APITimeoutError(request=httpx.Request("POST", "https://example.com"))

    def raise_it(**kwargs):
        raise exc

    adapter = _adapter_with_fake_client(raise_it)

    result = adapter.complete(_A_REQUEST)

    assert isinstance(result, ModelError)
    assert result.kind == "timeout"
    # The original exception must be preserved as `cause`, not swallowed -- a caller
    # inspecting the failure (or re-raising it for a traceback) needs the real SDK exception.
    assert result.cause is exc


def test_construction_forwards_an_explicit_timeout_to_the_sdk_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    adapter = AnthropicModelAdapter(api_key="test-key-not-a-real-credential", timeout=5.0)

    assert adapter._client.timeout == 5.0


def test_construction_leaves_the_sdk_default_timeout_alone_when_unset(monkeypatch):
    # Boundaries & Constraints: "No adapter's default behavior changes when timeout is left
    # unset" -- passing `timeout=None` to the SDK client would override its own internal
    # default with "no timeout", which is not the same thing as leaving it alone. Compares
    # against a bare `anthropic.Anthropic(api_key=...)` construction (no `timeout` kwarg at
    # all) to prove this adapter doesn't pass one either when its own `timeout` is unset.
    import anthropic

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = AnthropicModelAdapter(api_key="test-key-not-a-real-credential")
    reference_client = anthropic.Anthropic(api_key="test-key-not-a-real-credential")

    assert adapter._client.timeout == reference_client.timeout


# --- Patch-level fix (review round): reject a non-finite/non-positive timeout ------------
# before it's forwarded straight to the SDK client unchecked. Mirrors
# adapters/render_window.py's `grace_period <= 0` precondition -- a programmer-error
# precondition, not a modeled runtime outcome.


@pytest.mark.parametrize("bad_timeout", [0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_construction_rejects_a_non_finite_or_non_positive_timeout(bad_timeout, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError):
        AnthropicModelAdapter(api_key="test-key-not-a-real-credential", timeout=bad_timeout)


def test_construction_accepts_a_small_positive_timeout(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    adapter = AnthropicModelAdapter(api_key="test-key-not-a-real-credential", timeout=0.1)

    assert adapter._client.timeout == 0.1


# --- spec-ai-scene-agent story 18: needs_clarification sentinel detection -----------------


def test_extract_scene_text_detects_the_needs_clarification_sentinel():
    result = extract_scene_text("NEEDS_CLARIFICATION: 'fribbly' is not a defined DSL term")

    assert isinstance(result, ModelError)
    assert result.kind == "needs_clarification"
    assert result.message == "'fribbly' is not a defined DSL term"


def test_extract_scene_text_strips_whitespace_around_the_sentinel_and_its_reason():
    result = extract_scene_text(
        "  \n NEEDS_CLARIFICATION:   redder and greener are contradictory  \n  "
    )

    assert isinstance(result, ModelError)
    assert result.kind == "needs_clarification"
    assert result.message == "redder and greener are contradictory"


def test_extract_scene_text_does_not_match_the_sentinel_mentioned_in_passing():
    raw = "I could return NEEDS_CLARIFICATION: but here's a scene instead.\n```scala\nobject Foo:\n  val scene = Scene()\n```"

    result = extract_scene_text(raw)

    assert result == "object Foo:\n  val scene = Scene()"


def test_extract_scene_text_does_not_match_the_sentinel_wrapped_in_a_code_fence():
    # I/O & Edge-Case Matrix: malformed sentinel emission (fenced) is not specially handled --
    # it falls through to the existing object-declaration check and is rejected the same way
    # any other non-scene fenced content would be.
    raw = "```\nNEEDS_CLARIFICATION: fribbly is undefined\n```"

    result = extract_scene_text(raw)

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_extract_scene_text_does_not_match_the_sentinel_followed_by_extra_prose():
    # The sentinel must be the *entire* stripped text (single line) to match -- extra
    # content after it means the whole response falls through to the ordinary
    # object-declaration/fence checks instead. In this case it actually finds a top-level
    # object declaration, so it is accepted as scene text, not specially rejected -- exactly
    # the "not specially handled" contract (I/O & Edge-Case Matrix).
    raw = "NEEDS_CLARIFICATION: fribbly is undefined\nHere is my best guess anyway:\nobject Foo:\n  val scene = Scene()\n"

    result = extract_scene_text(raw)

    assert result == raw


def test_extract_scene_text_sentinel_match_is_case_sensitive():
    result = extract_scene_text("needs_clarification: lowercase should not match")

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_extract_scene_text_ordinary_scene_response_is_unaffected_by_the_sentinel_check():
    raw = "```scala\nobject Foo:\n  val scene = Scene()\n```"

    result = extract_scene_text(raw)

    assert result == "object Foo:\n  val scene = Scene()"


def test_extract_scene_text_does_not_match_an_empty_reason():
    # Nothing meaningful after the colon (once outer whitespace is stripped) -- the
    # sentinel's capture group requires at least one character, so this falls through to
    # the ordinary fence/object-declaration checks instead of matching with an empty reason.
    result = extract_scene_text("NEEDS_CLARIFICATION:")

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"


def test_extract_scene_text_does_not_truncate_a_reason_containing_a_colon():
    result = extract_scene_text("NEEDS_CLARIFICATION: ratio 3:2 is ambiguous")

    assert isinstance(result, ModelError)
    assert result.kind == "needs_clarification"
    assert result.message == "ratio 3:2 is ambiguous"


def test_extract_scene_text_matches_the_sentinel_with_no_space_after_the_colon():
    result = extract_scene_text("NEEDS_CLARIFICATION:fribbly is undefined")

    assert isinstance(result, ModelError)
    assert result.kind == "needs_clarification"
    assert result.message == "fribbly is undefined"


def test_extract_scene_text_does_not_match_sentinel_split_across_two_lines():
    # Review-round patch: the regex's post-colon gap is `[ \t]*`, not `\s*` -- a newline
    # right after the colon (keyword on its own line, reason on the next) must not match,
    # since the sentinel is documented as single-line-only.
    result = extract_scene_text("NEEDS_CLARIFICATION:\nfribbly is undefined")

    assert isinstance(result, ModelError)
    assert result.kind == "invalid_output"

"""Unit tests for core/readback.py -- a fake ModelAdapter only, zero network access."""

from __future__ import annotations

from adapters.model import ModelError
from core.readback import semantic_readback
from core.types import ReadbackError
from tests.fakes import FakeModelAdapter

SCENE_TEXT = (
    "package examples.dsl\n\n"
    "object GlassSponge:\n"
    "  val scene = Scene(\n"
    "    camera = Camera(position = Vec3(0f, 0f, 5f)),\n"
    "    objects = List(MengerSponge(level = 3f, material = Some(Material.Glass)))\n"
    "  )\n"
)
READBACK_TEXT = "level-3 sponge, glass, camera 5 units out"


# --- happy path ---------------------------------------------------------------------------


def test_semantic_readback_happy_path_returns_the_scripted_text():
    adapter = FakeModelAdapter(result=READBACK_TEXT)

    result = semantic_readback(SCENE_TEXT, adapter)

    assert result == READBACK_TEXT


def test_semantic_readback_passes_the_scene_text_into_the_user_prompt():
    adapter = FakeModelAdapter(result=READBACK_TEXT)

    semantic_readback(SCENE_TEXT, adapter)

    assert adapter.last_request is not None
    assert SCENE_TEXT in adapter.last_request.user_prompt


def test_semantic_readback_composes_the_style_and_rules_into_the_system_prompt():
    adapter = FakeModelAdapter(result=READBACK_TEXT)

    semantic_readback(SCENE_TEXT, adapter)

    system_prompt = adapter.last_request.system_prompt
    assert "one plain-language sentence" in system_prompt
    assert "warm area light upper-left" in system_prompt  # the style example


def test_semantic_readback_delimits_scene_text_with_scene_tags_not_a_code_fence():
    # Review round: a markdown code fence would prematurely close if scene_text itself
    # contained a triple-backtick sequence, splicing the remainder out of the quoted block.
    adapter = FakeModelAdapter(result=READBACK_TEXT)

    semantic_readback(SCENE_TEXT, adapter)

    user_prompt = adapter.last_request.user_prompt
    assert "<scene>" in user_prompt
    assert "</scene>" in user_prompt
    assert "```" not in user_prompt


def test_semantic_readback_handles_scene_text_containing_a_triple_backtick():
    # The exact case a code fence would have broken: scene_text containing "```" would
    # prematurely close a markdown fence, splicing text out of the quoted block.
    adapter = FakeModelAdapter(result=READBACK_TEXT)
    tricky_scene_text = SCENE_TEXT + '\n  // a comment mentioning ``` for good measure\n'

    semantic_readback(tricky_scene_text, adapter)

    assert tricky_scene_text in adapter.last_request.user_prompt


# --- Edge-Case Matrix: model call fails/times out ----------------------------------------


def test_model_call_failure_is_a_typed_error_not_an_exception():
    adapter = FakeModelAdapter(result=ModelError(kind="call_failed", message="timeout"))

    result = semantic_readback(SCENE_TEXT, adapter)

    assert isinstance(result, ReadbackError)
    assert result.kind == "model_call_failed"
    assert result.message == "timeout"


def test_adapter_raising_is_also_a_typed_error_not_an_exception():
    class RaisingAdapter:
        def complete(self, request):
            raise RuntimeError("adapter blew up")

    result = semantic_readback(SCENE_TEXT, RaisingAdapter())

    assert isinstance(result, ReadbackError)
    assert result.kind == "model_call_failed"


# --- Edge-Case Matrix: model returns empty/unusable response ------------------------------


def test_empty_model_response_is_a_typed_error_not_a_silent_success():
    adapter = FakeModelAdapter(result="")

    result = semantic_readback(SCENE_TEXT, adapter)

    assert isinstance(result, ReadbackError)
    assert result.kind == "invalid_model_output"


def test_whitespace_only_model_response_is_a_typed_error():
    adapter = FakeModelAdapter(result="   \n  ")

    result = semantic_readback(SCENE_TEXT, adapter)

    assert isinstance(result, ReadbackError)
    assert result.kind == "invalid_model_output"


def test_model_invalid_output_error_maps_to_typed_readback_error():
    adapter = FakeModelAdapter(
        result=ModelError(kind="invalid_output", message="model refused")
    )

    result = semantic_readback(SCENE_TEXT, adapter)

    assert isinstance(result, ReadbackError)
    assert result.kind == "invalid_model_output"
    assert result.message == "model refused"

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


# --- Usability review 2026-09 (F7): facts-grounded readback ------------------------------


def test_semantic_readback_includes_a_facts_block_with_material_and_camera():
    # Uses a real "Sponge(...)" call, not the module's shared SCENE_TEXT fixture (which uses
    # the non-DSL placeholder name "MengerSponge" -- not one of scene_facts's known object
    # types, so it yields no facts and would make this assertion vacuous).
    scene_text = (
        "object GlassSponge:\n"
        "  val scene = Scene(\n"
        "    camera = Camera(position = Vec3(0f, 0f, 5f), lookAt = Vec3(0f, 0f, 0f)),\n"
        "    objects = List(Sponge(level = 3f, material = Some(Material.Glass)))\n"
        "  )\n"
    )
    adapter = FakeModelAdapter(result=READBACK_TEXT)

    semantic_readback(scene_text, adapter)

    user_prompt = adapter.last_request.user_prompt
    assert "<facts>" in user_prompt
    assert "</facts>" in user_prompt
    assert "material=Glass" in user_prompt
    assert "transparent" in user_prompt
    assert "Camera at" in user_prompt


def test_semantic_readback_facts_block_states_directional_light_travel_phrase():
    scene_text = (
        "object LitFromBelow:\n"
        "  val scene = Scene(\n"
        "    lights = List(Directional(direction = Vec3(0f, 1f, 0f))),\n"
        "    objects = List(Sphere())\n"
        "  )\n"
    )
    adapter = FakeModelAdapter(result=READBACK_TEXT)

    semantic_readback(scene_text, adapter)

    user_prompt = adapter.last_request.user_prompt
    assert "shines from below" in user_prompt


def test_semantic_readback_system_prompt_declares_facts_authoritative():
    adapter = FakeModelAdapter(result=READBACK_TEXT)

    semantic_readback(SCENE_TEXT, adapter)

    system_prompt = adapter.last_request.system_prompt
    assert "authoritative" in system_prompt


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


# --- Usability review 2026-09, session 2 (F45, msa#2) -------------------------------------


def test_semantic_readback_facts_state_where_objects_are_relative_to_each_other():
    # Session 2, task 4.5: "a 24-cell above the sponge" was placed along +x and the readback
    # still said "above" -- the facts had positions, but nothing relating them.
    scene_text = (
        "object Beside:\n"
        "  val scene = Scene(objects = List(\n"
        "    TesseractSponge(pos = Vec3(0f, 0f, 0f)),\n"
        "    Icositetrachoron(pos = Vec3(2.5f, 0f, 0f), projection = Some(Projection4DSpec(rotXW = 30f)))\n"
        "  ))\n"
    )
    adapter = FakeModelAdapter(result=READBACK_TEXT)

    semantic_readback(scene_text, adapter)

    user_prompt = adapter.last_request.user_prompt
    assert "Icositetrachoron is beside the TesseractSponge (+2.5 along x), not above it" in user_prompt
    assert "4D projection=Some(Projection4DSpec(rotXW = 30f))" in user_prompt


def test_semantic_readback_facts_call_a_plus_y_offset_above():
    scene_text = (
        "object Above:\n"
        "  val scene = Scene(objects = List(\n"
        "    Sponge(pos = Vec3(0f, 0f, 0f)),\n"
        "    Sphere(pos = Vec3(0f, 2f, 0f), rotation = Vec3(0f, 0.5f, 0f))\n"
        "  ))\n"
    )
    adapter = FakeModelAdapter(result=READBACK_TEXT)

    semantic_readback(scene_text, adapter)

    user_prompt = adapter.last_request.user_prompt
    assert "Sphere is above the Sponge (+2.0 along y)" in user_prompt
    assert "rotation=(0.0, 0.5, 0.0)" in user_prompt

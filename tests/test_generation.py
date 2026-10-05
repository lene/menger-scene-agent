"""Unit tests for core/generation.py -- a fake ModelAdapter only, zero network access."""

from __future__ import annotations

from adapters.model import ModelError
from core.generation import generate, revise
from core.types import EXPECTED_MANIFEST_SCHEMA_VERSION, GenerationError
from tests.fakes import FakeModelAdapter

VALID_MANIFEST = {
    "schemaVersion": EXPECTED_MANIFEST_SCHEMA_VERSION,
    "objects": [{"name": "Sphere", "fields": []}],
}
VALID_CORPUS = {
    "schemaVersion": "1.0.0",
    "scenes": [
        {
            "name": "GlassSphere",
            "path": "GlassSphere.scala",
            "source": "object GlassSphere:\n  val scene = Scene()\n",
        }
    ],
}
SCENE_TEXT = "package examples.dsl\n\nobject Foo:\n  val scene = Scene()\n"


# --- CAP-1 happy path -------------------------------------------------------------------


def test_generate_happy_path_returns_non_empty_scene_text_with_top_level_object():
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    result = generate("a glass sphere", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert result == SCENE_TEXT
    first_60_lines = result.splitlines()[:60]
    assert any(line.strip().startswith("object ") for line in first_60_lines)


def test_generate_passes_the_prompt_into_the_request():
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    generate("a glass sphere that spins", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert adapter.last_request is not None
    assert "a glass sphere that spins" in adapter.last_request.user_prompt


def test_generate_composes_manifest_and_corpus_into_the_system_prompt():
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    system_prompt = adapter.last_request.system_prompt
    assert "Sphere" in system_prompt
    assert "GlassSphere" in system_prompt


def test_generate_system_prompt_requires_a_duration_in_seconds_for_animated_scenes():
    # Usability review 2026-09 (F3): menger's window plays an animated scene only when the
    # scene declares `val duration` (t in seconds); without it the user saw a still image.
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    generate("a sponge that turns for ten seconds", VALID_MANIFEST, VALID_CORPUS, adapter)

    system_prompt = adapter.last_request.system_prompt
    assert "val duration = <seconds>f" in system_prompt
    assert "t / duration" in system_prompt


def test_generate_system_prompt_treats_complaints_as_change_requests():
    # Usability review 2026-09 (F10): "that's not glass, it looks like matte plastic" and
    # "shouldn't there be shadows on the floor?" were bounced as questions.
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    system_prompt = adapter.last_request.system_prompt
    assert "is a CHANGE REQUEST, not a question" in system_prompt
    assert "that's not glass, it looks like matte plastic" in system_prompt


def test_generate_system_prompt_names_impossible_effects_and_the_nearest_option():
    # Usability review 2026-09 (F21, F25): "glow" silently became flat emission.
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    system_prompt = adapter.last_request.system_prompt
    assert "do not silently approximate it" in system_prompt
    assert "AND the nearest thing that is" in system_prompt


def test_revise_forbids_removing_properties_the_request_does_not_mention():
    # Usability review 2026-09 (F29): the glass repair deleted the xyz colouring.
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    revise("make it glass", SCENE_TEXT, VALID_MANIFEST, VALID_CORPUS, adapter)

    assert "Never remove a property the request does not ask to remove" in (
        adapter.last_request.user_prompt
    )


def test_generate_system_prompt_warns_against_examples_dsl_cross_imports():
    # gauntlet/allowlist.py (story 4, review round 1) correctly rejects
    # `import examples.dsl.common.Lighting._` -- it resolves only inside the renderer's
    # own example-source tree, never for a standalone generated scene. The system prompt
    # must steer the model away from imitating that one corpus scene's import pattern.
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert "examples.dsl" in adapter.last_request.system_prompt


def test_generate_request_carries_no_prior_scene_marker():
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    # A generate() request must not look like a revision -- no "current scene" framing.
    assert "current scene" not in adapter.last_request.user_prompt.lower()


# --- CAP-2 happy path: revise -----------------------------------------------------------


def test_revise_returns_modified_not_byte_identical_scene_text():
    prior = "object Prior:\n  val scene = Scene()\n"
    revised = "object Prior:\n  val scene = Scene(objects = List(Sphere()))\n"
    adapter = FakeModelAdapter(result=revised)

    result = revise("add a sphere", prior, VALID_MANIFEST, VALID_CORPUS, adapter)

    assert result == revised
    assert result != prior


def test_revise_passes_prior_scene_through_verbatim_in_the_request():
    prior = "object Prior:\n  val scene = Scene(camera = Camera(position = Vec3(0,0,5)))\n"
    adapter = FakeModelAdapter(result="object Prior:\n  val scene = Scene()\n")

    revise("make it darker", prior, VALID_MANIFEST, VALID_CORPUS, adapter)

    assert adapter.last_request is not None
    # Exact-request-shape assertion: the prior scene's full text appears unchanged, and the
    # change-request prompt is present too -- this is not a from-scratch regeneration.
    assert prior in adapter.last_request.user_prompt
    assert "make it darker" in adapter.last_request.user_prompt


def test_revise_composes_manifest_and_corpus_into_the_system_prompt_too():
    adapter = FakeModelAdapter(result="object Prior:\n  val scene = Scene()\n")

    revise("tweak it", "object Prior:\n  val scene = Scene()\n", VALID_MANIFEST, VALID_CORPUS, adapter)

    system_prompt = adapter.last_request.system_prompt
    assert "Sphere" in system_prompt
    assert "GlassSphere" in system_prompt


# --- Usability review 2026-09 (F20): scene-extent fact for "zoom to fit" -------------------


def test_revise_prompt_includes_the_scene_extent_when_objects_are_positioned():
    prior = (
        "object Prior:\n  val scene = Scene(objects = List(\n"
        "    Sphere(pos = Vec3(-2f, 0f, 0f), size = 1f),\n"
        "    Sphere(pos = Vec3(2f, 0f, 0f), size = 1f)\n"
        "  ))\n"
    )
    adapter = FakeModelAdapter(result=prior)

    revise("zoom out so everything fits", prior, VALID_MANIFEST, VALID_CORPUS, adapter)

    user_prompt = adapter.last_request.user_prompt
    assert "Current scene extent" in user_prompt
    assert "(0.0, 0.0, 0.0)" in user_prompt  # center
    assert "3.0" in user_prompt  # radius: 2 to each center + its own size 1


def test_revise_prompt_omits_the_extent_fact_for_a_scene_with_no_positioned_objects():
    adapter = FakeModelAdapter(result="object Prior:\n  val scene = Scene()\n")

    revise("tweak it", "object Prior:\n  val scene = Scene()\n", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert "Current scene extent" not in adapter.last_request.user_prompt


def test_system_prompt_carries_the_manifest_s_fixed_fov_framing_convention():
    # The FOV/framing formula itself lives in the manifest's own `conventions` (DslSemantics,
    # F20/T1#3) -- already flows into every generate()/revise() system prompt via the
    # manifest JSON dump, so this pins that it's actually there for the model to use.
    manifest_with_conventions = {
        **VALID_MANIFEST,
        "conventions": ["Camera: the horizontal field of view is fixed at 45 degrees."],
    }
    adapter = FakeModelAdapter(result="object Prior:\n  val scene = Scene()\n")

    revise(
        "zoom out", "object Prior:\n  val scene = Scene()\n", manifest_with_conventions,
        VALID_CORPUS, adapter,
    )

    assert "45 degrees" in adapter.last_request.system_prompt


# --- Edge-Case Matrix: model call fails/times out ----------------------------------------


def test_model_call_failure_is_a_typed_error_not_an_exception():
    adapter = FakeModelAdapter(result=ModelError(kind="call_failed", message="timeout"))

    result = generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "model_call_failed"
    assert result.message == "timeout"


def test_revise_model_call_failure_is_also_a_typed_error():
    adapter = FakeModelAdapter(result=ModelError(kind="call_failed", message="network error"))

    result = revise("prompt", "prior scene text", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "model_call_failed"


# --- spec-ai-scene-agent story 15: model-call timeout, typed and distinct -----------------


def test_model_call_timeout_maps_to_a_distinct_generation_error_kind_not_call_failed():
    adapter = FakeModelAdapter(result=ModelError(kind="timeout", message="request timed out"))

    result = generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "model_call_timeout"
    assert result.message == "request timed out"


def test_revise_model_call_timeout_is_also_mapped_to_the_distinct_kind():
    adapter = FakeModelAdapter(result=ModelError(kind="timeout", message="request timed out"))

    result = revise("prompt", "prior scene text", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "model_call_timeout"


# --- spec-ai-scene-agent story 18: needs_clarification outcome ----------------------------


def test_generate_needs_clarification_maps_to_a_distinct_generation_error_kind():
    adapter = FakeModelAdapter(
        result=ModelError(kind="needs_clarification", message="'fribbly' is not a defined term")
    )

    result = generate("make it more fribbly", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "needs_clarification"
    assert result.message == "'fribbly' is not a defined term"


def test_revise_needs_clarification_is_also_mapped_to_the_distinct_kind():
    adapter = FakeModelAdapter(
        result=ModelError(kind="needs_clarification", message="redder and greener contradict")
    )

    result = revise("make it redder and greener", "prior scene text", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "needs_clarification"
    assert result.message == "redder and greener contradict"


def test_generate_system_prompt_actually_tells_the_model_about_the_sentinel():
    # Review-round patch: a regression guard on the outbound prompt text itself, not just
    # on how a returned ModelError maps through -- a typo/rewording here would silently break
    # the whole feature with no other test catching it.
    adapter = FakeModelAdapter(result="object Foo:\n  val scene = Scene()")

    generate("a scene", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert "NEEDS_CLARIFICATION:" in adapter.last_request.system_prompt


def test_revise_system_prompt_also_tells_the_model_about_the_sentinel():
    adapter = FakeModelAdapter(result="object Foo:\n  val scene = Scene()")

    revise("a change", "object Prior:\n  val scene = Scene()", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert "NEEDS_CLARIFICATION:" in adapter.last_request.system_prompt


# --- Edge-Case Matrix: model returns non-scene text ---------------------------------------


def test_model_non_scene_output_is_a_typed_error_not_a_best_effort_guess():
    adapter = FakeModelAdapter(
        result=ModelError(kind="invalid_output", message="prose, no code block")
    )

    result = generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "invalid_model_output"
    assert result.message == "prose, no code block"


# --- Edge-Case Matrix: corpus/manifest artifact missing or stale schema version -----------


def test_generate_fails_fast_on_stale_manifest_schema_version_before_any_model_call():
    adapter = FakeModelAdapter(result=SCENE_TEXT)
    stale_manifest = {"schemaVersion": "0.9.0"}

    result = generate("prompt", stale_manifest, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "stale_manifest"
    assert "0.9.0" in result.message
    assert adapter.requests == []


def test_generate_fails_fast_on_stale_corpus_schema_version_before_any_model_call():
    adapter = FakeModelAdapter(result=SCENE_TEXT)
    stale_corpus = {"schemaVersion": "0.9.0"}

    result = generate("prompt", VALID_MANIFEST, stale_corpus, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "stale_corpus"
    assert adapter.requests == []


def test_generate_fails_fast_on_missing_manifest_schema_version_field():
    adapter = FakeModelAdapter(result=SCENE_TEXT)
    manifest_without_version = {"objects": []}

    result = generate("prompt", manifest_without_version, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "stale_manifest"


def test_revise_fails_fast_on_stale_manifest_schema_version():
    adapter = FakeModelAdapter(result=SCENE_TEXT)
    stale_manifest = {"schemaVersion": "0.1.0"}

    result = revise("prompt", "prior text", stale_manifest, VALID_CORPUS, adapter)

    assert isinstance(result, GenerationError)
    assert result.kind == "stale_manifest"
    assert adapter.requests == []


# --- usability review 2026-09, session 2 (F53, msa#13): ask instead of guessing -----------


def test_generate_system_prompt_lists_when_to_ask_instead_of_guessing():
    # F53: session 2 had zero clarification turns; the agent guessed placements, moved objects
    # the request didn't name and animated past the renderer's warning level.
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    system_prompt = adapter.last_request.system_prompt
    assert "Ask instead of guessing" in system_prompt
    assert "`warnAt`" in system_prompt
    assert "can only approximate" in system_prompt
    # F45: "above" is not ambiguous, it is +y.
    assert '"above"/"below" is +y/-y' in system_prompt


def test_generate_system_prompt_composes_what_the_request_explicitly_names():
    # Balance against over-asking: the MVP acceptance prompt names glass explicitly; the
    # caveat reaches the user as an automatic warning, not as a question.
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    generate("prompt", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert "do not ask about a value or material the request names explicitly" in (
        adapter.last_request.system_prompt
    )


def test_revise_prompt_asks_before_changing_what_the_request_does_not_name():
    # F8 recurrence: the sponge was moved to make world = local coordinates.
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    revise("colour it by position", "prior scene text", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert "would also change something the request does not name" in (
        adapter.last_request.user_prompt
    )


# --- usability review 2026-09, session 2 (F37/F40, msa#14) -------------------------------


def test_rules_defer_the_procedural_presets_to_the_manifest():
    # The preset list now lives in the manifest's proceduralType description (schema 1.3.0);
    # a second hand-kept copy in the rules can only drift.
    from core.generation import _RULES

    assert "1 value_noise, 2 fbm, 3 worley" not in _RULES
    assert "`proceduralType` description in the manifest" in _RULES


def test_revise_prompt_drops_a_pattern_that_imitated_the_replaced_material():
    # F40: "make it aluminium" kept the wood grain, because the prompt said never to remove a
    # procedural texture.
    adapter = FakeModelAdapter(result=SCENE_TEXT)

    revise("make it aluminium", "prior scene text", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert "replaces the look the old one imitated" in adapter.last_request.user_prompt


# --- Usability review 2026-10, session 3 (INBOX 2026-10-04): stale name and doc comment -----


def test_revise_prompt_keeps_the_doc_comment_and_name_true_to_the_scene():
    # Session 3, Task 3: a level-4.75 SurfaceUnfolding sponge was still `MengerLevel2`,
    # registered as "menger-level-2", with a level-2 doc comment.
    prior = "object MengerLevel2:\n  val scene = Scene()\n"
    adapter = FakeModelAdapter(result=prior)

    revise("go to level 4.75", prior, VALID_MANIFEST, VALID_CORPUS, adapter)

    user_prompt = adapter.last_request.user_prompt
    assert "doc comment" in user_prompt
    assert "SceneRegistry.register" in user_prompt


def test_system_prompt_names_scenes_by_subject_not_parameter_values():
    adapter = FakeModelAdapter(result="object GlassSponge:\n  val scene = Scene()\n")

    revise("tweak it", "object Prior:\n  val scene = Scene()\n", VALID_MANIFEST, VALID_CORPUS, adapter)

    assert "not by a parameter value" in adapter.last_request.system_prompt

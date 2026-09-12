"""Unit tests for core/generation.py -- a fake ModelAdapter only, zero network access."""

from __future__ import annotations

from adapters.model import ModelError
from core.generation import generate, revise
from core.types import GenerationError
from tests.fakes import FakeModelAdapter

VALID_MANIFEST = {"schemaVersion": "1.0.0", "objects": [{"name": "Sphere", "fields": []}]}
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

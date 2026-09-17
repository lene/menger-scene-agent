"""Unit tests for core/consult.py -- a fake ModelAdapter only, zero network access.

One test per I/O & Edge-Case Matrix row (spec-ai-scene-agent story 19), mirroring
tests/test_readback.py's structure."""

from __future__ import annotations

from adapters.model import ModelError
from core.consult import answer_consult
from core.types import ConsultError
from tests.fakes import FakeModelAdapter

MANIFEST = {"schemaVersion": "1.0.0", "types": ["MengerSponge"]}
CORPUS = {
    "schemaVersion": "1.0.0",
    "scenes": [{"path": "examples/glass.scala", "source": "object GlassSponge: ..."}],
}
SCENE_TEXT = (
    "package examples.dsl\n\n"
    "object GlassSponge:\n"
    "  val scene = Scene(\n"
    "    camera = Camera(position = Vec3(0f, 0f, 5f)),\n"
    "    objects = List(MengerSponge(level = 3f, material = Some(Material.Glass)))\n"
    "  )\n"
)
ANSWER_TEXT = "Position the light upper-left of the sponge for a warm highlight."


# --- happy path, no prior scene ------------------------------------------------------------


def test_answer_consult_happy_path_returns_the_scripted_text():
    adapter = FakeModelAdapter(result=ANSWER_TEXT)

    result = answer_consult("where should the light go?", MANIFEST, CORPUS, adapter)

    assert result == ANSWER_TEXT


def test_answer_consult_passes_the_question_into_the_user_prompt():
    adapter = FakeModelAdapter(result=ANSWER_TEXT)

    answer_consult("where should the light go?", MANIFEST, CORPUS, adapter)

    assert adapter.last_request is not None
    assert "where should the light go?" in adapter.last_request.user_prompt


def test_answer_consult_composes_manifest_and_corpus_into_the_system_prompt():
    adapter = FakeModelAdapter(result=ANSWER_TEXT)

    answer_consult("where should the light go?", MANIFEST, CORPUS, adapter)

    system_prompt = adapter.last_request.system_prompt
    assert "MengerSponge" in system_prompt  # from the manifest
    assert "examples/glass.scala" in system_prompt  # from the corpus


def test_answer_consult_with_no_prior_scene_does_not_mention_a_scene_in_the_prompt():
    # I/O & Edge-Case Matrix: "Consult turn with no scene yet (fresh session)" -- grounded
    # in manifest/corpus only, no scene reference.
    adapter = FakeModelAdapter(result=ANSWER_TEXT)

    answer_consult("where should the light go?", MANIFEST, CORPUS, adapter, prior_scene=None)

    user_prompt = adapter.last_request.user_prompt
    assert "<scene>" not in user_prompt
    assert "no current scene" in user_prompt.lower()


# --- with an existing scene -----------------------------------------------------------------


def test_answer_consult_with_prior_scene_includes_it_in_the_user_prompt():
    # I/O & Edge-Case Matrix: "Consult turn with an existing scene" -- answer may reference
    # the current scene's content.
    adapter = FakeModelAdapter(result=ANSWER_TEXT)

    answer_consult(
        "where should the light go?", MANIFEST, CORPUS, adapter, prior_scene=SCENE_TEXT
    )

    user_prompt = adapter.last_request.user_prompt
    assert SCENE_TEXT in user_prompt


def test_answer_consult_delimits_prior_scene_with_scene_tags_not_a_code_fence():
    # Review round in readback.py: a markdown code fence would prematurely close if
    # prior_scene itself contained a triple-backtick sequence, splicing the remainder out of
    # the quoted block -- the same reasoning applies here (Code Map cites readback.py:85).
    adapter = FakeModelAdapter(result=ANSWER_TEXT)

    answer_consult(
        "where should the light go?", MANIFEST, CORPUS, adapter, prior_scene=SCENE_TEXT
    )

    user_prompt = adapter.last_request.user_prompt
    assert "<scene>" in user_prompt
    assert "</scene>" in user_prompt


def test_answer_consult_handles_prior_scene_containing_a_triple_backtick():
    adapter = FakeModelAdapter(result=ANSWER_TEXT)
    tricky_scene_text = SCENE_TEXT + "\n  // a comment mentioning ``` for good measure\n"

    answer_consult(
        "where should the light go?", MANIFEST, CORPUS, adapter, prior_scene=tricky_scene_text
    )

    assert tricky_scene_text in adapter.last_request.user_prompt


# --- Edge-Case Matrix: model call fails/times out ------------------------------------------


def test_model_call_failure_is_a_typed_error_not_an_exception():
    underlying = RuntimeError("connection reset")
    adapter = FakeModelAdapter(
        result=ModelError(kind="call_failed", message="timeout", cause=underlying)
    )

    result = answer_consult("where should the light go?", MANIFEST, CORPUS, adapter)

    assert isinstance(result, ConsultError)
    assert result.kind == "model_call_failed"
    assert result.message == "timeout"
    # Review-round patch: cause must survive the ModelError -> ConsultError mapping, not be
    # silently dropped.
    assert result.cause is underlying


def test_adapter_raising_is_also_a_typed_error_not_an_exception():
    class RaisingAdapter:
        def complete(self, request):
            raise RuntimeError("adapter blew up")

    result = answer_consult("where should the light go?", MANIFEST, CORPUS, RaisingAdapter())

    assert isinstance(result, ConsultError)
    assert result.kind == "model_call_failed"
    assert isinstance(result.cause, RuntimeError)
    assert str(result.cause) == "adapter blew up"


def test_model_call_timeout_is_a_distinct_kind_not_invalid_model_output():
    # Review-round patch: mirrors core/generation.py's story-15 precedent -- a timeout must
    # not be indistinguishable from empty/garbage model output.
    adapter = FakeModelAdapter(result=ModelError(kind="timeout", message="request timed out"))

    result = answer_consult("where should the light go?", MANIFEST, CORPUS, adapter)

    assert isinstance(result, ConsultError)
    assert result.kind == "model_call_timeout"
    assert result.kind != "invalid_model_output"
    assert result.message == "request timed out"


# --- Edge-Case Matrix: model returns empty/unusable response --------------------------------


def test_empty_model_response_is_a_typed_error_not_a_silent_success():
    adapter = FakeModelAdapter(result="")

    result = answer_consult("where should the light go?", MANIFEST, CORPUS, adapter)

    assert isinstance(result, ConsultError)
    assert result.kind == "invalid_model_output"


def test_whitespace_only_model_response_is_a_typed_error():
    adapter = FakeModelAdapter(result="   \n  ")

    result = answer_consult("where should the light go?", MANIFEST, CORPUS, adapter)

    assert isinstance(result, ConsultError)
    assert result.kind == "invalid_model_output"


def test_model_invalid_output_error_maps_to_typed_consult_error():
    adapter = FakeModelAdapter(
        result=ModelError(kind="invalid_output", message="model refused")
    )

    result = answer_consult("where should the light go?", MANIFEST, CORPUS, adapter)

    assert isinstance(result, ConsultError)
    assert result.kind == "invalid_model_output"
    assert result.message == "model refused"

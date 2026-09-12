"""Unit tests for core/turn.py's run_turn() -- the turn pipeline orchestration story (20).

One test per I/O & Edge-Case Matrix row, plus the `generation_failed` precondition path
(a real failure mode -- gauntlet.check_*() requires a str -- not itself an enumerated matrix
row). `validate_scene` is monkey-patched directly (Code Map: "run_turn's tests should
monkeypatch validate_scene itself, not subprocess, since story 10 already covers the
subprocess layer") -- this suite never shells out, never touches Docker. `SceneStore` is
real, backed by `tmp_path`, so every assertion about "no scene file remains"/"a fresh ordinal
exists" is a real filesystem check, matching tests/test_scene_store.py's own pattern."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

import core.turn as turn_module
from adapters.model import ModelError, ModelRequest, ModelResult
from adapters.scene_store import SceneStore, SceneStoreError
from core.turn import run_turn
from core.types import TurnResult, ValidationError, ValidationResult
from tests.fakes import FakeModelAdapter

VALID_MANIFEST = {"schemaVersion": "1.0.0", "objects": [{"name": "Sphere", "fields": []}]}
VALID_CORPUS = {"schemaVersion": "1.0.0", "scenes": []}

# Trips no local gauntlet check (gauntlet/allowlist.py, resource_bounds.py, clean_code.py,
# lint.py): no imports, no resource-bounded constructor calls, no placeholder text, the
# top-level `object` is the first line, and there's no Camera/object/light for lint.py's
# heuristics to reason about at all.
CLEAN_SCENE_TEXT = "object Simple:\n  val scene = Scene()\n"

# Trips clean_code.py's placeholder-text check (a leftover "TODO") -- deliberately the only
# thing wrong with it, so this scripts exactly one local gauntlet finding.
SCENE_TEXT_WITH_TODO = "object Simple:\n  val scene = Scene() // TODO fix this\n"

_SCRIPT_PATH = "/fake/menger/docker/scene-validator/run-sandboxed.sh"


@dataclass
class _SequencedModelAdapter:
    """Returns a different scripted `ModelResult` per call, in order. Needed only for the
    "ok tag but semantic_readback() fails" row: generate() must succeed (return scene text)
    while the *later* semantic_readback() call on the same adapter fails --
    `FakeModelAdapter`'s single scripted `result` answers every call identically and can't
    express that."""

    results: List[ModelResult]

    def __post_init__(self) -> None:
        self._index = 0

    def complete(self, request: ModelRequest) -> ModelResult:
        result = self.results[self._index]
        self._index += 1
        return result


def _make_store(tmp_path) -> SceneStore:
    return SceneStore.create_session(tmp_path, slug="turn-test")


def _staging_path(store: SceneStore):
    return store.session_dir / ".candidate.scala"


def _ordinal_paths(store: SceneStore):
    return sorted(store.session_dir.glob("[0-9]*.scala"))


def _history_entries(store: SceneStore) -> List[dict]:
    import json

    if not store.history_path.exists():
        return []
    return [json.loads(line) for line in store.history_path.read_text(encoding="utf-8").splitlines()]


def _refuse_validate_scene(monkeypatch) -> None:
    """Fails the test loudly if validate_scene() is ever invoked -- used on every path that
    must short-circuit before the renderer is reached."""

    def _boom(*args, **kwargs):
        raise AssertionError("validate_scene() must not be called on this path")

    monkeypatch.setattr(turn_module, "validate_scene", _boom)


# --- Agent-side finding: local gauntlet check trips, renderer never invoked ----------------


def test_local_finding_short_circuits_before_any_renderer_call(tmp_path, monkeypatch):
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=SCENE_TEXT_WITH_TODO)
    store = _make_store(tmp_path)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert isinstance(result, TurnResult)
    assert result.tag == "local_finding"
    assert result.findings  # at least the TODO placeholder finding
    assert any("TODO" in f.message for f in result.findings)
    # No scene file -- staging or accepted -- remains on disk afterward.
    assert not _staging_path(store).exists()
    assert _ordinal_paths(store) == []
    entries = _history_entries(store)
    assert len(entries) == 1
    assert entries[0]["outcome"] == "rejected"
    assert entries[0]["ordinal"] is None


# --- Agent-side clean, renderer ok: readback runs, turn is accepted ------------------------


def test_clean_pass_renderer_ok_readback_ok_is_accepted(tmp_path, monkeypatch):
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        assert script_path == _SCRIPT_PATH
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "accepted"
    assert result.ordinal == 1
    # core/readback.py's semantic_readback() strips the model's raw response -- the scripted
    # FakeModelAdapter result is CLEAN_SCENE_TEXT itself, stripped of its trailing newline.
    assert result.readback_summary == CLEAN_SCENE_TEXT.strip()
    # A fresh ordinal exists in session_dir.
    assert _ordinal_paths(store) == [store.session_dir / "001.scala"]
    # The staging file never survives past run_turn's return.
    assert not _staging_path(store).exists()
    # history.jsonl's new entry includes the readback summary.
    entries = _history_entries(store)
    assert len(entries) == 1
    assert entries[0]["outcome"] == "accepted"
    assert entries[0]["ordinal"] == 1
    assert entries[0]["readback_summary"] == CLEAN_SCENE_TEXT.strip()


# --- Agent-side clean, renderer non-ok: compile_errors / lint_findings / refused -----------


@pytest.mark.parametrize("wire_tag", ["compile_errors", "refused"])
def test_clean_pass_renderer_non_ok_is_rejected_with_tag_and_messages(tmp_path, monkeypatch, wire_tag):
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag=wire_tag, messages=["renderer says no"], findings=[])

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == wire_tag
    assert result.messages == ["renderer says no"]
    assert _ordinal_paths(store) == []
    assert not _staging_path(store).exists()
    entries = _history_entries(store)
    assert entries[-1]["outcome"] == "rejected"
    assert entries[-1]["ordinal"] is None


def test_lint_findings_with_no_messages_still_produces_an_informative_rejection(tmp_path, monkeypatch):
    # story 10's I/O & Edge-Case Matrix: a lint_findings result populates `findings`, not
    # `messages` -- run_turn must still fold the structured findings into a real reason,
    # never record an empty/uninformative rejection.
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        from core.types import ValidationFinding

        return ValidationResult(
            tag="lint_findings",
            messages=[],
            findings=[ValidationFinding(invariant="frustum", message="object behind camera")],
        )

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "lint_findings"
    assert any("object behind camera" in m for m in result.messages)
    entries = _history_entries(store)
    assert entries[-1]["reason"]  # non-empty -- not silently dropped
    assert "object behind camera" in entries[-1]["reason"]
    assert not _staging_path(store).exists()


# --- validate_scene() returns a ValidationError: timeout/malformed/subprocess_failed -------


def test_validate_scene_error_is_rejected_and_staging_file_is_removed(tmp_path, monkeypatch):
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationError(kind="timeout", message="renderer subprocess timed out")

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "timeout"
    assert result.messages == ["renderer subprocess timed out"]
    assert _ordinal_paths(store) == []
    assert not _staging_path(store).exists()
    entries = _history_entries(store)
    assert entries[-1]["outcome"] == "rejected"
    assert "renderer subprocess timed out" in entries[-1]["reason"]


# --- ok tag but semantic_readback() fails: rejection, never a partial accept ---------------


def test_ok_tag_but_readback_failure_is_rejected_not_partially_accepted(tmp_path, monkeypatch):
    # generate() must succeed; the *second* adapter.complete() call (semantic_readback())
    # must fail -- FakeModelAdapter can't express two different outcomes from one instance.
    adapter = _SequencedModelAdapter(
        results=[CLEAN_SCENE_TEXT, ModelError(kind="call_failed", message="readback model down")]
    )
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "readback_failed"
    assert result.messages == ["readback model down"]
    assert result.readback_summary is None
    assert result.ordinal is None
    # No ordinal was consumed -- store.accept() must never have been called.
    assert _ordinal_paths(store) == []
    assert not _staging_path(store).exists()
    entries = _history_entries(store)
    assert entries[-1]["outcome"] == "rejected"
    assert "readback model down" in entries[-1]["reason"]


# --- Precondition: generate()/revise() itself fails --------------------------------------


def test_generation_failure_short_circuits_before_any_local_check_or_renderer_call(tmp_path, monkeypatch):
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=ModelError(kind="call_failed", message="model unreachable"))
    store = _make_store(tmp_path)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "generation_failed"
    assert result.messages == ["model unreachable"]
    assert _ordinal_paths(store) == []
    assert not _staging_path(store).exists()
    entries = _history_entries(store)
    assert entries[-1]["outcome"] == "rejected"


# --- spec-ai-scene-agent story 15: model-call timeout gets its own distinct TurnTag --------


def test_generation_timeout_maps_to_a_distinct_turn_tag_not_generation_failed(tmp_path, monkeypatch):
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=ModelError(kind="timeout", message="model call timed out"))
    store = _make_store(tmp_path)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "generation_timeout"
    assert result.tag != "generation_failed"
    assert result.messages == ["model call timed out"]
    assert _ordinal_paths(store) == []
    assert not _staging_path(store).exists()
    entries = _history_entries(store)
    assert entries[-1]["outcome"] == "rejected"


def test_generation_timeout_via_revise_maps_to_a_distinct_turn_tag_not_generation_failed(
    tmp_path, monkeypatch
):
    # The existing generation-timeout test only covers the generate()/prior_scene=None path
    # -- mirrors tests/test_generation.py testing both generate() and revise() for the same
    # mapping (core/generation.py's _model_error_to_generation_error is shared by both).
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=ModelError(kind="timeout", message="model call timed out"))
    store = _make_store(tmp_path)

    result = run_turn(
        "change it",
        "object Prior:\n  val scene = Scene()\n",
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
    )

    assert result.tag == "generation_timeout"
    assert result.tag != "generation_failed"
    assert result.messages == ["model call timed out"]
    assert _ordinal_paths(store) == []
    assert not _staging_path(store).exists()
    entries = _history_entries(store)
    assert entries[-1]["outcome"] == "rejected"


# --- revise() path: prior_scene is not None -------------------------------------------------


def test_prior_scene_present_uses_revise_not_generate(tmp_path, monkeypatch):
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    result = run_turn(
        "change it",
        "object Prior:\n  val scene = Scene()\n",
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
    )

    assert result.tag == "accepted"
    # revise()'s user prompt carries the prior scene verbatim -- generate()'s does not.
    assert "object Prior" in adapter.requests[0].user_prompt


# --- Patch-level fixes (post-review) --------------------------------------------------------


def test_non_ok_tag_with_empty_messages_and_findings_falls_back_to_tag_string(tmp_path, monkeypatch):
    # A bare "refused" (or any non-"ok" tag) with neither `messages` nor `findings`
    # populated must still leave `TurnResult.messages` non-empty -- the module's own
    # docstring claims `messages` is "always populated with at least one human-readable
    # string on any non-accepted tag."
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag="refused", messages=[], findings=[])

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "refused"
    assert result.messages == ["refused"]
    entries = _history_entries(store)
    assert entries[-1]["reason"] == "refused"


def test_run_turn_forwards_a_non_default_timeout_to_validate_scene(tmp_path, monkeypatch):
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)
    observed = {}

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        observed["timeout"] = timeout
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        timeout=42.5,
    )

    assert observed["timeout"] == 42.5


def test_run_turn_forwards_a_non_default_image_to_validate_scene(tmp_path, monkeypatch):
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)
    observed = {}

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        observed["image"] = image
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        image="base64-image-data",
    )

    assert observed["image"] == "base64-image-data"


def test_store_accept_failure_is_reported_as_storage_failed_not_raised(tmp_path, monkeypatch):
    # store.accept() can raise SceneStoreError after exhausting its ordinal-claim retries
    # under contention (documented on SceneStore.accept()) -- run_turn() must never let
    # that propagate.
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    def failing_accept(self, scene_text, prompt, readback_summary=None):
        raise SceneStoreError("ordinal-claim retries exhausted")

    monkeypatch.setattr(SceneStore, "accept", failing_accept)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "storage_failed"
    assert any("ordinal-claim retries exhausted" in m for m in result.messages)
    assert result.ordinal is None
    assert _ordinal_paths(store) == []
    assert not _staging_path(store).exists()
    entries = _history_entries(store)
    assert entries[-1]["outcome"] == "rejected"


def test_record_rejected_failure_is_folded_into_messages_not_raised(tmp_path, monkeypatch):
    # store.record_rejected() could itself raise (e.g. an I/O error inside SceneStore) --
    # run_turn() must fold both the original rejection reason and the storage failure into
    # the returned TurnResult's messages, never let the exception propagate.
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=SCENE_TEXT_WITH_TODO)
    store = _make_store(tmp_path)

    def failing_record_rejected(self, prompt, reason):
        raise SceneStoreError("history.jsonl append failed: disk full")

    monkeypatch.setattr(SceneStore, "record_rejected", failing_record_rejected)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "local_finding"
    assert any("TODO" in m for m in result.messages)
    assert any("history.jsonl append failed" in m for m in result.messages)


def test_staging_file_write_failure_is_reported_as_storage_failed_not_raised(tmp_path, monkeypatch):
    # write_text() itself can raise OSError (disk full, permission error) -- run_turn()
    # must not let that propagate uncaught.
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    import pathlib

    def failing_write_text(self, data, encoding=None, errors=None, newline=None):
        raise OSError("disk full")

    monkeypatch.setattr(pathlib.Path, "write_text", failing_write_text)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "storage_failed"
    assert any("disk full" in m for m in result.messages)
    assert _ordinal_paths(store) == []


def test_multiple_local_gauntlet_checks_fire_together(tmp_path, monkeypatch):
    # The module's own comment claims findings from all four gauntlet.check_*() stages are
    # "aggregated together ... so a rejected turn's messages/findings report everything
    # wrong with the candidate at once" -- trip clean_code.py's placeholder check and
    # resource_bounds.py's ceiling check simultaneously and confirm both show up.
    _refuse_validate_scene(monkeypatch)
    scene_text = (
        "object Simple:\n"
        "  val scene = Scene(objects = List(Sponge(level = 50f))) // TODO fix this\n"
    )
    adapter = FakeModelAdapter(result=scene_text)
    store = _make_store(tmp_path)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "local_finding"
    stages = {finding.stage for finding in result.findings}
    assert "clean_code" in stages
    assert "resource_bounds" in stages
    assert len(result.findings) >= 2


def test_generation_failure_via_revise_short_circuits_before_any_local_check_or_renderer_call(
    tmp_path, monkeypatch
):
    # The existing generation-failure test only covers the generate()/prior_scene=None path
    # -- this covers the revise() path (prior_scene is not None).
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=ModelError(kind="call_failed", message="model unreachable"))
    store = _make_store(tmp_path)

    result = run_turn(
        "change it",
        "object Prior:\n  val scene = Scene()\n",
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
    )

    assert result.tag == "generation_failed"
    assert result.messages == ["model unreachable"]
    assert _ordinal_paths(store) == []
    assert not _staging_path(store).exists()
    entries = _history_entries(store)
    assert entries[-1]["outcome"] == "rejected"


def test_staging_file_content_matches_candidate_scene_text_when_validate_scene_is_invoked(
    tmp_path, monkeypatch
):
    # Every existing test only asserts the staging file is gone *after* run_turn returns --
    # this captures its content at the moment validate_scene() is actually invoked.
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)
    observed_content = {}

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        from pathlib import Path

        observed_content["text"] = Path(scene_file).read_text(encoding="utf-8")
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    run_turn("make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH)

    assert observed_content["text"] == CLEAN_SCENE_TEXT


# --- Live status line (story 12): on_stage fires at each stage transition, never skipped ---
# and never fired for a stage that's short-circuited past.


def test_on_stage_fires_generating_validating_reading_back_in_order_on_accepted_turn(
    tmp_path, monkeypatch
):
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    stages: List[str] = []
    result = run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        on_stage=stages.append,
    )

    assert result.tag == "accepted"
    assert stages == ["generating", "validating", "reading back"]


def test_on_stage_fires_generating_only_when_generation_fails(tmp_path, monkeypatch):
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=ModelError(kind="call_failed", message="model unreachable"))
    store = _make_store(tmp_path)

    stages: List[str] = []
    result = run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        on_stage=stages.append,
    )

    assert result.tag == "generation_failed"
    assert stages == ["generating"]


def test_on_stage_never_fires_reading_back_on_local_finding(tmp_path, monkeypatch):
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=SCENE_TEXT_WITH_TODO)
    store = _make_store(tmp_path)

    stages: List[str] = []
    result = run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        on_stage=stages.append,
    )

    assert result.tag == "local_finding"
    assert stages == ["generating", "validating"]


def test_on_stage_never_fires_reading_back_on_renderer_rejection(tmp_path, monkeypatch):
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag="compile_errors", messages=["renderer says no"], findings=[])

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    stages: List[str] = []
    result = run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        on_stage=stages.append,
    )

    assert result.tag == "compile_errors"
    assert stages == ["generating", "validating"]


def test_on_stage_omitted_is_a_no_op_and_behaves_exactly_as_before(tmp_path, monkeypatch):
    # Boundaries & Constraints: "Defaults to None (a no-op) so every existing caller/test is
    # unaffected without modification." -- no TypeError, no stray output, same TurnResult as
    # every pre-story-12 test already asserts for this exact scenario.
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    result = run_turn(
        "make a scene", None, VALID_MANIFEST, VALID_CORPUS, adapter, store, _SCRIPT_PATH
    )

    assert result.tag == "accepted"
    assert result.ordinal == 1


# --- Patch-level fixes (post-review, story 12) ----------------------------------------------


def test_on_stage_exception_does_not_abort_turn_and_result_is_still_correct(
    tmp_path, monkeypatch
):
    # An on_stage callback's only job is cosmetic reporting -- a raise from it (or from a
    # non-callable value) must never abort a turn that had otherwise succeeded up to that
    # point.
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    calls: List[str] = []

    def _raising_on_stage(stage: str) -> None:
        calls.append(stage)
        if len(calls) == 1:
            raise RuntimeError("boom: rendering the status line failed")

    result = run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        on_stage=_raising_on_stage,
    )

    assert result.tag == "accepted"
    assert result.ordinal == 1
    assert result.readback_summary == CLEAN_SCENE_TEXT.strip()
    # All three stages were still attempted, despite the first call raising.
    assert calls == ["generating", "validating", "reading back"]


def test_on_stage_sequence_is_exactly_three_stages_on_readback_failure(tmp_path, monkeypatch):
    # Previously untested path: the ReadbackError branch (rejected, not partially accepted)
    # must still fire on_stage for all three stages, in order, never re-fired, never altered.
    adapter = _SequencedModelAdapter(
        results=[CLEAN_SCENE_TEXT, ModelError(kind="call_failed", message="readback model down")]
    )
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationResult(tag="ok", messages=[], findings=[], scene=str(scene_file))

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    stages: List[str] = []
    result = run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        on_stage=stages.append,
    )

    assert result.tag == "readback_failed"
    assert stages == ["generating", "validating", "reading back"]


def test_on_stage_never_fires_reading_back_on_staging_write_failure(tmp_path, monkeypatch):
    # Two independent reviewers flagged this exact path as untested and a real
    # silent-regression risk: a pure storage failure (the staging file write raises
    # OSError) must never fire "reading back" -- nothing would catch a future reordering
    # of the write/emit calls that made it fire spuriously.
    _refuse_validate_scene(monkeypatch)
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    import pathlib

    def failing_write_text(self, data, encoding=None, errors=None, newline=None):
        raise OSError("disk full")

    monkeypatch.setattr(pathlib.Path, "write_text", failing_write_text)

    stages: List[str] = []
    result = run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        on_stage=stages.append,
    )

    assert result.tag == "storage_failed"
    assert stages == ["generating", "validating"]


def test_on_stage_never_fires_reading_back_on_validation_error(tmp_path, monkeypatch):
    # Same reviewers found this branch untested too: validate_scene() returning a
    # ValidationError (not a ValidationResult with a non-"ok" tag -- a structurally
    # distinct rejection path) must also never fire "reading back".
    adapter = FakeModelAdapter(result=CLEAN_SCENE_TEXT)
    store = _make_store(tmp_path)

    def fake_validate_scene(scene_file, script_path, image=None, timeout=None):
        return ValidationError(kind="timeout", message="renderer subprocess timed out")

    monkeypatch.setattr(turn_module, "validate_scene", fake_validate_scene)

    stages: List[str] = []
    result = run_turn(
        "make a scene",
        None,
        VALID_MANIFEST,
        VALID_CORPUS,
        adapter,
        store,
        _SCRIPT_PATH,
        on_stage=stages.append,
    )

    assert result.tag == "timeout"
    assert stages == ["generating", "validating"]

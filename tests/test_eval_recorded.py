"""The recorded-response eval (evals/recorded.py, msa#30) as part of the unit suite: every
recorded case must still pass, so a change to the agent that breaks a turn the model once got
right fails the gate."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

import core.turn
import evals.recorded as recorded
from adapters.artifacts import load_corpus, load_manifest
from evals.recorded import (
    ROOT,
    _turn_problems,
    load_cases,
    main,
    previous_runs,
    ratchet_drops,
    run_case,
)
from gauntlet.types import Finding

MANIFEST = load_manifest(ROOT / "reference" / "dsl-manifest.json")
CORPUS = load_corpus(ROOT / "reference" / "dsl-corpus.json")
CASES = load_cases()
# A two-turn sponge case the tests below mutate; picked by id, not by sorted position.
SPONGE = next(c for c in CASES if c["id"] == "luna-answered-material-is-not-asked-again")


def test_there_are_cases():
    assert len(CASES) >= 3


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_recorded_case_passes(case):
    result = run_case(case, MANIFEST, CORPUS)
    assert result.problems == []


def test_a_wrong_expectation_fails():
    case = copy.deepcopy(SPONGE)
    case["turns"][0]["expect"] = {"objects": ["NoSuchType"]}
    assert run_case(case, MANIFEST, CORPUS).problems


def test_an_agent_that_rejects_every_scene_fails_every_case(monkeypatch):
    monkeypatch.setattr(
        core.turn, "_run_local_checks",
        lambda scene: [Finding(stage="lint", message="rejected by the mutation")],
    )
    for case in CASES:
        assert not run_case(case, MANIFEST, CORPUS).passed, case["id"]


def test_live_mode_counts_the_questions_the_model_asks():
    from tests.fakes import FakeModelAdapter

    asker = FakeModelAdapter(result="NEEDS_CLARIFICATION: which material?")
    result = run_case(SPONGE, MANIFEST, CORPUS, live=asker)
    assert not result.passed
    assert result.questions[0] == "NEEDS_CLARIFICATION: which material?"


# --- msa#57: conditions per usability-script request ---------------------------------------

_TURN_KEYS = {"prompt", "answer", "answered", "responses", "expect", "validator"}
_EXPECT_KEYS = {
    "questions_at_most", "tag", "message", "objects", "no_objects", "object_count", "materials",
    "lights", "levels", "above", "contains", "not_contains", "min_count", "changed",
    "camera_farther", "dark_background",
}


@pytest.mark.parametrize("case", load_cases(live=True), ids=lambda c: c["id"])
def test_every_case_uses_only_known_keys(case):
    # A misspelt condition would be ignored silently and the case would pass on nothing.
    for turn in case["turns"]:
        assert set(turn) <= _TURN_KEYS, turn["prompt"]
        assert set(turn.get("expect", {})) <= _EXPECT_KEYS, turn["prompt"]


def test_the_script_cases_are_live_only():
    script = [c for c in load_cases(live=True) if c not in CASES]
    assert len(script) >= 4
    assert all("responses" not in t for c in script for t in c["turns"])


_LOW = "object A:\n  val scene = Scene(camera = Camera(position = (0f, 0f, 4f), lookAt = (0f, 0f, 0f)), objects = List(Sponge(pos = (0f, 0f, 0f), level = 2f), Sphere(pos = (0f, 2f, 0f))))\n"
_FAR = _LOW.replace("(0f, 0f, 4f)", "(0f, 0f, 9f)").replace("level = 2f", "level = 2.5f")


@pytest.mark.parametrize("expect", [
    {"levels": {"Sponge": [2.5, 2.5]}},
    {"no_objects": ["Sphere"]},
    {"object_count": {"Sphere": [2, 3]}},
    {"above": [["Sponge", "Sphere"]]},
    {"min_count": {"Sphere\\(": 2}},
    {"not_contains": ["level = 2f"]},
    {"changed": ["position = \\([^)]*\\)"]},
    {"camera_farther": True},
    {"questions_at_most": 0},
    {"tag": ["unsupported", "unchanged"]},
])
def test_each_condition_can_fail(expect):
    assert _turn_problems(expect, _LOW, "accepted", questions=1, prior=_LOW)


def test_the_conditions_hold_on_a_scene_that_meets_them():
    expect = {"levels": {"Sponge": [2.5, 2.5]}, "above": [["Sphere", "Sponge"]],
              "changed": ["position = \\([^)]*\\)"], "camera_farther": True,
              "object_count": {"Sphere": [1, 1]}, "tag": ["accepted", "unchanged"]}
    assert _turn_problems(expect, _FAR, "accepted", questions=1, prior=_LOW) == []


def test_an_answer_is_threaded_when_the_model_asks():
    scene = SPONGE["turns"][0]["responses"]
    case = {"id": "answer", "turns": [{
        "prompt": "a menger sponge level 2", "answer": "yes",
        "responses": ["NEEDS_CLARIFICATION: gold or glass?", *scene],
        "expect": {"objects": ["Sponge"]},
    }]}
    result = run_case(case, MANIFEST, CORPUS)
    assert result.problems == []
    assert result.questions == ["NEEDS_CLARIFICATION: gold or glass?"]


# --- kept live runs, compare, ratchet (msa#57) ---------------------------------------------


def _keep(monkeypatch, tmp_path, stamp, results):
    """Writes one live run of `{case: [passed?, ...]}` as `_write_live_results` would."""
    monkeypatch.setattr(recorded, "LIVE_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(recorded, "selected_provider", lambda: "litellm")
    runs = {case: [recorded.CaseResult(case, [] if ok else ["wrong"]) for ok in oks]
            for case, oks in results.items()}
    return recorded._write_live_results(SimpleNamespace(model_name="vendor/luna"), runs,
                                        stamp=stamp, revision="abc1234")


def test_live_runs_are_kept_per_model_with_a_latest_copy(monkeypatch, tmp_path):
    first = _keep(monkeypatch, tmp_path, "20261023T100000Z", {"a": [True]})
    second = _keep(monkeypatch, tmp_path, "20261023T110000Z", {"a": [False]})
    assert first.parent == tmp_path / "litellm-vendor_luna"
    assert first.name == "20261023T100000Z-abc1234.json" and first.is_file()
    assert (first.parent / "latest.json").read_text() == second.read_text()


def test_compare_prints_each_case_s_rate_and_change(monkeypatch, tmp_path, capsys):
    a = _keep(monkeypatch, tmp_path, "20261023T100000Z", {"a": [True, True], "b": [True]})
    b = _keep(monkeypatch, tmp_path, "20261023T110000Z", {"a": [True, False], "c": [False]})
    assert main(["--compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["100% ->  50%  -50%  a", "100% ->    -  b", "   - ->   0%  c"]


def test_the_ratchet_flags_a_drop_of_more_than_one_run_s_share(monkeypatch, tmp_path):
    for n, oks in enumerate(([True, True, True], [True, True, False], [True, False, False])):
        _keep(monkeypatch, tmp_path, f"20261023T10000{n}Z", {"a": oks, "b": [True] * 3})
    latest = _keep(monkeypatch, tmp_path, "20261023T100009Z",
                   {"a": [True, False, False], "b": [True, True, False], "new": [False] * 3})
    previous = previous_runs(latest)
    assert [p.name[:16] for p in previous] == [f"20261023T10000{n}Z" for n in range(3)]
    # a: 33% against a best of 100% -- flagged; b: 67% against 100% is one run's share; a case
    # without history has no best.
    assert ratchet_drops(latest, previous) == [
        "a: 1/3 runs, best of the last 3 kept run(s) 100%"]


# --- recorded validator results (menger#78) -------------------------------------------------


def test_every_recorded_case_has_its_validator_results():
    for case in CASES:
        # A turn answered by a question before any model call has no validator call: [].
        assert all("validator" in turn for turn in case["turns"]), case["id"]


def test_the_recorded_validator_results_answer_the_replay():
    case = copy.deepcopy(SPONGE)
    case["turns"][0]["validator"] = [{"exit": 1, "result": {
        "tag": "lint-findings", "messages": [], "schemaVersion": "1.2.0",
        "findings": [{"invariant": "scene-budget", "message": "over budget"}]}}]
    problems = run_case(case, MANIFEST, CORPUS).problems
    assert any("tag lint_findings" in p for p in problems), problems


def test_a_real_validator_that_answers_differently_is_reported(tmp_path):
    # The stub prints a bare "ok": no snapshots, unlike the recorded results.
    validator = tmp_path / "validator.sh"
    validator.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{recorded.STUB_VALIDATOR}" "$@"\n')
    validator.chmod(0o755)
    problems = run_case(SPONGE, MANIFEST, CORPUS, validator=validator).problems
    assert any("snapshots differs from the recording" in p for p in problems), problems


def test_a_newer_result_schema_version_is_not_drift():
    # menger#113: an additive schema bump changes only schemaVersion in every recorded answer.
    old = {"exit": 0, "result": {"tag": "ok", "messages": [], "schemaVersion": "1.2.0"}}
    new = {"exit": 0, "result": {"tag": "ok", "messages": [], "schemaVersion": "1.3.0"}}

    assert recorded._drift([old], [new]) == []


def test_a_changed_answer_is_still_drift_when_the_schema_version_differs_too():
    old = {"exit": 0, "result": {"tag": "ok", "messages": [], "schemaVersion": "1.2.0"}}
    new = {"exit": 1, "result": {"tag": "lint-findings", "messages": ["over budget"],
                                 "schemaVersion": "1.3.0"}}

    drift = recorded._drift([old], [new])

    assert "validator call 1: tag differs from the recording" in drift
    assert "validator call 1: messages differs from the recording" in drift
    assert not any("schemaVersion" in d for d in drift)


def test_record_writes_the_validator_results_into_the_case(tmp_path):
    case = copy.deepcopy(SPONGE)
    for turn in case["turns"]:
        del turn["validator"]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    assert main(["--record", str(path)]) == 0
    written = json.loads(path.read_text())
    assert [len(t["validator"]) for t in written["turns"]] == [1] * len(case["turns"])
    assert written["turns"][0]["validator"][0]["result"]["tag"] == "ok"
    assert {k: v for k, v in written.items() if k != "turns"} == {
        k: v for k, v in case.items() if k != "turns"}

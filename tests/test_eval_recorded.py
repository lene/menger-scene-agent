"""The recorded-response eval (evals/recorded.py, msa#30) as part of the unit suite: every
recorded case must still pass, so a change to the agent that breaks a turn the model once got
right fails the gate."""

from __future__ import annotations

import copy

import pytest

import core.turn
from adapters.artifacts import load_corpus, load_manifest
from evals.recorded import ROOT, _turn_problems, load_cases, run_case
from gauntlet.types import Finding

MANIFEST = load_manifest(ROOT / "reference" / "dsl-manifest.json")
CORPUS = load_corpus(ROOT / "reference" / "dsl-corpus.json")
CASES = load_cases()


def test_there_are_cases():
    assert len(CASES) >= 3


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_recorded_case_passes(case):
    result = run_case(case, MANIFEST, CORPUS)
    assert result.problems == []


def test_a_wrong_expectation_fails():
    case = copy.deepcopy(CASES[0])
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
    result = run_case(CASES[0], MANIFEST, CORPUS, live=asker)
    assert not result.passed
    assert result.questions[0] == "NEEDS_CLARIFICATION: which material?"


# --- msa#57: conditions per usability-script request ---------------------------------------

_TURN_KEYS = {"prompt", "answer", "answered", "responses", "expect"}
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
    scene = CASES[0]["turns"][0]["responses"]
    case = {"id": "answer", "turns": [{
        "prompt": "a menger sponge level 2", "answer": "yes",
        "responses": ["NEEDS_CLARIFICATION: gold or glass?", *scene],
        "expect": {"objects": ["Sponge"]},
    }]}
    result = run_case(case, MANIFEST, CORPUS)
    assert result.problems == []
    assert result.questions == ["NEEDS_CLARIFICATION: gold or glass?"]

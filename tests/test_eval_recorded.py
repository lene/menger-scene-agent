"""The recorded-response eval (evals/recorded.py, msa#30) as part of the unit suite: every
recorded case must still pass, so a change to the agent that breaks a turn the model once got
right fails the gate."""

from __future__ import annotations

import copy

import pytest

import core.turn
from adapters.artifacts import load_corpus, load_manifest
from evals.recorded import ROOT, load_cases, run_case
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

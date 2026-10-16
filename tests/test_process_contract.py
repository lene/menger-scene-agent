"""The agent's side of menger's process contract (reference/process-result.json, review
2026-10-14, menger#82, #100, #116): both parsers accept exactly what the contract allows."""

from __future__ import annotations

import json

import pytest

from adapters.process_contract import CONTRACT, EXIT_CODES, RESULT_SCHEMA_VERSION, result_problem
from adapters.scene_validator import _parse_stdout
from core.types import ValidationError, ValidationResult


def _result(tag: str, **fields) -> str:
    return json.dumps({"tag": tag, "messages": [], "findings": [],
                       "schemaVersion": RESULT_SCHEMA_VERSION, **fields})


@pytest.mark.parametrize("tag", sorted(EXIT_CODES))
def test_every_tag_with_its_exit_code_is_accepted(tag):
    outcome = _parse_stdout(_result(tag), "", EXIT_CODES[tag])
    assert isinstance(outcome, ValidationResult)


def test_an_exit_code_that_disagrees_with_the_tag_is_malformed():
    outcome = _parse_stdout(_result("ok"), "", EXIT_CODES["refused"])
    assert isinstance(outcome, ValidationError)
    assert outcome.kind == "malformed_output"
    assert "does not match tag 'ok'" in outcome.message


def test_a_result_without_a_schema_version_points_at_the_image():
    payload = json.loads(_result("ok"))
    del payload["schemaVersion"]
    outcome = _parse_stdout(json.dumps(payload), "", 0)
    assert isinstance(outcome, ValidationError)
    assert "rebuild it" in outcome.message


def test_a_newer_major_is_refused_and_a_newer_minor_accepted():
    major = int(RESULT_SCHEMA_VERSION.split(".")[0])
    assert result_problem("ok", f"{major + 1}.0.0", 0) is not None
    assert result_problem("ok", f"{major}.99.0", 0) is None


def test_the_contract_copy_describes_the_window_events_the_adapter_reads():
    from adapters.render_window import LAUNCH_EVENTS, RELOAD_EVENTS

    kinds = set(CONTRACT["windowEvents"]["kinds"])
    assert set(LAUNCH_EVENTS) | set(RELOAD_EVENTS) | {"startup_failed", "frame_failed"} <= kinds

"""The repair feedback built from a validator result: compile errors come one per line when the
validator says where they are (result schema 1.3.0, menger#113)."""

from __future__ import annotations

from core.turn import _validation_messages
from core.types import ValidationResult


def _compile_error(**fields) -> ValidationResult:
    return ValidationResult(
        tag="compile_errors",
        messages=["Compilation of 'scene.scala' failed:\nline 3: Found: Int\nline 7: not found: x"],
        **fields,
    )


def test_each_compile_error_is_its_own_line_numbered_message():
    load_error = {"kind": "compile_failed", "diagnostics": [
        {"line": 3, "message": "Found: Int"}, {"line": 7, "message": "not found: x"}]}

    assert _validation_messages(_compile_error(load_error=load_error)) == [
        "line 3: Found: Int", "line 7: not found: x"]


def test_a_compile_error_without_a_position_is_just_its_message():
    load_error = {"kind": "compile_failed", "diagnostics": [{"line": 0, "message": "bad option"}]}

    assert _validation_messages(_compile_error(load_error=load_error)) == ["bad option"]


def test_an_older_validator_result_keeps_the_validators_own_message():
    result = _compile_error()

    assert _validation_messages(result) == result.messages


def test_a_load_error_that_is_not_a_compile_failure_keeps_the_validators_message():
    result = ValidationResult(
        tag="lint_findings", messages=["scene construction failed: Size must be positive"],
        load_error={"kind": "construction_failed", "diagnostics": []})

    assert _validation_messages(result) == result.messages

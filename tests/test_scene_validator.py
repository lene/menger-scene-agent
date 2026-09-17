"""Unit tests for adapters/scene_validator.py -- the renderer-side gauntlet adapter (story
10). One test per I/O & Edge-Case Matrix row. The subprocess boundary (`subprocess.run`) is
always monkey-patched with a scripted fake `subprocess.CompletedProcess` or a scripted
exception -- this suite never invokes real Docker or a real `run-sandboxed.sh`."""

from __future__ import annotations

import json
import subprocess
from typing import Any, List

import pytest

from adapters.scene_validator import validate_scene
from core.types import ValidationError, ValidationFinding, ValidationResult

_SCRIPT_PATH = "/fake/menger/docker/scene-validator/run-sandboxed.sh"
_SCENE_FILE = "/fake/scene-workspace/session/001.scala"


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[_SCRIPT_PATH, _SCENE_FILE], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _patch_run(monkeypatch, fake_run) -> List[Any]:
    """Monkey-patches `adapters.scene_validator.subprocess.run` with `fake_run` and returns
    the list of positional `cmd` arguments each call was invoked with, for assertions on the
    command actually built."""
    calls: List[Any] = []

    def recording_run(cmd, **kwargs):
        calls.append(cmd)
        return fake_run(cmd, **kwargs)

    import adapters.scene_validator as scene_validator

    monkeypatch.setattr(scene_validator.subprocess, "run", recording_run)
    return calls


# --- Valid scene, no issues (exit 0, tag ok) ------------------------------------------------


def test_exit_0_ok_returns_typed_result_tag_ok_scene_populated(monkeypatch):
    payload = {
        "tag": "ok",
        "messages": [],
        "findings": [],
        "scene": _SCENE_FILE,
        "schemaVersion": "1.0.0",
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationResult)
    assert result.tag == "ok"
    assert result.scene == _SCENE_FILE
    assert result.messages == []
    assert result.findings == []


# --- Compile failure (exit 1, tag compile-errors) -------------------------------------------


def test_exit_1_compile_errors_returns_typed_result_messages_populated(monkeypatch):
    payload = {
        "tag": "compile-errors",
        "messages": ["Compilation of scene.scala failed: not found: value Sponge2"],
        "findings": [],
        "scene": None,
        "schemaVersion": "1.0.0",
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(1, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationResult)
    assert result.tag == "compile_errors"
    assert result.messages == ["Compilation of scene.scala failed: not found: value Sponge2"]


# --- Geometric check failure (exit 1, tag lint-findings) ------------------------------------


def test_exit_1_lint_findings_returns_typed_result_findings_populated(monkeypatch):
    payload = {
        "tag": "lint-findings",
        "messages": ["edge-length-invariant: edges must be equal length"],
        "findings": [{"invariant": "edge-length-invariant", "message": "edges must be equal length"}],
        "scene": _SCENE_FILE,
        "schemaVersion": "1.0.0",
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(1, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationResult)
    assert result.tag == "lint_findings"
    assert result.findings == [
        ValidationFinding(invariant="edge-length-invariant", message="edges must be equal length")
    ]


# --- Renderer refuses (exit 2, tag refused) -------------------------------------------------


def test_exit_2_refused_returns_typed_result_tag_refused(monkeypatch):
    payload = {
        "tag": "refused",
        "messages": ["restricted classpath is not usable: stale manifest"],
        "findings": [],
        "scene": None,
        "schemaVersion": "1.0.0",
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(2, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationResult)
    assert result.tag == "refused"


# --- Subprocess times out (exit 124, no JSON on stdout) -------------------------------------


def test_exit_124_returns_a_distinct_typed_timeout_error_not_a_parse_crash(monkeypatch):
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(124, stdout=""))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "timeout"


def test_exit_124_never_attempts_to_parse_stdout_as_json_even_if_stdout_is_not_empty(monkeypatch):
    # Defensive: even if something ends up on stdout alongside a 124 exit, the timeout path
    # must be taken unconditionally, never a JSON-parse attempt.
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(124, stdout="not valid json{"))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "timeout"


def test_subprocess_run_raising_timeout_expired_is_also_a_typed_timeout_error(monkeypatch):
    def raise_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    _patch_run(monkeypatch, raise_timeout)

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH, timeout=5)

    assert isinstance(result, ValidationError)
    assert result.kind == "timeout"


# --- Malformed/unexpected stdout on any exit code --------------------------------------------


def test_malformed_json_returns_a_typed_parse_error_not_an_unhandled_exception(monkeypatch):
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout="not valid json{"))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_empty_stdout_on_a_non_timeout_exit_returns_a_typed_parse_error(monkeypatch):
    # E.g. run-sandboxed.sh's own "Scene file not found" usage-error path: exit 1, a plain
    # message on stderr, and nothing at all on stdout.
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(1, stdout="", stderr="Scene file not found"))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_valid_json_but_not_an_object_returns_a_typed_parse_error(monkeypatch):
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(["ok"])))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_unexpected_tag_value_returns_a_typed_parse_error(monkeypatch):
    payload = {"tag": "not-a-real-tag", "messages": [], "findings": [], "schemaVersion": "1.0.0"}
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_missing_tag_field_returns_a_typed_parse_error(monkeypatch):
    payload = {"messages": [], "findings": [], "schemaVersion": "1.0.0"}
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_malformed_finding_shape_returns_a_typed_parse_error(monkeypatch):
    payload = {
        "tag": "lint-findings",
        "messages": ["bad"],
        "findings": [{"invariant": "only-invariant-no-message"}],
        "schemaVersion": "1.0.0",
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(1, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_finding_with_non_string_invariant_or_message_returns_a_typed_parse_error(monkeypatch):
    # Both keys present (no KeyError), but the wrong type -- distinct from the missing-key
    # case above, and only caught by an explicit type check on the finding fields.
    payload = {
        "tag": "lint-findings",
        "messages": ["bad"],
        "findings": [{"invariant": 42, "message": "edges must be equal length"}],
        "schemaVersion": "1.0.0",
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(1, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_null_messages_and_findings_default_to_empty_list(monkeypatch):
    # An explicit JSON null is treated the same as the field being omitted -- consistent with
    # how `scene: null` is already allowed.
    payload = {
        "tag": "ok",
        "messages": None,
        "findings": None,
        "scene": None,
        "schemaVersion": "1.0.0",
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationResult)
    assert result.messages == []
    assert result.findings == []


def test_non_list_non_str_messages_returns_a_typed_parse_error(monkeypatch):
    payload = {
        "tag": "ok",
        "messages": [1, 2, 3],
        "findings": [],
        "schemaVersion": "1.0.0",
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_non_string_scene_returns_a_typed_parse_error(monkeypatch):
    payload = {
        "tag": "ok",
        "messages": [],
        "findings": [],
        "scene": 12345,
        "schemaVersion": "1.0.0",
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_non_string_schema_version_returns_a_typed_parse_error(monkeypatch):
    payload = {
        "tag": "ok",
        "messages": [],
        "findings": [],
        "schemaVersion": 100,
    }
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(payload)))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"


def test_empty_stdout_error_message_includes_stderr_diagnostic(monkeypatch):
    # run-sandboxed.sh's own "Scene file not found" usage-error path: the real diagnostic is
    # on stderr while stdout is empty -- the error message must not drop it.
    _patch_run(
        monkeypatch,
        lambda cmd, **kwargs: _completed(1, stdout="", stderr="Scene file not found"),
    )

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "malformed_output"
    assert "Scene file not found" in result.message


# --- Non-UTF-8 subprocess output ------------------------------------------------------------


def test_subprocess_run_raising_unicode_decode_error_is_a_typed_subprocess_failed_error(
    monkeypatch,
):
    def raise_unicode_decode_error(cmd, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    _patch_run(monkeypatch, raise_unicode_decode_error)

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)  # must not raise

    assert isinstance(result, ValidationError)
    assert result.kind == "subprocess_failed"


# --- Non-str `image` argument -----------------------------------------------------------------


def test_non_str_image_returns_a_typed_error_without_invoking_subprocess(monkeypatch):
    calls = _patch_run(
        monkeypatch,
        lambda cmd, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess.run must not be called")),
    )

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH, image=123)  # type: ignore[arg-type]

    assert isinstance(result, ValidationError)
    assert result.kind == "subprocess_failed"
    assert calls == []


# --- Subprocess itself cannot be launched (never in the I/O matrix by name, but "never
# raises" per Boundaries & Constraints applies to this failure mode too) --------------------


def test_subprocess_launch_failure_returns_a_typed_error_not_an_exception(monkeypatch):
    def raise_oserror(cmd, **kwargs):
        raise FileNotFoundError("run-sandboxed.sh not found")

    _patch_run(monkeypatch, raise_oserror)

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert isinstance(result, ValidationError)
    assert result.kind == "subprocess_failed"


# --- The command actually built -------------------------------------------------------------


def test_builds_the_command_with_scene_file_and_script_path(monkeypatch):
    payload = {"tag": "ok", "messages": [], "findings": [], "schemaVersion": "1.0.0"}
    calls = _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(payload)))

    validate_scene(_SCENE_FILE, _SCRIPT_PATH)

    assert calls == [[_SCRIPT_PATH, _SCENE_FILE]]


def test_builds_the_command_with_an_image_tag_when_given(monkeypatch):
    payload = {"tag": "ok", "messages": [], "findings": [], "schemaVersion": "1.0.0"}
    calls = _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, stdout=json.dumps(payload)))

    validate_scene(_SCENE_FILE, _SCRIPT_PATH, image="menger-scene-validator:test")

    assert calls == [[_SCRIPT_PATH, _SCENE_FILE, "--image", "menger-scene-validator:test"]]


def test_subprocess_run_is_invoked_with_capture_output_text_and_timeout_kwargs(monkeypatch):
    payload = {"tag": "ok", "messages": [], "findings": [], "schemaVersion": "1.0.0"}
    recorded_kwargs: List[Any] = []

    def fake_run(cmd, **kwargs):
        recorded_kwargs.append(kwargs)
        return _completed(0, stdout=json.dumps(payload))

    import adapters.scene_validator as scene_validator

    monkeypatch.setattr(scene_validator.subprocess, "run", fake_run)

    validate_scene(_SCENE_FILE, _SCRIPT_PATH, timeout=42)

    assert len(recorded_kwargs) == 1
    assert recorded_kwargs[0]["capture_output"] is True
    assert recorded_kwargs[0]["text"] is True
    assert recorded_kwargs[0]["timeout"] == 42


def test_subprocess_run_is_invoked_with_a_plain_argv_list_and_never_shell_true(monkeypatch):
    # Protects the sandboxing/no-shell-interpolation guarantee the module's docstring claims.
    payload = {"tag": "ok", "messages": [], "findings": [], "schemaVersion": "1.0.0"}
    recorded: List[Any] = []

    def fake_run(cmd, **kwargs):
        recorded.append((cmd, kwargs))
        return _completed(0, stdout=json.dumps(payload))

    import adapters.scene_validator as scene_validator

    monkeypatch.setattr(scene_validator.subprocess, "run", fake_run)

    validate_scene(_SCENE_FILE, _SCRIPT_PATH, image="menger-scene-validator:test")

    assert len(recorded) == 1
    cmd, kwargs = recorded[0]
    assert isinstance(cmd, list)
    assert all(isinstance(part, str) for part in cmd)
    assert kwargs.get("shell") is not True
    assert "shell" not in kwargs


@pytest.mark.parametrize("returncode", [0, 1, 2])
def test_never_raises_regardless_of_exit_code(monkeypatch, returncode):
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(returncode, stdout="garbage"))

    result = validate_scene(_SCENE_FILE, _SCRIPT_PATH)  # must not raise

    assert isinstance(result, ValidationError)

"""Unit tests for adapters/render_window.py -- the render-window launch adapter (story 13).
One test per I/O & Edge-Case Matrix row. `subprocess.Popen` is always monkey-patched with a
scripted `FakePopen` (never launches a real window/process), and `grace_period` is set tiny so
these tests run fast."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any, List, Optional

import pytest

from adapters.render_window import refresh_render_window
from core.types import RenderWindowError, RenderWindowResult

_LAUNCHER_PATH = "/fake/menger-app/target/universal/stage/bin/menger-app"
_SCENE_FILE = "/fake/scene-workspace/session/001.scala"
_GRACE_PERIOD = 0.01


class FakePopen:
    """A scripted stand-in for `subprocess.Popen` -- never launches a real process.

    `still_running=True` simulates a launch that survives the grace period: `wait()` raises
    `subprocess.TimeoutExpired` until `terminate()`/`kill()` has been called (mirroring a real
    process that only exits once asked to). `still_running=False` simulates a fast exit:
    `wait()` returns `returncode` immediately, with `stdout`/`stderr` available to read.
    """

    def __init__(
        self, returncode: int = 0, stdout: str = "", stderr: str = "", still_running: bool = False
    ):
        self.returncode = returncode
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self._still_running = still_running
        self.terminated = False
        self.killed = False
        self.wait_calls: List[Optional[float]] = []
        self.communicate_calls: List[Optional[float]] = []

    def poll(self) -> Optional[int]:
        return None if self._still_running else self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        self.wait_calls.append(timeout)
        if self._still_running and not (self.terminated or self.killed):
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        self._still_running = False
        return self.returncode

    def communicate(self, timeout: Optional[float] = None):
        self.communicate_calls.append(timeout)
        if self._still_running and not (self.terminated or self.killed):
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        self._still_running = False
        return self.stdout.read(), self.stderr.read()

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _patch_popen(monkeypatch, fake_popen: FakePopen) -> List[Any]:
    """Monkey-patches `adapters.render_window.subprocess.Popen` to return `fake_popen`
    (or raise, if `fake_popen` is an exception instance) and returns the list of positional
    `cmd` arguments each call was invoked with, for assertions on the command actually built."""
    calls: List[Any] = []

    def recording_popen(cmd, **kwargs):
        calls.append(cmd)
        if isinstance(fake_popen, BaseException):
            raise fake_popen
        return fake_popen

    import adapters.render_window as render_window

    monkeypatch.setattr(render_window.subprocess, "Popen", recording_popen)
    return calls


# --- No prior process, launch succeeds -------------------------------------------------------


def test_no_prior_process_launch_succeeds_returns_running_handle(monkeypatch):
    fake = FakePopen(still_running=True)
    calls = _patch_popen(monkeypatch, fake)

    result = refresh_render_window(
        _SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD
    )

    assert isinstance(result, RenderWindowResult)
    assert result.process is fake
    assert calls == [[_LAUNCHER_PATH, "--scene", _SCENE_FILE]]
    # The caller-supplied grace_period must be the exact value passed through to the
    # underlying communicate() call, not e.g. a default or a mangled value.
    assert fake.communicate_calls == [_GRACE_PERIOD]


def test_no_prior_process_accepts_path_objects_for_scene_file_and_launcher_path(monkeypatch):
    fake = FakePopen(still_running=True)
    calls = _patch_popen(monkeypatch, fake)

    result = refresh_render_window(
        Path(_SCENE_FILE), Path(_LAUNCHER_PATH), grace_period=_GRACE_PERIOD
    )

    assert isinstance(result, RenderWindowResult)
    assert calls == [[_LAUNCHER_PATH, "--scene", _SCENE_FILE]]


def test_builds_command_with_display_only_when_render_lock_path_not_given(monkeypatch):
    fake = FakePopen(still_running=True)
    calls = _patch_popen(monkeypatch, fake)

    refresh_render_window(
        _SCENE_FILE, _LAUNCHER_PATH, display=":1", grace_period=_GRACE_PERIOD
    )

    assert calls == [[_LAUNCHER_PATH, "--scene", _SCENE_FILE, "--display", ":1"]]


def test_builds_command_with_render_lock_path_only_when_display_not_given(monkeypatch):
    fake = FakePopen(still_running=True)
    calls = _patch_popen(monkeypatch, fake)

    refresh_render_window(
        _SCENE_FILE,
        _LAUNCHER_PATH,
        render_lock_path="/fake/render.lock",
        grace_period=_GRACE_PERIOD,
    )

    assert calls == [
        [_LAUNCHER_PATH, "--scene", _SCENE_FILE, "--render-lock-path", "/fake/render.lock"]
    ]


def test_builds_command_with_display_and_render_lock_path_when_given(monkeypatch):
    fake = FakePopen(still_running=True)
    calls = _patch_popen(monkeypatch, fake)

    refresh_render_window(
        _SCENE_FILE,
        _LAUNCHER_PATH,
        display=":1",
        render_lock_path="/fake/render.lock",
        grace_period=_GRACE_PERIOD,
    )

    assert calls == [
        [
            _LAUNCHER_PATH,
            "--scene",
            _SCENE_FILE,
            "--display",
            ":1",
            "--render-lock-path",
            "/fake/render.lock",
        ]
    ]


# --- `previous_process` still running ---------------------------------------------------------


def test_still_running_previous_process_is_terminated_before_fresh_launch(monkeypatch):
    previous = FakePopen(still_running=True)
    fresh = FakePopen(still_running=True)
    calls = _patch_popen(monkeypatch, fresh)

    result = refresh_render_window(
        _SCENE_FILE, _LAUNCHER_PATH, previous_process=previous, grace_period=_GRACE_PERIOD
    )

    assert previous.terminated is True
    assert isinstance(result, RenderWindowResult)
    assert result.process is fresh
    assert calls == [[_LAUNCHER_PATH, "--scene", _SCENE_FILE]]
    # The same grace_period the caller supplied must reach the previous_process termination
    # wait, not a default or a different value.
    assert previous.wait_calls == [_GRACE_PERIOD]


def test_already_exited_previous_process_is_not_terminated(monkeypatch):
    # poll() returning a real exit code (not None) means it's already dead -- terminate()
    # must not be called on a process that isn't running.
    previous = FakePopen(still_running=False, returncode=0)
    fresh = FakePopen(still_running=True)
    _patch_popen(monkeypatch, fresh)

    refresh_render_window(
        _SCENE_FILE, _LAUNCHER_PATH, previous_process=previous, grace_period=_GRACE_PERIOD
    )

    assert previous.terminated is False


def test_previous_process_not_responding_to_terminate_is_killed(monkeypatch):
    # A previous process that ignores terminate() (still "running" even after it) must be
    # killed rather than left orphaned -- never left un-reaped.
    class StubbornFakePopen(FakePopen):
        def wait(self, timeout: Optional[float] = None) -> int:
            self.wait_calls.append(timeout)
            if self._still_running and not self.killed:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            self._still_running = False
            return self.returncode

    previous = StubbornFakePopen(still_running=True)
    fresh = FakePopen(still_running=True)
    _patch_popen(monkeypatch, fresh)

    refresh_render_window(
        _SCENE_FILE, _LAUNCHER_PATH, previous_process=previous, grace_period=_GRACE_PERIOD
    )

    assert previous.terminated is True
    assert previous.killed is True


# --- Another process holds the AD-16 lock -----------------------------------------------------


def test_lock_held_by_another_process_returns_typed_refused_rejection(monkeypatch):
    payload = {
        "tag": "refused",
        "messages": ["render lock already held"],
        "findings": [],
        "scene": None,
        "schemaVersion": "1.0.0",
    }
    fake = FakePopen(returncode=1, stdout=json.dumps(payload), still_running=False)
    _patch_popen(monkeypatch, fake)

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "refused"
    assert "render lock already held" in result.message


def test_refused_with_no_previous_process_terminates_nothing(monkeypatch):
    payload = {"tag": "refused", "messages": ["render lock already held"]}
    fake = FakePopen(returncode=1, stdout=json.dumps(payload), still_running=False)
    _patch_popen(monkeypatch, fake)

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "refused"
    # Nothing else exists to terminate in this scenario; the fast-exiting launcher process
    # itself was never a `previous_process` and must not be touched beyond reading its output.
    assert fake.terminated is False
    assert fake.killed is False


# --- Launcher exits fast with unexpected/malformed output --------------------------------------


def test_malformed_json_on_fast_exit_returns_typed_malformed_output_error(monkeypatch):
    fake = FakePopen(returncode=1, stdout="not valid json{", still_running=False)
    _patch_popen(monkeypatch, fake)

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "malformed_output"


def test_unexpected_tag_on_fast_exit_returns_typed_malformed_output_error(monkeypatch):
    # A fast exit reporting "ok" (or any tag other than "refused") is itself unexpected --
    # story 8's launcher never exits fast except to report `refused`.
    payload = {"tag": "ok", "messages": [], "schemaVersion": "1.0.0"}
    fake = FakePopen(returncode=0, stdout=json.dumps(payload), still_running=False)
    _patch_popen(monkeypatch, fake)

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "malformed_output"


def test_empty_stdout_on_fast_exit_returns_typed_malformed_output_error(monkeypatch):
    fake = FakePopen(
        returncode=1, stdout="", stderr="native windowing failure", still_running=False
    )
    _patch_popen(monkeypatch, fake)

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "malformed_output"
    assert "native windowing failure" in result.message


def test_valid_json_but_not_an_object_returns_typed_malformed_output_error(monkeypatch):
    fake = FakePopen(returncode=1, stdout=json.dumps(["refused"]), still_running=False)
    _patch_popen(monkeypatch, fake)

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "malformed_output"


def test_missing_tag_field_returns_typed_malformed_output_error(monkeypatch):
    payload = {"messages": ["something"], "schemaVersion": "1.0.0"}
    fake = FakePopen(returncode=1, stdout=json.dumps(payload), still_running=False)
    _patch_popen(monkeypatch, fake)

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "malformed_output"


def test_non_list_messages_returns_typed_malformed_output_error(monkeypatch):
    payload = {"tag": "refused", "messages": "not a list"}
    fake = FakePopen(returncode=1, stdout=json.dumps(payload), still_running=False)
    _patch_popen(monkeypatch, fake)

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "malformed_output"


# --- `launcher_path` doesn't exist / can't be executed -----------------------------------------


def test_launcher_path_not_found_returns_typed_launch_failed_error_not_an_oserror(monkeypatch):
    _patch_popen(monkeypatch, FileNotFoundError("menger-app not found"))

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "launch_failed"


def test_launcher_path_not_executable_returns_typed_launch_failed_error(monkeypatch):
    _patch_popen(monkeypatch, PermissionError("menger-app is not executable"))

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)

    assert isinstance(result, RenderWindowError)
    assert result.kind == "launch_failed"


# --- Never raises, regardless of scenario -------------------------------------------------------


@pytest.mark.parametrize("returncode", [0, 1, 2])
def test_never_raises_regardless_of_fast_exit_code(monkeypatch, returncode):
    fake = FakePopen(returncode=returncode, stdout="garbage", still_running=False)
    _patch_popen(monkeypatch, fake)

    result = refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)  # must not raise

    assert isinstance(result, RenderWindowError)


# --- Non-OSError exceptions from Popen are not swallowed --------------------------------------


def test_non_oserror_from_popen_propagates_uncaught(monkeypatch):
    # launch_failed handling must only catch OSError -- a ValueError (e.g. a bad argument to
    # Popen) is a programmer error, not a modeled "launcher couldn't be started" outcome, and
    # must be left to propagate rather than accidentally swallowed.
    _patch_popen(monkeypatch, ValueError("bad Popen argument"))

    with pytest.raises(ValueError):
        refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=_GRACE_PERIOD)


# --- grace_period precondition -----------------------------------------------------------------


@pytest.mark.parametrize("grace_period", [0, -1, -0.5])
def test_grace_period_not_positive_raises_value_error(monkeypatch, grace_period):
    result_calls = _patch_popen(monkeypatch, FakePopen(still_running=True))

    with pytest.raises(ValueError):
        refresh_render_window(_SCENE_FILE, _LAUNCHER_PATH, grace_period=grace_period)

    # The precondition check must happen before anything is launched or terminated.
    assert result_calls == []

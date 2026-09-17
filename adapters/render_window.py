"""Render window launch adapter (spec-ai-scene-agent story 13): launches `menger`'s staged
interactive-window binary (story 8, `menger` repo) non-blockingly and reports whether it's
still running past a short grace period.

AD-16 (GPU exclusivity) is enforced entirely on the `menger` side (story 8's `RenderLock` +
`--render-lock-path`): this adapter never inspects or holds the lock itself, it only launches
the process and classifies the result. A lock conflict with a *different* process surfaces as
the `refused` JSON story 8's `Main.refusedResultJson` prints on stdout before a fast, non-zero
exit -- this adapter parses exactly that shape (reusing `adapters.scene_validator`'s own
`_parse_stdout` reasoning, not inventing a second JSON contract).

A successful launch has no "I'm ready" signal on stdout -- a render window just keeps running
until the user closes it (Design Notes). So the only success/failure signal this adapter has
is: exits fast (parse the JSON) vs. still running after `grace_period` seconds (success, the
live `Popen` handle returned for the caller to track across turns).

No pipeline wiring here -- calling this from `cli.py`'s turn loop is a separate, deferred
story (see `_bmad-output/implementation-artifacts/deferred-work.md` in `menger-toplevel`, the
workspace root).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from core.types import RenderWindowError, RenderWindowOutcome, RenderWindowResult

# Bounds how much of a malformed/unexpected stdout payload is echoed back in a
# RenderWindowError's message, so a pathological or adversarial response can't blow up the
# error message itself. Mirrors `adapters.scene_validator._STDOUT_PREVIEW_LENGTH`.
_STDOUT_PREVIEW_LENGTH = 500

# The only wire tag this adapter ever expects on a fast exit -- story 8's `Main` only ever
# reports `refused` before opening the window; any other tag (including `ok`) on a *fast* exit
# is itself an unexpected shape (a real "ok" launch never exits within the grace period, per
# Design Notes), reported as `malformed_output` rather than silently accepted.
_REFUSED_WIRE_TAG = "refused"


def refresh_render_window(
    scene_file: Union[str, Path],
    launcher_path: Union[str, Path],
    previous_process: Optional[subprocess.Popen[str]] = None,
    display: Optional[str] = None,
    render_lock_path: Optional[Union[str, Path]] = None,
    grace_period: float = 1.5,
) -> RenderWindowOutcome:
    """Terminates a still-running `previous_process` (a same-caller replace, never a
    cross-session preempt -- Design Notes), then launches `<launcher_path> --scene
    <scene_file> [--display <display>] [--render-lock-path <render_lock_path>]` via
    `subprocess.Popen` (non-blocking -- a render window stays open, it is not a one-shot
    call). Waits up to `grace_period` seconds: still running past it is treated as a
    successful launch (the live `Popen` handle returned); an exit within it is parsed as
    AD-5's tagged `refused`/error JSON. Never raises for an expected outcome (Boundaries &
    Constraints) -- `launcher_path` not existing/executable, or the launcher producing
    unexpected output, are both typed errors, never an unhandled `OSError`/exception.

    `grace_period` is intentionally reused for both waits this function performs: the
    new-process launch-success check below, and (via `_terminate`) the wait for a still-running
    `previous_process` to exit. This is a single knob for now, not an oversight -- there is no
    story-level need yet for the two to differ independently.

    Note on ordering: a still-running `previous_process` is terminated *before* the new launch
    is attempted or confirmed. If the new launch subsequently fails (`refused`, `launch_failed`,
    `malformed_output`), the caller has already lost their previously-working render window with
    nothing to show for it. This is inherent to AD-16's one-active-session exclusivity model --
    the old lock must be given up before a new one can be acquired -- not a bug, but a real
    caller-facing consequence worth documenting explicitly.
    """
    if grace_period <= 0:
        raise ValueError(f"grace_period must be positive, got {grace_period!r}")

    if previous_process is not None:
        try:
            still_running = previous_process.poll() is None
        except Exception:
            # Best-effort: a racy poll() failure is treated the same as "it's probably
            # already gone" -- this function must never raise.
            still_running = False
        if still_running:
            _terminate(previous_process, grace_period)

    cmd: List[str] = [str(launcher_path), "--scene", str(scene_file)]
    if display is not None:
        cmd += ["--display", display]
    if render_lock_path is not None:
        cmd += ["--render-lock-path", str(render_lock_path)]

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except OSError as e:
        return RenderWindowError(
            kind="launch_failed",
            message=f"could not launch render window process '{launcher_path}': {e}",
            cause=e,
        )

    try:
        # `communicate()`, not `wait()`: both streams are opened as `subprocess.PIPE`, and a
        # process that writes enough to either before exiting/timing out would block on a full
        # OS pipe buffer -- a deadlock `wait()` alone can't detect (it would misclassify a
        # stuck process as "still running = success"). `communicate()` drains both streams
        # concurrently while waiting, and still raises `subprocess.TimeoutExpired` exactly like
        # `wait()` when the process is still running past `grace_period` -- the normal,
        # expected "success" case here.
        stdout, stderr = process.communicate(timeout=grace_period)
    except subprocess.TimeoutExpired:
        # Still running past the grace period -- a successful launch (Design Notes: no other
        # "ready" signal exists for a render window).
        return RenderWindowResult(process=process)

    return _parse_fast_exit(stdout or "", stderr or "", process.returncode)


def _terminate(process: subprocess.Popen[str], grace_period: float) -> None:
    """Terminates a caller-supplied `previous_process`, escalating to `kill()` if it doesn't
    exit promptly -- never left as an orphaned/zombie process (Boundaries & Constraints:
    "Only a caller-supplied `previous_process` is ever terminated").

    Every step here is best-effort: `terminate()`/`kill()`/`wait()` can all race with the
    process exiting on its own (e.g. a `ProcessLookupError`/`PermissionError`), and this
    function's whole contract is "never raises, never blocks indefinitely" -- so any exception
    along the way is treated as "it's probably already gone, proceed" rather than escaping to
    the caller. The final wait (after `kill()`) is bounded by `grace_period` too, and proceeds
    regardless of whether it actually confirms the process's death.
    """
    try:
        process.terminate()
    except Exception:
        pass

    try:
        process.wait(timeout=grace_period)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return

    try:
        process.kill()
    except Exception:
        pass

    try:
        process.wait(timeout=grace_period)
    except Exception:
        # Best-effort cleanup only -- if the killed process still isn't reaped within
        # grace_period, proceed anyway; this function must never block indefinitely.
        pass


def _parse_fast_exit(stdout: str, stderr: str, returncode: int) -> RenderWindowOutcome:
    """Parses a fast-exiting launcher's stdout as AD-5's tagged JSON (story 8's
    `Main.refusedResultJson`, reusing `SceneValidator.ValidationResult`'s shape) -- or returns
    a typed `malformed_output` error for corrupted JSON, an unexpected shape, or any tag other
    than `refused` (I/O & Edge-Case Matrix: "Launcher exits fast with unexpected/malformed
    output" -> "typed error naming the problem, never an unhandled exception")."""
    try:
        payload = json.loads(stdout)
        if not isinstance(payload, dict):
            raise TypeError(f"expected a JSON object, got {type(payload).__name__}")

        tag = payload["tag"]

        # An explicit JSON `null` is treated the same as the field being omitted entirely,
        # same convention as `adapters.scene_validator._parse_stdout`.
        messages = payload.get("messages")
        if messages is None:
            messages = []
        if not isinstance(messages, list) or not all(isinstance(m, str) for m in messages):
            raise TypeError("'messages' was not a list of strings")
    except (json.JSONDecodeError, TypeError, KeyError, RecursionError, MemoryError) as e:
        return RenderWindowError(
            kind="malformed_output",
            message=_unexpected_output_message(stdout, stderr, returncode, str(e)),
            cause=e,
        )

    if tag != _REFUSED_WIRE_TAG:
        return RenderWindowError(
            kind="malformed_output",
            message=_unexpected_output_message(
                stdout, stderr, returncode, f"unexpected tag {tag!r} on a fast exit"
            ),
        )

    message = "; ".join(messages) if messages else "render window launch refused"
    return RenderWindowError(kind="refused", message=message)


def _unexpected_output_message(stdout: str, stderr: str, returncode: int, detail: str) -> str:
    preview = stdout[:_STDOUT_PREVIEW_LENGTH]
    message = (
        f"render window launcher exited unexpectedly (exit {returncode}): {detail} "
        f"-- stdout was {preview!r}"
    )
    stderr_preview = stderr[:_STDOUT_PREVIEW_LENGTH]
    if stderr_preview:
        message += f" -- stderr was {stderr_preview!r}"
    return message

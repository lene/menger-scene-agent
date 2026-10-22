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
import logging
import os
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union

from adapters.child_env import child_environment
from adapters.process_contract import (
    WINDOW_EVENT_VERSION,
    WINDOW_STATE_SCHEMA_VERSION,
    result_problem,
    same_major,
)
from adapters.scene_validator import REFERENCE_DSL_REVISION

# Bounds how much of a malformed/unexpected stdout payload is echoed back in a
# RenderWindowError's message, so a pathological or adversarial response can't blow up the
# error message itself. Mirrors `adapters.scene_validator._STDOUT_PREVIEW_LENGTH`.
# The window's result types live with the adapter that produces them (review 2026-10-14,
# msa#34): a live `Popen` handle is not a core type.
# `adapters.render_window.refresh_render_window()`'s own typed failure vocabulary (story 13):
#   - "refused" -- AD-16's own lock-conflict tag, reused verbatim from `menger`'s
#     `Main.refusedResultJson` (story 8, itself reusing `SceneValidator`'s AD-5 `Tag` shape).
#     A real conflict with a *different* process's render session -- nothing is terminated.
#   - "malformed_output" -- the launcher exited within the grace period but its stdout wasn't
#     the expected tagged JSON (mirrors `ValidationErrorKind`'s own tag of the same name).
#   - "launch_failed" -- `launcher_path` doesn't exist or can't be executed at all (the
#     `subprocess.Popen(...)` call itself raised `OSError`), never a raised exception.
RenderWindowErrorKind = Literal["refused", "malformed_output", "launch_failed"]


@dataclass(frozen=True)
class RenderWindowError:
    """A typed failure result from `adapters.render_window.refresh_render_window()` -- never
    an exception escaping to the caller (Boundaries & Constraints: "typed error ... never an
    unhandled exception"). Mirrors `GenerationError`/`ReadbackError`/`TicketError`/
    `ValidationError`'s exact shape (story 13)."""

    kind: RenderWindowErrorKind
    message: str
    cause: Optional[BaseException] = None


@dataclass(frozen=True)
class RenderWindowResult:
    """A successful render-window launch (story 13): still running past `grace_period`, per
    the Design Notes' own "no ready signal beyond staying alive" reasoning. Carries the live
    `subprocess.Popen` handle itself -- unlike `ValidationResult`'s `messages`/`findings`
    payload, there is nothing else to report on success, and the caller needs the handle to
    track/replace this window on a later call (`previous_process`).

    Note: this module's own "no I/O at runtime" guarantee (AD-1/AD-3, module header) is about
    this module's *own code* never performing I/O itself -- it says nothing about whether the
    dataclasses it defines can *reference* objects that do. `process` is exactly that case: a
    live `subprocess.Popen` that performs real I/O, just not I/O this module initiates.

    Also note: `frozen=True` only prevents reassigning the `process` attribute itself -- it has
    no effect on the wrapped `Popen`'s own mutable OS-level state. The underlying process can
    still exit, be waited on, terminated, etc. independently of this wrapper; a reader should
    not assume stronger immutability than that.
    """

    process: "subprocess.Popen[str]"


# refresh_render_window() returns either the running launch's handle, or a typed error --
# never raise. Mirrors GenerationResult/ReadbackResult/TicketResult/ValidationOutcome's exact
# convention.
RenderWindowOutcome = Union[RenderWindowResult, RenderWindowError]


_STDOUT_PREVIEW_LENGTH = 500
# Earlier launches' window logs kept next to the current one (render.stderr.1.log ...).
_KEPT_LOGS = 5

logger = logging.getLogger(__name__)

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

    `grace_period` bounds the new-process launch-success check only. Terminating a still-running
    `previous_process` waits `_TERMINATE_GRACE_SECONDS`: the contained window removes its
    container on SIGTERM, which takes longer than a launch check (review 2026-10-14, menger#104
    -- the two used to be one knob).

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
            _terminate(previous_process, _TERMINATE_GRACE_SECONDS)

    cmd: List[str] = [
        str(launcher_path), "--scene", str(scene_file),
        # Every frame rendered, looping (usability review 2026-10, session 4, F95): the default
        # real-time mode skipped whole phases when a frame was slow.
        "--playback", "every-frame",
    ]
    if display is not None:
        cmd += ["--display", display]
    if render_lock_path is not None:
        cmd += ["--render-lock-path", str(render_lock_path)]

    # The window's output goes to log files next to the scene, never to pipes: a window runs
    # for as long as the user keeps it open, and nothing reads a pipe after the grace period,
    # so once it had written ~64 KB (e.g. a per-frame error with a stack trace) its next write
    # blocked forever -- a hung window, and a hidden error.
    stdout_log, stderr_log = log_paths(scene_file)
    # Earlier launches' logs are kept, not overwritten (review 2026-10-14, msa#31).
    _rotate(stdout_log)
    _rotate(stderr_log)
    # The previous launch's events would otherwise answer this launch's first wait.
    with suppress(OSError):
        events_path(scene_file).unlink()
    try:
        with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
            # Its own session: the terminal's Ctrl-C (SIGINT to the foreground process group)
            # interrupts the turn, not the window (usability review 2026-10, session 4, F88).
            # The REPL still closes it on exit (`close_render_window`).
            process = subprocess.Popen(
                cmd, stdout=out, stderr=err, text=True, start_new_session=True,
                env=_window_environment(),
            )
    except OSError as e:
        return RenderWindowError(
            kind="launch_failed",
            message=f"could not launch render window process '{launcher_path}': {e}",
            cause=e,
        )

    logger.info("render window launched, pid %s: %s", getattr(process, "pid", None), cmd)
    try:
        process.wait(timeout=grace_period)
    except subprocess.TimeoutExpired:
        # Still running past the grace period -- a successful launch (Design Notes: no other
        # "ready" signal exists for a render window).
        return RenderWindowResult(process=process)

    return _parse_fast_exit(
        stdout_log.read_text(errors="replace"),
        stderr_log.read_text(errors="replace"),
        process.returncode,
    )


def _window_environment() -> dict:
    """The window's environment: no credentials, and the manifest the agent holds, so the
    contained window (run-render-window.sh) refuses an image built from another DSL -- the same
    identity check the validator gets, so validation and rendering are one build (review
    2026-10-14, menger#77)."""
    env = child_environment()
    if REFERENCE_DSL_REVISION is not None:
        env["MENGER_RENDER_EXPECT_DSL_REVISION"] = REFERENCE_DSL_REVISION
    return env


def render_window_scene_path(session_dir: Union[str, Path]) -> Path:
    """The stable file menger's interactive window is launched against and watches for live
    reload (usability review 2026-09, F5) -- distinct from `SceneStore`'s immutable,
    ordinal-numbered turn history (`001.scala`, `002.scala`, ...), which never touches an
    already-written file (AD-8/AD-14) and so can't itself be the thing a long-running window
    watches across turns. Kept in sync with the latest ordinal's content by
    `sync_render_window_scene()` after every accepted turn; the user's hand-editing workflow
    (editing the current ordinal file directly, per `check_hand_edit()`) is unaffected -- this
    path is purely the render window's own launch target, never read by the storage layer.

    It lives in its own `window/` directory (review 2026-10-14, menger#77): the contained
    window mounts the scene's directory read-write, and that directory must hold nothing but
    what the window reads and writes -- not the session's history, ordinals or tickets."""
    return Path(session_dir) / "window" / "current.scala"


def read_window_state(session_dir: Union[str, Path]) -> Optional[dict]:
    """What the render window currently shows -- its camera, the 4D rotation added with
    Shift+drag, and (animated scenes) the time and play state -- as menger writes it to
    `<current.scala>.window-state.json` (usability review 2026-10, session 4, F96). `None` when
    there is no such file, it is unreadable, or it is of another schema major (no window, or a
    menger this agent cannot read -- the file has carried a version since menger#82)."""
    path = Path(str(render_window_scene_path(session_dir)) + ".window-state.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data if same_major(data.get("schemaVersion"), WINDOW_STATE_SCHEMA_VERSION) else None


def _vec(value: object) -> str:
    return "(" + ", ".join(f"{round(float(v), 3):g}" for v in value) + ")"  # type: ignore[attr-defined]


def describe_window_state(state: dict) -> str:
    """The window state as one sentence for the model: camera, 4D rotation (degrees, added on
    top of the scene file's own projection), playback."""
    parts = [
        f"camera at {_vec(state['cameraEye'])} looking at {_vec(state['cameraLookAt'])}",
        "4D rotation added in the window: "
        f"rotXW {round(float(state['rotXW']), 2):g}, rotYW {round(float(state['rotYW']), 2):g}, "
        f"rotZW {round(float(state['rotZW']), 2):g} degrees (in addition to the file's own "
        "Projection4DSpec rotations)",
    ]
    if state.get("t") is not None:
        parts.append(
            f"animation at t = {round(float(state['t']), 2):g} s, "
            + ("playing" if state.get("playing") else "paused")
        )
    return "; ".join(parts)


def sync_render_window_scene(scene_file: Union[str, Path], session_dir: Union[str, Path]) -> Path:
    """Copies the just-accepted ordinal's content into `render_window_scene_path()`,
    atomically (write-to-temp, flush, fsync, `os.replace()`). Unlike `SceneStore`'s own
    writes (`os.link()`, exclusive-create-only -- AD-14), this file is a mutable pointer
    that's meant to be overwritten every turn, so an atomic *replace* is the right primitive,
    not an atomic *create*. Returns the stable path, ready to pass to
    `refresh_render_window()`."""
    target = render_window_scene_path(session_dir)
    target.parent.mkdir(exist_ok=True)
    text = Path(scene_file).read_text(encoding="utf-8")
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".current.scala.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    finally:
        with suppress(OSError):
            os.unlink(tmp_name)
    return target


def _rotate(log: Path, keep: int = _KEPT_LOGS) -> None:
    """`render.stderr.log` -> `render.stderr.1.log` -> ... -> `.<keep>.log`, the oldest
    dropped. Best effort: a log that cannot be moved is overwritten as before."""
    def numbered(n: int) -> Path:
        return log.with_name(f"{log.stem}.{n}{log.suffix}")
    with suppress(OSError):
        for n in range(keep - 1, 0, -1):
            if numbered(n).exists():
                numbered(n).replace(numbered(n + 1))
        if log.exists():
            log.replace(numbered(1))


def log_paths(scene_file: Union[str, Path]) -> Tuple[Path, Path]:
    """Where a render window launched for `scene_file` writes its stdout and stderr: next to
    the scene, in its session directory. Overwritten by each launch."""
    session_dir = Path(scene_file).parent
    return session_dir / "render.stdout.log", session_dir / "render.stderr.log"


def close_render_window(
    process: Optional[subprocess.Popen[str]], grace_period: Optional[float] = None
) -> None:
    """Closes and reaps the session's render window when the REPL exits. Without this the
    window outlived the REPL, kept holding the GPU and the render lock, and lingered as a
    zombie once closed (usability review 2026-09, W4). `None` is a no-op."""
    if process is not None:
        logger.info("closing render window, pid %s", getattr(process, "pid", None))
        _terminate(process, _TERMINATE_GRACE_SECONDS if grace_period is None else grace_period)


def crash_report(
    process: Optional[subprocess.Popen[str]], scene_file: Union[str, Path]
) -> Optional[str]:
    """`None` while the tracked render window runs or after the user closed it cleanly;
    otherwise its exit status plus the most telling line of `render.stderr.log`, so a crash
    is reported in the REPL instead of only in the log (usability review 2026-09, F12)."""
    try:
        returncode = process.poll() if process is not None else None
    except Exception:
        return None
    if returncode is None:
        return None
    # A window whose first scene fails exits non-zero (it exited 0 until menger#100, F61), so
    # a zero exit is the user closing it.
    if returncode == 0:
        return None
    startup_failures = [e.detail for e in _read_events(scene_file, 0) if e.kind == "startup_failed"]
    try:
        text = log_paths(scene_file)[1].read_text(errors="replace")
    except OSError:
        text = ""
    # Stack-frame lines ("at ...") never name the cause; prefer the last line that does.
    lines = [s for s in (raw.strip() for raw in text.splitlines()) if s and not s.startswith("at ")]
    errors = [s for s in lines if "error" in s.lower() or "exception" in s.lower()]
    silent = _SIGNAL_NOTES.get(-returncode if returncode < 0 else returncode - 128,
                               "no output in render.stderr.log")
    detail = (startup_failures or errors or lines or [silent])[-1]
    return f"exit {returncode}: {detail[:_STDOUT_PREVIEW_LENGTH]}"


# A window ended by a signal writes nothing to its log (session 5, F113, menger#130): name the
# signal. -N from Popen, 128+N from a wrapper shell.
_SIGNAL_NOTES = {
    9: "killed (SIGKILL), often by the out-of-memory killer -- see `journalctl -k`",
    15: "terminated (SIGTERM) by another process",
}


RELOAD_EVENTS = ("reloaded", "reload_failed", "kind_changed")
LAUNCH_EVENTS = ("up",)


def events_path(scene_file: Union[str, Path]) -> Path:
    """Where the window writes its life-cycle events: `<scene file>.events.jsonl`, one JSON
    object per line (menger's contracts/process-result.json `windowEvents`). It replaced reading
    seven English substrings of the window's log, which already disagreed between the two windows
    (review 2026-10-14, menger#82). Scene code runs in that process and could write one too, so
    an event is the window's report, not proof."""
    return Path(str(scene_file) + ".events.jsonl")


@dataclass(frozen=True)
class WindowEvent:
    """What the window reported about itself: `up`, `reloaded`, `reload_failed` (the playing
    scene stays), `kind_changed` (static <-> animated, needs a restart), `startup_failed` or
    `exited` (the process ended), or `unavailable` (no log to read)."""

    kind: str
    detail: str = ""


def log_size(scene_file: Union[str, Path]) -> int:
    """How much of the events file exists now: the offset from which a wait reads only what the
    window reports after an action. 0 when there is no file yet."""
    try:
        return events_path(scene_file).stat().st_size
    except OSError:
        return 0


def _read_events(scene_file: Union[str, Path], offset: int) -> List[WindowEvent]:
    """The window's events after `offset`. A line of another event version, or not an event at
    all, is skipped: never a guess at what it meant."""
    try:
        with open(events_path(scene_file), "rb") as handle:
            handle.seek(offset)
            text = handle.read().decode(errors="replace")
    except OSError:
        return []
    events = []
    for line in text.splitlines():
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        if not isinstance(raw, dict) or raw.get("v") != WINDOW_EVENT_VERSION:
            continue
        kind, detail = raw.get("kind"), raw.get("detail", "")
        if isinstance(kind, str) and isinstance(detail, str):
            events.append(WindowEvent(kind, detail[:_STDOUT_PREVIEW_LENGTH]))
    return events


def wait_for_window_event(
    scene_file: Union[str, Path],
    process: Optional[subprocess.Popen[str]],
    offset: int,
    wanted: Iterable[str],
    timeout: float,
    poll_interval: float = 0.2,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Optional[WindowEvent]:
    """Waits until the window reports one of `wanted` (or a startup failure) after `offset` in
    its events file, or its process ends, or `timeout` seconds pass (then `None`). A window this
    adapter did not start (no log next to the scene) has nothing to wait for: `unavailable`.
    Never raises; this is how "Render window: reloaded" stops being a guess (F57)."""
    if not log_paths(scene_file)[1].exists():
        return WindowEvent("unavailable")
    wanted = tuple(wanted) + ("startup_failed",)
    deadline = monotonic() + timeout
    while True:
        # The process status first: once it has ended its events are complete, so one it wrote
        # just before exiting is found by the read below, not lost to a race.
        try:
            returncode = process.poll() if process is not None else None
        except Exception:
            returncode = None
        for event in _read_events(scene_file, offset):
            if event.kind in wanted:
                return event
        if returncode is not None:
            return WindowEvent("exited", str(returncode))
        if monotonic() >= deadline:
            return None
        sleep(poll_interval)


def frame_build_failures(scene_file: Union[str, Path]) -> List[str]:
    """The frames the running window failed to build, oldest first (`frame_failed` events):
    menger keeps the last good frame on screen (menger#54), so without this a failure looked
    like a frozen window (usability review 2026-09, F44)."""
    return [e.detail for e in _read_events(scene_file, 0) if e.kind == "frame_failed"]


# How long a SIGTERMed window gets before SIGKILL: the contained window's wrapper removes its
# container on SIGTERM (`docker kill` + `docker rm`), which takes longer than the 1.5 s launch
# check; killed mid-cleanup, the container would keep running (menger#104).
_TERMINATE_GRACE_SECONDS = 10.0

# Only a real child's process group is signalled -- never whatever a test passed in its place.
_REAL_POPEN = subprocess.Popen


def _signal_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    """Best effort: a real window's process group, else the process itself."""
    if isinstance(process, _REAL_POPEN):
        try:
            os.killpg(process.pid, sig)
            return
        except OSError:  # no such group: already gone, or not a session leader
            pass
    try:
        (process.terminate if sig == signal.SIGTERM else process.kill)()
    except Exception:  # noqa: BLE001 -- already gone
        pass


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

    The window runs in its own session (`start_new_session=True`), so its process group is the
    window and everything it started -- the contained window's wrapper script and its `docker`
    client among them. The group is signalled, not just the PID (review 2026-10-14, menger#104):
    SIGTERM lets the wrapper's trap remove the container; signalling only the wrapper's PID, or
    killing outright, could leave the container running and holding the GPU.
    """
    _signal_group(process, signal.SIGTERM)

    try:
        process.wait(timeout=grace_period)
        return
    except Exception:  # noqa: BLE001 -- timed out, or a racy wait: either way, escalate below
        pass

    _signal_group(process, signal.SIGKILL)

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

    problem = result_problem(tag, payload.get("schemaVersion"), returncode)
    if problem is not None:
        return RenderWindowError(
            kind="malformed_output",
            message=_unexpected_output_message(stdout, stderr, returncode, problem),
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

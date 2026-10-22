"""The render window as the REPL drives it (review 2026-10-14, msa#33/#39/#34: split out of
cli.py): launch, live reload, crash reports and frame failures. The adapter calls the turn's
outcome depends on -- launch, wait for an event, read the crash -- come in through
`WindowPorts`, which cli.py builds (its composition root), so they can be replaced."""

from __future__ import annotations

import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from adapters.render_window import (
    LAUNCH_EVENTS,
    RELOAD_EVENTS,
    RenderWindowOutcome,
    RenderWindowResult,
    WindowEvent,
    frame_build_failures,
    log_size,
    render_window_scene_path,
    sync_render_window_scene,
)
from adapters.scene_store import SceneStore
from repl.formatting import _format_render_outcome


@dataclass(frozen=True)
class WindowPorts:
    """`refresh` launches (or relaunches) a window, `wait_for_event` waits for what it says
    about a launch or reload, `crash_report` explains a window that ended."""

    refresh: Callable[..., RenderWindowOutcome]
    wait_for_event: Callable[..., Optional[WindowEvent]]
    crash_report: Callable[..., Optional[str]]


# Seconds the REPL waits for the render window to say what became of a reload or a launch: a
# reload compiles the scene first (about 5 s), a launch also builds the first frame (usability
# review 2026-10, session 3, F57/F61). Past them the window is reported as unconfirmed.
_RELOAD_CONFIRM_SECONDS = 20.0
_LAUNCH_CONFIRM_SECONDS = 25.0


class WindowWatcher:
    """Reports a render window crash while the REPL waits for input (usability review 2026-10,
    session 4, F88: 3.6's out-of-memory crash came minutes after the turn and was never
    reported, because the check ran only between turns). A daemon thread polls the window the
    loop last handed it; each window (by object identity) is reported at most once, here or by
    `_report_render_crash`.

    It acts only while the REPL is idle at its prompt: the REPL holds `busy()` while it handles
    a line and releases it in `idle()` around `input()`. Session 5 (F113, menger#130): a
    relaunch terminates the old window while a turn runs, and the watcher reported that as
    "crashed - exit 143". The idle hook (`on_idle`) is how a hand edit reloads the window
    without a typed line (F108, msa#47); it returns True when it printed something."""

    POLL_SECONDS = 2.0

    def __init__(
        self, ports: WindowPorts, store: SceneStore, poll_seconds: float = POLL_SECONDS
    ) -> None:
        self._ports = ports
        self._store = store
        self._poll_seconds = poll_seconds
        self._process: Optional[subprocess.Popen[str]] = None
        self._reported: set[int] = set()
        self._lock = threading.Lock()
        self._busy = threading.Lock()
        self._prompt = ""
        self._on_idle: Optional[Callable[[], bool]] = None
        self._stop = threading.Event()
        threading.Thread(target=self._run, name="render-window-watcher", daemon=True).start()

    def watch(self, process: Optional[subprocess.Popen[str]]) -> None:
        with self._lock:
            self._process = process

    def already_reported(self, process: Optional[subprocess.Popen[str]]) -> bool:
        with self._lock:
            return process is not None and id(process) in self._reported

    def on_idle(self, callback: Callable[[], bool]) -> None:
        self._on_idle = callback

    @contextmanager
    def busy(self) -> Iterator[None]:
        with self._busy:
            yield

    @contextmanager
    def idle(self, prompt: str) -> Iterator[None]:
        """Inside `busy()`: lets the watcher act while the REPL waits at `prompt`."""
        self._prompt = prompt
        self._busy.release()
        try:
            yield
        finally:
            self._busy.acquire()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            if not self._busy.acquire(blocking=False):
                continue
            try:
                printed = self._check_crash()
                if self._on_idle is not None:
                    printed = self._on_idle() or printed
                if printed:
                    _redraw_prompt(self._prompt)
            except Exception:  # noqa: BLE001 -- the watcher must never kill the REPL
                pass
            finally:
                self._busy.release()

    def _check_crash(self) -> bool:
        with self._lock:
            process = self._process
        if process is None or id(process) in self._reported or process.poll() is None:
            return False
        scene_path = self._store.current_scene_path()
        report = self._ports.crash_report(process, scene_path) if scene_path is not None else None
        with self._lock:
            self._reported.add(id(process))
        if report is None:
            return False
        print(f"\nRender window: crashed - {report}", flush=True)
        return True


def _redraw_prompt(prompt: str) -> None:
    """After output printed over a waiting prompt: the prompt and what the user had typed."""
    try:
        import readline

        typed = readline.get_line_buffer()
    except (ImportError, AttributeError):
        typed = ""
    print(prompt + typed, end="", flush=True)


def _report_render_crash(
    ports: WindowPorts,
    store: SceneStore,
    render_process: Optional[subprocess.Popen[str]],
    watcher: Optional[WindowWatcher] = None,
) -> Optional[subprocess.Popen[str]]:
    """Prints a crashed render window's cause once and stops tracking it (usability review
    2026-09, F12: a crash used to be visible only in render.stderr.log). A window the user
    closed cleanly just stops being tracked. Must not kill the REPL."""
    if render_process is None:
        return None
    if watcher is not None and watcher.already_reported(render_process):
        return None
    try:
        scene_path = store.current_scene_path()
        report = ports.crash_report(render_process, scene_path) if scene_path is not None else None
        if report is not None:
            print(f"Render window: crashed - {report}")
            return None
        return None if render_process.poll() is not None else render_process
    except Exception:  # noqa: BLE001 -- a crash check must not kill the REPL
        return render_process


def _report_frame_build_failures(store: SceneStore, reported: int) -> int:
    """Prints each frame the render window failed to build since the last check (usability
    review 2026-09, F44: menger keeps the last good frame on screen, which looked frozen) and
    returns how many are reported now. A relaunch overwrites render.stderr.log, so a shorter
    list than `reported` starts the count over. Must not kill the REPL."""
    try:
        # The window's logs sit next to the file it was launched against, not the ordinal.
        failures = frame_build_failures(render_window_scene_path(store.session_dir))
    except Exception:  # noqa: BLE001 -- a failure check must not kill the REPL
        return reported
    if len(failures) < reported:
        reported = 0
    for failure in failures[reported:]:
        print(f"Render window: frame failed to build - {failure}")
    return len(failures)


def _launch_window(
    ports: WindowPorts,
    window_path: Path,
    render_launcher_path: str,
    previous_process: Optional[subprocess.Popen[str]],
    note: str = "",
) -> Optional[subprocess.Popen[str]]:
    """Launches (or relaunches) the window and waits until it reports that it is up -- or that
    it could not build its first scene (menger then exits with status 0, which used to look
    like a window the user had closed: F61). Returns the process to track, or None."""
    outcome = ports.refresh(window_path, render_launcher_path, previous_process=previous_process)
    if not isinstance(outcome, RenderWindowResult):
        print(_format_render_outcome(outcome))
        return None
    event = ports.wait_for_event(window_path, outcome.process, 0, LAUNCH_EVENTS, _LAUNCH_CONFIRM_SECONDS)
    if event is not None and event.kind in ("startup_failed", "exited"):
        report = ports.crash_report(outcome.process, window_path) or event.detail or "the window ended"
        print(f"Render window: crashed - {report}")
        return None
    print(f"Render window: refreshed{note}")
    return outcome.process


def _after_reload(
    ports: WindowPorts,
    event: Optional[WindowEvent],
    window_path: Path,
    render_launcher_path: str,
    render_process: Optional[subprocess.Popen[str]],
) -> Optional[subprocess.Popen[str]]:
    """Reports what became of a live reload from what the window said about it (F57): the old
    "Render window: reloaded" was printed for any running process, including an animated window
    that never reloaded."""
    kind = event.kind if event is not None else "timeout"
    if kind in ("reloaded", "unavailable"):
        # "unavailable": no log to read (a window not started by this adapter) -- as before.
        print("Render window: reloaded")  # the file is already on the "Turn N" line
        return render_process
    if kind == "reload_failed":
        print(
            f"Render window: reload failed - {event.detail if event else ''}; "
            "the window keeps the previous scene"
        )
        return render_process
    if kind == "kind_changed":
        return _launch_window(
            ports,
            window_path,
            render_launcher_path,
            render_process,
            note=" (the scene changed between static and animated)",
        )
    if kind in ("startup_failed", "exited"):
        report = ports.crash_report(render_process, window_path) or "the window ended"
        print(f"Render window: crashed - {report}")
        return None
    print(
        "Render window: file synced, but the window has not confirmed the reload "
        "(an older menger build, or still compiling)"
    )
    return render_process


def _accept_and_refresh_render(
    ports: WindowPorts,
    store: SceneStore,
    render_launcher_path: str,
    render_process: Optional[subprocess.Popen[str]],
) -> Optional[subprocess.Popen[str]]:
    """Shared post-acceptance helper (story 17 -- factored out of the inline block story 21
    introduced, so the hand-edit-accepted path and the normal run_turn()-accepted path both
    call it instead of duplicating the refresh_render_window()/render_process-update dance).
    Called after ANY accepted TurnResult (from run_turn() or from check_hand_edit()); prints
    the one-line render status and returns the render_process value the caller should track
    from here on.

    F5 (usability review 2026-09): a render window launched against
    `render_window_scene_path()` (a stable path, `sync_render_window_scene()`'d from the
    just-accepted ordinal every turn) is watched and live-reloaded in place by menger itself
    -- so once a window is confirmed running, subsequent turns only need the sync, never a
    terminate-and-relaunch. `refresh_render_window()` (terminate the old process, launch a
    new one) is still used for the session's first turn and to recover from a crashed/closed
    window (`render_process` is `None` or has already exited) -- this coupling requires a
    menger build with F5 support (this sprint's or later); there is no runtime capability
    probe, so an older `menger-app` binary would simply never see its window update in
    place (it would keep working, just showing the first turn's scene until the process is
    replaced by some other path, e.g. a crash).

    Guarded exactly like the call site it replaces: current_scene_path() does a fresh
    filesystem scan (AD-13) and can theoretically raise SceneStoreError, and
    refresh_render_window's own contract doesn't cover exceptions raised before it's even
    called -- a single turn's post-processing must not be able to kill the whole REPL any
    more than run_turn() itself can."""
    try:
        scene_path = store.current_scene_path()
        if scene_path is None:
            # Should never happen: an "accepted" tag is only ever returned after
            # store.accept() has already written the ordinal this reads back -- but
            # current_scene_path()'s own return type is Optional, so this is handled rather
            # than silently trusted.
            raise RuntimeError(
                "accepted turn but current_scene_path() returned None -- this should be "
                "impossible; SceneStore/run_turn's accept-then-report invariant may be "
                "broken"
            )
        # Where the window's log ends now: only what it writes after the sync tells whether
        # this reload worked (F57).
        offset = log_size(render_window_scene_path(store.session_dir))
        window_path = sync_render_window_scene(scene_path, store.session_dir)

        try:
            still_running = render_process is not None and render_process.poll() is None
        except Exception:
            # Best-effort, same convention as refresh_render_window()'s own previous_process
            # check: a racy/unexpected poll() failure falls through to a fresh launch rather
            # than risking a sync with no window to show it.
            still_running = False
        if still_running:
            event = ports.wait_for_event(
                window_path, render_process, offset, RELOAD_EVENTS, _RELOAD_CONFIRM_SECONDS
            )
            return _after_reload(ports, event, window_path, render_launcher_path, render_process)
        return _launch_window(ports, window_path, render_launcher_path, render_process)
    except Exception as e:  # noqa: BLE001 -- a render-refresh failure must not kill the REPL
        print(f"Render window: error - {e}")
        return None

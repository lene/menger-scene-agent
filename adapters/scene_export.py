"""`/export image|video` (msa#56): renders the accepted scene to a PNG or an mp4 with menger's
own batch flags (`--headless --save-name`, `--frames --video`), through the same contained
launcher as the render window (`run-render-window.sh`: GPU, no network, only the scene's
directory mounted). So the output lands in `<session>/window/export/`, the one directory the
container may write, and the run gets the window's DSL-revision check.

A batch render takes no render lock (menger `Main`), so it runs next to an open window.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from adapters import window_files
from adapters.render_window import _TERMINATE_GRACE_SECONDS, _terminate, _window_environment

# menger's `--video` frame rate (WithVideoExport.videoFps).
VIDEO_FPS = 24
IMAGE_TIMEOUT_SECONDS = 600.0
VIDEO_TIMEOUT_SECONDS = 3600.0
_MESSAGE_LINES = 5


@dataclass(frozen=True)
class ExportResult:
    path: Path


@dataclass(frozen=True)
class ExportError:
    message: str


def export_dir(scene_file: Union[str, Path]) -> Path:
    return Path(scene_file).parent / "export"


def _next_stem(directory: Path) -> str:
    numbers = [int(p.stem) for p in directory.glob("[0-9][0-9][0-9].*") if p.stem.isdigit()]
    return f"{max(numbers, default=0) + 1:03d}"


def export_args(kind: str, out: Path, t: Optional[float], seconds: Optional[float]) -> List[str]:
    """menger's arguments for one export: a PNG at `t`, or an mp4 of t in [0, seconds] at
    menger's video frame rate (its frames are written next to it and deleted after encoding)."""
    if kind == "image":
        return ["--headless", "--save-name", str(out)] + (["--t", f"{t:g}"] if t is not None else [])
    if seconds is None or seconds <= 0:
        raise ValueError(f"a video needs a positive length, got {seconds!r}")
    frames = max(1, round(seconds * VIDEO_FPS))
    frame_pattern = out.with_name(f"{out.stem}-%04d.png")
    return ["--headless", "--start-t", "0", "--end-t", f"{seconds:g}", "--frames", str(frames),
            "--save-name", str(frame_pattern), "--video", str(out)]


def _failure_message(log: Path, returncode: int, kind: str) -> str:
    """menger's own `Error:` line when it printed one (it can exit 0 after one), else the
    log's last lines."""
    lines = [ln for ln in window_files.read_text(log).splitlines() if ln.strip()]
    errors = [ln for ln in lines if ln.startswith("Error:")]
    if errors:
        return f"{errors[-1]} (log: {log})"
    if returncode == 0:
        what = "no video (a still scene has none; /export image)" if kind == "video" else "no image"
        return f"menger wrote {what} (log: {log})"
    tail = " | ".join(lines[-_MESSAGE_LINES:]) or "no output"
    return f"menger exited with {returncode}: {tail} (log: {log})"


def export_scene(
    scene_file: Union[str, Path],
    launcher_path: Union[str, Path],
    kind: str,
    t: Optional[float] = None,
    seconds: Optional[float] = None,
) -> Union[ExportResult, ExportError]:
    """Never raises for an expected outcome: a launcher that won't start, a timeout and a run
    that writes no file are all `ExportError`s. Ctrl-C stops the render (and its container)
    and propagates."""
    directory = export_dir(scene_file)
    # The window directory is the scene's to write (menger#88): `export` may be a symlink to a
    # host directory by now. Nothing is written through one.
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return ExportError(f"cannot use {directory}: {e}")
    if directory.is_symlink():
        return ExportError(f"{directory} is a symlink, not a directory: nothing written there")
    stem = _next_stem(directory)
    out = directory / f"{stem}.{'png' if kind == 'image' else 'mp4'}"
    log = directory / f"{stem}.log"
    cmd = [str(launcher_path), "--scene", str(scene_file)] + export_args(kind, out, t, seconds)
    timeout = IMAGE_TIMEOUT_SECONDS if kind == "image" else VIDEO_TIMEOUT_SECONDS
    try:
        with window_files.create(log) as output:
            # Its own session, like the window: stopping it signals the group, so the
            # launcher's trap removes the container.
            process = subprocess.Popen(
                cmd, stdout=output, stderr=subprocess.STDOUT, text=True,
                start_new_session=True, env=_window_environment(),
            )
    except OSError as e:
        return ExportError(f"could not launch '{launcher_path}': {e}")
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate(process, _TERMINATE_GRACE_SECONDS)
        return ExportError(f"timed out after {timeout:g} s (log: {log})")
    except KeyboardInterrupt:
        _terminate(process, _TERMINATE_GRACE_SECONDS)
        raise
    if returncode != 0 or not window_files.is_file(out):
        return ExportError(_failure_message(log, returncode, kind))
    return ExportResult(out)

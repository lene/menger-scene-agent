"""`/export image|video` (msa#56): parsing, the REPL handler's refusals, and the adapter
against fake launchers (the live run in the contained window is a manual check)."""

from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import adapters.scene_export as scene_export
from adapters.scene_export import ExportError, ExportResult, export_args, export_scene
from core.scene_facts import scene_duration
from repl.session import TurnState, _export, _export_request, command_for

ANIMATED = "val durationSeconds = 2.5f\ndef scene(t: Float) = Scene(objects = List(Sphere()))\n"
STILL = "val scene = Scene(objects = List(Sphere()))\n"


def test_export_takes_arguments_and_parses_kind_and_value():
    assert command_for("/export image 1.5") is _export
    assert command_for("/export") is _export  # the handler prints the usage
    assert _export_request("/export image") == ("image", None)
    assert _export_request("/export image 1.5") == ("image", 1.5)
    assert _export_request("/export video 3") == ("video", 3.0)
    for line in ("/export", "/export gif", "/export video soon", "/export image 1 2"):
        assert _export_request(line) is None


def test_scene_duration_reads_the_declared_length():
    assert scene_duration(ANIMATED) == pytest.approx(2.5)
    assert scene_duration(STILL) is None


def test_video_args_cover_the_length_at_menger_s_frame_rate(tmp_path):
    args = export_args("video", tmp_path / "001.mp4", None, 2.0)
    assert args[args.index("--frames") + 1] == "48"
    assert args[args.index("--end-t") + 1] == "2"
    assert args[args.index("--save-name") + 1] == str(tmp_path / "001-%04d.png")
    assert args[args.index("--video") + 1] == str(tmp_path / "001.mp4")
    assert export_args("image", tmp_path / "a.png", 0.5, None)[-2:] == ["--t", "0.5"]
    with pytest.raises(ValueError):
        export_args("video", tmp_path / "001.mp4", None, None)


# --- the REPL handler ------------------------------------------------------------------------


def _session(tmp_path: Path, scene: str | None):
    if scene is not None:
        (tmp_path / "window").mkdir()
        (tmp_path / "window" / "current.scala").write_text(scene)
    calls = []

    def export(scene_file, launcher, kind, t=None, seconds=None):
        calls.append((kind, t, seconds))
        return ExportResult(Path(scene_file).parent / "export" / "001.png")

    ctx = SimpleNamespace(store=SimpleNamespace(session_dir=tmp_path), render_launcher_path="l")
    return ctx, SimpleNamespace(export_scene=export), TurnState(prior_scene=scene), calls


def test_no_accepted_scene_is_refused_without_a_render(tmp_path, capsys):
    ctx, ports, state, calls = _session(tmp_path, None)
    _export(ctx, ports, state, "/export image")
    assert "no accepted scene" in capsys.readouterr().out
    assert calls == []


def test_a_video_of_a_scene_without_duration_asks_for_the_length(tmp_path, capsys):
    ctx, ports, state, calls = _session(tmp_path, STILL)
    _export(ctx, ports, state, "/export video")
    assert "/export video <seconds>" in capsys.readouterr().out
    assert calls == []


def test_a_video_defaults_to_the_scene_s_duration_and_an_image_to_t(tmp_path, capsys):
    ctx, ports, state, calls = _session(tmp_path, ANIMATED)
    _export(ctx, ports, state, "/export video")
    _export(ctx, ports, state, "/export video 1")
    _export(ctx, ports, state, "/export image 0.75")
    assert calls == [("video", None, pytest.approx(2.5)), ("video", None, 1.0),
                     ("image", 0.75, None)]
    assert "Exported: " in capsys.readouterr().out


def test_an_export_error_is_printed(tmp_path, capsys):
    ctx, ports, state, _ = _session(tmp_path, ANIMATED)
    ports.export_scene = lambda *a, **k: ExportError("Error: it broke")
    _export(ctx, ports, state, "/export image")
    assert "Export error: Error: it broke" in capsys.readouterr().out


# --- the adapter, with fake launchers ----------------------------------------------------------


def _launcher(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "launcher.sh"
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# Writes the file named after --video, else after --save-name, as menger does.
_WRITES_OUTPUT = """out=""
while [ $# -gt 0 ]; do
  case "$1" in --save-name) [ -z "$out" ] && out="$2"; shift;; --video) out="$2"; shift;; esac
  shift
done
echo rendered > "$out"
"""


def _scene(tmp_path: Path) -> Path:
    scene = tmp_path / "window" / "current.scala"
    scene.parent.mkdir()
    scene.write_text(ANIMATED)
    return scene


def test_the_written_file_is_returned_and_numbered(tmp_path):
    scene = _scene(tmp_path)
    launcher = _launcher(tmp_path, _WRITES_OUTPUT)
    first = export_scene(scene, launcher, "image")
    second = export_scene(scene, launcher, "video", seconds=1.0)
    assert first == ExportResult(scene.parent / "export" / "001.png")
    assert second == ExportResult(scene.parent / "export" / "002.mp4")
    assert second.path.read_text() == "rendered\n"


def test_menger_s_error_line_is_reported(tmp_path):
    launcher = _launcher(tmp_path, 'echo "INFO noise"; echo "Error: too many instances"; exit 1\n')
    outcome = export_scene(_scene(tmp_path), launcher, "image")
    assert isinstance(outcome, ExportError)
    assert outcome.message.startswith("Error: too many instances")


def test_a_run_that_writes_no_video_says_so(tmp_path):
    outcome = export_scene(_scene(tmp_path), _launcher(tmp_path, "exit 0\n"), "video", seconds=1.0)
    assert isinstance(outcome, ExportError)
    assert "no video" in outcome.message


def test_a_missing_launcher_and_a_timeout_are_errors(tmp_path, monkeypatch):
    scene = _scene(tmp_path)
    assert isinstance(export_scene(scene, tmp_path / "nope", "image"), ExportError)
    monkeypatch.setattr(scene_export, "IMAGE_TIMEOUT_SECONDS", 0.3)
    outcome = export_scene(scene, _launcher(tmp_path, "sleep 30\n"), "image")
    assert isinstance(outcome, ExportError)
    assert "timed out" in outcome.message

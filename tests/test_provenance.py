"""Session provenance (review 2026-10-14, msa#31): session.json, model-calls.jsonl, agent.log,
the run's identity on every history entry, and rotated window logs. The fitness function the
issue names -- every history.jsonl entry carries the provenance keys -- is
`test_every_history_entry_carries_ts_and_provenance`."""

from __future__ import annotations

import json
import logging

import pytest

import cli
from adapters import provenance
from adapters.model import ModelError, ModelRequest
from adapters.provenance import HISTORY_KEYS, RecordingModelAdapter, write_session_json
from adapters.render_window import _rotate
from adapters.scene_store import SceneStore
from core.types import TurnResult

_RUN = {"agent_version": "0.1.0", "provider": "fake", "model": "m-1", "dsl_revision": "abc"}


def _history(store: SceneStore) -> list:
    return [json.loads(line) for line in store.history_path.read_text().splitlines()]


def test_every_history_entry_carries_ts_and_provenance(tmp_path):
    store = SceneStore.create_session(tmp_path, session_id="s")
    store = SceneStore(session_dir=store.session_dir, provenance=_RUN)
    store.accept("object Scene\n", "make a scene", readback_summary="a scene")
    store.record_rejected("again", "local_finding: x", scene_text="object Bad\n")
    store.record_consult("why?", answer="because")
    entries = _history(store)
    assert [e["outcome"] for e in entries] == ["accepted", "rejected", "consult"]
    for entry in entries:
        assert entry["ts"].endswith("+00:00")
        assert set(entry["provenance"]) == set(HISTORY_KEYS)
        assert entry["provenance"] == _RUN


class _Fake:
    model_name = "m-1"

    def __init__(self, result):
        self.result = result

    def complete(self, request):
        return self.result


def test_recording_adapter_keeps_raw_responses_and_errors(tmp_path):
    path = tmp_path / "model-calls.jsonl"
    request = ModelRequest(system_prompt="manifest ...", user_prompt="a red sphere")
    raw = "Here you go:\n```scala\nobject Scene\n```\nEnjoy."
    assert RecordingModelAdapter(_Fake(raw), path).complete(request) == raw
    error = ModelError(kind="timeout", message="slow")
    assert RecordingModelAdapter(_Fake(error), path).complete(request) is error
    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert first["response"] == raw  # before extraction: the prose around the fence too
    assert first["user_prompt"] == "a red sphere"
    assert first["system_prompt_chars"] == len("manifest ...")
    assert second["error"] == {"kind": "timeout", "message": "slow"}
    assert RecordingModelAdapter(_Fake(raw), path).model_name == "m-1"


def test_session_json_is_written_once(tmp_path):
    write_session_json(tmp_path, {"model": "first"})
    write_session_json(tmp_path, {"model": "resumed with another"})
    data = json.loads((tmp_path / "session.json").read_text())
    assert data["model"] == "first"
    assert "created" in data


def test_window_logs_rotate_instead_of_being_overwritten(tmp_path):
    log = tmp_path / "render.stderr.log"
    for n in range(7):
        log.write_text(f"launch {n}")
        _rotate(log, keep=3)
    assert not log.exists()
    assert [(tmp_path / f"render.stderr.{n}.log").read_text() for n in (1, 2, 3)] == [
        "launch 6", "launch 5", "launch 4"
    ]
    assert not (tmp_path / "render.stderr.4.log").exists()


def test_main_writes_provenance_and_records_model_calls(monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_RENDER_LAUNCHER", "/fake/run-render-window.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions))
    monkeypatch.setattr(cli, "get_model_adapter", lambda: _Fake("```scala\nobject S\n```"))
    monkeypatch.setattr(
        provenance, "_image_identity", lambda image: {"id": "sha256:1", "dsl_revision": "r"}
    )
    seen = {}

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store, script_path,
                       on_stage=None, **_):
        seen["adapter"] = adapter
        adapter.complete(ModelRequest(system_prompt="s", user_prompt=prompt))
        store.record_consult(prompt, answer="ok")
        return TurnResult(tag="accepted", messages=[], ordinal=None)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(cli, "refresh_render_window", lambda *a, **k: None, raising=False)
    inputs = iter(["a blue cube"])

    def _input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", _input)
    try:
        cli.main([])
    finally:
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
    (session_dir,) = sessions.iterdir()
    session = json.loads((session_dir / "session.json").read_text())
    assert session["model"] == "m-1"
    assert session["validator_image"] == {
        "name": "menger-scene-validator:latest", "id": "sha256:1", "dsl_revision": "r"
    }
    assert session["dsl_revision"] and session["manifest_sha256"]
    assert isinstance(seen["adapter"], RecordingModelAdapter)
    (call,) = (session_dir / "model-calls.jsonl").read_text().splitlines()
    assert json.loads(call)["user_prompt"] == "a blue cube"
    (entry,) = [json.loads(line) for line in (session_dir / "history.jsonl").read_text().splitlines()]
    assert entry["provenance"]["model"] == "m-1"
    assert "turn: accepted" in (session_dir / "agent.log").read_text()


@pytest.fixture(autouse=True)
def _restore_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)

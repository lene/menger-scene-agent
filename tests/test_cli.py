"""Unit tests for cli.py -- the persistent REPL session story (11), extended by story 21
("render window refresh wiring") for the `refresh_render_window()` call site.

One test per I/O & Edge-Case Matrix row. The REPL loop is fed scripted input (not real
stdin, via a monkeypatched `builtins.input`) and asserted on printed output (`capsys`).
`run_turn()` itself is monkeypatched throughout (Tasks & Acceptance: "run_turn() itself
monkeypatched (already covered by tests/test_turn.py -- no need to re-exercise the
pipeline's internals here)"). `get_model_adapter()` is monkeypatched too, so no test
requires real provider credentials -- construction is stubbed, never exercised.
`refresh_render_window()` is likewise monkeypatched in every test where an accepted turn
occurs but the render-window call itself isn't under test, so no test shells out to a real
(nonexistent) launcher binary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import pytest

import cli
from adapters.model import MissingAPIKeyError
from adapters.scene_store import SceneStore, SceneStoreError
from core.types import (
    ConsultError,
    EXPECTED_CORPUS_SCHEMA_VERSION,
    EXPECTED_MANIFEST_SCHEMA_VERSION,
    RenderWindowError,
    RenderWindowResult,
    TurnResult,
)

_FAKE_RENDER_LAUNCHER = "/fake/menger-app/target/universal/stage/bin/menger-app"


def _scripted_input(lines: List[str]) -> Callable[..., str]:
    """Returns a fake `input()` that yields `lines` in order, then raises `EOFError` --
    the REPL loop's own exit condition, same as a real terminal's Ctrl-D."""
    it = iter(lines)

    def _fake_input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return _fake_input


def _refuse_run_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails the test loudly if run_turn() is ever invoked -- used on every path that must
    short-circuit before the pipeline is reached (mirrors tests/test_turn.py's
    `_refuse_validate_scene` pattern)."""

    def _boom(*args, **kwargs):
        raise AssertionError("run_turn() must not be called on this path")

    monkeypatch.setattr(cli, "run_turn", _boom)


def _stub_model_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests never need a real model adapter (run_turn() is monkeypatched, so nothing
    ever calls `.complete()` on it) -- stub construction so no test requires a real
    provider's API key."""
    monkeypatch.setattr(cli, "get_model_adapter", lambda: object())


def _refuse_model_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails the test loudly if get_model_adapter() is ever invoked -- used to prove a
    startup check happens strictly before model adapter construction."""

    def _boom(*args, **kwargs):
        raise AssertionError("get_model_adapter() must not be called on this path")

    monkeypatch.setattr(cli, "get_model_adapter", _boom)


def _refuse_create_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails the test loudly if a session is ever bootstrapped -- used to prove a startup
    check happens strictly before session creation."""

    def _boom(*args, **kwargs):
        raise AssertionError("SceneStore.create_session() must not be called on this path")

    monkeypatch.setattr(cli.SceneStore, "create_session", _boom)


def _set_render_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sets `MENGER_RENDER_LAUNCHER` -- story 21's new required startup env var, checked
    alongside `MENGER_SCENE_VALIDATOR_SCRIPT`. Every test that reaches past startup (session
    bootstrap, the REPL loop) needs this set, or it fails at the render-launcher check
    instead of exercising whatever it actually means to test."""
    monkeypatch.setenv("MENGER_RENDER_LAUNCHER", _FAKE_RENDER_LAUNCHER)


def _stub_render_window(monkeypatch: pytest.MonkeyPatch, outcome=None):
    """Stubs `refresh_render_window()` so an accepted-turn test that isn't specifically
    about render-window behavior doesn't shell out to a real (nonexistent) launcher binary.
    Defaults to a successful outcome carrying a plain sentinel in place of a real `Popen`
    handle -- `RenderWindowResult` never inspects it, just threads it through."""
    if outcome is None:
        outcome = RenderWindowResult(process=object())
    calls = []

    def _fake_refresh(*args, **kwargs):
        calls.append((args, kwargs))
        return outcome

    monkeypatch.setattr(cli, "refresh_render_window", _fake_refresh)
    return calls


def _refuse_render_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails the test loudly if refresh_render_window() is ever invoked -- used on every
    path that must never call it (a rejected turn, or a startup path that never reaches the
    loop at all)."""

    def _boom(*args, **kwargs):
        raise AssertionError("refresh_render_window() must not be called on this path")

    monkeypatch.setattr(cli, "refresh_render_window", _boom)


# --- Fresh start, no --session: new session created, loop accepts input immediately -------


def test_fresh_start_creates_new_session_and_prompts_immediately(monkeypatch, tmp_path):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)
    monkeypatch.setattr("builtins.input", _scripted_input([]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert sessions_dir.is_dir()
    created = list(sessions_dir.iterdir())
    assert len(created) == 1
    assert created[0].is_dir()


# --- --session <id>, valid existing dir: history replayed, then loop accepts new input ----


def test_resume_replays_history_then_accepts_new_input(monkeypatch, tmp_path, capsys):
    sessions_dir = tmp_path / "sessions"
    store = SceneStore.create_session(sessions_dir, session_id="resume-test")
    store.record_rejected("first prompt", "local_finding: bad thing")
    store.accept("object Scene:\n  val x = 1\n", "second prompt")
    # Review round, patch-level fix (story 19): a consult entry has its own replay shape --
    # seeded here so replay is actually exercised against a "consult"-outcome history line,
    # not just accepted/rejected ones.
    store.record_consult("where should the light go?", answer="upper-left, warm color")

    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    recorded_calls = []

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None
    ):
        recorded_calls.append((prompt, prior_scene, script_path, on_stage))
        return TurnResult(tag="accepted", messages=[], ordinal=2)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["a third prompt"]))

    exit_code = cli.main(["--session", "resume-test"])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines[0] == "Turn None: rejected - local_finding: bad thing"
    assert out_lines[1] == "Turn 1: accepted"
    assert out_lines[2] == "Consult: 'where should the light go?' -> upper-left, warm color"
    assert out_lines[3] == "Turn 2: accepted"

    # The turn issued after replay must see the resumed session's real current scene, not
    # None -- proves resume threads prior_scene through, not just replaying text.
    assert len(recorded_calls) == 1
    prompt, prior_scene, script_path, on_stage = recorded_calls[0]
    assert prompt == "a third prompt"
    assert prior_scene == "object Scene:\n  val x = 1\n"
    assert script_path == "/fake/validator.sh"
    # story 12 ("live status line"): cli.py's run_turn() call site forwards a callback.
    assert on_stage is not None


# --- --session <id>, dir doesn't exist: clear startup error, exits before any turn --------


def test_resume_with_missing_session_dir_exits_cleanly(monkeypatch, tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    # Session bootstrap is now the LAST startup step (AD-15 anti-orphan reordering), so the
    # model adapter *is* constructed before the missing-session-dir check runs -- stub it
    # rather than refusing it.
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--session", "does-not-exist"])

    message = str(exc_info.value)
    assert "does-not-exist" in message
    assert list(sessions_dir.iterdir()) == []


# --- MENGER_SCENE_VALIDATOR_SCRIPT unset: clear startup error, before any turn/model call --


def test_missing_script_path_env_var_exits_before_model_or_session(monkeypatch, tmp_path):
    monkeypatch.delenv("MENGER_SCENE_VALIDATOR_SCRIPT", raising=False)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _refuse_model_adapter(monkeypatch)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert "MENGER_SCENE_VALIDATOR_SCRIPT" in str(exc_info.value)
    assert not (tmp_path / "sessions").exists()


# --- Plain-text input, turn accepted: "Turn N: accepted" printed --------------------------


def test_plain_text_turn_accepted_prints_ordinal_and_tag(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, **kw):
        store_arg.accept("object A:\n  val x = 1\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=5)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["add a sphere"]))

    exit_code = cli.main([])

    assert exit_code == 0
    # story 21: an accepted turn also prints the render-refresh status line, right after
    # the turn result line.
    assert capsys.readouterr().out.splitlines() == [
        "Turn 5: accepted",
        "Render window: refreshed",
    ]


# --- Live status line (story 12): stage lines print, in order, before the result line -----


def test_stage_callback_output_appears_before_final_turn_result_line(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, **kwargs):
        on_stage = kwargs["on_stage"]
        on_stage("generating")
        on_stage("validating")
        on_stage("reading back")
        store_arg.accept("object A:\n  val x = 1\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["add a sphere"]))

    exit_code = cli.main([])

    assert exit_code == 0
    captured = capsys.readouterr()
    # Patch-level fix (post-review): stage lines are ephemeral progress output and go to
    # stderr (with an explicit flush), never stdout -- stdout stays clean for the actual
    # "Turn N: tag" result line.
    assert captured.err.splitlines() == [
        "... generating",
        "... validating",
        "... reading back",
    ]
    # story 21: the render-refresh status line follows the turn result line on stdout.
    assert captured.out.splitlines() == ["Turn 1: accepted", "Render window: refreshed"]


# --- Plain-text input, turn rejected: "Turn N: <tag>" printed with the tag/reason ----------


def test_plain_text_turn_rejected_prints_tag_and_reason(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    # story 21: a rejected turn must never reach refresh_render_window().
    _refuse_render_window(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_turn",
        lambda *a, **kw: TurnResult(
            tag="local_finding", messages=["allowlist: disallowed import"], ordinal=None
        ),
    )
    monkeypatch.setattr("builtins.input", _scripted_input(["add something forbidden"]))

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines == ["Turn None: local_finding - allowlist: disallowed import"]


# --- /ask or /question input: real answer_consult() dispatch, run_turn() never called -----
# spec-ai-scene-agent story 19: the stub is gone -- /ask/-question now reaches
# core.consult.answer_consult() for real, prints its result, and logs it via
# store.record_consult(), never run_turn().


def test_ask_and_question_prefixes_reach_answer_consult_not_run_turn(
    monkeypatch, tmp_path, capsys
):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    consult_calls = []

    def _fake_answer_consult(question, manifest, corpus, adapter, prior_scene):
        consult_calls.append((question, prior_scene))
        return f"answer to: {question}"

    monkeypatch.setattr(cli, "answer_consult", _fake_answer_consult)
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(["/ask what does this scene contain?", "/question why is this here?"]),
    )

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines == [
        "answer to: what does this scene contain?",
        "answer to: why is this here?",
    ]
    assert consult_calls == [
        ("what does this scene contain?", None),
        ("why is this here?", None),
    ]

    # No scene file written, no ordinal consumed, and both consult turns logged with
    # outcome="consult" and ordinal=None.
    session_dirs = list(sessions_dir.iterdir())
    assert len(session_dirs) == 1
    assert list(session_dirs[0].glob("*.scala")) == []
    history_lines = (
        (session_dirs[0] / "history.jsonl").read_text(encoding="utf-8").splitlines()
    )
    entries = [json.loads(line) for line in history_lines]
    assert len(entries) == 2
    assert entries[0]["outcome"] == "consult"
    assert entries[0]["ordinal"] is None
    assert entries[0]["file"] is None
    assert entries[0]["prompt"] == "what does this scene contain?"
    assert entries[0]["answer"] == "answer to: what does this scene contain?"
    assert "error" not in entries[0]
    assert entries[1]["prompt"] == "why is this here?"


def test_consult_turn_model_failure_prints_typed_error_and_logs_it(
    monkeypatch, tmp_path, capsys
):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    monkeypatch.setattr(
        cli,
        "answer_consult",
        lambda *a, **kw: ConsultError(kind="model_call_failed", message="model unavailable"),
    )
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["/ask where should the light go?"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines == ["Consult error: model_call_failed - model unavailable"]

    session_dirs = list(sessions_dir.iterdir())
    entry = json.loads(
        (session_dirs[0] / "history.jsonl").read_text(encoding="utf-8").strip()
    )
    assert entry["outcome"] == "consult"
    assert entry["ordinal"] is None
    assert entry["file"] is None
    assert entry["prompt"] == "where should the light go?"
    assert entry["error"] == "model unavailable"
    assert "answer" not in entry


def test_consult_turn_empty_question_never_calls_answer_consult(monkeypatch, tmp_path, capsys):
    # Review-round patch: a bare "/ask" (or "/ask   ") must fail fast rather than sending an
    # empty Question: to the model, which would waste a real, paid model call on nothing.
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    def _refuse_answer_consult(*a, **kw):
        raise AssertionError("answer_consult() must not be called for an empty question")

    monkeypatch.setattr(cli, "answer_consult", _refuse_answer_consult)
    monkeypatch.setattr("builtins.input", _scripted_input(["/ask", "/ask   "]))

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines == [
        "Consult error: empty question -- type a question after /ask or /question",
        "Consult error: empty question -- type a question after /ask or /question",
    ]


def test_consult_turn_storage_failure_prints_note_and_does_not_crash_the_repl(
    monkeypatch, tmp_path, capsys
):
    # spec-ai-scene-agent story 19, verification-gap review round: no test previously drove
    # _record_consult_safely's SceneStoreError-swallowing branch through the actual REPL loop.
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    monkeypatch.setattr(cli, "answer_consult", lambda *a, **kw: "an answer")

    def _raising_record_consult(self, prompt, answer=None, error=None):
        raise SceneStoreError("disk full")

    monkeypatch.setattr(SceneStore, "record_consult", _raising_record_consult)
    monkeypatch.setattr("builtins.input", _scripted_input(["/ask where should the light go?"]))

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines[0] == "an answer"
    assert any("Consult history: storage_failed" in line for line in out_lines)
    assert any("disk full" in line for line in out_lines)


def test_consult_turn_sees_the_sessions_existing_scene_as_prior_scene(
    monkeypatch, tmp_path, capsys
):
    # blind-hunter review round: every existing consult CLI test starts from a fresh
    # session, so prior_scene was only ever asserted None -- this confirms the real CLI
    # wiring (not just core.consult's own unit tests) threads an existing scene through.
    sessions_dir = tmp_path / "sessions"
    store = SceneStore.create_session(sessions_dir, session_id="has-a-scene")
    store.accept("object Sponge:\n  val x = 1\n", "make a sponge")

    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    seen_prior_scenes = []

    def _fake_answer_consult(question, manifest, corpus, adapter, prior_scene):
        seen_prior_scenes.append(prior_scene)
        return "an answer"

    monkeypatch.setattr(cli, "answer_consult", _fake_answer_consult)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["/ask where should the light go?"])
    )

    exit_code = cli.main(["--session", "has-a-scene"])

    assert exit_code == 0
    assert seen_prior_scenes == ["object Sponge:\n  val x = 1\n"]


# --- --session <id> path-escape validation: clear startup error, before session bootstrap -


@pytest.mark.parametrize(
    "bad_session_id",
    ["", "   ", "/etc/passwd", "../escape", "foo/../../escape"],
)
def test_invalid_session_id_exits_before_any_bootstrap(monkeypatch, tmp_path, bad_session_id):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _refuse_model_adapter(monkeypatch)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    with pytest.raises(SystemExit):
        cli.main(["--session", bad_session_id])

    assert not sessions_dir.exists()


# --- MENGER_SCENE_VALIDATOR_SCRIPT set to whitespace only: same as unset --------------------


def test_whitespace_only_script_path_env_var_is_treated_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "   ")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _refuse_model_adapter(monkeypatch)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert "MENGER_SCENE_VALIDATOR_SCRIPT" in str(exc_info.value)


# --- get_model_adapter() failure: clear startup error, session never bootstrapped ----------


def test_model_adapter_construction_failure_exits_cleanly(monkeypatch, tmp_path):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))

    def _boom():
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(cli, "get_model_adapter", _boom)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert "ANTHROPIC_API_KEY" in str(exc_info.value)
    assert not sessions_dir.exists()


# --- Staleness pre-flight (story 14): stale manifest/corpus exits before the model adapter -
# --- is constructed or a session directory is created; a valid pair passes silently --------


_VALID_MANIFEST = {"schemaVersion": EXPECTED_MANIFEST_SCHEMA_VERSION, "objects": []}
_VALID_CORPUS = {"schemaVersion": EXPECTED_CORPUS_SCHEMA_VERSION, "scenes": []}


def test_stale_manifest_exits_before_model_adapter_construction(monkeypatch, tmp_path):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setattr(
        cli, "load_manifest", lambda path: {"schemaVersion": "0.9.0", "objects": []}
    )
    monkeypatch.setattr(cli, "load_corpus", lambda path: dict(_VALID_CORPUS))
    _refuse_model_adapter(monkeypatch)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    message = str(exc_info.value)
    assert "schemaVersion" in message
    assert "0.9.0" in message
    assert not sessions_dir.exists()


def test_stale_corpus_exits_before_model_adapter_construction(monkeypatch, tmp_path):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setattr(cli, "load_manifest", lambda path: dict(_VALID_MANIFEST))
    monkeypatch.setattr(
        cli, "load_corpus", lambda path: {"schemaVersion": "0.1.0", "scenes": []}
    )
    _refuse_model_adapter(monkeypatch)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    message = str(exc_info.value)
    assert "schemaVersion" in message
    assert "0.1.0" in message
    assert not sessions_dir.exists()


def test_valid_manifest_and_corpus_preflight_passes_silently(monkeypatch, tmp_path, capsys):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setattr(cli, "load_manifest", lambda path: dict(_VALID_MANIFEST))
    monkeypatch.setattr(cli, "load_corpus", lambda path: dict(_VALID_CORPUS))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)
    monkeypatch.setattr("builtins.input", _scripted_input([]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert sessions_dir.is_dir()
    assert capsys.readouterr().out == ""


# --- _sessions_base_dir() default (MENGER_AGENT_SESSIONS_DIR unset): "./sessions" ----------


def test_sessions_base_dir_defaults_to_relative_sessions_dir(monkeypatch):
    monkeypatch.delenv("MENGER_AGENT_SESSIONS_DIR", raising=False)

    assert cli._sessions_base_dir() == Path("./sessions")


# --- Blank/whitespace-only input line: no output, run_turn() never called -----------------


def test_blank_input_line_produces_no_output_and_skips_run_turn(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)
    monkeypatch.setattr("builtins.input", _scripted_input(["   "]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


# --- Fresh session (no --session): _replay_history() is never invoked ---------------------


def test_replay_history_skipped_on_fresh_session(monkeypatch, tmp_path):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    replay_calls = []
    monkeypatch.setattr(cli, "_replay_history", lambda store: replay_calls.append(store))
    monkeypatch.setattr(
        cli, "run_turn", lambda *a, **kw: TurnResult(tag="accepted", messages=[], ordinal=1)
    )
    monkeypatch.setattr("builtins.input", _scripted_input(["hello"]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert replay_calls == []


# --- Malformed history.jsonl line: skipped with a warning, replay continues ---------------


def test_resume_with_malformed_history_line_skips_with_warning(monkeypatch, tmp_path, capsys):
    sessions_dir = tmp_path / "sessions"
    store = SceneStore.create_session(sessions_dir, session_id="bad-history")
    with open(store.history_path, "a", encoding="utf-8") as f:
        f.write("not valid json at all\n")
    store.record_rejected("prompt", "local_finding: bad thing")

    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)
    monkeypatch.setattr("builtins.input", _scripted_input([]))

    exit_code = cli.main(["--session", "bad-history"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Turn None: rejected - local_finding: bad thing" in captured.out
    assert "malformed" in captured.err.lower()


# --- run_turn() raises: "Turn error: ..." printed, loop continues instead of dying --------


def test_run_turn_exception_prints_turn_error_and_continues(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    # An exception from run_turn() never produces a TurnResult at all, so there is no
    # "accepted" tag to trigger a render-window refresh -- refuse the call to prove it.
    _refuse_render_window(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("model backend unreachable")

    monkeypatch.setattr(cli, "run_turn", _boom)
    monkeypatch.setattr("builtins.input", _scripted_input(["add a sphere"]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert "Turn error: model backend unreachable" in capsys.readouterr().out


# --- Fresh session, first turn: prior_scene=None is passed into run_turn() ----------------


def test_fresh_session_first_turn_receives_prior_scene_none(monkeypatch, tmp_path):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    recorded_prior_scenes = []

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None
    ):
        recorded_prior_scenes.append(prior_scene)
        return TurnResult(tag="accepted", messages=[], ordinal=1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["hello"]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert recorded_prior_scenes == [None]


# --- prior_scene is refreshed after an accepted turn, for the *next* run_turn() call ------


def test_prior_scene_refreshed_after_accepted_turn(monkeypatch, tmp_path):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    accepted_scene = "object Scene:\n  val y = 2\n"
    recorded_calls = []

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None
    ):
        recorded_calls.append((prompt, prior_scene))
        if len(recorded_calls) == 1:
            # Stub also performs the real acceptance side effect so
            # store.current_scene() actually changes, same as the real pipeline would.
            store_arg.accept(accepted_scene, prompt)
            return TurnResult(tag="accepted", messages=[], ordinal=1)
        return TurnResult(tag="accepted", messages=[], ordinal=2)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["first prompt", "second prompt"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    assert len(recorded_calls) == 2
    assert recorded_calls[0][1] is None
    assert recorded_calls[1][1] == accepted_scene


# --- story 21: MENGER_RENDER_LAUNCHER unset -- clear startup error, before any bootstrap --


def test_missing_render_launcher_env_var_exits_before_model_or_session(monkeypatch, tmp_path):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.delenv("MENGER_RENDER_LAUNCHER", raising=False)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _refuse_model_adapter(monkeypatch)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert "MENGER_RENDER_LAUNCHER" in str(exc_info.value)
    assert not (tmp_path / "sessions").exists()


def test_whitespace_only_render_launcher_env_var_is_treated_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_RENDER_LAUNCHER", "   ")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _refuse_model_adapter(monkeypatch)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert "MENGER_RENDER_LAUNCHER" in str(exc_info.value)


# --- story 21: an accepted turn calls refresh_render_window() with the just-accepted -------
# --- file's path and the previously tracked render_process --------------------------------


def test_accepted_turn_calls_refresh_render_window_with_scene_path_and_no_prior_process(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    render_calls = _stub_render_window(monkeypatch)

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None
    ):
        store_arg.accept("object A:\n  val x = 1\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["add a sphere"]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert len(render_calls) == 1
    args, kwargs = render_calls[0]
    scene_file, launcher_path = args
    assert Path(scene_file).name == "001.scala"
    assert launcher_path == _FAKE_RENDER_LAUNCHER
    assert kwargs["previous_process"] is None


def test_second_accepted_turn_passes_the_tracked_render_process_as_previous(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)

    first_handle = object()
    outcomes = iter([RenderWindowResult(process=first_handle), RenderWindowResult(process=object())])
    render_calls = []
    # Each call returns a distinct outcome so the *second* call's previous_process can be
    # asserted against the *first* call's returned handle.
    monkeypatch.setattr(
        cli,
        "refresh_render_window",
        lambda *a, **kw: (render_calls.append((a, kw)), next(outcomes))[1],
    )

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None
    ):
        store_arg.accept(f"object A:\n  val x = {len(render_calls)}\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=len(render_calls) + 1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["first prompt", "second prompt"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    assert len(render_calls) == 2
    assert render_calls[0][1]["previous_process"] is None
    assert render_calls[1][1]["previous_process"] is first_handle


def test_render_window_failure_resets_render_process_to_none_and_prints_status(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)

    failure = RenderWindowError(kind="refused", message="another session holds the lock")
    success = RenderWindowResult(process=object())
    # Third call's outcome doesn't matter for this test beyond proving it was invoked with
    # previous_process=None -- reuse `success`.
    outcomes = iter([success, failure, success])
    render_calls = []
    monkeypatch.setattr(
        cli,
        "refresh_render_window",
        lambda *a, **kw: (render_calls.append((a, kw)), next(outcomes))[1],
    )

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None
    ):
        store_arg.accept(f"object A:\n  val x = {len(render_calls)}\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=len(render_calls) + 1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["first prompt", "second prompt", "third prompt"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    assert len(render_calls) == 3
    # The second call's previous_process must be the FIRST call's successful handle, even
    # though the second call itself then fails -- render_process is always overwritten with
    # the new call's own outcome, never left stale.
    assert render_calls[1][1]["previous_process"] is success.process
    # The THIRD call's previous_process must be None -- proving the second call's failure
    # really did reset render_process to None, rather than leaving the (now-invalid) first
    # handle tracked.
    assert render_calls[2][1]["previous_process"] is None
    out_lines = capsys.readouterr().out.splitlines()
    assert "Render window: refused - another session holds the lock" in out_lines


# --- story 21: a rejected turn never calls refresh_render_window(), render_process untouched


def test_rejected_turn_between_two_accepted_turns_does_not_touch_render_process(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)

    first_handle = object()
    render_calls = []

    def _fake_refresh(scene_file, launcher_path, previous_process=None, **kw):
        render_calls.append(previous_process)
        return RenderWindowResult(process=first_handle)

    monkeypatch.setattr(cli, "refresh_render_window", _fake_refresh)

    responses = iter(
        [
            TurnResult(tag="accepted", messages=[], ordinal=1),
            TurnResult(tag="local_finding", messages=["disallowed import"], ordinal=None),
            TurnResult(tag="accepted", messages=[], ordinal=2),
        ]
    )

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None
    ):
        result = next(responses)
        if result.tag == "accepted":
            store_arg.accept(f"object A:\n  val x = {result.ordinal}\n", prompt)
        return result

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["first", "rejected one", "second"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # Only the two accepted turns ever reach refresh_render_window() -- the rejected one in
    # between is skipped entirely, and the second accepted call's previous_process is still
    # the handle from the first (proving the untouched render_process survived the
    # rejection unchanged).
    assert render_calls == [None, first_handle]


# --- Patch-level fix (post-review, story 21): a render-refresh failure must not kill the ---
# --- REPL, matching the existing run_turn() exception guard -------------------------------


def test_render_refresh_exception_prints_error_and_repl_continues(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("render launcher path is a directory, not a file")

    monkeypatch.setattr(cli, "refresh_render_window", _boom)

    calls = []

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None
    ):
        calls.append(prompt)
        store_arg.accept(f"object A:\n  val x = {len(calls)}\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=len(calls))

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["first prompt", "second prompt"]))

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    # Both turns still ran to completion (the REPL survived the first render-refresh
    # exception) -- proving one bad render refresh doesn't abort the session.
    assert "Turn 1: accepted" in out_lines
    assert "Turn 2: accepted" in out_lines
    assert any(line.startswith("Render window: error - ") for line in out_lines)


# --- story 17: hand-edit fallback -- checked once per loop iteration, before /ask/run_turn -


def test_hand_edit_accepted_refreshes_render_window_and_updates_prior_scene(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    render_calls = _stub_render_window(monkeypatch)
    _refuse_run_turn(monkeypatch)

    known_scenes_seen = []

    def _fake_check_hand_edit(store_arg, known_scene):
        known_scenes_seen.append(known_scene)
        if len(known_scenes_seen) == 1:
            store_arg.accept("object HandEdited:\n  val x = 1\n", "<hand-edit>")
            return TurnResult(tag="accepted", messages=[], ordinal=1)
        return None

    monkeypatch.setattr(cli, "check_hand_edit", _fake_check_hand_edit)
    monkeypatch.setattr("builtins.input", _scripted_input(["", ""]))

    exit_code = cli.main([])

    assert exit_code == 0
    # The second iteration's check_hand_edit() call receives the newly-accepted content as
    # known_scene -- proving the CLI's tracked prior_scene really advanced.
    assert known_scenes_seen == [None, "object HandEdited:\n  val x = 1\n"]
    assert len(render_calls) == 1
    out_lines = capsys.readouterr().out.splitlines()
    assert any("Hand edit: accepted" in line for line in out_lines)
    assert "Render window: refreshed" in out_lines


def test_hand_edit_rejected_does_not_refresh_render_window_or_change_prior_scene(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_render_window(monkeypatch)
    _refuse_run_turn(monkeypatch)

    known_scenes_seen = []

    def _fake_check_hand_edit(store_arg, known_scene):
        known_scenes_seen.append(known_scene)
        return TurnResult(
            tag="hand_edit_rejected",
            messages=["allowlist: disallowed import"],
            findings=[],
        )

    monkeypatch.setattr(cli, "check_hand_edit", _fake_check_hand_edit)
    monkeypatch.setattr("builtins.input", _scripted_input(["", ""]))

    exit_code = cli.main([])

    assert exit_code == 0
    # Both iterations see prior_scene unchanged (None) -- a rejected hand edit never
    # advances the CLI's tracked known_scene/prior_scene.
    assert known_scenes_seen == [None, None]
    out_lines = capsys.readouterr().out.splitlines()
    assert any("hand_edit_rejected" in line for line in out_lines)
    assert any("allowlist: disallowed import" in line for line in out_lines)
    assert not any("Render window" in line for line in out_lines)


def test_no_hand_edit_detected_produces_no_output_and_normal_turn_proceeds(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    hand_edit_calls = []

    def _fake_check_hand_edit(store_arg, known_scene):
        hand_edit_calls.append(known_scene)
        return None

    monkeypatch.setattr(cli, "check_hand_edit", _fake_check_hand_edit)

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, **kw
    ):
        store_arg.accept("object A:\n  val x = 1\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["add a sphere"]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert hand_edit_calls == [None]
    out_lines = capsys.readouterr().out.splitlines()
    # No hand-edit-related output at all -- just the normal turn result and render refresh,
    # exactly as story 21 left it (no behavior change when no external edit occurred).
    assert not any("hand edit" in line.lower() for line in out_lines)
    assert out_lines == ["Turn 1: accepted", "Render window: refreshed"]


def test_check_hand_edit_is_called_every_loop_iteration(monkeypatch, tmp_path):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    call_count = 0

    def _fake_check_hand_edit(store_arg, known_scene):
        nonlocal call_count
        call_count += 1
        return None

    monkeypatch.setattr(cli, "check_hand_edit", _fake_check_hand_edit)
    monkeypatch.setattr(
        cli, "run_turn", lambda *a, **kw: TurnResult(tag="local_finding", messages=["x"])
    )
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["/ask something", "", "a real prompt"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # Called once per iteration -- including the /ask (consult-turn) and blank-line
    # iterations, which never reach run_turn() at all.
    assert call_count == 3


def test_hand_edit_check_exception_prints_error_and_repl_continues(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("session directory vanished")

    monkeypatch.setattr(cli, "check_hand_edit", _boom)

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, **kw
    ):
        store_arg.accept("object A:\n  val x = 1\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["add a sphere"]))

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert "Hand edit check error: session directory vanished" in out_lines
    # The REPL survived the exception and still ran the real turn.
    assert "Turn 1: accepted" in out_lines


# --- Patch-level fixes (post-review, story 17) ---------------------------------------------
# --- Test gaps three parallel automated reviewers corroborated ------------------------------


def test_hand_edit_storage_failure_does_not_refresh_render_window_or_change_prior_scene(
    monkeypatch, tmp_path, capsys
):
    # No test previously drove the "storage_failed" tag through the actual cli.py REPL loop
    # (only tests/test_turn.py exercised check_hand_edit() directly for this) -- simulates
    # check_hand_edit() itself failing with a storage error (standing in for its underlying
    # store.accept() call exhausting its ordinal-claim retries) and confirms prior_scene/
    # render_process are left untouched, exactly like the hand_edit_rejected case, and the
    # render window is never refreshed.
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_render_window(monkeypatch)
    _refuse_run_turn(monkeypatch)

    known_scenes_seen = []

    def _fake_check_hand_edit(store_arg, known_scene):
        known_scenes_seen.append(known_scene)
        return TurnResult(
            tag="storage_failed",
            messages=["ordinal-claim retries exhausted"],
        )

    monkeypatch.setattr(cli, "check_hand_edit", _fake_check_hand_edit)
    monkeypatch.setattr("builtins.input", _scripted_input(["", ""]))

    exit_code = cli.main([])

    assert exit_code == 0
    # Both iterations see known_scene/prior_scene unchanged (None) -- a storage failure never
    # advances the CLI's tracked value, same as a hand_edit_rejected tag.
    assert known_scenes_seen == [None, None]
    out_lines = capsys.readouterr().out.splitlines()
    assert any("Hand edit: storage_failed" in line for line in out_lines)
    assert any("ordinal-claim retries exhausted" in line for line in out_lines)
    assert not any("Render window" in line for line in out_lines)


def test_hand_edit_render_refresh_exception_prints_error_and_repl_continues(
    monkeypatch, tmp_path, capsys
):
    # No test previously exercised a render-window exception specifically on the
    # hand-edit-accepted path through the shared _accept_and_refresh_render() helper -- the
    # existing exception-safety test (test_render_refresh_exception_prints_error_and_repl_
    # continues) only covers the normal run_turn()-accepted path, even though both paths now
    # share this helper.
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("render launcher path is a directory, not a file")

    monkeypatch.setattr(cli, "refresh_render_window", _boom)

    def _fake_check_hand_edit(store_arg, known_scene):
        if known_scene is None:
            store_arg.accept("object HandEdited:\n  val x = 1\n", "<hand-edit>")
            return TurnResult(tag="accepted", messages=[], ordinal=1)
        return None

    monkeypatch.setattr(cli, "check_hand_edit", _fake_check_hand_edit)

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, **kw
    ):
        store_arg.accept("object B:\n  val x = 2\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=2)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["add a sphere"]))

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    # The hand-edit-accepted path's own render-refresh call raised -- proving the REPL
    # survived it via _accept_and_refresh_render()'s own guard, the same way the existing
    # test already proves for the normal run_turn()-accepted path.
    assert any(line.startswith("Render window: error - ") for line in out_lines)
    assert any("Hand edit: accepted" in line for line in out_lines)
    # The real prompt this same iteration still ran to completion afterward.
    assert "Turn 2: accepted" in out_lines


def test_hand_edit_accepted_and_same_iterations_real_prompt_see_it_as_prior_scene_end_to_end(
    monkeypatch, tmp_path
):
    # The cli.py module docstring describes "an accepted hand edit and a real prompt
    # processed in the same loop iteration, with the latter seeing the former's
    # freshly-promoted content as prior_scene" as supported behavior, but this was only ever
    # tested at the core/turn.py level (calling run_turn() directly) -- never end to end
    # through cli.main(). This scripts the scenario through main() itself: check_hand_edit()
    # is NOT mocked (only run_turn() is, matching every other test in this module), so the
    # real detection-and-promotion logic runs; the fake input() simulates a hand edit landing
    # on disk immediately before the user's real prompt is submitted, both within the same
    # loop iteration.
    sessions_dir = tmp_path / "sessions"
    store = SceneStore.create_session(sessions_dir, session_id="hand-edit-e2e")
    original_text = "object Old:\n  val scene = Scene()\n"
    store.accept(original_text, "seed turn")
    edited_text = "object New:\n  val scene = Scene()\n"

    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    call_count = [0]

    def _fake_input(prompt: str = "") -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            # Simulates the user hand-editing the scene file in their editor, then typing a
            # real prompt and hitting enter -- both land within this single input() call, so
            # the loop iteration that follows sees the edit AND processes the real prompt.
            (store.session_dir / "001.scala").write_text(edited_text, encoding="utf-8")
            return "a real prompt"
        raise EOFError

    monkeypatch.setattr("builtins.input", _fake_input)

    recorded_calls = []

    def _fake_run_turn(
        prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None
    ):
        recorded_calls.append((prompt, prior_scene))
        return TurnResult(tag="accepted", messages=[], ordinal=3)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)

    exit_code = cli.main(["--session", "hand-edit-e2e"])

    assert exit_code == 0
    assert len(recorded_calls) == 1
    prompt, prior_scene = recorded_calls[0]
    assert prompt == "a real prompt"
    # The real prompt's run_turn() call received the hand-edit-promoted content as
    # prior_scene, not the pre-edit value -- proving the wiring, not just core/turn.py's own
    # already-tested internals.
    assert prior_scene == edited_text
    # The hand edit really was promoted under a brand-new ordinal (002.scala), never an
    # in-place rewrite of 001.scala.
    assert (store.session_dir / "002.scala").read_text(encoding="utf-8") == edited_text


# --- story 22: /retry command and its confirmation-gate semantics (PRD FR7) ----------------
# One test per I/O & Edge-Case Matrix row, driven through cli.main() like every other
# dispatch branch in this file.


def test_retry_with_no_prior_timeout_reports_nothing_to_retry(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    _refuse_render_window(monkeypatch)
    monkeypatch.setattr("builtins.input", _scripted_input(["/retry"]))

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines == ["Retry: nothing to retry -- no prior generation_timeout to resend."]


def test_first_retry_after_timeout_with_unchanged_scene_arms_gate_without_resending(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_render_window(monkeypatch)

    call_log = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None):
        call_log.append(prompt)
        return TurnResult(tag="generation_timeout", messages=["model hung"])

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["timeout prompt", "/retry"]))

    exit_code = cli.main([])

    assert exit_code == 0
    # The first /retry must NOT call run_turn() again -- only the original prompt's own
    # failed attempt is in the log.
    assert call_log == ["timeout prompt"]
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines == [
        "Turn None: generation_timeout - model hung",
        "Retry: nothing has changed since the last generation_timeout. Type /retry "
        "again to resend the identical prompt.",
    ]


def test_second_retry_with_still_unchanged_scene_resends_exactly_once(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    call_log = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None):
        call_log.append(prompt)
        if len(call_log) == 1:
            return TurnResult(tag="generation_timeout", messages=["model hung"])
        store_arg.accept("object A:\n  val x = 1\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["timeout prompt", "/retry", "/retry"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # The resend must reuse the exact same remembered prompt -- not the literal "/retry"
    # text -- and must happen only once (on the SECOND /retry, not the first).
    assert call_log == ["timeout prompt", "timeout prompt"]
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines == [
        "Turn None: generation_timeout - model hung",
        "Retry: nothing has changed since the last generation_timeout. Type /retry "
        "again to resend the identical prompt.",
        "Turn 1: accepted",
        "Render window: refreshed",
    ]


def test_retry_with_scene_changed_via_hand_edit_resends_immediately_no_gate(
    monkeypatch, tmp_path, capsys
):
    # The frozen I/O & Edge-Case Matrix names both triggers for "scene changed since the
    # failure" -- "a hand edit OR another accepted turn" -- and check_hand_edit() never
    # goes through run_turn()/_execute_turn() at all, so this is a distinct code path from
    # test_retry_with_scene_changed_since_failure_resends_immediately_no_gate above.
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    run_turn_calls = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None):
        run_turn_calls.append(prompt)
        if len(run_turn_calls) == 1:
            return TurnResult(tag="generation_timeout", messages=["model hung"])
        # Second call: the /retry resend of the original failed prompt.
        store_arg.accept("object C:\n  val x = 3\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=2)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)

    hand_edit_calls = []

    def _fake_check_hand_edit(store_arg, known_scene):
        hand_edit_calls.append(known_scene)
        if len(hand_edit_calls) == 2:
            store_arg.accept("object HandEdited:\n  val x = 1\n", "<hand-edit>")
            return TurnResult(tag="accepted", messages=[], ordinal=1)
        return None

    monkeypatch.setattr(cli, "check_hand_edit", _fake_check_hand_edit)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["timeout prompt", "", "/retry"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # The hand edit (detected on the second loop iteration) changed the tracked scene --
    # the single /retry on the third iteration resends the ORIGINAL failed prompt right
    # away, no confirmation-gate message in between.
    assert run_turn_calls == ["timeout prompt", "timeout prompt"]
    out_lines = capsys.readouterr().out.splitlines()
    assert not any("nothing has changed" in line for line in out_lines)
    assert "Turn 2: accepted" in out_lines


def test_retry_with_scene_changed_since_failure_resends_immediately_no_gate(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    call_log = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None):
        call_log.append(prompt)
        idx = len(call_log)
        if idx == 1:
            return TurnResult(tag="generation_timeout", messages=["model hung"])
        if idx == 2:
            store_arg.accept("object A:\n  val x = 1\n", prompt)
            return TurnResult(tag="accepted", messages=[], ordinal=1)
        # idx == 3: the /retry resend of the original failed prompt.
        store_arg.accept("object B:\n  val x = 2\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=2)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(["timeout prompt", "unrelated accepted prompt", "/retry"]),
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # An intervening accepted turn changed the scene -- the single /retry resends the
    # ORIGINAL failed prompt right away, no confirmation-gate message in between.
    assert call_log == ["timeout prompt", "unrelated accepted prompt", "timeout prompt"]
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines == [
        "Turn None: generation_timeout - model hung",
        "Turn 1: accepted",
        "Render window: refreshed",
        "Turn 2: accepted",
        "Render window: refreshed",
    ]
    assert not any("nothing has changed" in line for line in out_lines)


def test_retry_resend_that_times_out_again_resets_an_unarmed_gate(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_render_window(monkeypatch)

    call_log = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None):
        call_log.append(prompt)
        return TurnResult(tag="generation_timeout", messages=["model hung again"])

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(["timeout prompt", "/retry", "/retry", "/retry"]),
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # Only two run_turn() calls total: the original failing prompt, and the one resend
    # triggered by the SECOND /retry. The fourth input ("/retry" again) must NOT resend a
    # third time -- a fresh, un-armed retry state replaced the old one when the resend
    # itself timed out again, so it must be re-confirmed from scratch.
    assert call_log == ["timeout prompt", "timeout prompt"]
    out_lines = capsys.readouterr().out.splitlines()
    arm_message = (
        "Retry: nothing has changed since the last generation_timeout. Type /retry "
        "again to resend the identical prompt."
    )
    assert out_lines == [
        "Turn None: generation_timeout - model hung again",
        arm_message,
        "Turn None: generation_timeout - model hung again",
        arm_message,
    ]


# --- Story 22 review round: gaps two independent reviewers (blind-hunter, verification-gap)
# corroborated, plus one edge-case-hunter reject-adjacent gap worth locking down anyway --------


def test_retry_resend_that_raises_leaves_retry_state_untouched(monkeypatch, tmp_path, capsys):
    # blind-hunter + verification-gap, independently corroborated: an untyped exception
    # during a /retry resend must not silently clear the pending retry -- the user should
    # still be able to /retry again rather than being told "nothing to retry".
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_render_window(monkeypatch)

    call_log = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None):
        call_log.append(prompt)
        if len(call_log) == 1:
            return TurnResult(tag="generation_timeout", messages=["model hung"])
        if len(call_log) == 2:
            raise RuntimeError("transient network blip")
        return TurnResult(tag="generation_timeout", messages=["model hung"])

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["timeout prompt", "/retry", "/retry", "/retry"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # Sequence: 1) fails with a timeout, arming the gate. 2) first /retry only arms/confirms
    # (no run_turn call). 3) second /retry resends -- run_turn RAISES this time. 4) a third
    # /retry must still see the pending retry_state (not wiped by the exception) and resend
    # again, rather than reporting "nothing to retry".
    assert call_log == ["timeout prompt", "timeout prompt", "timeout prompt"]
    out_lines = capsys.readouterr().out.splitlines()
    assert not any("nothing to retry" in line for line in out_lines)
    assert any("Turn error: transient network blip" in line for line in out_lines)


def test_normal_turn_exception_does_not_clear_a_pending_retry_state(
    monkeypatch, tmp_path, capsys
):
    # verifies the main-loop guard itself: an unrelated prompt's run_turn() raising (result is
    # None) must leave an existing pending retry_state untouched, exactly like any other
    # non-generation_timeout tag would.
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_render_window(monkeypatch)

    call_log = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None):
        call_log.append(prompt)
        if len(call_log) == 1:
            return TurnResult(tag="generation_timeout", messages=["model hung"])
        if len(call_log) == 2:
            raise RuntimeError("unrelated turn's own failure")
        return TurnResult(tag="accepted", messages=[], ordinal=1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(["timeout prompt", "an unrelated prompt", "/retry", "/retry"]),
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # The gate is still unconfirmed after the unrelated prompt's own exception (retry_state
    # untouched) -- the first post-exception /retry only arms it, the second resends
    # "timeout prompt". Either way, retry_state survived the exception in between.
    assert call_log == ["timeout prompt", "an unrelated prompt", "timeout prompt"]
    out_lines = capsys.readouterr().out.splitlines()
    assert not any("nothing to retry" in line for line in out_lines)


def test_retry_resend_landing_on_a_non_timeout_rejection_clears_the_gate(
    monkeypatch, tmp_path, capsys
):
    # A resend that lands on some other rejection tag (not generation_timeout, not accepted)
    # resolves the pending retry either way -- a further /retry must report "nothing to
    # retry", never silently resend the same already-resolved prompt again.
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_render_window(monkeypatch)

    call_log = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None):
        call_log.append(prompt)
        if len(call_log) == 1:
            return TurnResult(tag="generation_timeout", messages=["model hung"])
        if len(call_log) == 2:
            return TurnResult(tag="local_finding", messages=["allowlist: disallowed import"])
        raise AssertionError("must not be called a third time")

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["timeout prompt", "/retry", "/retry", "/retry"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # 1) timeout, arms gate. 2) first /retry only confirms. 3) second /retry resends -- lands
    # on local_finding, resolving (clearing) the gate. 4) third /retry must report "nothing
    # to retry", never call run_turn() a third time.
    assert call_log == ["timeout prompt", "timeout prompt"]
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines[-1] == "Retry: nothing to retry -- no prior generation_timeout to resend."


def test_double_retry_after_a_successful_resend_reports_nothing_to_retry(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    _set_render_launcher_env(monkeypatch)
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _stub_render_window(monkeypatch)

    call_log = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path, on_stage=None):
        call_log.append(prompt)
        if len(call_log) == 1:
            return TurnResult(tag="generation_timeout", messages=["model hung"])
        store_arg.accept("object A:\n  val x = 1\n", prompt)
        return TurnResult(tag="accepted", messages=[], ordinal=1)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr(
        "builtins.input", _scripted_input(["timeout prompt", "/retry", "/retry", "/retry"])
    )

    exit_code = cli.main([])

    assert exit_code == 0
    # The resend (second /retry) succeeds. A THIRD /retry immediately after must not
    # re-resend the now-resolved prompt.
    assert call_log == ["timeout prompt", "timeout prompt"]
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines[-1] == "Retry: nothing to retry -- no prior generation_timeout to resend."

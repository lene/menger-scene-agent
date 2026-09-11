"""Unit tests for cli.py -- the persistent REPL session story (11).

One test per I/O & Edge-Case Matrix row. The REPL loop is fed scripted input (not real
stdin, via a monkeypatched `builtins.input`) and asserted on printed output (`capsys`).
`run_turn()` itself is monkeypatched throughout (Tasks & Acceptance: "run_turn() itself
monkeypatched (already covered by tests/test_turn.py -- no need to re-exercise the
pipeline's internals here)"). `get_model_adapter()` is monkeypatched too, so no test
requires real provider credentials -- construction is stubbed, never exercised."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import pytest

import cli
from adapters.model import MissingAPIKeyError
from adapters.scene_store import SceneStore
from core.types import TurnResult


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


# --- Fresh start, no --session: new session created, loop accepts input immediately -------


def test_fresh_start_creates_new_session_and_prompts_immediately(monkeypatch, tmp_path):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
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

    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)

    recorded_calls = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path):
        recorded_calls.append((prompt, prior_scene, script_path))
        return TurnResult(tag="accepted", messages=[], ordinal=2)

    monkeypatch.setattr(cli, "run_turn", _fake_run_turn)
    monkeypatch.setattr("builtins.input", _scripted_input(["a third prompt"]))

    exit_code = cli.main(["--session", "resume-test"])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines[0] == "Turn None: rejected - local_finding: bad thing"
    assert out_lines[1] == "Turn 1: accepted"
    assert out_lines[2] == "Turn 2: accepted"

    # The turn issued after replay must see the resumed session's real current scene, not
    # None -- proves resume threads prior_scene through, not just replaying text.
    assert len(recorded_calls) == 1
    prompt, prior_scene, script_path = recorded_calls[0]
    assert prompt == "a third prompt"
    assert prior_scene == "object Scene:\n  val x = 1\n"
    assert script_path == "/fake/validator.sh"


# --- --session <id>, dir doesn't exist: clear startup error, exits before any turn --------


def test_resume_with_missing_session_dir_exits_cleanly(monkeypatch, tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    # Session bootstrap is now the LAST startup step (AD-15 anti-orphan reordering), so the
    # model adapter *is* constructed before the missing-session-dir check runs -- stub it
    # rather than refusing it.
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)

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

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert "MENGER_SCENE_VALIDATOR_SCRIPT" in str(exc_info.value)
    assert not (tmp_path / "sessions").exists()


# --- Plain-text input, turn accepted: "Turn N: accepted" printed --------------------------


def test_plain_text_turn_accepted_prints_ordinal_and_tag(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    monkeypatch.setattr(
        cli, "run_turn", lambda *a, **kw: TurnResult(tag="accepted", messages=[], ordinal=5)
    )
    monkeypatch.setattr("builtins.input", _scripted_input(["add a sphere"]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == ["Turn 5: accepted"]


# --- Plain-text input, turn rejected: "Turn N: <tag>" printed with the tag/reason ----------


def test_plain_text_turn_rejected_prints_tag_and_reason(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
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


# --- /ask or /question input: stub response, run_turn() never called ----------------------


def test_ask_and_question_prefixes_route_to_stub_not_run_turn(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(["/ask what does this scene contain?", "/question why is this here?"]),
    )

    exit_code = cli.main([])

    assert exit_code == 0
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines == [cli._CONSULT_STUB_MESSAGE, cli._CONSULT_STUB_MESSAGE]


# --- --session <id> path-escape validation: clear startup error, before session bootstrap -


@pytest.mark.parametrize(
    "bad_session_id",
    ["", "   ", "/etc/passwd", "../escape", "foo/../../escape"],
)
def test_invalid_session_id_exits_before_any_bootstrap(monkeypatch, tmp_path, bad_session_id):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _refuse_model_adapter(monkeypatch)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)

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

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert "MENGER_SCENE_VALIDATOR_SCRIPT" in str(exc_info.value)


# --- get_model_adapter() failure: clear startup error, session never bootstrapped ----------


def test_model_adapter_construction_failure_exits_cleanly(monkeypatch, tmp_path):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))

    def _boom():
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(cli, "get_model_adapter", _boom)
    _refuse_create_session(monkeypatch)
    _refuse_run_turn(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert "ANTHROPIC_API_KEY" in str(exc_info.value)
    assert not sessions_dir.exists()


# --- _sessions_base_dir() default (MENGER_AGENT_SESSIONS_DIR unset): "./sessions" ----------


def test_sessions_base_dir_defaults_to_relative_sessions_dir(monkeypatch):
    monkeypatch.delenv("MENGER_AGENT_SESSIONS_DIR", raising=False)

    assert cli._sessions_base_dir() == Path("./sessions")


# --- Blank/whitespace-only input line: no output, run_turn() never called -----------------


def test_blank_input_line_produces_no_output_and_skips_run_turn(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    monkeypatch.setattr("builtins.input", _scripted_input(["   "]))

    exit_code = cli.main([])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


# --- Fresh session (no --session): _replay_history() is never invoked ---------------------


def test_replay_history_skipped_on_fresh_session(monkeypatch, tmp_path):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)

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
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(sessions_dir))
    _stub_model_adapter(monkeypatch)
    _refuse_run_turn(monkeypatch)
    monkeypatch.setattr("builtins.input", _scripted_input([]))

    exit_code = cli.main(["--session", "bad-history"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Turn None: rejected - local_finding: bad thing" in captured.out
    assert "malformed" in captured.err.lower()


# --- run_turn() raises: "Turn error: ..." printed, loop continues instead of dying --------


def test_run_turn_exception_prints_turn_error_and_continues(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", "/fake/validator.sh")
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)

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
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)

    recorded_prior_scenes = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path):
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
    monkeypatch.setenv("MENGER_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    _stub_model_adapter(monkeypatch)

    accepted_scene = "object Scene:\n  val y = 2\n"
    recorded_calls = []

    def _fake_run_turn(prompt, prior_scene, manifest, corpus, adapter, store_arg, script_path):
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

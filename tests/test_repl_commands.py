"""The REPL's command table (review 2026-10-14, msa#33/#39): commands are looked up, not an
if-ladder in main(), so adding one is a handler plus a table entry. The behaviour of each
command is covered end to end in test_cli.py."""

from __future__ import annotations

import re

from repl.session import COMMANDS, _consult, _status_echo, _unknown, command_for


def test_exact_commands_dispatch_and_tolerate_surrounding_space():
    assert set(COMMANDS) == {"/retry", "/open", "/drop", "/new"}
    for name, handler in COMMANDS.items():
        assert command_for(name) is handler
        assert command_for(f"  {name}  ") is handler


def test_consult_prefixes_take_arguments():
    assert command_for("/ask where should the light go?") is _consult
    assert command_for("/question why is it dark") is _consult
    assert command_for("/ask") is _consult  # the handler reports the empty question


def test_anything_else_is_a_request():
    for line in ("", "   ", "a red sphere", "ask nicely", "make it ... generating"):
        assert command_for(line) is None


def test_unknown_slash_words_and_pasted_status_lines_never_reach_the_model(capsys):
    # Session 5 (F107): /exit, /new, a pasted file path and "... generating" were requests.
    for line in ("/unknown", "/retry now", "/home/me/session/002.scala"):
        assert command_for(line) is _unknown
    for line in ("... generating", "... generating, validating, reading back"):
        assert command_for(line) is _status_echo
    _unknown(None, None, None, "/frobnicate now")
    out = capsys.readouterr().out
    assert "Unknown command '/frobnicate'" in out
    assert "/exit" in out and "/retry" in out


def test_an_export_request_in_words_is_answered_locally(capsys):
    # Session 6 (msa#56): "save the rendered animation as mp4" went to the model, which declined
    # it and drafted a ticket for /export itself.
    from repl.session import _export_hint

    for line in ("save the rendered animation as mp4", "export a png of the scene",
                 "please write it out to an mp4", "save an image at t=2"):
        assert command_for(line) is _export_hint, line
    for line in ("render the animation slower", "add a picture frame around the sponge",
                 "make the image brighter", "store more light in the glass"):
        assert command_for(line) is None, line
    _export_hint(None, None, None, "save it as mp4")
    assert "/export video [seconds]" in capsys.readouterr().out


def test_the_request_after_an_ask_carries_it_once(tmp_path, capsys):
    # Session 6 (msa#49): "keep them like that" after "/ask what are the 4D parameters now?"
    # reached the model without the question, so "them" had no referent.
    from types import SimpleNamespace

    from adapters.scene_store import SceneStore
    from core.types import TurnResult
    from repl.session import TurnState, execute_turn

    calls: list = []

    def run_turn(prompt, *args, **kwargs):
        calls.append(kwargs.get("context_note"))
        return TurnResult(tag="unchanged", messages=["No change made."])

    ports = SimpleNamespace(answer_consult=lambda *a, **k: "rotXW 93.6, rotYW 19.2",
                            run_turn=run_turn)
    ctx = SimpleNamespace(manifest={}, corpus={}, adapter=None,
                          store=SceneStore.create_session(tmp_path), script_path=None,
                          render_launcher_path="menger-app")
    state = TurnState(prior_scene=None)

    _consult(ctx, ports, state, "/ask what are the 4D parameters now?")
    execute_turn(ctx, ports, state, "ok, keep them like that")
    execute_turn(ctx, ports, state, "make it red")

    assert "what are the 4D parameters now?" in calls[0] and "rotXW 93.6" in calls[0]
    assert calls[1] is None


def test_a_turn_logs_how_long_each_stage_took(tmp_path, caplog):
    # menger#91: turn latency was never measured; agent.log now has it per stage.
    import logging
    from types import SimpleNamespace

    from adapters.scene_store import SceneStore
    from core.types import TurnResult
    from repl.session import TurnState, execute_turn

    def run_turn(prompt, *args, on_stage, **kwargs):
        for stage in ("generating", "validating", "reading back"):
            on_stage(stage)
        return TurnResult(tag="unchanged", messages=["No change made."])

    ctx = SimpleNamespace(manifest={}, corpus={}, adapter=None,
                          store=SceneStore.create_session(tmp_path), script_path=None,
                          render_launcher_path="menger-app")
    with caplog.at_level(logging.INFO, logger="cli"):
        execute_turn(ctx, SimpleNamespace(run_turn=run_turn), TurnState(prior_scene=None), "x")

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("turn:"))
    assert re.search(r"generating \d+\.\ds, validating \d+\.\ds, reading back \d+\.\ds, "
                     r"total \d+\.\ds", line), line

"""The REPL's command table (review 2026-10-14, msa#33/#39): commands are looked up, not an
if-ladder in main(), so adding one is a handler plus a table entry. The behaviour of each
command is covered end to end in test_cli.py."""

from __future__ import annotations

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

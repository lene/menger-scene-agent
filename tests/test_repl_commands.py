"""The REPL's command table (review 2026-10-14, msa#33/#39): commands are looked up, not an
if-ladder in main(), so adding one is a handler plus a table entry. The behaviour of each
command is covered end to end in test_cli.py."""

from __future__ import annotations

from repl.session import COMMANDS, _consult, command_for


def test_exact_commands_dispatch_and_tolerate_surrounding_space():
    assert set(COMMANDS) == {"/retry", "/open", "/drop"}
    for name, handler in COMMANDS.items():
        assert command_for(name) is handler
        assert command_for(f"  {name}  ") is handler


def test_consult_prefixes_take_arguments():
    assert command_for("/ask where should the light go?") is _consult
    assert command_for("/question why is it dark") is _consult
    assert command_for("/ask") is _consult  # the handler reports the empty question


def test_anything_else_is_a_request():
    for line in ("", "   ", "a red sphere", "/retry now", "/unknown", "ask nicely"):
        assert command_for(line) is None

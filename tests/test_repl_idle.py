"""What the render-window watcher does while the REPL waits at its prompt (session 5: F108,
msa#47 -- a hand edit reloads the window without a typed line; F113, menger#130 -- a relaunch
during a turn is not a crash)."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from adapters.scene_store import SceneStore
from core.types import TurnResult
from repl.session import TurnState, _hand_edit_while_idle
from repl.window import WindowPorts, WindowWatcher


class _Ended:
    pid = 4242

    def poll(self):
        return 143


def _watcher(store, report="exit 143: terminated"):
    ports = WindowPorts(refresh=None, wait_for_event=None, crash_report=lambda p, s: report)
    return WindowWatcher(ports, store, poll_seconds=0.01)


def test_the_watcher_stays_quiet_while_the_repl_is_busy(tmp_path, capsys):
    # The window a relaunch terminates mid-turn exits 143; the loop then hands over the new one.
    store = SceneStore.create_session(tmp_path)
    store.accept("object A:\n  val x = 1\n", "first")
    watcher = _watcher(store)
    old = _Ended()

    with watcher.busy():
        watcher.watch(old)
        time.sleep(0.1)
        watcher.watch(None)
        with watcher.idle("> "):
            time.sleep(0.1)
    watcher.stop()

    assert "crashed" not in capsys.readouterr().out
    assert not watcher.already_reported(old)


def test_the_watcher_reports_a_crash_while_idle_and_redraws_the_prompt(tmp_path, capsys):
    store = SceneStore.create_session(tmp_path)
    store.accept("object A:\n  val x = 1\n", "first")
    watcher = _watcher(store)

    with watcher.busy():
        watcher.watch(_Ended())
        with watcher.idle("> "):
            time.sleep(0.1)
    watcher.stop()

    out = capsys.readouterr().out
    assert out.count("Render window: crashed - exit 143: terminated") == 1
    assert out.endswith("> ")


def _ports(results, refreshed):
    def check_hand_edit(store, known, script):
        return results.pop(0) if results else None

    def accept_and_refresh_render(store, launcher, process):
        refreshed.append(store.current_scene())
        return "new window"

    return SimpleNamespace(
        check_hand_edit=check_hand_edit, accept_and_refresh_render=accept_and_refresh_render
    )


def _ctx(store):
    return SimpleNamespace(store=store, script_path=None, render_launcher_path="menger-app")


def test_an_idle_hand_edit_is_promoted_and_the_window_refreshed(tmp_path, capsys):
    store = SceneStore.create_session(tmp_path)
    store.accept("object A:\n  val x = 1\n", "first")
    state = TurnState(prior_scene=store.current_scene())
    refreshed: list = []
    watched: list = []
    watcher = SimpleNamespace(watch=watched.append)
    edited = "object A:\n  val x = 2\n"
    store.current_scene_path().write_text(edited)
    ports = _ports([TurnResult(tag="accepted", messages=[], ordinal=2)], refreshed)

    assert _hand_edit_while_idle(_ctx(store), ports, state, watcher) is True
    assert "Hand edit: accepted (ordinal 2)" in capsys.readouterr().out
    assert refreshed == [edited] and state.prior_scene == edited
    assert watched == ["new window"]
    assert _hand_edit_while_idle(_ctx(store), ports, state, watcher) is False


def test_an_idle_broken_hand_edit_is_reported_once(tmp_path, capsys):
    store = SceneStore.create_session(tmp_path)
    store.accept("object A:\n  val x = 1\n", "first")
    state = TurnState(prior_scene=store.current_scene())
    store.current_scene_path().write_text("object A:\n  import java.io._\n")
    rejected = TurnResult(tag="hand_edit_rejected", messages=["disallowed import"])
    ports = _ports([rejected, rejected], [])
    watcher = SimpleNamespace(watch=lambda p: None)

    assert _hand_edit_while_idle(_ctx(store), ports, state, watcher) is True
    assert _hand_edit_while_idle(_ctx(store), ports, state, watcher) is False
    assert capsys.readouterr().out.count("hand_edit_rejected") == 1


def test_the_watcher_reports_new_slow_frames_once_while_idle(tmp_path, capsys):
    # Session 5 (F125, menger#133): 3.4 s per frame, the window looked blank.
    import json

    from adapters.render_window import render_window_scene_path

    store = SceneStore.create_session(tmp_path)
    events = Path(str(render_window_scene_path(store.session_dir)) + ".events.jsonl")
    events.parent.mkdir()

    def slow(detail):
        return json.dumps({"v": 1, "kind": "frame_slow", "detail": detail, "time": "x"}) + "\n"

    events.write_text(slow("frame 3/501 took 3.4 s") + slow("frame 5/501 took 3.5 s"))
    watcher = _watcher(store)
    with watcher.busy():
        with watcher.idle("> "):
            time.sleep(0.1)
    watcher.stop()

    out = capsys.readouterr().out
    assert out.count("Render window: slow") == 1
    assert "frame 5/501 took 3.5 s (1 earlier report(s) like it)" in out

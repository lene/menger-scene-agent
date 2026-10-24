"""core/prompt_checks.py: deterministic reasons to ask before generating (session 4, F93)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.prompt_checks import cut_off_reason, unknown_material_hint

_MANIFEST = json.loads(
    (Path(__file__).resolve().parent.parent / "reference" / "dsl-manifest.json").read_text()
)


@pytest.mark.parametrize("line", [
    # Session 4, 8b: the two halves of a split paste.
    "A camera flight through a 3D Menger sponge while the material cycles steel,",
    "  glass, diamond, ruby, gold",
    "make the sponge glass and",
    ", and make it gold",
])
def test_a_line_that_looks_like_half_a_request_gets_a_question(line):
    assert cut_off_reason(line) is not None


@pytest.mark.parametrize("line", [
    "make it from wood",
    "a menger sponge level 2",
    "slower, and start it already at level 1",
    "tunnel 7.5 to 9 s, then out; ease the speed",
])
def test_a_complete_request_gets_no_question(line):
    assert cut_off_reason(line) is None


def test_the_question_quotes_where_the_line_stops():
    reason = cut_off_reason("the material cycles steel,")

    assert reason == 'your request looks cut off after "the material cycles steel," -- what is the rest?'


def test_a_material_without_a_preset_asks_for_the_approximation():
    hint = unknown_material_hint("the material cycles steel, glass, diamond, ruby, gold", _MANIFEST)

    assert hint is not None and "steel, ruby" in hint and "NEEDS_CLARIFICATION" in hint


def test_preset_materials_need_no_hint():
    assert unknown_material_hint("make it glass, then gold, then diamond", _MANIFEST) is None


def test_a_procedural_preset_is_a_look_the_dsl_has():
    # Session 5 (F104): wood and marble are procedural presets, composed without a question.
    assert unknown_material_hint("make it from wood, or marble", _MANIFEST) is None
    hint = unknown_material_hint("make it aluminium", _MANIFEST)
    assert hint is not None and "Material.metal" in hint


def test_a_request_about_what_the_window_shows_is_recognised():
    from core.prompt_checks import refers_to_window

    assert refers_to_window("ok, keep them like that for further prompts")
    assert refers_to_window("use the current view")
    assert not refers_to_window("make it glass")


def test_the_window_hint_keeps_an_animated_value_at_the_windows_t():
    # Session 6 follow-up (msa#49): "keep them like that" wrote the shown rotation as the
    # animation's start, so the paused frame jumped 40 degrees.
    from core.prompt_checks import window_state_hint

    hint = window_state_hint("animation at t = 8.88 s, paused")
    assert "AT THE WINDOW'S t" in hint and "not as its value at t = 0" in hint


def test_keep_them_like_that_asks_for_the_window_s_4d_view_and_camera():
    # Session 6 follow-up (msa#49): the requested rotation came back as "also changed".
    from core.scene_facts import _asked_for

    assert _asked_for("projection", "ok, keep them like that for further prompts")
    assert _asked_for("camera", "keep it as it is")
    assert not _asked_for("projection", "make it glass")


def test_play_once_on_an_animated_scene_notes_that_the_preview_loops():
    # Session 5 (F119): "one 360 degree rotation ..., then stop" was accepted silently.
    from core.prompt_checks import play_once_note

    animated = "object A:\n  val durationSeconds = 4f\n  def scene(t: Float): Scene = Scene()\n"
    prompt = "play one 360 degree rotation about the xw plane, then stop for further instructions"
    assert "loops" in play_once_note(prompt, animated)
    assert play_once_note("rotate it about the xw plane", animated) is None
    assert play_once_note(prompt, "object A:\n  val scene = Scene()\n") is None


def test_colouring_before_the_4d_rotation_gets_the_unsupported_hint():
    # Session 6: accepted 3/3 with the false claim that xyz_rgb_local is pre-projection (F80b).
    from core.prompt_checks import pre_projection_colour_hint

    hint = pre_projection_colour_hint(
        "x,y,z in the xyz -> RGB should be the original x,y,z values before 4D rotation and "
        "projection to 3D, not *after*")
    assert hint is not None and "UNSUPPORTED" in hint
    assert pre_projection_colour_hint("colour it by its local x, y, z") is None
    assert pre_projection_colour_hint("rotate it in 4D before the next step") is None

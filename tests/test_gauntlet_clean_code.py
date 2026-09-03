"""Unit tests for gauntlet/clean_code.py -- stage 3's file-length bound, placeholder-text
scan, and independent object-declaration-position re-check. Pure string processing, no
compiler, no network."""

from __future__ import annotations

from gauntlet.clean_code import check_clean_code

CLEAN_SCENE = (
    "package examples.dsl\n\n"
    "import menger.dsl._\n\n"
    "object GlassSphere:\n"
    "  val scene = Scene(objects = List(Sphere(material = Some(Material.Glass))))\n"
)


# --- Acceptance: clean scene has no findings -------------------------------------------------


def test_clean_scene_has_no_findings():
    findings = check_clean_code(CLEAN_SCENE)

    assert findings == []


# --- line-count bound ------------------------------------------------------------------------


def test_file_within_the_line_count_bound_is_not_flagged():
    scene = "object Foo:\n" + "  // padding\n" * 100

    findings = check_clean_code(scene)

    assert not any("line" in f.message and "bound" in f.message for f in findings)


def test_file_over_the_line_count_bound_is_flagged():
    scene = "object Foo:\n" + "  // padding line\n" * 200

    findings = check_clean_code(scene)

    length_findings = [f for f in findings if "150-line" in f.message or "bound" in f.message]
    assert len(length_findings) == 1
    assert length_findings[0].stage == "clean_code"
    assert length_findings[0].field is None
    assert length_findings[0].identifier is None


# --- Edge-Case Matrix: stray TODO/FIXME/XXX placeholder text --------------------------------


def test_stray_todo_comment_is_flagged():
    scene = "object Foo:\n  // TODO: finish this scene\n  val scene = Scene()\n"

    findings = check_clean_code(scene)

    todo_findings = [f for f in findings if f.identifier == "TODO"]
    assert len(todo_findings) == 1
    assert todo_findings[0].line == 2


def test_fixme_and_xxx_are_both_flagged():
    scene = "object Foo:\n  // FIXME broken\n  // XXX hack\n  val scene = Scene()\n"

    findings = check_clean_code(scene)

    identifiers = {f.identifier for f in findings if f.identifier in ("FIXME", "XXX")}
    assert identifiers == {"FIXME", "XXX"}


def test_placeholder_substring_inside_a_real_identifier_is_not_flagged():
    # "TodoList" and "FIXMEHelper" are case-sensitive-distinct identifiers, but a same-case
    # substring embedded in a larger identifier must not be treated as the whole-word token.
    scene = "object Foo:\n  val TODOItem = 1\n  val scene = Scene()\n"

    findings = check_clean_code(scene)

    # "TODOItem" is not a whole-word match for "TODO" (word boundary requires a transition
    # into a non-word character) -- must not be reported as a placeholder.
    assert not any(f.identifier == "TODO" for f in findings)


# --- Edge-Case Matrix: object declaration past line 60 / not first -------------------------


def test_object_declaration_within_first_60_lines_is_not_flagged():
    findings = check_clean_code(CLEAN_SCENE)

    assert not any("60" in f.message for f in findings)


def test_object_declaration_past_line_60_is_flagged():
    padding = "\n".join(f"// line {i}" for i in range(65))
    scene = f"{padding}\nobject Foo:\n  val scene = Scene()\n"

    findings = check_clean_code(scene)

    position_findings = [f for f in findings if "60" in f.message]
    assert len(position_findings) == 1
    assert position_findings[0].stage == "clean_code"


def test_no_object_declaration_at_all_is_flagged():
    scene = "package examples.dsl\n\nval x = 1\n"

    findings = check_clean_code(scene)

    assert any("object" in f.message.lower() for f in findings)


# --- Edge-Case Matrix: unparseable/garbage input --------------------------------------------


def test_garbage_input_never_raises():
    findings = check_clean_code("not even remotely valid scala {{{ ]][[")

    assert isinstance(findings, list)
    # No 'object' declaration in garbage input -- flagged, but never a crash.
    assert any("object" in f.message.lower() for f in findings)


def test_empty_input_never_raises():
    findings = check_clean_code("")

    assert isinstance(findings, list)
    assert any("object" in f.message.lower() for f in findings)


# --- Verification: story 2's real, already-compiling poc-run fixtures -----------------------


def test_poc_run_turn1_generated_scala_has_no_clean_code_findings():
    scene_text = open("poc-run/turn1-generated.scala").read()

    findings = check_clean_code(scene_text)

    assert findings == []


def test_poc_run_turn2_refined_scala_has_no_clean_code_findings():
    scene_text = open("poc-run/turn2-refined.scala").read()

    findings = check_clean_code(scene_text)

    assert findings == []


# --- Verification: the full 32-scene reference corpus (review round 1) --------------------
#
# `ParametricScenes.scala` is the corpus's own documented outlier (dsl-surface.md: "one
# 248-line outlier holding several objects") -- a demonstration file holding multiple scene
# definitions, not a single-scene template. It legitimately exceeds the 150-line ceiling; a
# single generated scene imitating just *one* of its patterns should not.


def test_full_corpus_passes_except_the_documented_length_outlier():
    import json

    corpus = json.load(open("reference/dsl-corpus.json"))
    scenes = corpus["scenes"]
    assert len(scenes) == 32

    flagged = {s["name"]: check_clean_code(s["source"]) for s in scenes}
    flagged = {name: f for name, f in flagged.items() if f}

    assert set(flagged) == {"ParametricScenes"}
    assert "150-line" in flagged["ParametricScenes"][0].message

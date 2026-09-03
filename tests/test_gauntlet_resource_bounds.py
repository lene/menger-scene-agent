"""Unit tests for gauntlet/resource_bounds.py -- AD-4 rule 3 static resource-bound check.
Pure string processing, no compiler, no network."""

from __future__ import annotations

from gauntlet.resource_bounds import check_resource_bounds

CLEAN_SCENE = (
    "object GlassSponge:\n"
    "  val scene = Scene(\n"
    "    objects = List(\n"
    "      TesseractSponge(level = 3f, size = 1.5f)\n"
    "    )\n"
    "  )\n"
)


# --- Acceptance: clean, in-bound scene ------------------------------------------------------


def test_clean_scene_with_in_bound_literals_has_no_findings():
    findings = check_resource_bounds(CLEAN_SCENE)

    assert findings == []


def test_level_at_exactly_the_ceiling_is_not_flagged():
    scene = "object Foo:\n  val scene = Scene(objects = List(Sponge(level = 10f)))\n"

    findings = check_resource_bounds(scene)

    assert findings == []


# --- Edge-Case Matrix: resource-bound violation ---------------------------------------------


def test_level_above_ceiling_is_flagged_naming_field_and_value():
    scene = "object Foo:\n  val scene = Scene(objects = List(Sponge(level = 50f)))\n"

    findings = check_resource_bounds(scene)

    assert len(findings) == 1
    assert findings[0].stage == "resource_bounds"
    assert findings[0].field == "level"
    assert "50" in findings[0].message


def test_u_steps_above_ceiling_is_flagged():
    scene = "object Foo:\n  val s = ParametricSurface(uSteps = 500)\n"

    findings = check_resource_bounds(scene)

    assert len(findings) == 1
    assert findings[0].field == "uSteps"


def test_v_steps_above_ceiling_is_flagged():
    scene = "object Foo:\n  val s = ParametricSurface(vSteps = 999)\n"

    findings = check_resource_bounds(scene)

    assert len(findings) == 1
    assert findings[0].field == "vSteps"


def test_iterations_above_ceiling_is_flagged():
    scene = "object Foo:\n  val s = LSystem(iterations = 25)\n"

    findings = check_resource_bounds(scene)

    assert len(findings) == 1
    assert findings[0].field == "iterations"


def test_finding_names_the_correct_line():
    scene = "object Foo:\n  val scene = Scene(\n    objects = List(Sponge(level = 99f))\n  )\n"

    findings = check_resource_bounds(scene)

    assert findings[0].line == 3


def test_multiple_violations_are_all_reported():
    scene = (
        "object Foo:\n"
        "  val a = Sponge(level = 50f)\n"
        "  val b = ParametricSurface(uSteps = 500)\n"
    )

    findings = check_resource_bounds(scene)

    assert {f.field for f in findings} == {"level", "uSteps"}


# --- Edge-Case Matrix: animated (non-literal) field is skipped, not failed -----------------


def test_animated_level_expression_is_skipped_not_flagged():
    scene = "object Foo:\n  val level = progress * 3f\n  val s = Sponge(level = level)\n"

    findings = check_resource_bounds(scene)

    assert findings == []


def test_level_as_additive_animated_expression_is_skipped():
    scene = "object Foo:\n  val s = Sponge(level = 1f + progress * 3f)\n"

    findings = check_resource_bounds(scene)

    assert findings == []


# --- Edge-Case Matrix: unparseable/garbage input --------------------------------------------


def test_garbage_input_never_raises_and_returns_a_list():
    findings = check_resource_bounds("not valid scala {{{ ]][[")

    assert findings == []


def test_empty_input_never_raises():
    findings = check_resource_bounds("")

    assert findings == []


# --- Verification: story 2's real, already-compiling poc-run fixtures -----------------------


def test_poc_run_turn1_generated_scala_has_no_resource_bound_findings():
    scene_text = open("poc-run/turn1-generated.scala").read()

    findings = check_resource_bounds(scene_text)

    assert findings == []


def test_poc_run_turn2_refined_scala_has_no_resource_bound_findings():
    scene_text = open("poc-run/turn2-refined.scala").read()

    findings = check_resource_bounds(scene_text)

    assert findings == []


# --- Verification: the full 32-scene reference corpus (review round 1) --------------------
#
# Run against every real corpus scene, an earlier unscoped version of this check
# false-positived on `CausticsCanonical.scala`/`TwoSpheres.scala`'s
# `Caustics(iterations = 20, ...)` -- `Caustics.iterations` (a photon-mapping setting)
# shares a name with `LSystem.iterations` but is a different field entirely. Constructor-
# scoped extraction (this module's `_FIELD_CEILINGS_BY_TYPE`) fixes it; this test pins
# that the whole corpus now passes cleanly.


def test_full_corpus_has_no_resource_bound_findings():
    import json

    corpus = json.load(open("reference/dsl-corpus.json"))
    scenes = corpus["scenes"]
    assert len(scenes) == 32

    for scene in scenes:
        findings = check_resource_bounds(scene["source"])
        assert findings == [], f"{scene['name']}: {findings}"

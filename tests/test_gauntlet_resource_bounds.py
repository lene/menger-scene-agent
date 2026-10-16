"""Unit tests for gauntlet/resource_bounds.py -- AD-4 rule 3 static resource-bound check.
Pure string processing, no compiler, no network."""

from __future__ import annotations

from core.vocabulary import FIELD_BOUNDS
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
    # Ceiling comes from reference/dsl-manifest.json's Sponge.level max (menger's
    # ResourceLimits.cubeSpongeLevel.max, currently 5) -- not a value hand-picked here.
    ceiling = FIELD_BOUNDS["Sponge"]["level"][1]
    scene = f"object Foo:\n  val scene = Scene(objects = List(Sponge(level = {ceiling}f)))\n"

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
    scene = "object Foo:\n  val s = ParametricSurface(uSteps = 2000000)\n"

    findings = check_resource_bounds(scene)

    assert len(findings) == 1
    assert findings[0].field == "uSteps"


def test_v_steps_above_ceiling_is_flagged():
    scene = "object Foo:\n  val s = ParametricSurface(vSteps = 2000000)\n"

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
        "  val b = ParametricSurface(uSteps = 2000000)\n"
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


# --- Usability review 2026-09 (T1#3): ceilings sourced from the manifest, not hand-picked ---


# Review 2026-10-14 (msa#28): the check covers every field the manifest bounds, not a hand list
# of five; the hand-picked fallback ceilings (200 uSteps, level 10, ...) are gone.
def test_every_field_the_manifest_bounds_is_checked():
    import json

    manifest = json.load(open("reference/dsl-manifest.json"))
    bounded = {
        (t["name"], f["name"])
        for t in manifest["objects"] + manifest["lights"] + manifest["sceneComposition"]
        + [manifest["renderSettings"]]
        for f in t["fields"] if f.get("max") is not None
    }
    checked = {(t, f) for t, fields in FIELD_BOUNDS.items() for f, (_, hi) in fields.items() if hi}
    assert bounded <= checked
    assert ("IBL", "samples") in checked and ("RenderSettings", "aaMaxDepth") in checked


def test_a_cap_outside_the_old_hand_list_is_flagged():
    scene = "object Foo:\n  val scene = Scene(ibl = Some(IBL(samples = 9)))\n"

    findings = check_resource_bounds(scene)

    assert [f.field for f in findings] == ["samples"]


def test_a_value_below_the_minimum_is_flagged():
    scene = "object Foo:\n  val r = RenderSettings(aaMaxDepth = 0)\n"

    assert [f.field for f in check_resource_bounds(scene)] == ["aaMaxDepth"]


def test_the_product_cap_is_checked_when_both_factors_are_constants():
    over = "object Foo:\n  val s = ParametricSurface(uSteps = 2000, vSteps = 1000)\n"
    within = "object Foo:\n  val s = ParametricSurface(uSteps = 1000, vSteps = 1000)\n"
    animated = "object Foo:\n  def s(t: Float) = ParametricSurface(uSteps = 2000, vSteps = n)\n"

    assert [f.field for f in check_resource_bounds(over)] == ["uSteps"]
    assert "exceeds the cap of 1e+06" in check_resource_bounds(over)[0].message
    assert check_resource_bounds(within) == []
    assert check_resource_bounds(animated) == []


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
    assert len(scenes) == 33  # + PolytopeGallery (usability review 2026-09, F17)

    for scene in scenes:
        findings = check_resource_bounds(scene["source"])
        assert findings == [], f"{scene['name']}: {findings}"


# --- usability review 2026-10, session 3: F59 per-spongeType bounds (manifest 1.4.0) ----------


def _sponge_scene(sponge_type: str, level: str) -> str:
    return (
        "object Foo:\n  val scene = Scene(objects = List("
        f"Sponge(spongeType = {sponge_type}, level = {level})))\n"
    )


def test_recursive_ias_accepts_a_level_above_the_cube_sponge_ceiling():
    # The engine accepts [1, 14) for sponge-recursive-ias; 5.8 used to be "repaired" to 5.
    assert check_resource_bounds(_sponge_scene("RecursiveIAS", "5.8f")) == []


def test_volume_filling_level_above_the_ceiling_is_still_flagged():
    findings = check_resource_bounds(_sponge_scene("VolumeFilling", "5.8f"))

    assert len(findings) == 1
    assert "exceeds the resource-bound ceiling of 5" in findings[0].message


def test_a_too_high_level_names_the_sponge_type_that_accepts_it():
    # Usability review 2026-10, session 4 (F87): the repair lowered 5.8 to VolumeFilling's 5,
    # a level the window could not build then, although RecursiveIAS accepts it.
    findings = check_resource_bounds(_sponge_scene("VolumeFilling", "5.8f"))

    assert "spongeType RecursiveIAS (1 to 13) accepts 5.8" in findings[0].message
    assert "rather than lowering a value the user asked for" in findings[0].message


def test_no_hint_when_no_sponge_type_accepts_the_level():
    findings = check_resource_bounds(_sponge_scene("RecursiveIAS", "14f"))

    assert "accepts" not in findings[0].message


def test_recursive_ias_level_below_its_minimum_is_flagged():
    findings = check_resource_bounds(_sponge_scene("RecursiveIAS", "0f"))

    assert len(findings) == 1
    assert "below the minimum of 1" in findings[0].message


def test_recursive_ias_level_above_its_own_ceiling_is_flagged():
    findings = check_resource_bounds(_sponge_scene("RecursiveIAS", "14f"))

    assert len(findings) == 1
    assert "ceiling of 13" in findings[0].message


def test_an_unknown_or_missing_sponge_type_keeps_the_conservative_ceiling():
    unnamed = "object Foo:\n  val scene = Scene(objects = List(Sponge(level = 5.8f)))\n"

    assert len(check_resource_bounds(unnamed)) == 1
    assert len(check_resource_bounds(_sponge_scene("SpongeType.VolumeFilling", "5.8f"))) == 1


# --- Review 2026-10-14, menger-scene-agent#41: non-plain literals --------------------------


def _surface_scene(u_steps: str) -> str:
    return (
        "object Foo:\n"
        f"  val scene = Scene(objects = List(ParametricSurface(uSteps = {u_steps}, vSteps = 10)))\n"
    )


def test_arithmetic_hex_and_scientific_literals_are_evaluated_and_flagged():
    for over in ("1000 * 1001", "0xFFFFFF", "1e7", "2e6.toInt", "(1000 + 1) * 1000"):
        findings = check_resource_bounds(_surface_scene(over))
        if over.endswith(".toInt"):
            assert findings == [], over  # a method call is not a constant expression
        else:
            # The uSteps ceiling, read from the constant (the product cap may flag it too).
            assert any(f.field == "uSteps" and "ceiling" in f.message for f in findings), over


def test_a_t_driven_expression_is_still_skipped():
    assert check_resource_bounds(_surface_scene("(t * 1000000).toInt")) == []


def test_as_float_reads_scala_constant_expressions():
    from gauntlet._scala_text import as_float

    assert as_float("1000 * 1000") == 1_000_000
    assert as_float("0x3E8") == 1000
    assert as_float("0xFFL") == 255
    assert as_float("1.5e3f") == 1500
    assert as_float("-2f") == -2
    assert as_float("100_000") == 100_000
    assert as_float("1e400") == float("inf")
    assert as_float("x2") is None
    assert as_float("t * 3f") is None
    assert as_float("1 / 0") is None
    assert as_float("") is None


# Review 2026-10-14 (msa#38): a brace block with a top-level comma split the argument list and
# shifted every named argument after it.
def test_split_top_level_keeps_brace_blocks_whole():
    from gauntlet._scala_text import named_args, split_top_level

    block = "{ import menger.dsl.{Sphere, Cube}; Sphere() }"
    inner = f"f = {block}, uSteps = 2000000, vSteps = 10"
    assert split_top_level(inner)[0] == f"f = {block}"
    assert named_args(inner)["uSteps"] == "2000000"


def test_split_top_level_on_corpus_shaped_input_keeps_every_named_argument():
    import random

    from gauntlet._scala_text import named_args

    rng = random.Random(1234)
    nested = ["Vec3(1f, 2f, 3f)", "Some(Color(0f, 0f, 0f))", "{ (u, v) => (u, v) }",
              "List(Sphere(), Cube())", "[1, 2]", "0.5f", "{ import x.{A, B}; A() }"]
    for _ in range(200):
        names = rng.sample(["a", "b", "c", "d", "e"], k=rng.randint(1, 5))
        values = [rng.choice(nested) for _ in names]
        inner = ", ".join(f"{n} = {v}" for n, v in zip(names, values))
        assert named_args(inner) == dict(zip(names, values)), inner

"""Unit tests for gauntlet/lint.py -- stage 2's five named lint checks (object outside the
camera frustum, light inside geometry, camera inside an object, black-on-black material
against the background, degenerate scale). Pure string processing, no compiler, no
network."""

from __future__ import annotations

from gauntlet.lint import check_lint

CLEAN_SCENE = (
    "object CleanScene:\n"
    "  val scene = Scene(\n"
    "    camera = Camera(position = Vec3(0f, 2f, 6f), lookAt = Vec3(0f, 0f, 0f)),\n"
    "    objects = List(\n"
    "      Sphere(pos = Vec3(0f, 0f, 0f), size = 1f, color = Some(Color(0.8f, 0.2f, 0.2f)))\n"
    "    ),\n"
    "    lights = List(Point(position = Vec3(5f, 5f, 5f))),\n"
    "    background = Some(Color(0.1f, 0.1f, 0.2f))\n"
    "  )\n"
)


# --- Acceptance: clean scene passes all five checks -----------------------------------------


def test_clean_scene_has_no_findings():
    findings = check_lint(CLEAN_SCENE)

    assert findings == []


# --- Acceptance / Edge-Case Matrix: camera inside an object ---------------------------------


def test_camera_inside_object_is_flagged_naming_the_collision():
    scene = (
        "object CameraInside:\n"
        "  val scene = Scene(\n"
        "    camera = Camera(position = Vec3(0f, 0f, 0.5f), lookAt = Vec3(0f, 0f, -5f)),\n"
        "    objects = List(Sphere(pos = Vec3(0f, 0f, 0f), size = 2f))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert any(f.field == "camera_inside_object" for f in findings)
    hit = next(f for f in findings if f.field == "camera_inside_object")
    assert hit.stage == "lint"
    assert hit.identifier == "Sphere"


def test_camera_well_outside_object_is_not_flagged_for_that_check():
    scene = (
        "object CameraOutside:\n"
        "  val scene = Scene(\n"
        "    camera = Camera(position = Vec3(0f, 0f, 20f), lookAt = Vec3(0f, 0f, 0f)),\n"
        "    objects = List(Sphere(pos = Vec3(0f, 0f, 0f), size = 1f))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert not any(f.field == "camera_inside_object" for f in findings)


# --- light inside geometry --------------------------------------------------------------


def test_light_inside_geometry_is_flagged():
    scene = (
        "object LightInside:\n"
        "  val scene = Scene(\n"
        "    camera = Camera(position = Vec3(0f, 0f, 10f), lookAt = Vec3(0f, 0f, 0f)),\n"
        "    objects = List(Sphere(pos = Vec3(0f, 0f, 0f), size = 3f)),\n"
        "    lights = List(Point(position = Vec3(0f, 0f, 1f)))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert any(f.field == "light_inside_geometry" for f in findings)


def test_directional_light_is_never_checked_for_inside_geometry():
    # Directional has only a `direction`, no location in space -- never flagged.
    scene = (
        "object DirectionalOnly:\n"
        "  val scene = Scene(\n"
        "    objects = List(Sphere(pos = Vec3(0f, 0f, 0f), size = 3f)),\n"
        "    lights = List(Directional(direction = Vec3(1f, -1f, -1f), intensity = 1f))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert not any(f.field == "light_inside_geometry" for f in findings)


# --- object outside the camera frustum ----------------------------------------------------


def test_object_behind_camera_is_flagged_as_outside_the_frustum():
    scene = (
        "object BehindCamera:\n"
        "  val scene = Scene(\n"
        "    camera = Camera(position = Vec3(0f, 0f, 5f), lookAt = Vec3(0f, 0f, 0f)),\n"
        "    objects = List(Sphere(pos = Vec3(0f, 0f, 10f), size = 1f))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert any(f.field == "frustum" for f in findings)


def test_object_in_front_of_camera_is_not_flagged_for_frustum():
    findings = check_lint(CLEAN_SCENE)

    assert not any(f.field == "frustum" for f in findings)


def test_degenerate_camera_view_direction_is_never_a_crash_and_skips_frustum():
    scene = (
        "object DegenerateView:\n"
        "  val scene = Scene(\n"
        "    camera = Camera(position = Vec3(0f, 0f, 0f), lookAt = Vec3(0f, 0f, 0f)),\n"
        "    objects = List(Sphere(pos = Vec3(0f, 0f, 10f), size = 1f))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert not any(f.field == "frustum" for f in findings)


# --- black-on-black material against the background -----------------------------------------


def test_black_object_against_black_background_is_flagged():
    scene = (
        "object BlackOnBlack:\n"
        "  val scene = Scene(\n"
        "    camera = Camera(position = Vec3(0f, 0f, 5f), lookAt = Vec3(0f, 0f, 0f)),\n"
        "    objects = List(\n"
        "      Sphere(pos = Vec3(0f, 0f, 0f), size = 1f, color = Some(Color(0.01f, 0.01f, 0.01f)))\n"
        "    ),\n"
        "    background = Some(Color(0.02f, 0.02f, 0.02f))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert any(f.field == "black_on_black" for f in findings)


def test_bright_object_against_black_background_is_not_flagged():
    scene = (
        "object NotBlack:\n"
        "  val scene = Scene(\n"
        "    objects = List(\n"
        "      Sphere(pos = Vec3(0f, 0f, 0f), size = 1f, color = Some(Color(0.9f, 0.1f, 0.1f)))\n"
        "    ),\n"
        "    background = Some(Color(0.02f, 0.02f, 0.02f))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert not any(f.field == "black_on_black" for f in findings)


def test_object_with_no_explicit_literal_color_is_never_checked_for_black_on_black():
    # No `color` field at all -- a material preset supplies appearance; can't reason about it.
    scene = (
        "object NoColor:\n"
        "  val scene = Scene(\n"
        "    objects = List(Sphere(pos = Vec3(0f, 0f, 0f), size = 1f, material = Some(Material.Glass))),\n"
        "    background = Some(Color(0.02f, 0.02f, 0.02f))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert not any(f.field == "black_on_black" for f in findings)


# --- degenerate (near-zero) scale ------------------------------------------------------------


def test_near_zero_size_is_flagged_as_degenerate_scale():
    scene = "object TinySphere:\n  val scene = Scene(objects = List(Sphere(pos = Vec3(5f, 5f, 5f), size = 0.0001f)))\n"

    findings = check_lint(scene)

    assert any(f.field == "degenerate_scale" for f in findings)


def test_normal_size_is_not_flagged_as_degenerate_scale():
    scene = "object NormalSphere:\n  val scene = Scene(objects = List(Sphere(pos = Vec3(5f, 5f, 5f), size = 1.5f)))\n"

    findings = check_lint(scene)

    assert not any(f.field == "degenerate_scale" for f in findings)


# --- Edge-Case Matrix: animated (non-literal) field is skipped for that check, not failed ---


def test_animated_size_expression_skips_degenerate_scale_check_not_failed():
    scene = (
        "object AnimatedSize:\n"
        "  val scene = Scene(objects = List(Sphere(pos = Vec3(0f, 0f, 0f), size = baseSize * t)))\n"
    )

    findings = check_lint(scene)

    assert findings == []


def test_animated_camera_position_skips_camera_dependent_checks():
    scene = (
        "object AnimatedCamera:\n"
        "  val scene = Scene(\n"
        "    camera = Camera(position = camPos(t), lookAt = Vec3(0f, 0f, 0f)),\n"
        "    objects = List(Sphere(pos = Vec3(0f, 0f, 0f), size = 1f))\n"
        "  )\n"
    )

    findings = check_lint(scene)

    assert findings == []


# --- Edge-Case Matrix: unparseable/garbage input --------------------------------------------


def test_garbage_input_never_raises_and_returns_a_list():
    findings = check_lint("not even remotely valid scala {{{ ]][[")

    assert findings == []


def test_empty_input_never_raises():
    findings = check_lint("")

    assert findings == []


def test_scene_with_no_camera_never_raises_camera_checks():
    scene = "object NoCamera:\n  val scene = Scene(objects = List(Sphere(pos = Vec3(0f,0f,0f), size = 1f)))\n"

    findings = check_lint(scene)

    assert findings == []


# --- Verification: story 2's real, already-compiling poc-run fixtures -----------------------


def test_poc_run_turn1_generated_scala_has_no_lint_findings():
    scene_text = open("poc-run/turn1-generated.scala").read()

    findings = check_lint(scene_text)

    assert findings == []


def test_poc_run_turn2_refined_scala_has_no_lint_findings():
    scene_text = open("poc-run/turn2-refined.scala").read()

    findings = check_lint(scene_text)

    assert findings == []


# --- Verification: the full 32-scene reference corpus (review round 1) --------------------


def test_full_corpus_has_no_lint_findings():
    import json

    corpus = json.load(open("reference/dsl-corpus.json"))
    scenes = corpus["scenes"]
    assert len(scenes) == 32

    for scene in scenes:
        findings = check_lint(scene["source"])
        assert findings == [], f"{scene['name']}: {findings}"

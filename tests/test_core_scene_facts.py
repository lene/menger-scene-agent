"""Unit tests for core/scene_facts.py -- ground-truth extraction shared by lint.py and the
F7/F8/F20/F23 usability follow-ups. Pure string processing, no compiler, no network."""

from __future__ import annotations

from core.scene_facts import extract_scene_facts, facts_diff, occlusion_warnings, parse_color_rgb, parse_vec3


# --- parse_vec3 / parse_color_rgb -----------------------------------------------------------


def test_parse_vec3_reads_three_literal_components():
    assert parse_vec3("Vec3(1f, 2f, -3f)") == (1.0, 2.0, -3.0)


def test_parse_vec3_returns_none_for_non_literal():
    assert parse_vec3("Vec3(t, 2f, 3f)") is None


def test_parse_color_rgb_ignores_a_fourth_alpha_component():
    assert parse_color_rgb("Color(1f, 0f, 0f, 0.5f)") == (1.0, 0.0, 0.0)


# --- object facts: pos/size/color/rotation defaults and literals ----------------------------


def test_object_takes_dsl_default_pos_and_size_when_absent():
    scene = "Sphere(color = Some(Color(1f, 1f, 1f)))"
    facts = extract_scene_facts(scene)

    assert len(facts.objects) == 1
    assert facts.objects[0].pos == (0.0, 0.0, 0.0)
    assert facts.objects[0].size == 1.0


def test_object_rotation_is_extracted_when_present():
    scene = "Cube(rotation = Vec3(0f, 1.5f, 0f))"
    facts = extract_scene_facts(scene)

    assert facts.objects[0].rotation == (0.0, 1.5, 0.0)


# --- material + opacity (F14 rule) -----------------------------------------------------------


def test_transparent_material_preset_yields_low_opacity_regardless_of_color():
    scene = "Sponge(material = Some(Material.Glass), color = Some(Color(1f, 0f, 0f)))"
    facts = extract_scene_facts(scene)

    obj = facts.objects[0]
    assert obj.material == "Glass"
    assert obj.opacity < 0.9
    assert not obj.is_opaque


def test_opaque_material_preset_yields_full_opacity():
    scene = "Sponge(material = Some(Material.Chrome))"
    facts = extract_scene_facts(scene)

    assert facts.objects[0].is_opaque


def test_no_material_falls_back_to_color_alpha():
    scene = "Sphere(color = Some(Color(1f, 0f, 0f, 0.1f)))"
    facts = extract_scene_facts(scene)

    obj = facts.objects[0]
    assert obj.material is None
    assert obj.opacity == 0.1
    assert not obj.is_opaque


def test_no_material_and_no_color_is_conservatively_opaque():
    scene = "Sphere(size = 2f)"
    facts = extract_scene_facts(scene)

    assert facts.objects[0].opacity is None
    assert facts.objects[0].is_opaque


# --- Directional light direction phrase (F7) -------------------------------------------------


def test_directional_light_from_above_travelling_down():
    scene = "Directional(direction = Vec3(0f, -1f, 0f))"
    facts = extract_scene_facts(scene)

    assert facts.lights[0].direction_phrase == "from above"


def test_directional_light_from_below_and_sideways():
    scene = "Directional(direction = Vec3(1f, 1f, 0f))"
    facts = extract_scene_facts(scene)

    phrase = facts.lights[0].direction_phrase
    assert "from below" in phrase
    assert "+x" in phrase


def test_positioned_light_has_no_direction_phrase():
    scene = "Point(position = Vec3(0f, 5f, 0f))"
    facts = extract_scene_facts(scene)

    assert facts.lights[0].direction_phrase is None


# --- scene extent (F20) -----------------------------------------------------------------------


def test_extent_center_and_radius_cover_every_positioned_object():
    scene = (
        "Sphere(pos = Vec3(-2f, 0f, 0f), size = 1f)\n"
        "Sphere(pos = Vec3(2f, 0f, 0f), size = 1f)\n"
    )
    facts = extract_scene_facts(scene)

    assert facts.extent_center == (0.0, 0.0, 0.0)
    assert facts.extent_radius == 3.0  # distance 2 to each center + its own size 1


def test_extent_is_none_for_a_scene_with_no_objects():
    facts = extract_scene_facts("Camera(position = Vec3(0f, 0f, 5f), lookAt = Vec3(0f, 0f, 0f))")

    assert facts.extent_center is None
    assert facts.extent_radius is None


# --- camera / background ----------------------------------------------------------------------


def test_camera_position_and_look_at_extracted():
    scene = "Camera(position = Vec3(0f, 0f, 5f), lookAt = Vec3(0f, 0f, 0f))"
    facts = extract_scene_facts(scene)

    assert facts.camera == ((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))


def test_background_color_extracted():
    scene = "background = Some(Color(0.1f, 0.1f, 0.1f))"
    facts = extract_scene_facts(scene)

    assert facts.background == (0.1, 0.1, 0.1)


# --- facts_diff (F8) --------------------------------------------------------------------------


def test_facts_diff_flags_a_moved_object_by_same_type_and_position():
    before = extract_scene_facts("Sphere(pos = Vec3(0f, 0f, 0f))")
    after = extract_scene_facts("Sphere(pos = Vec3(3f, 0f, 0f))")

    warnings = facts_diff(before, after)

    assert len(warnings) == 1
    assert "Sphere" in warnings[0]
    assert "(0.0, 0.0, 0.0)" in warnings[0]
    assert "(3.0, 0.0, 0.0)" in warnings[0]


def test_facts_diff_flags_a_changed_material():
    before = extract_scene_facts("Sponge(material = Some(Material.Glass))")
    after = extract_scene_facts("Sponge(material = Some(Material.Chrome))")

    warnings = facts_diff(before, after)

    assert any("Glass" in w and "Chrome" in w for w in warnings)


def test_facts_diff_flags_a_moved_camera():
    before = extract_scene_facts("Camera(position = Vec3(0f, 0f, 5f), lookAt = Vec3(0f, 0f, 0f))")
    after = extract_scene_facts("Camera(position = Vec3(0f, 0f, 9f), lookAt = Vec3(0f, 0f, 0f))")

    warnings = facts_diff(before, after)

    assert any("camera" in w for w in warnings)


def test_facts_diff_is_empty_when_nothing_relevant_changed():
    before = extract_scene_facts("Sphere(pos = Vec3(0f, 0f, 0f))")
    after = extract_scene_facts("Sphere(pos = Vec3(0f, 0f, 0f))")

    assert facts_diff(before, after) == []


def test_facts_diff_skips_objects_whose_type_changed_at_the_same_index():
    # No stable identity to match by (module docstring) -- a same-index type change is an
    # add/remove, not a "move", so it's silently not compared rather than misreported.
    before = extract_scene_facts("Sphere(pos = Vec3(0f, 0f, 0f))")
    after = extract_scene_facts("Cube(pos = Vec3(5f, 0f, 0f))")

    assert facts_diff(before, after) == []


# --- occlusion_warnings (F23) ------------------------------------------------------------------


def test_occlusion_warning_for_a_small_orb_centered_inside_an_opaque_sponge():
    scene = (
        "Sponge(pos = Vec3(0f, 0f, 0f), size = 5f, material = Some(Material.Chrome))\n"
        "Sphere(pos = Vec3(0f, 0f, 0f), size = 0.5f)\n"
    )
    facts = extract_scene_facts(scene)

    warnings = occlusion_warnings(facts)

    assert len(warnings) == 1
    assert "Sphere" in warnings[0]
    assert "Sponge" in warnings[0]


def test_no_occlusion_warning_when_the_containing_object_is_transparent():
    scene = (
        "Sponge(pos = Vec3(0f, 0f, 0f), size = 5f, material = Some(Material.Glass))\n"
        "Sphere(pos = Vec3(0f, 0f, 0f), size = 0.5f)\n"
    )
    facts = extract_scene_facts(scene)

    assert occlusion_warnings(facts) == []


def test_no_occlusion_warning_for_two_separate_objects():
    scene = (
        "Sponge(pos = Vec3(-5f, 0f, 0f), size = 1f)\n"
        "Sphere(pos = Vec3(5f, 0f, 0f), size = 1f)\n"
    )
    facts = extract_scene_facts(scene)

    assert occlusion_warnings(facts) == []


# --- garbage input never raises ----------------------------------------------------------------


def test_garbage_input_never_raises():
    facts = extract_scene_facts("not valid scala {{{ ]][[")

    assert facts.objects == []
    assert facts.lights == []
    assert facts.camera is None


# --- usability review 2026-09, session 2: F36 (msa#15), #4c/F55 (msa#13) -------------------

from core.scene_facts import caveat_warnings, manifest_warn_levels  # noqa: E402

_WARN_LEVELS = {("TesseractSponge", "level"): 2.0, ("Sponge", "level"): 3.0}

_ANIMATED_SPONGE = """
object S:
  val duration = 10f
  def scene(t: Float): Scene =
    val progress = math.max(0f, math.min(t / duration, 1f))
    val level = 1f + progress * 2f
    Scene(objects = List(TesseractSponge(spongeType = VolumeRemoving, level = level,
      material = Some(Material.%s))))
"""


def test_object_level_is_the_upper_bound_of_an_animated_level_expression():
    facts = extract_scene_facts(_ANIMATED_SPONGE % "Gold")
    assert facts.objects[0].level == 3.0


def test_object_level_of_a_literal_and_of_t_over_duration():
    literal = extract_scene_facts("Sponge(level = 2.5f)")
    direct = extract_scene_facts(
        "val duration = 4f\ndef scene(t: Float) = Sponge(level = 2f * t / duration)"
    )
    assert literal.objects[0].level == 2.5
    assert direct.objects[0].level == 2.0


def test_object_level_is_none_for_an_expression_it_cannot_evaluate():
    facts = extract_scene_facts("Sponge(level = someHelper(t))")
    assert facts.objects[0].level is None


def test_caveat_warns_when_an_animated_level_passes_the_manifest_warn_level():
    warnings = caveat_warnings(extract_scene_facts(_ANIMATED_SPONGE % "Gold"), _WARN_LEVELS)
    assert any("level 3" in w and "slow" in w for w in warnings)


def test_caveat_warns_about_glass_on_a_tesseract_sponge_from_level_2():
    warnings = caveat_warnings(extract_scene_facts(_ANIMATED_SPONGE % "Glass"), _WARN_LEVELS)
    assert any("Glass" in w and "chaotic" in w for w in warnings)


def test_caveat_is_silent_below_the_warn_level_and_for_glass_at_level_1():
    facts = extract_scene_facts("TesseractSponge(level = 1f, material = Some(Material.Glass))")
    assert caveat_warnings(facts, _WARN_LEVELS) == []


def test_manifest_warn_levels_reads_field_warn_at():
    manifest = {"objects": [{"name": "Sponge", "fields": [
        {"name": "level", "warnAt": 3.0}, {"name": "size"}
    ]}]}
    assert manifest_warn_levels(manifest) == {("Sponge", "level"): 3.0}


# --- usability review 2026-09, session 2: F8 recurrence + F39 (msa#7) -----------------------


def test_facts_diff_matches_objects_within_their_type_when_another_type_is_added():
    # Comparing by overall index paired the moved Sponge with the new Sphere and skipped it.
    before = extract_scene_facts("Sponge(pos = Vec3(0f, 0f, 0f))")
    after = extract_scene_facts("Sphere(pos = Vec3(0f, 3f, 0f))\nSponge(pos = Vec3(1f, 1f, 1f))")

    assert any("moved the Sponge" in w for w in facts_diff(before, after))


def test_facts_diff_reports_a_change_of_a_non_literal_pos():
    # Session 2, task 3.5: the sponge moved via `SpongeSize / 2`-style expressions, unreported.
    before = extract_scene_facts("Sponge(pos = Vec3(0f, 0f, 0f))")
    after = extract_scene_facts("Sponge(pos = Vec3(s / 2, s / 2, s / 2))")

    warnings = facts_diff(before, after)

    assert any("moved the Sponge" in w and "s / 2" in w for w in warnings)


def test_facts_diff_reports_a_changed_4d_projection():
    before = extract_scene_facts("Tesseract(size = 1f)")
    after = extract_scene_facts("Tesseract(size = 1f, projection = Some(Projection4DSpec(rotXW = 30f)))")

    assert any("4D projection" in w for w in facts_diff(before, after))


def test_facts_diff_does_not_report_what_the_request_asked_for():
    # F39: "also changed the Sponge's material from Gold to metal" after "make it aluminium".
    before = extract_scene_facts("Sponge(material = Some(Material.Gold))")
    after = extract_scene_facts("Sponge(material = Some(Material.Chrome))")

    assert facts_diff(before, after, prompt="make it aluminium") == []
    assert facts_diff(before, after, prompt="make it bigger") != []

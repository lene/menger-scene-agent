"""Unit tests for core/scene_facts.py -- ground-truth extraction shared by lint.py and the
F7/F8/F20/F23 usability follow-ups. Pure string processing, no compiler, no network."""

from __future__ import annotations

from core.scene_facts import (
    camera_gaze_warnings, extract_scene_facts, facts_diff, occlusion_warnings, parse_color_rgb,
    parse_vec3, scene_changes,
)


# --- parse_vec3 / parse_color_rgb -----------------------------------------------------------


def test_parse_vec3_reads_three_literal_components():
    assert parse_vec3("Vec3(1f, 2f, -3f)") == (1.0, 2.0, -3.0)


def test_parse_vec3_returns_none_for_non_literal():
    assert parse_vec3("Vec3(t, 2f, 3f)") is None


def test_parse_vec3_reads_a_tuple_literal():
    # Session 3's scenes all wrote `pos = (0f, 0f, 0f)`; only `Vec3(...)` was parsed, so every
    # position and the camera were unknown and the occlusion check never ran (F78).
    assert parse_vec3("(1f, 2f, -3f)") == (1.0, 2.0, -3.0)


def test_camera_with_tuple_vectors_is_extracted():
    facts = extract_scene_facts("Camera(position = (9f, 4.75f, 13.5f), lookAt = (0f, 3.25f, 0f))")

    assert facts.camera == ((9.0, 4.75, 13.5), (0.0, 3.25, 0.0))


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


def test_local_material_is_named_by_its_val_not_none():
    # F75: `material = Some(emissiveGreen)` read as "none", so a turn reported "changed the
    # material from Film to none".
    scene = (
        "val emissiveGreen = Material(color = Color(0.1f, 0.8f, 0.3f), emission = 1.0f)\n"
        "Sponge(material = Some(emissiveGreen))\n"
    )

    fact = extract_scene_facts(scene).objects[0]

    assert fact.material == "emissiveGreen"
    assert fact.is_opaque


def test_local_copy_of_a_preset_keeps_the_preset_name():
    scene = (
        "private val tinted = Material.Glass.copy(ior = 1.6f)\n"
        "Sphere(material = Some(tinted))\n"
    )

    fact = extract_scene_facts(scene).objects[0]

    assert fact.material == "Glass"
    assert not fact.is_opaque


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


def test_occlusion_warning_for_a_small_orb_centered_inside_an_opaque_cube():
    scene = (
        "Cube(pos = Vec3(0f, 0f, 0f), size = 5f, material = Some(Material.Chrome))\n"
        "Sphere(pos = Vec3(0f, 0f, 0f), size = 0.5f)\n"
    )
    facts = extract_scene_facts(scene)

    warnings = occlusion_warnings(facts)

    assert len(warnings) == 1
    assert "Sphere" in warnings[0]
    assert "Cube" in warnings[0]


def test_occlusion_warning_for_an_orb_inside_a_cube_with_a_local_material():
    scene = (
        "val stone = Material(color = Color(0.5f, 0.5f, 0.5f), roughness = 1.0f)\n"
        "Cube(pos = (0f, 0f, 0f), size = 5f, material = Some(stone))\n"
        "Sphere(pos = (0f, 0f, 0f), size = 0.5f)\n"
    )

    assert len(occlusion_warnings(extract_scene_facts(scene))) == 1


def test_no_occlusion_warning_inside_a_sponge_because_it_has_holes():
    # Session 3, 4.13: the orb inside the tesseract sponge was visible through its holes (F78).
    for occluder in ("Sponge", "TesseractSponge"):
        scene = (
            f"{occluder}(pos = (0f, 0f, 0f), size = 2.5f, material = Some(Material.Chrome))\n"
            "Sphere(pos = (0f, 0f, 0f), size = 0.8f)\n"
        )

        assert occlusion_warnings(extract_scene_facts(scene)) == [], occluder


def test_no_occlusion_warning_when_the_containing_object_is_transparent():
    scene = (
        "Cube(pos = Vec3(0f, 0f, 0f), size = 5f, material = Some(Material.Glass))\n"
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

from core.scene_facts import (  # noqa: E402
    caveat_warnings,
    manifest_subtype_warn_levels,
    manifest_warn_levels,
)

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


# --- usability review 2026-10, session 3: F59 per-subtype warnAt (manifest 1.4.0) -------------

_SUBTYPE_MANIFEST = {"objects": [{"name": "Sponge", "fields": [{
    "name": "level", "warnAt": 3.0,
    "limitsBy": {"field": "spongeType", "values": {
        "VolumeFilling": {"min": 0, "max": 5, "warnAt": 3},
        "RecursiveIAS": {"min": 1, "max": 13},
    }},
}]}]}


def test_object_fact_carries_the_sponge_type_argument():
    facts = extract_scene_facts("Sponge(spongeType = SpongeType.RecursiveIAS, level = 5f)")
    assert facts.objects[0].subtype == "RecursiveIAS"


def test_manifest_subtype_warn_levels_reads_limits_by_and_keeps_missing_warn_at_as_none():
    assert manifest_subtype_warn_levels(_SUBTYPE_MANIFEST) == {
        ("Sponge", "level"): {"VolumeFilling": 3.0, "RecursiveIAS": None}
    }


def test_caveat_does_not_warn_about_slowness_for_a_recursive_ias_sponge():
    facts = extract_scene_facts("Sponge(spongeType = RecursiveIAS, level = 6f)")
    warnings = caveat_warnings(
        facts,
        manifest_warn_levels(_SUBTYPE_MANIFEST),
        manifest_subtype_warn_levels(_SUBTYPE_MANIFEST),
    )
    assert warnings == []


def test_caveat_still_warns_for_a_volume_filling_sponge_at_the_warn_level():
    facts = extract_scene_facts("Sponge(spongeType = VolumeFilling, level = 3f)")
    warnings = caveat_warnings(
        facts,
        manifest_warn_levels(_SUBTYPE_MANIFEST),
        manifest_subtype_warn_levels(_SUBTYPE_MANIFEST),
    )
    assert any("slow" in w for w in warnings)


# --- scene_changes (F62) -----------------------------------------------------------------------


def test_scene_changes_lists_changed_values_and_added_objects():
    before = extract_scene_facts("Sponge(level = 2f, material = Some(Material.Gold))")
    after = extract_scene_facts(
        "Sponge(level = 3f, material = Some(Material.Gold))\nSphere(pos = (0f, 2f, 0f))"
    )

    assert scene_changes(before, after) == ["added 1 Sphere", "Sponge level: 2.0 -> 3.0"]


def test_scene_changes_sees_an_animated_rotation():
    before = extract_scene_facts("Cube(size = 1f)")
    after = extract_scene_facts("Cube(size = 1f, rotation = Vec3(0f, angle, 0f))")

    assert scene_changes(before, after) == ["Cube rotation: default -> Vec3(0f, angle, 0f)"]


def test_scene_changes_is_empty_for_the_same_scene():
    facts = extract_scene_facts("Cube(size = 1f)")

    assert scene_changes(facts, facts) == []


# --- camera_gaze_warnings (F83) ----------------------------------------------------------------

_FLIGHT = """
object Flight:
  val duration = 12f
  def scene(t: Float): Scene =
    val progress = math.max(0f, math.min(t, duration)) / duration
    val camZ     = 12f - progress * 24f
    Scene(
      camera = Camera(position = (0f, 0f, camZ), lookAt = (0f, 0f, camZ - 1f)),
      objects = List(Sponge(pos = (0f, 0f, 0f), size = 2.5f))
    )
"""


def test_an_animated_camera_that_ends_up_looking_at_nothing_is_reported():
    # Session 3, Task 8b turn 4: the camera left the sponge at about 6.6 s and looked at an
    # empty horizon; the material and level phases after that were never seen.
    warnings = camera_gaze_warnings(extract_scene_facts(_FLIGHT), _FLIGHT)

    assert len(warnings) == 1
    assert "t = 7.2 s to 12.0 s" in warnings[0]


def test_a_camera_that_keeps_looking_at_the_object_is_fine():
    scene = _FLIGHT.replace("lookAt = (0f, 0f, camZ - 1f)", "lookAt = (0f, 0f, 0f)")

    assert camera_gaze_warnings(extract_scene_facts(scene), scene) == []


def test_a_static_camera_is_left_to_the_frustum_lint():
    scene = "Camera(position = (0f, 0f, 5f), lookAt = (0f, 0f, 9f))\nSponge(size = 1f)"

    assert camera_gaze_warnings(extract_scene_facts(scene), scene) == []


def test_a_durationSeconds_animation_is_bounded_like_a_duration_one():
    # menger#65 (F84): `val durationSeconds` replaces `val duration`; both are read.
    for name in ("durationSeconds", "duration"):
        scene = f"val {name} = 4f\nSponge(level = 1f + t / 2f)"

        assert extract_scene_facts(scene).objects[0].level == 3.0, name

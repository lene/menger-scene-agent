"""core/snapshot_facts.py against real validator snapshots of usability session 4's scenes
(tests/fixtures/snapshots/*.json, produced by menger's SceneValidator, result schema 1.1.0)."""

from __future__ import annotations

import json
from pathlib import Path

from core.snapshot_facts import (
    camera_path_check,
    check_turn,
    material_name,
    snapshot_changes,
    unrequested_changes,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "snapshots"
_MANIFEST = json.loads(
    (Path(__file__).resolve().parent.parent / "reference" / "dsl-manifest.json").read_text()
)


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / f"{name}.json").read_text())


def test_task_2_reports_exactly_the_duration_and_level_floor():
    # Session 4 Task 2 "slower, and start it already at level 1": the readback said "Nothing
    # the facts cover changed".
    changes = snapshot_changes(
        _load("task2-001")["snapshots"], _load("task2-002")["snapshots"], _MANIFEST
    )

    assert changes == [
        "duration: 10 s -> 20 s",
        "tesseract-sponge-surface level: animated from 0.5 to 2 -> animated from 1 to 2",
    ]


def test_a_material_change_is_reported_with_preset_names_and_values():
    changes = snapshot_changes(
        _load("t4-015")["snapshots"], _load("t4-016")["snapshots"], _MANIFEST
    )

    assert (
        "24-cell material: Glass, colour (1, 1, 1), alpha 0.02 -> "
        "Glass, colour (1, 0.45, 0), alpha 0.5, emission 0.35"
    ) in changes


def test_an_added_object_is_one_line_with_its_main_values():
    changes = snapshot_changes(
        _load("8a-004")["snapshots"], _load("8a-005")["snapshots"], _MANIFEST
    )

    assert changes[0] == (
        "added tesseract-sponge-surface (position (0, 0, 0), size 4, "
        "level animated 0 -> peak 3 -> 0, material Glass, colour (1, 1, 1), alpha 0.02)"
    )


def test_identical_snapshots_have_no_changes():
    snapshots = _load("8b-010")["snapshots"]

    assert snapshot_changes(snapshots, snapshots, _MANIFEST) == []


def test_summarise_reports_only_non_default_render_settings():
    from core.snapshot_facts import summarise

    snapshot = [{"t": 0, "render": "RenderSettings(true,false,true,3,0.1,None,4,Off)"}]

    assert summarise(snapshot, _MANIFEST)[("scene", "render settings")] == (
        "antialiasing true, AA depth 3, accumulation 4"
    )


def test_summarise_omits_default_render_settings():
    from core.snapshot_facts import summarise

    snapshot = [{"t": 0, "render": "RenderSettings(true,false,false,2,0.1,None,1,Off)"}]

    assert ("scene", "render settings") not in summarise(snapshot, _MANIFEST)


def test_a_kept_procedural_under_a_new_colour_is_a_finding():
    # Session 4, 4.9: the sponge "glow green" kept xyz_rgb_local, which replaces the colour.
    result = check_turn(
        _load("t4-015")["snapshots"],
        _load("t4-016")["snapshots"],
        [],
        "make the 24-cell glow orange and the sponge glow green",
        _MANIFEST,
    )

    assert len(result.findings) == 1
    assert "proceduralType 11 (xyz_rgb_local)" in result.findings[0]
    assert "replaces the new colour" in result.findings[0]


def test_no_procedural_finding_when_the_request_asks_for_the_pattern():
    result = check_turn(
        _load("t4-015")["snapshots"],
        _load("t4-016")["snapshots"],
        [],
        "make it green but keep the xyz colouring",
        _MANIFEST,
    )

    assert result.findings == []


def test_camera_jumps_between_legs_are_findings():
    # Session 4, 8b turn 10: the legs did not join at 7.5 s and 10.5 s.
    data = _load("8b-010")
    result = camera_path_check(data["cameraPath"], data["snapshots"], "fly through the tunnel")

    assert any("jumps at t = 7.55 s, 10.55 s" in f for f in result.findings)


def test_looking_away_from_the_motion_is_a_finding_when_asked_to_look_ahead():
    # Session 4, 8b turn 3: the look direction was 90 degrees off during the turn.
    data = _load("8b-003")
    result = camera_path_check(
        data["cameraPath"], data["snapshots"], "fly the turn; look along the flight direction"
    )

    assert any("4.8-6.4 s" in f and "tangent" in f for f in result.findings)


def test_looking_away_is_only_a_warning_without_a_look_ahead_request():
    data = _load("8b-003")
    result = camera_path_check(data["cameraPath"], data["snapshots"], "fly through the tunnel")

    assert not any("degrees away" in f for f in result.findings)
    assert any("degrees away" in w for w in result.warnings)


def test_a_correct_flight_has_no_camera_findings():
    # Session 4, 8b turn 4 fixed the look direction (constant speed, no jumps).
    data = _load("8b-004")
    result = camera_path_check(
        data["cameraPath"], data["snapshots"], "fly through; look along the flight direction"
    )

    assert result.findings == []


def test_an_orbit_is_not_checked_like_a_flight():
    data = _load("task2-001")

    assert camera_path_check(data["cameraPath"], data["snapshots"], "slower").findings == []


def test_a_refactor_that_keeps_the_values_is_no_unrequested_change():
    # Session 4, 8a turn 5: moving the projection into a local val gave "also changed the
    # Tesseract's 4D projection to projection".
    snapshots = _load("8a-004")["snapshots"]

    assert unrequested_changes(snapshots, snapshots, "add a sponge", _MANIFEST) == []


def test_an_unasked_camera_move_is_reported_with_values():
    warnings = unrequested_changes(
        _load("8b-004")["snapshots"], _load("8b-010")["snapshots"], "make it 12 s long", _MANIFEST
    )

    assert any(w.startswith("also changed camera position:") for w in warnings)


def test_material_name_ignores_tint_and_emission():
    glass = next(m for m in _MANIFEST["materials"] if m["name"] == "Glass")
    tinted = {**glass, "color": [0.8, 0.06, 0.12, 0.05], "emission": 0.35}

    assert material_name(tinted, _MANIFEST) == (
        "Glass, colour (0.8, 0.06, 0.12), alpha 0.05, emission 0.35"
    )


def test_unasked_size_and_edge_choices_are_noted():
    after = [
        {
            "objects": [
                {
                    "objectType": "sponge-volume",
                    "pos": [0, 0, 0],
                    "size": 2.5,
                    "proceduralType": 0,
                    "edgeMaterial": {
                        **next(m for m in _MANIFEST["materials"] if m["name"] == "Copper"),
                        "color": [1, 0.5, 0.2, 1],
                    },
                }
            ]
        }
    ]

    result = check_turn([], after, [], "a menger sponge level 2", _MANIFEST)

    assert result.notes == [
        "chose for the sponge-volume, unasked: size 2.5 (default 1), Copper edges -- "
        "say if you want otherwise"
    ]


# --- Result schema 1.2.0 (review 2026-10-14, menger#92): DSL names and planes ---------------


def _snap(objects: list[dict], planes: list[dict]) -> list[dict]:
    return [
        {
            "t": 0,
            "camera": {"position": [0, 0, 5], "lookAt": [0, 0, 0]},
            "objects": objects,
            "lights": [],
            "planes": planes,
        }
    ]


_SPONGE = {
    "objectType": "sponge-volume",
    "dslType": "Sponge",
    "pos": [0, 0, 0],
    "size": 1,
    "level": 2,
    "proceduralType": 0,
}
_FLOOR = {
    "axis": "Y",
    "positive": True,
    "value": -1,
    "color": [0.5, 0.5, 0.5, 1],
    "checkerColor": None,
    "material": None,
}


def test_a_plane_change_is_reported():
    # The Tier 2 live run: a plane made blue was missing from Changed:, and the readback
    # claimed it "was already blue".
    blue = dict(_FLOOR, color=[0.2, 0.4, 0.9, 1])
    changes = snapshot_changes(_snap([_SPONGE], [_FLOOR]), _snap([_SPONGE], [blue]), _MANIFEST)
    assert changes == ["plane colour: (0.5, 0.5, 0.5, 1) -> (0.2, 0.4, 0.9, 1)"]


def test_objects_are_named_as_the_dsl_names_them_and_a_subtype_change_still_shows():
    surface = dict(_SPONGE, objectType="sponge-surface")
    changes = snapshot_changes(_snap([_SPONGE], []), _snap([surface], []), _MANIFEST)
    assert changes == ["Sponge subtype: sponge-volume -> sponge-surface"]


def test_same_material_between_4d_objects_names_missing_edge_tubes():
    # Session 5 (F120, msa#53): the 24-cell got the sponge's Film but not its edge tubes.
    from core.snapshot_facts import same_material_edge_findings

    projection = {"eyeW": 3, "screenW": 1.5, "rotXW": 0, "rotYW": 10, "rotZW": 0, "wScale": 1}
    after = [{"t": 0, "objects": [
        {"objectType": "tesseract-sponge-volume", "dslType": "TesseractSponge",
         "edgeRadius": 0.005, "edgeMaterial": None, "projection": projection},
        {"objectType": "sphere", "dslType": "Sphere", "edgeRadius": None, "projection": None},
        {"objectType": "24-cell", "dslType": "Icositetrachoron", "edgeRadius": None,
         "projection": projection},
    ]}]
    prompt = "make the 24-cell the same size and material as the sponge"

    findings = same_material_edge_findings(after, prompt)

    assert len(findings) == 1
    assert findings[0].startswith("Icositetrachoron: ") and "edgeRadius 0.005" in findings[0]
    assert same_material_edge_findings(after, "make the 24-cell bigger") == []

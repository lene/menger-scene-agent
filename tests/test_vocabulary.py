"""The manifest-closure test (review 2026-10-14, msa#28): a regenerated reference manifest that
brings vocabulary this agent does not handle fails here, instead of the agent silently ignoring
it (the five regular polytopes went six days without facts). Adding a name below is the
decision that the agent handles it, or deliberately ignores it."""

from __future__ import annotations

from core import vocabulary

# Every top-level section the agent reads or has decided it does not need.
KNOWN_SECTIONS = {
    "schemaVersion", "contentRevision", "scalaVersion", "optixJniVersion", "minDriverVersion",
    "conventions", "objects", "enums", "sceneComposition", "materials", "lights", "placement",
    "colorPresets", "plane", "planeAxes", "cameraPath", "cameraPathHelpers", "renderSettings",
    "renderSettingsPresets", "proceduralPresets", "productCaps", "compileSurface",
}
KNOWN_TRAITS = {"holed", "positioned"}
KNOWN_PRESET_EFFECTS = {"modulates", "replaces"}


def test_no_manifest_section_is_unknown_to_the_agent():
    assert set(vocabulary.MANIFEST) <= KNOWN_SECTIONS


def test_no_type_trait_is_unknown_to_the_agent():
    traits = {
        trait for t in vocabulary.MANIFEST["objects"] + vocabulary.MANIFEST["lights"]
        for trait in t.get("traits", [])
    }
    assert traits <= KNOWN_TRAITS


def test_no_preset_effect_is_unknown_to_the_agent():
    effects = {p["effect"] for p in vocabulary.MANIFEST["proceduralPresets"]}
    assert effects <= KNOWN_PRESET_EFFECTS


def test_the_vocabulary_is_read_from_the_manifest():
    assert "Icositetrachoron" in vocabulary.OBJECT_TYPES  # the polytopes that went missing
    assert set(vocabulary.POSITIONED_LIGHT_TYPES) == {"Point", "AreaLight"}
    assert {"Glass", "Film"} <= vocabulary.TRANSPARENT_MATERIALS
    assert vocabulary.REPLACING_PRESETS[11] == "xyz_rgb_local"
    assert "menger.dsl" in vocabulary.IMPORT_ROOTS
    assert "menger.Projection4DSpec" in vocabulary.IMPORT_EXACT
    assert len(vocabulary.DSL_REVISION) == 64

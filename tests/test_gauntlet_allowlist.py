"""Unit tests for gauntlet/allowlist.py -- AD-4 rule 1 static allowlist check. Pure string
processing, no compiler, no network."""

from __future__ import annotations

from gauntlet.allowlist import check_allowlist

CLEAN_SCENE = (
    "package examples.dsl\n\n"
    "import scala.language.implicitConversions\n\n"
    "import menger.dsl._\n\n"
    "object GlassSphere:\n"
    "  val scene = Scene(\n"
    "    camera = Camera(position = Vec3(0f, 0f, 5f), lookAt = Vec3(0f, 0f, 0f)),\n"
    "    objects = List(Sphere(material = Some(Material.Glass)))\n"
    "  )\n"
)


# --- Acceptance: clean, in-vocabulary scene -----------------------------------------------


def test_clean_scene_with_only_allowed_imports_has_no_findings():
    findings = check_allowlist(CLEAN_SCENE)

    assert findings == []


def test_scala_math_import_is_allowed():
    scene = "import scala.math._\n\nobject Foo:\n  val x = math.Pi\n"

    findings = check_allowlist(scene)

    assert findings == []


def test_menger_common_import_is_allowed():
    scene = "import menger.common.SomeType\n\nobject Foo:\n  val scene = Scene()\n"

    findings = check_allowlist(scene)

    assert findings == []


# --- Edge-Case Matrix: out-of-allowlist import ---------------------------------------------


def test_disallowed_import_is_flagged_and_names_the_offending_import():
    scene = "import java.io.File\n\nobject Foo:\n  val scene = Scene()\n"

    findings = check_allowlist(scene)

    assert len(findings) == 1
    assert findings[0].stage == "allowlist"
    assert findings[0].identifier == "java.io.File"
    assert "java.io.File" in findings[0].message


def test_disallowed_import_finding_names_the_correct_line():
    scene = "package examples.dsl\n\nimport java.io.File\n\nobject Foo:\n  val scene = Scene()\n"

    findings = check_allowlist(scene)

    assert findings[0].line == 3


def test_wildcard_import_of_a_disallowed_root_is_flagged():
    scene = "import java.io._\n\nobject Foo:\n  val scene = Scene()\n"

    findings = check_allowlist(scene)

    assert len(findings) == 1
    assert findings[0].identifier == "java.io._"


def test_grouped_import_of_a_disallowed_root_is_flagged():
    scene = "import java.util.{List, ArrayList}\n\nobject Foo:\n  val scene = Scene()\n"

    findings = check_allowlist(scene)

    assert len(findings) == 1


def test_fully_qualified_inline_reference_bypassing_wildcard_import_is_flagged():
    scene = (
        "import menger.dsl._\n\n"
        "object Foo:\n"
        "  val f = java.io.File(\"x\")\n"
    )

    findings = check_allowlist(scene)

    assert any(f.identifier == "java.io.File" for f in findings)


def test_unqualified_dsl_identifier_is_never_flagged_no_symbol_table():
    # `Sphere(...)` is a bare, unqualified identifier -- not syntactically resolvable to a
    # root package without a symbol table, so it must never be flagged (module docstring).
    scene = "import menger.dsl._\n\nobject Foo:\n  val scene = Scene(objects = List(Sphere()))\n"

    findings = check_allowlist(scene)

    assert findings == []


def test_reference_inside_a_line_comment_is_not_flagged():
    scene = (
        "import menger.dsl._\n\n"
        "object Foo:\n"
        "  // see java.io.File for inspiration\n"
        "  val scene = Scene()\n"
    )

    findings = check_allowlist(scene)

    assert findings == []


def test_duplicate_disallowed_qualified_reference_is_reported_only_once():
    scene = (
        "import menger.dsl._\n\n"
        "object Foo:\n"
        "  val a = java.io.File(\"x\")\n"
        "  val b = java.io.File(\"y\")\n"
    )

    findings = check_allowlist(scene)

    assert len(findings) == 1


# --- Edge-Case Matrix: unparseable/garbage input --------------------------------------------


def test_garbage_input_never_raises_and_returns_a_list():
    findings = check_allowlist("not even remotely valid scala {{{ ]][[")

    assert findings == []


def test_empty_input_never_raises():
    findings = check_allowlist("")

    assert findings == []


# --- Verification: story 2's real, already-compiling poc-run fixtures -----------------------
#
# Both fixtures carry `import menger.Projection4DSpec` (required for any object's
# `projection: Option[menger.Projection4DSpec]` field, reference/dsl-manifest.json). This was
# a real gap in AD-4's original allowed-roots list, found by this check's first real-fixture
# run -- resolved by the maintainer adding `menger.Projection4DSpec` specifically (not bare
# `menger`) to AD-4's exceptions, so the execution surface AD-2 forbids stays closed. See
# ARCHITECTURE-SPINE.md AD-4 and this story's Spec Change Log.


def test_poc_run_turn1_generated_scala_passes_cleanly():
    scene_text = open("poc-run/turn1-generated.scala").read()

    findings = check_allowlist(scene_text)

    assert findings == []


def test_poc_run_turn2_refined_scala_passes_cleanly():
    scene_text = open("poc-run/turn2-refined.scala").read()

    findings = check_allowlist(scene_text)

    assert findings == []


# --- Verification: the full 32-scene reference corpus (review round 1) --------------------
#
# Run against every real corpus scene, not just the two poc-run fixtures, this check found
# a real false positive: `ReusableComponents.scala` uses `import examples.dsl.common.
# Lighting._` / `Materials._`, valid only inside the renderer's own example-source tree.
# That import would not resolve for a standalone generated scene either -- this is a
# correct rejection, not a bug, and `core/generation.py`'s system prompt now warns the
# model against imitating this one scene's cross-file-import pattern (see
# test_generation.py). All other 31 corpus scenes pass cleanly.


def test_full_corpus_passes_except_the_documented_cross_import_case():
    import json

    corpus = json.load(open("reference/dsl-corpus.json"))
    scenes = corpus["scenes"]
    assert len(scenes) == 32

    flagged = {s["name"]: check_allowlist(s["source"]) for s in scenes}
    flagged = {name: f for name, f in flagged.items() if f}

    assert set(flagged) == {"ReusableComponents"}
    identifiers = {f.identifier for f in flagged["ReusableComponents"]}
    assert identifiers == {"examples.dsl.common.Lighting._", "examples.dsl.common.Materials._"}

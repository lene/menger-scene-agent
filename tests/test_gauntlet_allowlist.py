"""Unit tests for gauntlet/allowlist.py -- AD-4 rule 1 static allowlist check. Pure string
processing, no compiler, no network."""

from __future__ import annotations

import pytest

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


# --- Verification: the full reference corpus ----------------------------------------------
#
# Every corpus scene must pass the allowlist: the corpus is what the model imitates. Until
# review 2026-10-14 (menger#115) `ReusableComponents.scala` imported `examples.dsl.common.*`,
# which resolves only inside menger's own example tree, and was the one documented rejection;
# menger now ships it self-contained and pins that in CorpusExporterSuite.


def test_full_corpus_passes_the_allowlist():
    import json

    corpus = json.load(open("reference/dsl-corpus.json"))
    scenes = corpus["scenes"]
    assert len(scenes) == 34  # + CubeToTesseract (usability review 2026-10, F124)

    flagged = {s["name"]: check_allowlist(s["source"]) for s in scenes}
    assert {name: f for name, f in flagged.items() if f} == {}


# --- Review 2026-10-14, menger-scene-agent#36: Scala 3 imports and implicit names -----------

HOSTILE_SNIPPETS = [
    "import scala.sys.process.*",
    "import scala.sys.process.{given, *}",
    "import java.lang.Runtime as R",
    "import menger.dsl.*, java.io.File",
    "import _root_.java.net.URL",
    "import scala.sys.process.given",
    "import scala.sys.process.`*`",
    "val p = scala.sys.process.stringToProcess(\"ls\")",
    "val r = _root_.java.lang.Runtime.getRuntime",
    "val e = System.getenv(\"HOME\")",
    "val r = Runtime.getRuntime",
    "val p = ProcessBuilder(\"ls\")",
    "val c = Class.forName(\"x\")",
    "sys.exit(1)",
    "val t = Thread(() => ())",
    "val l = getClass.getClassLoader",
]


def test_every_hostile_snippet_is_flagged():
    for snippet in HOSTILE_SNIPPETS:
        scene = f"import menger.dsl._\nobject Foo:\n  {snippet}\n"
        assert check_allowlist(scene), snippet


def test_scala3_wildcard_and_rename_of_allowed_roots_pass():
    scene = (
        "import menger.dsl.*\n"
        "import menger.common.{Color as C, *}\n"
        "import scala.math.Pi as P\n"
        "object Foo:\n"
        "  val scene = Scene(objects = List(Sphere(size = math.sin(P).toFloat)))\n"
    )
    assert check_allowlist(scene) == []


def test_member_access_on_local_values_is_not_a_package_reference():
    scene = "import menger.dsl._\nobject Foo:\n  val io = 1\n  val x = scene.objects.size + io\n"
    assert check_allowlist(scene) == []


# --- menger#78: references hidden by string splices, spacing or member calls -------------

HIDDEN_REFERENCES = {
    "interpolated splice": 'val k = s"${System.getenv("KEY")}"',
    "brace inside splice string": 'val k = s"${ "}" + System.getenv("KEY") }"',
    "splice inside splice": 'val k = s"${ s"${System.getenv("KEY")}" }"',
    "triple-quoted splice": 'val k = s"""${System.getenv("KEY")}"""',
    "member getClass": "val l = this.getClass.getClassLoader",
    "spaced dots": "val f = java . nio . file . Files . readString(null)",
    "comment between segments": "val f = java/**/.nio.file.Files.readString(null)",
    "dot on the next line": "val f = java\n    .nio.file.Files.readString(null)",
}


@pytest.mark.parametrize("snippet", HIDDEN_REFERENCES.values(), ids=HIDDEN_REFERENCES.keys())
def test_reference_hidden_by_syntax_is_flagged(snippet):
    scene = f"import menger.dsl._\nobject Foo:\n  {snippet}\n"
    assert check_allowlist(scene), snippet


def test_spaced_reference_is_reported_without_its_spaces():
    scene = "import menger.dsl._\nobject Foo:\n  val f = java . io . File(null)\n"
    assert [f.identifier for f in check_allowlist(scene)] == ["java.io.File"]


def test_splice_finding_names_the_splice_line():
    scene = 'import menger.dsl._\nobject Foo:\n  val a = 1\n  val k = s"${System.getenv("K")}"\n'
    assert [f.line for f in check_allowlist(scene)] == [4]


def test_plain_splices_and_strings_naming_dangerous_things_pass():
    scene = (
        "import menger.dsl._\n"
        "object Foo:\n"
        "  val i = 3\n"
        '  val label = s"frame ${i + 1} of $i, see System.getenv"\n'
        '  val note = "java.io.File and this.getClass"\n'
    )
    assert check_allowlist(scene) == []

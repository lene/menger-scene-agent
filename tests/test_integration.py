"""One integration test: a real Anthropic API round-trip using mvp-acceptance.md's exact
Turn 1 prompt. Skipped cleanly (not failed) when ANTHROPIC_API_KEY is unset. This is a smoke
test that the real adapter round-trips -- NOT a compile/load check (AD-2 forbids compiling or
loading a generated scene in the agent domain; that's story 5's job). It only asserts the
output looks like Scala scene text with a top-level `object` in the first 60 lines."""

from __future__ import annotations

import difflib
import os

import pytest

from adapters.artifacts import load_corpus, load_manifest
from adapters.model import AnthropicModelAdapter
from adapters.scene_store import SceneStore
from core.generation import generate, revise
from core.readback import semantic_readback
from core.types import GenerationError, ReadbackError

TURN_1_PROMPT = (
    "a tesseract sponge that gets more intricate as it turns, about ten seconds, "
    "glass, dark background"
)

# mvp-acceptance.md's Turn 2 prompt -- the concrete CAP-3 refinement request.
TURN_2_PROMPT = "slower, and start it already at level 1"

# A distinctive, unambiguous marker unrelated to "make it darker" -- CAP-2's bar is that a
# revise() request only touches what it names; this camera position should survive verbatim.
_MARKER_CAMERA_Z = "7.319f"
_PRIOR_SCENE_WITH_MARKER = f"""package poc

import menger.dsl._

object MarkerScene:
  val scene = Scene(
    camera = Camera(position = Vec3(0f, 2f, {_MARKER_CAMERA_Z}), lookAt = Vec3(0f, 0f, 0f)),
    objects = List(Sphere(material = Some(Material.Gold))),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 1.5f)),
    background = Some(Color(0.5f, 0.5f, 0.5f))
  )
"""

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- skipping the real Anthropic API integration test",
)


def test_generate_turn_1_prompt_against_the_real_anthropic_api():
    manifest = load_manifest("reference/dsl-manifest.json")
    corpus = load_corpus("reference/dsl-corpus.json")
    adapter = AnthropicModelAdapter()

    result = generate(TURN_1_PROMPT, manifest, corpus, adapter)

    assert not isinstance(result, GenerationError), (
        result.message if isinstance(result, GenerationError) else ""
    )
    assert isinstance(result, str)
    assert result.strip() != ""

    first_60_lines = result.splitlines()[:60]
    assert any(line.strip().startswith("object ") for line in first_60_lines), (
        "Expected a top-level 'object' declaration within the first 60 lines "
        "(SceneLoader.detectObjectName's heuristic); got:\n" + result
    )


def test_revise_preserves_an_unrelated_marker_against_the_real_anthropic_api():
    """CAP-2's actual bar -- content the request doesn't implicate survives -- can't be
    unit-tested against a fake adapter (the fake's output is scripted, not derived). This is
    the only place in the story that checks it against a real model."""
    manifest = load_manifest("reference/dsl-manifest.json")
    corpus = load_corpus("reference/dsl-corpus.json")
    adapter = AnthropicModelAdapter()

    result = revise(
        "make it darker", _PRIOR_SCENE_WITH_MARKER, manifest, corpus, adapter
    )

    assert not isinstance(result, GenerationError), (
        result.message if isinstance(result, GenerationError) else ""
    )
    assert isinstance(result, str)
    assert _MARKER_CAMERA_Z in result, (
        "Expected the unrelated camera-position marker to survive a 'make it darker' "
        "revision untouched; got:\n" + result
    )


def test_semantic_readback_against_the_real_anthropic_api():
    """Story 6: a real round-trip of `semantic_readback()` against the harness spike's own
    generated scene (`poc-run/turn1-generated.scala`, story 2 output for the Turn 1 prompt).
    Not over-asserting on exact wording since this is a real model call -- only that the
    response is non-trivial prose and plausibly names something from the scene it describes
    (tesseract/sponge/glass), not an exact-string match."""
    with open("poc-run/turn1-generated.scala", encoding="utf-8") as f:
        scene_text = f.read()
    adapter = AnthropicModelAdapter()

    result = semantic_readback(scene_text, adapter)

    assert not isinstance(result, ReadbackError), (
        result.message if isinstance(result, ReadbackError) else ""
    )
    assert isinstance(result, str)
    stripped = result.strip()
    assert stripped != ""
    # A real, plausible readback of this scene is well short of a paragraph, but long enough
    # to name object/material/camera -- not a one-word non-answer.
    assert len(stripped) >= 20
    lowered = stripped.lower()
    assert any(
        keyword in lowered for keyword in ("tesseract", "sponge", "glass")
    ), f"Expected the readback to name something recognizable from the scene; got:\n{result}"


def test_turn_1_then_turn_2_through_a_real_session_diff_touches_only_timing_and_level(tmp_path):
    """Story 7's own bar: run mvp-acceptance.md's exact Turn 1 -> Turn 2 flow through a real
    `SceneStore` session and the real Anthropic API, then diff the two real on-disk files.
    Mirrors `test_revise_preserves_an_unrelated_marker_against_the_real_anthropic_api`'s
    pattern, but does a real `difflib` diff between real files instead of a marker-substring
    check -- the store makes real files available to diff, which that earlier test's
    hand-built fixture never had.

    Turn 2's `prior_scene` is read back from `001.scala` on disk via the store, never the
    in-memory string `generate()` returned in this same process (Boundaries & Constraints) --
    proving the store's read-back path actually works, not just its write path."""
    manifest = load_manifest("reference/dsl-manifest.json")
    corpus = load_corpus("reference/dsl-corpus.json")
    adapter = AnthropicModelAdapter()
    store = SceneStore.create_session(tmp_path, slug="mvp-acceptance")

    turn_1_result = generate(TURN_1_PROMPT, manifest, corpus, adapter)
    assert not isinstance(turn_1_result, GenerationError), (
        turn_1_result.message if isinstance(turn_1_result, GenerationError) else ""
    )
    ordinal_1 = store.accept(turn_1_result, TURN_1_PROMPT)
    assert ordinal_1 == 1

    # Read back from disk -- not the `turn_1_result` string above -- so Turn 2 is derived
    # from the store's real read-back path, exactly as a real second turn would be.
    prior_scene = store.read_ordinal(1)

    turn_2_result = revise(TURN_2_PROMPT, prior_scene, manifest, corpus, adapter)
    assert not isinstance(turn_2_result, GenerationError), (
        turn_2_result.message if isinstance(turn_2_result, GenerationError) else ""
    )
    ordinal_2 = store.accept(turn_2_result, TURN_2_PROMPT)
    assert ordinal_2 == 2

    file_1_lines = store.read_ordinal(1).splitlines(keepends=True)
    file_2_lines = store.read_ordinal(2).splitlines(keepends=True)

    matcher = difflib.SequenceMatcher(a=file_1_lines, b=file_2_lines, autojunk=False)
    changed_lines = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_lines.extend(file_1_lines[i1:i2])
        changed_lines.extend(file_2_lines[j1:j2])

    diff_text = "".join(
        difflib.unified_diff(
            file_1_lines, file_2_lines, fromfile="001.scala", tofile="002.scala"
        )
    )

    assert changed_lines, f"Expected turn 2 to change something; diff was empty:\n{diff_text}"

    # mvp-acceptance.md's CAP-3 bar: camera, material, lights and background must be
    # byte-identical between the two turns -- only the timing expression and the level floor
    # may differ.
    forbidden_keywords = ("camera", "material", "directional", "light", "background")
    for line in changed_lines:
        lowered = line.lower()
        assert not any(keyword in lowered for keyword in forbidden_keywords), (
            "CAP-3 diff bar violated: camera/material/lights/background must be "
            f"byte-identical between turns, but a changed line mentions one of "
            f"{forbidden_keywords}:\n{line!r}\n\nFull diff:\n{diff_text}"
        )

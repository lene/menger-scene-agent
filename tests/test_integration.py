"""One integration test: a real Anthropic API round-trip using mvp-acceptance.md's exact
Turn 1 prompt. Skipped cleanly (not failed) when ANTHROPIC_API_KEY is unset. This is a smoke
test that the real adapter round-trips -- NOT a compile/load check (AD-2 forbids compiling or
loading a generated scene in the agent domain; that's story 5's job). It only asserts the
output looks like Scala scene text with a top-level `object` in the first 60 lines."""

from __future__ import annotations

import os

import pytest

from adapters.artifacts import load_corpus, load_manifest
from adapters.model import AnthropicModelAdapter
from core.generation import generate, revise
from core.readback import semantic_readback
from core.types import GenerationError, ReadbackError

TURN_1_PROMPT = (
    "a tesseract sponge that gets more intricate as it turns, about ten seconds, "
    "glass, dark background"
)

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

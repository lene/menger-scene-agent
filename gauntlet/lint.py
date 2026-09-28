"""Stage 2 scene-graph lint: the five checks `validation-gauntlet.md` names under "Must be
built -- Scene linter" -- object outside the camera frustum, light inside geometry, camera
inside an object, black-on-black material against the background, degenerate (near-zero)
scale. "These are the failures that produce a black or empty frame and no explanation."

Every check operates ONLY on literal numeric values extracted from the relevant DSL
constructor calls (`Camera(position = Vec3(...), lookAt = Vec3(...))`, an object's
`pos =`/`size =`/`color =` fields, `Color(...)` values, `background = Some(Color(...))`).
When a relevant value is absent, it takes the DSL's own documented default
(`reference/dsl-manifest.json`: `pos` defaults to `Vec3(0,0,0)`, `size` to `1.0`) -- when it
is present but a non-literal expression, this module cannot reason about it at all (AD-2:
no compiler, no evaluation), so that specific check is skipped for that specific
object/light, never guessed or failed.

Each heuristic is intentionally simple and its threshold documented inline -- proportionate
to a static pass over source text, not a substitute for stage 4's real geometric checks
(story 5, renderer domain) which can actually evaluate the renderer's math.

Usability review 2026-09 (Phase 3b): the extraction (`_extract_objects` et al.) that used to
live here was promoted to `core/scene_facts.py`, shared with the F7/F8/F20/F23 follow-ups.
This module is now a thin caller: `extract_scene_facts` for parsing, its own five checks for
the actual lint rules. Behavior is unchanged; this file's own tests are the safety net.
"""

from __future__ import annotations

import math

from core.scene_facts import ObjectFact, extract_scene_facts
from gauntlet._scala_text import line_of, strip_comments_and_strings
from gauntlet.types import Finding

_STAGE = "lint"

# Heuristic thresholds, chosen to be simple and defensible over a static text pass:
#  - degenerate scale: a `size` literal below this is visually indistinguishable from a
#    point, well below any of the corpus's real scenes (which use size >= ~0.5).
_DEGENERATE_SCALE_EPSILON = 1e-3
#  - black-on-black: an RGB component below this reads as "black" on any display.
_NEAR_BLACK_EPSILON = 0.05
#  - camera/light "inside" an object: distance from the point to the object's center is
#    less than the object's `size`. `size` is used directly as a conservative bounding
#    radius (rather than attempting a per-object-type exact bounding-volume formula) --
#    proportionate to a static heuristic pass, and conservative in the sense that it can
#    under-flag a thin/elongated object more readily than it over-flags a compact one.
_INSIDE_DISTANCE_THRESHOLD_IS_SIZE = True  # documents the rule above; not a tunable knob


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _check_camera_inside_object(camera, objects: list[ObjectFact], text) -> list[Finding]:
    findings = []
    if camera is None:
        return findings
    position, _ = camera
    for obj in objects:
        if obj.pos is None or obj.size is None:
            continue  # non-literal pos/size -- can't reason about this object
        if _distance(position, obj.pos) < obj.size:
            findings.append(
                Finding(
                    stage=_STAGE,
                    message=(
                        f"Camera position falls inside {obj.type_name}'s bounding volume "
                        f"(distance < size={obj.size})"
                    ),
                    field="camera_inside_object",
                    identifier=obj.type_name,
                    line=line_of(text, obj.offset),
                )
            )
    return findings


def _check_light_inside_geometry(lights, objects, text) -> list[Finding]:
    findings = []
    for light in lights:
        if light.position is None:
            continue  # absent or non-literal position -- can't reason about this light
        for obj in objects:
            if obj.pos is None or obj.size is None:
                continue
            if _distance(light.position, obj.pos) < obj.size:
                findings.append(
                    Finding(
                        stage=_STAGE,
                        message=(
                            f"{light.type_name} light position falls inside "
                            f"{obj.type_name}'s bounding volume (distance < size={obj.size})"
                        ),
                        field="light_inside_geometry",
                        identifier=light.type_name,
                        line=line_of(text, light.offset),
                    )
                )
    return findings


def _check_frustum(camera, objects, text) -> list[Finding]:
    findings = []
    if camera is None:
        return findings
    position, look_at = camera
    view_dir = tuple(la - p for la, p in zip(look_at, position))
    view_len = math.sqrt(sum(c * c for c in view_dir))
    if view_len == 0:
        return findings  # camera position == lookAt -- no well-defined view direction
    for obj in objects:
        if obj.pos is None:
            continue
        to_object = tuple(o - p for o, p in zip(obj.pos, position))
        dot = sum(a * b for a, b in zip(view_dir, to_object))
        if dot <= 0:
            findings.append(
                Finding(
                    stage=_STAGE,
                    message=(
                        f"{obj.type_name} lies behind or level with the camera's view "
                        f"direction -- outside the frustum"
                    ),
                    field="frustum",
                    identifier=obj.type_name,
                    line=line_of(text, obj.offset),
                )
            )
    return findings


def _check_black_on_black(objects, background, text) -> list[Finding]:
    findings = []
    if background is None or not all(c < _NEAR_BLACK_EPSILON for c in background):
        return findings
    for obj in objects:
        if obj.color is None:
            continue  # no explicit literal color -- material preset supplies appearance
        if all(c < _NEAR_BLACK_EPSILON for c in obj.color):
            findings.append(
                Finding(
                    stage=_STAGE,
                    message=(
                        f"{obj.type_name}'s color and the background are both near-black "
                        f"-- invisible against the background"
                    ),
                    field="black_on_black",
                    identifier=obj.type_name,
                    line=line_of(text, obj.offset),
                )
            )
    return findings


def _check_degenerate_scale(objects, text) -> list[Finding]:
    findings = []
    for obj in objects:
        if obj.size is None:
            continue
        if obj.size < _DEGENERATE_SCALE_EPSILON:
            findings.append(
                Finding(
                    stage=_STAGE,
                    message=f"{obj.type_name}'s size={obj.size} is degenerate (near-zero scale)",
                    field="degenerate_scale",
                    identifier=obj.type_name,
                    line=line_of(text, obj.offset),
                )
            )
    return findings


def check_lint(scene_text: str) -> list[Finding]:
    """The five stage-2 lint checks from `validation-gauntlet.md`, each evaluated only
    where the relevant DSL call uses literal values -- a `t`-driven expression is skipped
    for that specific check, never failed or guessed at.

    Runs against comment/string-stripped text (review round 1): a stray `(`/`)` inside a
    string argument (e.g. `texture = "shape (v2).png"`) previously desynced
    `scan_balanced`'s paren-depth count and silently corrupted parsing of that call and
    everything after it. Line numbers passed to `Finding` still use the offsets from this
    stripped text, which are aligned with the original since stripping only blanks
    content, never removes a newline."""
    text = strip_comments_and_strings(scene_text)
    facts = extract_scene_facts(text)
    camera, objects, lights, background = facts.camera, facts.objects, facts.lights, facts.background

    findings: list[Finding] = []
    findings.extend(_check_frustum(camera, objects, text))
    findings.extend(_check_light_inside_geometry(lights, objects, text))
    findings.extend(_check_camera_inside_object(camera, objects, text))
    findings.extend(_check_black_on_black(objects, background, text))
    findings.extend(_check_degenerate_scale(objects, text))
    return findings

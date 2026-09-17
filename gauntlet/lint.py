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
"""

from __future__ import annotations

import math
import re
from typing import Optional

from gauntlet._scala_text import (
    as_float,
    find_call_bodies,
    line_of,
    named_args,
    scan_balanced,
    split_top_level,
    strip_comments_and_strings,
)
from gauntlet.types import Finding

_STAGE = "lint"

# The DSL's own documented defaults (reference/dsl-manifest.json: every object's `pos`
# field defaults to `Vec3(0.0,0.0,0.0)`, `size` to `1.0`) -- used only when the field is
# absent from a constructor call, never when it's present but non-literal.
_DEFAULT_POS = (0.0, 0.0, 0.0)
_DEFAULT_SIZE = 1.0

_OBJECT_TYPES = (
    "Sphere",
    "Cube",
    "Sponge",
    "Tesseract",
    "TesseractSponge",
    "Sierpinski4D",
    "ParametricSurface",
    "Curve",
    "LSystem",
)
# Lights with a `position` field can meaningfully be "inside" geometry; `Directional` has
# only a `direction` (no location in space), so it is never checked for this.
_POSITIONED_LIGHT_TYPES = ("Point", "AreaLight")

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


def _parse_vec3(value: str) -> Optional[tuple[float, float, float]]:
    value = value.strip()
    match = re.match(r"^Vec3\((.*)\)$", value, re.DOTALL)
    if not match:
        return None
    parts = split_top_level(match.group(1))
    if len(parts) != 3:
        return None
    nums = [as_float(p) for p in parts]
    if any(n is None for n in nums):
        return None
    return (nums[0], nums[1], nums[2])


def _parse_color_rgb(value: str) -> Optional[tuple[float, float, float]]:
    value = value.strip()
    match = re.match(r"^Color\((.*)\)$", value, re.DOTALL)
    if not match:
        return None
    parts = split_top_level(match.group(1))
    if len(parts) < 3:
        return None
    nums = [as_float(p) for p in parts[:3]]
    if any(n is None for n in nums):
        return None
    return (nums[0], nums[1], nums[2])


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class _Object:
    def __init__(self, type_name: str, offset: int, args: dict[str, str]):
        self.type_name = type_name
        self.offset = offset
        self.pos = _parse_vec3(args["pos"]) if "pos" in args else _DEFAULT_POS
        self.size = as_float(args["size"]) if "size" in args else _DEFAULT_SIZE
        color_arg = args.get("color")
        self.color = _parse_color_literal_option(color_arg) if color_arg is not None else None


class _Light:
    def __init__(self, type_name: str, offset: int, args: dict[str, str]):
        self.type_name = type_name
        self.offset = offset
        position_arg = args.get("position")
        self.position = _parse_vec3(position_arg) if position_arg is not None else None


def _parse_color_literal_option(value: str) -> Optional[tuple[float, float, float]]:
    """Parses an object's `color` field, which is `Option[Color]` -- typically written as
    `Some(Color(r, g, b[, a]))`. Returns `None` when absent or non-literal (no explicit
    `Some(Color(...))`, e.g. a material preset supplies appearance instead)."""
    stripped = value.strip()
    match = re.match(r"^Some\((.*)\)$", stripped, re.DOTALL)
    inner = match.group(1) if match else stripped
    return _parse_color_rgb(inner)


def _extract_objects(text: str) -> list[_Object]:
    objects = []
    for type_name in _OBJECT_TYPES:
        for offset, inner in find_call_bodies(text, type_name):
            objects.append(_Object(type_name, offset, named_args(inner)))
    return objects


def _extract_lights(text: str) -> list[_Light]:
    lights = []
    for type_name in _POSITIONED_LIGHT_TYPES:
        for offset, inner in find_call_bodies(text, type_name):
            lights.append(_Light(type_name, offset, named_args(inner)))
    return lights


def _extract_camera(text: str) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    calls = find_call_bodies(text, "Camera")
    if not calls:
        return None
    _, inner = calls[0]  # a scene has exactly one camera; the first call is authoritative
    args = named_args(inner)
    if "position" not in args or "lookAt" not in args:
        return None
    position = _parse_vec3(args["position"])
    look_at = _parse_vec3(args["lookAt"])
    if position is None or look_at is None:
        return None
    return position, look_at


def _extract_background(text: str) -> Optional[tuple[float, float, float]]:
    match = re.search(r"\bbackground\s*=\s*Some\(", text)
    if not match:
        return None
    open_idx = match.end() - 1
    close_idx = scan_balanced(text, open_idx)
    return _parse_color_rgb(text[open_idx + 1 : close_idx])


def _check_camera_inside_object(camera, objects, text) -> list[Finding]:
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
    camera = _extract_camera(text)
    objects = _extract_objects(text)
    lights = _extract_lights(text)
    background = _extract_background(text)

    findings: list[Finding] = []
    findings.extend(_check_frustum(camera, objects, text))
    findings.extend(_check_light_inside_geometry(lights, objects, text))
    findings.extend(_check_camera_inside_object(camera, objects, text))
    findings.extend(_check_black_on_black(objects, background, text))
    findings.extend(_check_degenerate_scale(objects, text))
    return findings

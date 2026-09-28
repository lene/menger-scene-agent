"""Ground-truth facts extracted directly from a scene's DSL source text: pure string
processing, no compiler, no evaluation (AD-2) -- the same discipline `gauntlet/lint.py`'s
stage-2 checks already followed. Promoted out of `lint.py` (usability review 2026-09) because
four different consumers all need the same underlying facts and were at risk of drifting into
four slightly different re-derivations of "where is this object, what does it look like":

- `gauntlet/lint.py`'s stage-2 checks (frustum, light-inside-geometry, camera-inside-object,
  black-on-black, degenerate-scale) -- unchanged behavior, now a thin caller of this module.
- F7 (`core/readback.py`): a `<facts>` block as ground truth for the readback prompt.
- F8 (`core/turn.py`): before/after diffing to warn about a moved object the user didn't ask
  to move.
- F23 (`gauntlet/*`, occlusion warning): is one object's bounding sphere inside another's.
- F20 (prompt grounding for `generate()`/`revise()`): the scene's extent, for "zoom to fit".

When a relevant value is absent, a fact takes the DSL's own documented default (`pos` ->
`(0,0,0)`, `size` -> `1.0`, per `reference/dsl-manifest.json`); when it is present but a
non-literal expression (a `t`-driven animation), this module cannot reason about it at all,
so that fact is `None` -- never guessed, never failed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from gauntlet._scala_text import as_float, find_call_bodies, named_args, scan_balanced, split_top_level

Vec3 = tuple[float, float, float]

OBJECT_TYPES = (
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
POSITIONED_LIGHT_TYPES = ("Point", "AreaLight")
DEFAULT_POS: Vec3 = (0.0, 0.0, 0.0)
DEFAULT_SIZE = 1.0

# Material presets whose defining characteristic is transparency (the DSL's 12 presets, per
# reference/dsl-manifest.json's `materials` list) -- used only for the coarse "is this object
# opaque enough to occlude something behind it" judgment (F23); not a substitute for the
# renderer's own physically-based transmission.
_TRANSPARENT_MATERIAL_PRESETS = frozenset({
    "Glass", "Water", "Diamond", "GlassDispersive", "DiamondDispersive", "Film",
})
_TRANSPARENT_MATERIAL_OPACITY_HINT = 0.3


def parse_vec3(value: str) -> Optional[Vec3]:
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


def parse_color_rgb(value: str) -> Optional[Vec3]:
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


def _parse_color_alpha(value: str) -> Optional[float]:
    """The 4th `Color(r, g, b, a)` component, or `None` when absent/non-literal (the DSL's
    own default alpha is 1.0, applied by the caller, not here -- this only reports what's
    textually present)."""
    value = value.strip()
    match = re.match(r"^Color\((.*)\)$", value, re.DOTALL)
    if not match:
        return None
    parts = split_top_level(match.group(1))
    if len(parts) < 4:
        return None
    return as_float(parts[3])


def _parse_color_literal_option(value: str) -> Optional[Vec3]:
    """Parses an object's `color` field, `Option[Color]`, typically `Some(Color(r, g, b[,
    a]))`. `None` when absent or non-literal (no explicit `Some(Color(...))`, e.g. a
    material preset supplies appearance instead)."""
    stripped = value.strip()
    match = re.match(r"^Some\((.*)\)$", stripped, re.DOTALL)
    inner = match.group(1) if match else stripped
    return parse_color_rgb(inner)


def _material_name(value: str) -> Optional[str]:
    """`material = Some(Material.Glass)` (or a bare `Material.Glass`) -> `"Glass"`."""
    match = re.search(r"\bMaterial\.(\w+)", value)
    return match.group(1) if match else None


def _distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class ObjectFact:
    type_name: str
    offset: int
    pos: Optional[Vec3] = DEFAULT_POS
    size: Optional[float] = DEFAULT_SIZE
    color: Optional[Vec3] = None
    rotation: Optional[Vec3] = None
    material: Optional[str] = None
    # Coarse opacity in [0, 1] used for the occlusion check (F23) only -- see module
    # docstring's `_TRANSPARENT_MATERIAL_PRESETS` note. `None` when it can't be determined
    # (non-literal color and no material).
    opacity: Optional[float] = None

    @property
    def is_opaque(self) -> bool:
        """Conservative: an indeterminate opacity is treated as opaque, since F23 exists to
        warn about occlusion, and a false "might be hidden" is far cheaper than a missed one."""
        return self.opacity is None or self.opacity >= 0.9


@dataclass
class LightFact:
    type_name: str
    offset: int
    position: Optional[Vec3] = None
    direction: Optional[Vec3] = None

    @property
    def direction_phrase(self) -> Optional[str]:
        """A short natural-language phrase for a `Directional` light's travel direction
        (F7): "(0,-1,0)" means nothing to a model without also stating the travel-direction
        convention every time; this says it once, in the fact itself. `None` when there is
        no direction (not a Directional light, or a non-literal expression)."""
        if self.direction is None:
            return None
        dx, dy, dz = self.direction
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length == 0:
            return None
        dx, dy, dz = dx / length, dy / length, dz / length
        parts = []
        if dy < -0.3:
            parts.append("from above")
        elif dy > 0.3:
            parts.append("from below")
        horizontal = []
        if abs(dx) > 0.3:
            horizontal.append(f"toward {'+x' if dx > 0 else '-x'}")
        if abs(dz) > 0.3:
            horizontal.append(f"toward {'+z' if dz > 0 else '-z'}")
        if horizontal:
            parts.append(" and ".join(horizontal))
        return ", travelling ".join(parts) if parts else "travelling straight along its axis"


@dataclass
class SceneFacts:
    camera: Optional[tuple[Vec3, Vec3]]  # (position, lookAt)
    objects: list[ObjectFact] = field(default_factory=list)
    lights: list[LightFact] = field(default_factory=list)
    background: Optional[Vec3] = None

    @property
    def extent_center(self) -> Optional[Vec3]:
        """Centroid of every object with a literal `pos` -- `None` if the scene has no such
        object (F20: nothing to frame)."""
        positioned = [o.pos for o in self.objects if o.pos is not None]
        if not positioned:
            return None
        n = len(positioned)
        return (
            sum(p[0] for p in positioned) / n,
            sum(p[1] for p in positioned) / n,
            sum(p[2] for p in positioned) / n,
        )

    @property
    def extent_radius(self) -> Optional[float]:
        """Radius of the bounding sphere, centered at `extent_center`, that contains every
        object's own bounding sphere (`pos`, radius `size`) -- an approximation (not the
        exact minimal enclosing sphere), proportionate to a "does this roughly fit in frame"
        judgment (F20), not a collision/culling primitive. `None` when `extent_center` is."""
        center = self.extent_center
        if center is None:
            return None
        return max(
            (_distance(center, o.pos) + (o.size or 0.0) for o in self.objects if o.pos is not None),
            default=0.0,
        )


def _extract_objects(text: str) -> list[ObjectFact]:
    objects = []
    for type_name in OBJECT_TYPES:
        for offset, inner in find_call_bodies(text, type_name):
            args = named_args(inner)
            pos = parse_vec3(args["pos"]) if "pos" in args else DEFAULT_POS
            size = as_float(args["size"]) if "size" in args else DEFAULT_SIZE
            color_arg = args.get("color")
            color = _parse_color_literal_option(color_arg) if color_arg is not None else None
            material = _material_name(args["material"]) if "material" in args else None
            opacity = None
            if material is not None:
                opacity = _TRANSPARENT_MATERIAL_OPACITY_HINT if material in _TRANSPARENT_MATERIAL_PRESETS else 1.0
            elif color_arg is not None:
                alpha = _parse_color_alpha(
                    re.match(r"^Some\((.*)\)$", color_arg.strip(), re.DOTALL).group(1)
                    if re.match(r"^Some\(", color_arg.strip())
                    else color_arg
                )
                opacity = alpha if alpha is not None else 1.0
            rotation = parse_vec3(args["rotation"]) if "rotation" in args else None
            objects.append(ObjectFact(type_name, offset, pos, size, color, rotation, material, opacity))
    return objects


def _extract_lights(text: str) -> list[LightFact]:
    lights = []
    for type_name in POSITIONED_LIGHT_TYPES:
        for offset, inner in find_call_bodies(text, type_name):
            args = named_args(inner)
            position_arg = args.get("position")
            position = parse_vec3(position_arg) if position_arg is not None else None
            lights.append(LightFact(type_name, offset, position=position))
    for offset, inner in find_call_bodies(text, "Directional"):
        args = named_args(inner)
        direction_arg = args.get("direction")
        direction = parse_vec3(direction_arg) if direction_arg is not None else None
        lights.append(LightFact("Directional", offset, direction=direction))
    return lights


def _extract_camera(text: str) -> Optional[tuple[Vec3, Vec3]]:
    calls = find_call_bodies(text, "Camera")
    if not calls:
        return None
    _, inner = calls[0]  # a scene has exactly one camera; the first call is authoritative
    args = named_args(inner)
    if "position" not in args or "lookAt" not in args:
        return None
    position = parse_vec3(args["position"])
    look_at = parse_vec3(args["lookAt"])
    if position is None or look_at is None:
        return None
    return position, look_at


def _extract_background(text: str) -> Optional[Vec3]:
    match = re.search(r"\bbackground\s*=\s*Some\(", text)
    if not match:
        return None
    open_idx = match.end() - 1
    close_idx = scan_balanced(text, open_idx)
    return parse_color_rgb(text[open_idx + 1 : close_idx])


def facts_diff(before: SceneFacts, after: SceneFacts) -> list[str]:
    """Coarse before/after diff (F8): flags a moved object/camera or a changed material that
    a revise() request may not have asked for. Objects are matched positionally -- same index,
    same `type_name`, in the two facts' `objects` lists -- since the DSL source carries no
    stable per-object identity to match by; a reordered or added/removed object among same-type
    siblings is simply not compared, same "coarse by design, good enough to tell the user
    something changed" spirit as `core.turn.removed_properties` (F29)."""
    warnings: list[str] = []
    if before.camera is not None and after.camera is not None and before.camera != after.camera:
        warnings.append(f"also moved the camera from {before.camera[0]} to {after.camera[0]}")
    for b, a in zip(before.objects, after.objects):
        if b.type_name != a.type_name:
            continue
        if b.pos is not None and a.pos is not None and b.pos != a.pos:
            warnings.append(f"also moved the {a.type_name} from {b.pos} to {a.pos}")
        if b.material != a.material:
            warnings.append(
                f"also changed the {a.type_name}'s material from "
                f"{b.material or 'none'} to {a.material or 'none'}"
            )
    return warnings


def extract_scene_facts(stripped_text: str) -> SceneFacts:
    """Extracts every fact this module knows how to find from `stripped_text` -- comment/
    string-stripped scene source (`gauntlet._scala_text.strip_comments_and_strings`; callers
    that haven't already stripped must do so first, same requirement `lint.py` already had)."""
    return SceneFacts(
        camera=_extract_camera(stripped_text),
        objects=_extract_objects(stripped_text),
        lights=_extract_lights(stripped_text),
        background=_extract_background(stripped_text),
    )

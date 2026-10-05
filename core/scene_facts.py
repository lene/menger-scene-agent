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

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from gauntlet._scala_text import as_float, find_call_bodies, named_args, scan_balanced, split_top_level
from gauntlet.manifest_bounds import subtype_of

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
    # The regular 4D polytopes (menger F17); missing here, session 2's 24-cell had no facts.
    "Pentachoron",
    "Hexadecachoron",
    "Icositetrachoron",
    "Hexacosichoron",
    "Hecatonicosachoron",
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
# Fractals with holes through them: an object inside one stays visible through the holes, so
# they never occlude (F78: the orb inside a tesseract sponge was visible).
_HOLED_TYPES = frozenset({"Sponge", "TesseractSponge", "Sierpinski4D"})


def parse_vec3(value: str) -> Optional[Vec3]:
    """`Vec3(x, y, z)` or the tuple `(x, y, z)` the DSL converts implicitly (F78: session 3's
    scenes only used tuples)."""
    value = value.strip()
    match = re.match(r"^(?:Vec3)?\((.*)\)$", value, re.DOTALL)
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


def _material_name(value: str, local_vals: dict[str, str]) -> Optional[str]:
    """`material = Some(Material.Glass)` (or a bare `Material.Glass`) -> `"Glass"`. A local
    `val` resolves to its definition: a preset copy (`Material.Glass.copy(...)`) keeps the
    preset's name, a `Material(...)` of its own is named by the `val` (F75: it was "none")."""
    match = re.search(r"\bMaterial\.(\w+)", value)
    if match:
        return match.group(1)
    match = re.match(r"^(?:Some\()?\s*(\w+)\s*\)?$", value.strip())
    if not match or match.group(1) not in local_vals:
        return None
    name = match.group(1)
    return _material_name(local_vals[name], {}) or name


def _material_opacity(material: str, value: str, local_vals: dict[str, str]) -> float:
    """Coarse opacity of a material (F23): the alpha of a `Color` its definition sets (the DSL
    defaults alpha to 1), else the preset hint."""
    name = re.sub(r"^Some\(|\)$", "", value.strip()).strip()
    definition = local_vals.get(name, value)
    color = re.search(r"\bColor\(", definition)
    if color:
        close = scan_balanced(definition, color.end() - 1)
        alpha = _parse_color_alpha(definition[color.start() : close + 1])
        return alpha if alpha is not None else 1.0
    return _TRANSPARENT_MATERIAL_OPACITY_HINT if material in _TRANSPARENT_MATERIAL_PRESETS else 1.0


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
    # Highest `level` the object reaches: a literal, or an animated expression evaluated at
    # t = 0 and t = duration (F36). `None` when absent or not evaluable.
    level: Optional[float] = None
    # Raw argument text, whitespace-normalized, for changes a literal parse can't see: a `pos`
    # built from expressions (F8 recurrence) and the 4D `projection` (F45).
    pos_text: Optional[str] = None
    projection_text: Optional[str] = None
    # The value of the `spongeType` argument (`RecursiveIAS`), which selects the manifest's
    # per-subtype level bounds (F59); `None` when absent or not an identifier.
    subtype: Optional[str] = None

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


_LOCAL_VAL_RE = re.compile(r"\bval\s+(\w+)\s*(?::\s*\w+)?\s*=\s*([^\n]+)")
_SCALA_NUMBER_RE = re.compile(r"(\d+(?:\.\d*)?|\.\d+)[fFdD]\b")
_MAX_VAL_DEPTH = 5


def _evaluate(expr: str, env: dict[str, str], t: float, depth: int = 0) -> Optional[float]:
    """Evaluates a Scala arithmetic expression: numbers, + - * /, parentheses, math.max/min/abs,
    `t`, and local `val`s from `env` (resolved recursively). Anything else -> `None`."""
    if depth > _MAX_VAL_DEPTH:
        return None
    try:
        tree = ast.parse(_SCALA_NUMBER_RE.sub(r"\1", expr.strip()).replace("math.", ""), mode="eval")
    except SyntaxError:
        return None
    functions = {"max": max, "min": min, "abs": abs}
    operators = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                 ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = ev(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
            return functions[node.func.id](*(ev(a) for a in node.args))
        if isinstance(node, ast.Name):
            if node.id == "t":
                return t
            if node.id in env:
                value = _evaluate(env[node.id], env, t, depth + 1)
                if value is not None:
                    return value
        raise ValueError(node)

    try:
        return ev(tree.body)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _level_upper_bound(expr: str, env: dict[str, str]) -> Optional[float]:
    """Highest value of a `level` expression over an animation: it is evaluated at t = 0 and
    t = duration (as the renderer's validator does), so a monotonic ramp is bounded exactly."""
    duration = _evaluate(env.get("duration", "1"), env, 0.0) or 1.0
    ends = [_evaluate(expr, env, 0.0), _evaluate(expr, env, duration)]
    return None if None in ends else max(ends)


def _local_vals(text: str) -> dict[str, str]:
    """Local `val`s by name: the rest of the line, extended to the closing parenthesis when a
    call spans lines (a multi-line `Material(...)`, F75)."""
    vals = {}
    for match in _LOCAL_VAL_RE.finditer(text):
        value = match.group(2)
        if value.count("(") > value.count(")"):
            start = match.start(2)
            open_idx = text.index("(", start)
            value = text[start : scan_balanced(text, open_idx) + 1]
        vals[match.group(1)] = value
    return vals


def _extract_objects(text: str) -> list[ObjectFact]:
    objects = []
    local_vals = _local_vals(text)
    for type_name in OBJECT_TYPES:
        for offset, inner in find_call_bodies(text, type_name):
            args = named_args(inner)
            pos = parse_vec3(args["pos"]) if "pos" in args else DEFAULT_POS
            size = as_float(args["size"]) if "size" in args else DEFAULT_SIZE
            color_arg = args.get("color")
            color = _parse_color_literal_option(color_arg) if color_arg is not None else None
            material = _material_name(args["material"], local_vals) if "material" in args else None
            opacity = None
            if material is not None:
                opacity = _material_opacity(material, args["material"], local_vals)
            elif color_arg is not None:
                alpha = _parse_color_alpha(
                    re.match(r"^Some\((.*)\)$", color_arg.strip(), re.DOTALL).group(1)
                    if re.match(r"^Some\(", color_arg.strip())
                    else color_arg
                )
                opacity = alpha if alpha is not None else 1.0
            rotation = parse_vec3(args["rotation"]) if "rotation" in args else None
            level = _level_upper_bound(args["level"], local_vals) if "level" in args else None
            objects.append(ObjectFact(
                type_name, offset, pos, size, color, rotation, material, opacity, level,
                pos_text=_normalized(args.get("pos")),
                projection_text=_normalized(args.get("projection")),
                subtype=subtype_of(args.get("spongeType")),
            ))
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


def _normalized(text: Optional[str]) -> Optional[str]:
    return " ".join(text.split()) if text is not None else None


# Words that show the request asked for a kind of change, so facts_diff doesn't report it back
# as a side effect (F39). Coarse on purpose: a false silence costs less than a warning that
# tells the user their own request was an unrequested change.
_ASKED_FOR = {
    "pos": ("move", "position", "place", "put", "shift", "above", "below", "left", "right",
            "next to", "beside", "closer", "further", "higher", "lower", "centre", "center"),
    "material": ("material", "made of", "made from", "glass", "gold", "chrome", "copper",
                 "metal", "aluminium", "aluminum", "silver", "steel", "wood", "stone", "marble",
                 "plastic", "matte", "diamond", "water", "film", "transparent", "opaque"),
    "camera": ("camera", "view", "zoom", "frame", "angle", "closer", "further"),
    "projection": ("rotate", "rotation", "turn", "spin", "tilt", "4d", "projection"),
}


def _asked_for(kind: str, prompt: str) -> bool:
    lowered = prompt.lower()
    return any(word in lowered for word in _ASKED_FOR[kind])


def facts_diff(before: SceneFacts, after: SceneFacts, prompt: str = "") -> list[str]:
    """Coarse before/after diff (F8): flags a moved object/camera, a changed material or 4D
    projection that a revise() request didn't ask for (`prompt`; F39: a change the request
    names is not reported back). Objects are matched by occurrence within their type -- the
    DSL source carries no stable per-object identity -- so an object added or removed of
    another type doesn't shift the comparison (F8 recurrence); same "coarse by design, good
    enough to tell the user something changed" spirit as `core.turn.removed_properties`
    (F29)."""
    warnings: list[str] = []
    if (before.camera is not None and after.camera is not None and before.camera != after.camera
            and not _asked_for("camera", prompt)):
        warnings.append(f"also moved the camera from {before.camera[0]} to {after.camera[0]}")
    for type_name in OBJECT_TYPES:
        pairs = zip(
            (o for o in before.objects if o.type_name == type_name),
            (o for o in after.objects if o.type_name == type_name),
        )
        for b, a in pairs:
            if not _asked_for("pos", prompt):
                if b.pos is not None and a.pos is not None:
                    if b.pos != a.pos:
                        warnings.append(f"also moved the {type_name} from {b.pos} to {a.pos}")
                elif b.pos_text != a.pos_text:
                    warnings.append(
                        f"also moved the {type_name} from {b.pos_text or 'the origin'} to "
                        f"{a.pos_text or 'the origin'}"
                    )
            if b.material != a.material and not _asked_for("material", prompt):
                warnings.append(
                    f"also changed the {type_name}'s material from "
                    f"{b.material or 'none'} to {a.material or 'none'}"
                )
            if b.projection_text != a.projection_text and not _asked_for("projection", prompt):
                warnings.append(
                    f"also changed the {type_name}'s 4D projection to "
                    f"{a.projection_text or 'the default'}"
                )
    return warnings


def occlusion_warnings(facts: SceneFacts) -> list[str]:
    """F23: an object whose entire bounding sphere fits inside another, opaque object's
    bounding sphere is never visible from any angle -- a size-aware extension of the
    "distance from center < size" heuristic `gauntlet/lint.py`'s
    `_check_light_inside_geometry` uses for a (point-like) light; checking full containment
    (`distance + obj.size < occluder.size`), not just "center within occluder.size", matters
    here because two same-sized or larger-inside-smaller objects sharing a center are not an
    occlusion (review round: an unsized-heuristic version flagged a size-5 Sponge as "hidden
    inside" a same-centered size-0.5 Sphere). Only counts an opaque occluder -- a transparent
    one, e.g. Glass, doesn't hide anything. Coarse and conservative like the rest of this
    module's heuristics: a compact, indeterminate-opacity occluder is treated as opaque
    (`ObjectFact.is_opaque`), so this can under-warn on a thin/elongated occluder more
    readily than it over-warns on a compact one."""
    warnings: list[str] = []
    for occluder in facts.objects:
        if occluder.pos is None or occluder.size is None or not occluder.is_opaque:
            continue
        if occluder.type_name in _HOLED_TYPES:
            continue
        for obj in facts.objects:
            if obj is occluder or obj.pos is None or obj.size is None:
                continue
            if _distance(obj.pos, occluder.pos) + obj.size < occluder.size:
                warnings.append(
                    f"the {obj.type_name} at {obj.pos} may be entirely hidden inside the "
                    f"opaque {occluder.type_name} at {occluder.pos}"
                )
    return warnings


_CHAOTIC_TRANSPARENT_SPONGE_LEVEL = 2.0


def manifest_subtype_warn_levels(manifest: dict) -> dict[tuple[str, str], dict[str, Optional[float]]]:
    """(type, field) -> {subtype: its `warnAt` or None} from the manifest's `limitsBy` (schema
    1.4.0). A subtype with no `warnAt` (RecursiveIAS) never triggers the slow-level caveat,
    unlike the field-level default."""
    return {
        (obj["name"], f["name"]): {
            value: (None if bounds.get("warnAt") is None else float(bounds["warnAt"]))
            for value, bounds in f["limitsBy"].get("values", {}).items()
        }
        for obj in manifest.get("objects", [])
        for f in obj.get("fields", [])
        if f.get("limitsBy")
    }


def manifest_warn_levels(manifest: dict) -> dict[tuple[str, str], float]:
    """(type, field) -> the manifest's `warnAt` (schema 1.3.0): from that value on, rendering
    gets slow."""
    return {
        (obj["name"], f["name"]): float(f["warnAt"])
        for obj in manifest.get("objects", [])
        for f in obj.get("fields", [])
        if f.get("warnAt") is not None
    }


def caveat_warnings(
    facts: SceneFacts,
    warn_levels: dict[tuple[str, str], float],
    subtype_warn_levels: Optional[dict[tuple[str, str], dict[str, Optional[float]]]] = None,
) -> list[str]:
    """Caveats the user should hear about even when the request asked for exactly this:
    a level that reaches the renderer's slowness threshold (F36: animated past it silently) and
    a transparent material on a tesseract sponge from level 2 up, which renders as chaotic
    refraction (F55, #4c; the manifest's conventions say so, the model doesn't always)."""
    warnings: list[str] = []
    for obj in facts.objects:
        if obj.level is None:
            continue
        warn_at = warn_levels.get((obj.type_name, "level"))
        by_subtype = (subtype_warn_levels or {}).get((obj.type_name, "level"), {})
        if obj.subtype in by_subtype:
            warn_at = by_subtype[obj.subtype]
        if warn_at is not None and obj.level >= warn_at:
            warnings.append(
                f"the {obj.type_name} reaches level {obj.level:g}; rendering gets slow from "
                f"level {warn_at:g}"
            )
        if (obj.type_name == "TesseractSponge" and obj.material in _TRANSPARENT_MATERIAL_PRESETS
                and obj.level >= _CHAOTIC_TRANSPARENT_SPONGE_LEVEL):
            warnings.append(
                f"{obj.material} on a TesseractSponge at level {obj.level:g} renders as chaotic, "
                f"fragmented refraction; an opaque or metal material, or level 1, shows the shape"
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

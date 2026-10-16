"""Change reports and checks from the validator's evaluated scene (result schema 1.1.0).

Usability review 2026-10, session 4: the agent knew a scene only by parsing its text
(`core/scene_facts.py`), which cannot resolve local vals, constants, `if`/`else` camera legs or
`math.cos`. Its change lists came out empty ("Nothing the facts cover changed", F90), a colour
change kept a procedural that overrode it (F89), and 8b's camera path had a 90-degree look
error and three jumps nothing saw (F102). menger's `SceneValidator` now returns the evaluated
scene: `snapshots` (one per checked time) and, for an animated scene, `cameraPath` (every
0.05 s). Pure functions over those JSON objects; no I/O.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

# --- formatting -----------------------------------------------------------------------------


def _num(x: float) -> str:
    return f"{round(x, 3):g}"


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return _num(float(value))
    if isinstance(value, list):
        return "(" + ", ".join(_fmt(v) for v in value) + ")"
    return "none" if value is None else str(value)


# --- manifest lookups -----------------------------------------------------------------------

# Emission is left out: a glowing Glass is still Glass (it is shown separately).
_MATERIAL_KEYS = ("ior", "roughness", "metallic", "specular", "filmThickness", "dispersion")
_PRESET_TOLERANCE = 1e-4
# The presets come from the manifest's structured `proceduralPresets` (review 2026-10-14,
# menger#81). They were parsed out of the `proceduralType` description with a regex, so
# rewording that sentence silently broke the F89 repair.


def procedural_names(manifest: dict) -> dict[int, str]:
    return {p["id"]: p["name"] for p in manifest.get("proceduralPresets", [])}


def colour_replacing_procedurals(manifest: dict) -> set[int]:
    return {p["id"] for p in manifest.get("proceduralPresets", []) if p.get("effect") == "replaces"}


def material_name(material: Optional[dict], manifest: dict) -> str:
    """The preset whose non-colour values the material has (a tinted Glass is still Glass),
    else "custom"; plus its colour, alpha and emission."""
    if material is None:
        return "none"
    preset = next(
        (
            m["name"] for m in manifest.get("materials", [])
            if all(
                abs(float(m.get(k, 0)) - float(material.get(k, 0))) <= _PRESET_TOLERANCE
                for k in _MATERIAL_KEYS
            )
        ),
        "custom material",
    )
    colour = material.get("color") or [1, 1, 1, 1]
    text = f"{preset}, colour {_fmt(colour[:3])}, alpha {_num(colour[3])}"
    if float(material.get("emission", 0)) > 0:
        text += f", emission {_num(material['emission'])}"
    return text


# --- summaries and change lines -------------------------------------------------------------


def _summary(values: list[object]) -> str:
    """One value over all checked times: constant, or "animated from A to B" (with the peak
    when it is neither end)."""
    if all(v == values[0] for v in values):
        return _fmt(values[0])
    first, last = values[0], values[-1]
    if all(isinstance(v, (int, float)) for v in values):
        peak = max(values)  # type: ignore[type-var]
        low = min(values)  # type: ignore[type-var]
        if peak not in (first, last):
            return f"animated {_fmt(first)} -> peak {_fmt(peak)} -> {_fmt(last)}"
        if low not in (first, last):
            return f"animated {_fmt(first)} -> low {_fmt(low)} -> {_fmt(last)}"
    return f"animated from {_fmt(first)} to {_fmt(last)}"


def _kind(o: dict) -> str:
    """The DSL name the scene wrote (`Sponge`); the renderer's (`sponge-volume`) for snapshots
    from a validator image older than result schema 1.2.0 (review 2026-10-14, menger#92)."""
    return str(o.get("dslType") or o["objectType"])


def _labels(objects: list[dict]) -> list[str]:
    counts: dict[str, int] = {}
    for o in objects:
        counts[_kind(o)] = counts.get(_kind(o), 0) + 1
    seen: dict[str, int] = {}
    labels = []
    for o in objects:
        kind = _kind(o)
        seen[kind] = seen.get(kind, 0) + 1
        labels.append(kind if counts[kind] == 1 else f"{kind} #{seen[kind]}")
    return labels


def _object_fields(o: dict, manifest: dict) -> dict[str, object]:
    names = procedural_names(manifest)
    proc = int(o.get("proceduralType", 0))
    fields: dict[str, object] = {
        # The renderer's name carries the subtype (sponge-volume, sponge-surface): shown so a
        # subtype change is reported, now that the label is the DSL name.
        "subtype": o.get("objectType") if o.get("dslType") else None,
        "position": o.get("pos"),
        "size": o.get("size"),
        "level": o.get("level"),
        "rotation": o.get("rotation"),
        "colour": o.get("color"),
        "material": material_name(o.get("material"), manifest),
        "procedural": "none" if proc == 0 else f"{proc} {names.get(proc, '?')}",
        "edge tubes": o.get("edgeRadius"),
        "edge material": (
            material_name(o.get("edgeMaterial"), manifest) if o.get("edgeMaterial") else None
        ),
    }
    if proc:
        fields["procedural scale"] = o.get("proceduralScale")
    projection = o.get("projection")
    if projection:
        for key in ("eyeW", "screenW", "rotXW", "rotYW", "rotZW", "wScale"):
            fields[f"4D {key}"] = projection.get(key)
    return {k: v for k, v in fields.items() if v is not None}


def summarise(snapshots: list[dict], manifest: dict) -> dict[tuple[str, str], str]:
    """Every value of a scene, keyed (subject, field) -- subject "scene", "camera", an object
    label ("Sponge", "Icositetrachoron #2"), a plane ("plane #2") or a light ("Directional light") -- summarised over
    its checked times. Objects are matched by type and order between scenes."""
    if not snapshots:
        return {}
    result: dict[tuple[str, str], list[object]] = {}

    def put(subject: str, name: str, value: object) -> None:
        result.setdefault((subject, name), []).append(value)

    for snap in snapshots:
        camera = snap.get("camera") or {}
        put("camera", "position", camera.get("position"))
        put("camera", "looks at", camera.get("lookAt"))
        put("scene", "background", snap.get("background"))
        put("scene", "caustics", snap.get("caustics"))
        put("scene", "render settings", snap.get("render"))
        objects = snap.get("objects") or []
        for label, o in zip(_labels(objects), objects):
            for name, value in _object_fields(o, manifest).items():
                put(label, name, value)
        # Planes (result schema 1.2.0, menger#92): a plane change used to be missing from the
        # change list, and the readback guessed.
        planes = snap.get("planes") or []
        for i, plane in enumerate(planes, 1):
            subject = "plane" + (f" #{i}" if len(planes) > 1 else "")
            put(subject, "position", f"{plane.get('axis')} = {_fmt(plane.get('value'))}")
            put(subject, "colour", plane.get("color"))
            put(subject, "checker colour", plane.get("checkerColor"))
            if plane.get("material"):
                put(subject, "material", material_name(plane.get("material"), manifest))
        lights = snap.get("lights") or []
        for i, light in enumerate(lights, 1):
            subject = f"{light.get('kind', 'light')} light" + (f" #{i}" if len(lights) > 1 else "")
            for name in ("direction", "position", "intensity", "color"):
                if light.get(name) is not None:
                    put(subject, name, light.get(name))
    summaries = {k: _summary(v) for k, v in result.items() if any(x is not None for x in v)}
    duration = snapshots[-1].get("t", 0)
    if duration:
        summaries[("scene", "duration")] = f"{_num(duration)} s"
    return summaries


def _key_text(key: tuple[str, str]) -> str:
    subject, name = key
    return name if subject == "scene" else f"{subject} {name}"


def snapshot_changes(before: list[dict], after: list[dict], manifest: dict) -> list[str]:
    """What changed between two turns' evaluated scenes, with values: "<what>: old -> new",
    "added <object> (...)", "removed <object>". Empty means no evaluated value changed."""
    old, new = summarise(before, manifest), summarise(after, manifest)
    old_subjects = {s for s, _ in old}
    new_subjects = {s for s, _ in new}
    added, gone = new_subjects - old_subjects, old_subjects - new_subjects
    changes = []
    for subject in sorted(added):
        parts = [
            f"{name} {value}" for (s, name), value in new.items()
            if s == subject and name in ("position", "size", "level", "material")
        ]
        changes.append(f"added {subject} ({', '.join(parts)})")
    changes += [f"removed {subject}" for subject in sorted(gone)]
    for key in sorted(set(old) | set(new)):
        if key[0] in added or key[0] in gone:
            continue
        if old.get(key) != new.get(key):
            changes.append(f"{_key_text(key)}: {old.get(key, 'none')} -> {new.get(key, 'none')}")
    return changes


def _change_kind(key: tuple[str, str]) -> Optional[str]:
    """`core.scene_facts._ASKED_FOR`'s kind a changed value belongs to, if it is one the user
    should hear about when the request did not ask for it."""
    subject, name = key
    if subject == "camera":
        return "camera"
    if name == "position":
        return "pos"
    if name in ("material", "colour"):
        return "material"
    if name.startswith("4D "):
        return "projection"
    return None


def unrequested_changes(before: list[dict], after: list[dict], prompt: str, manifest: dict) -> list[str]:
    """F8/F39 with values (session 4, F90): a moved object or camera, a changed material or 4D
    projection the request did not ask for. A refactor that keeps the values (8a: a projection
    moved into a local val) is no change, so it is not reported."""
    from core.scene_facts import _asked_for  # same package; the keyword lists live there

    old, new = summarise(before, manifest), summarise(after, manifest)
    warnings = []
    for key in sorted(set(old) & set(new)):
        kind = _change_kind(key)
        if kind is not None and old[key] != new[key] and not _asked_for(kind, prompt):
            warnings.append(f"also changed {_key_text(key)}: {old[key]} -> {new[key]}")
    return warnings


# --- checks ---------------------------------------------------------------------------------


@dataclass
class CheckResult:
    """`findings` go back to the model like a lint finding (a repair round); `warnings` and
    `notes` are only shown."""

    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_PATTERN_WORDS = re.compile(r"\b(pattern|procedural|texture|grain|noise|marble|wood|xyz|rgb|colou?r(ed)? by)\b", re.I)


def procedural_override_findings(before: list[dict], after: list[dict], prompt: str, manifest: dict) -> list[str]:
    """F89: a turn changed an object's material or colour but kept a non-zero procedural, which
    either imitates the old material (wood on aluminium) or replaces the new colour (green
    under xyz_rgb_local)."""
    if not before or not after or _PATTERN_WORDS.search(prompt):
        return []
    replacing = colour_replacing_procedurals(manifest)
    names = procedural_names(manifest)
    old_objects, new_objects = before[0].get("objects") or [], after[0].get("objects") or []
    findings = []
    for label, o_old, o_new in zip(_labels(new_objects), old_objects, new_objects):
        if o_old.get("objectType") != o_new.get("objectType"):
            continue
        proc = int(o_new.get("proceduralType", 0))
        if proc == 0 or int(o_old.get("proceduralType", 0)) != proc:
            continue
        old_mat = material_name(o_old.get("material"), manifest)
        new_mat = material_name(o_new.get("material"), manifest)
        colour_changed = o_old.get("color") != o_new.get("color") or old_mat != new_mat
        if not colour_changed:
            continue
        effect = (
            "replaces the new colour, so it is never seen" if proc in replacing
            else "belongs to the look of the old material"
        )
        findings.append(
            f"{label}: the material/colour changed ({old_mat} -> {new_mat}) but proceduralType "
            f"{proc} ({names.get(proc, '?')}) stayed, and it {effect}; drop it (proceduralType = 0) "
            "unless the request asked for that pattern"
        )
    return findings


_FLIGHT_WORDS = re.compile(r"\b(fly|flies|flight|flying|through|travel|tunnel|follow)\b", re.I)
_LOOK_AHEAD_WORDS = re.compile(r"\b(look(s|ing)? (along|ahead|forward|where)|flight direction|facing forward)\b", re.I)
_INSIDE_WORDS = re.compile(r"\b(through|inside|into|tunnel)\b", re.I)
# A step this many times larger than both its neighbours and than the path's median step is a
# jump, not a fast leg (8b 010: 0.46 units at 7.55 s against 0.03 around it).
JUMP_FACTOR = 3.0
# Look direction and direction of travel further apart than this (8b 004: 90 degrees).
LOOK_ANGLE_DEGREES = 60.0
# Neighbouring steps whose lengths differ by more than this factor: an abrupt speed change.
SPEED_FACTOR = 3.0
# Share of a "through" flight spent outside every object before it is worth a warning.
OUTSIDE_SHARE = 0.5
_MOVING = 1e-6


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _angle(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na < _MOVING or nb < _MOVING:
        return 0.0
    cos = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))
    return math.degrees(math.acos(cos))


def _spans(times: list[float]) -> str:
    """"5-7.5 s, 10.5 s" from sorted sample times, merging neighbours."""
    if not times:
        return ""
    spans, start, prev = [], times[0], times[0]
    for t in times[1:]:
        if t - prev > 0.11:
            spans.append((start, prev))
            start = t
        prev = t
    spans.append((start, prev))
    return ", ".join(f"{_num(a)} s" if a == b else f"{_num(a)}-{_num(b)} s" for a, b in spans)


def _inside_any(p: list[float], objects: list[dict]) -> bool:
    for o in objects:
        half = float(o.get("size", 1)) / 2
        centre = o.get("pos") or [0, 0, 0]
        if all(abs(p[i] - centre[i]) <= half for i in range(3)):
            return True
    return False


def camera_path_check(camera_path: list[dict], snapshots: list[dict], prompt: str) -> CheckResult:
    """F102: jumps between the legs of a flight, a camera that looks away from where it moves,
    abrupt speed changes, and a "through" flight spent mostly outside."""
    result = CheckResult()
    points = [s.get("position") for s in camera_path]
    if len(points) < 3 or any(p is None for p in points):
        return result
    times = [float(s.get("t", 0)) for s in camera_path]
    steps = [_norm(_sub(points[i], points[i - 1])) for i in range(1, len(points))]
    moving = sorted(s for s in steps if s > _MOVING)
    if not moving:
        return result
    median = moving[len(moving) // 2]
    jumps = [
        times[i + 1] for i in range(1, len(steps) - 1)
        if steps[i] > JUMP_FACTOR * max(steps[i - 1], steps[i + 1], median)
    ]
    if jumps:
        result.findings.append(
            f"the camera jumps at t = {_spans(jumps)} (a leg does not start where the previous "
            "one ends): make each leg start at the previous leg's end point"
        )
    if _FLIGHT_WORDS.search(prompt):
        away = []
        for i in range(1, len(points) - 1):
            velocity = _sub(points[i + 1], points[i - 1])
            look = _sub(camera_path[i].get("lookAt") or points[i], points[i])
            if _norm(velocity) > _MOVING and _angle(velocity, look) > LOOK_ANGLE_DEGREES:
                away.append(times[i])
        if away:
            message = (
                f"the camera looks more than {_num(LOOK_ANGLE_DEGREES)} degrees away from its "
                f"direction of travel at t = {_spans(away)}"
            )
            if _LOOK_AHEAD_WORDS.search(prompt):
                result.findings.append(message + ": aim lookAt along the path's tangent")
            else:
                result.warnings.append(message)
        abrupt = [
            times[i + 1] for i in range(1, len(steps))
            if min(steps[i], steps[i - 1]) > _MOVING
            and max(steps[i], steps[i - 1]) > SPEED_FACTOR * min(steps[i], steps[i - 1])
            and times[i + 1] not in jumps and times[i] not in jumps
        ]
        if abrupt:
            result.warnings.append(f"the camera speed changes abruptly at t = {_spans(abrupt)}")
    if _INSIDE_WORDS.search(prompt) and snapshots:
        objects = snapshots[0].get("objects") or []
        outside = sum(1 for p in points if not _inside_any(p, objects))
        if objects and outside / len(points) > OUTSIDE_SHARE:
            result.warnings.append(
                f"the camera is outside the objects for {round(100 * outside / len(points))}% "
                "of the flight"
            )
    return result


_SIZE_WORDS = re.compile(r"\b(size|big|bigger|small|smaller|large|larger|tiny|huge|scale|scaled|units?)\b", re.I)
_EDGE_COLOUR_WORDS = re.compile(r"\bedges?\b.{0,40}\b(gold|chrome|copper|colou?r|material|metal|glass|black|white|red|green|blue)\b|\b(gold|chrome|copper|colou?r(ed)?|black|white|red|green|blue)\b.{0,20}\bedges?\b", re.I)
_DEFAULT_SIZE = 1.0


def unasked_choice_notes(before: list[dict], after: list[dict], prompt: str, manifest: dict) -> list[str]:
    """F93: values the agent chose for a new object that the request did not name -- shown, so
    the user can say otherwise (session 4: a size-2.5 sponge, copper edges, a glass 24-cell)."""
    if not after:
        return []
    old_labels = set(_labels(before[0].get("objects") or [])) if before else set()
    objects = after[0].get("objects") or []
    material_words = {m["name"].lower() for m in manifest.get("materials", [])}
    prompt_words = set(re.findall(r"[a-z]+", prompt.lower()))
    notes = []
    for label, o in zip(_labels(objects), objects):
        if label in old_labels:
            continue
        chosen = []
        size = float(o.get("size", _DEFAULT_SIZE))
        if abs(size - _DEFAULT_SIZE) > _PRESET_TOLERANCE and not _SIZE_WORDS.search(prompt):
            chosen.append(f"size {_num(size)} (default 1)")
        if o.get("material") and not (prompt_words & material_words) and "material" not in prompt_words:
            chosen.append(material_name(o.get("material"), manifest).split(",")[0])
        if o.get("edgeMaterial") and not _EDGE_COLOUR_WORDS.search(prompt):
            chosen.append(
                material_name(o.get("edgeMaterial"), manifest).split(",")[0] + " edges"
            )
        if chosen:
            notes.append(f"chose for the {label}, unasked: {', '.join(chosen)} -- say if you want otherwise")
    return notes


def check_turn(
    before: list[dict], after: list[dict], camera_path: list[dict], prompt: str, manifest: dict
) -> CheckResult:
    """All snapshot checks of one turn."""
    result = camera_path_check(camera_path, after, prompt)
    result.findings = procedural_override_findings(before, after, prompt, manifest) + result.findings
    result.notes = unasked_choice_notes(before, after, prompt, manifest)
    return result

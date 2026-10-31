"""The DSL vocabulary the agent works with, read from reference/dsl-manifest.json (review
2026-10-14, msa#28): object types, positioned lights, see-through materials, holed objects,
colour-replacing presets, field bounds and product caps, the compile surface and the DSL's
revision. These were lists kept by hand here and in the gauntlet, and fell behind the DSL
(the five regular polytopes had no facts for six days). `tests/test_vocabulary.py` fails when a
regenerated manifest brings vocabulary the agent does not handle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "reference" / "dsl-manifest.json"
MANIFEST: dict = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

Bounds = Tuple[Optional[float], Optional[float]]


def _names(types: List[dict], trait: Optional[str] = None) -> Tuple[str, ...]:
    return tuple(t["name"] for t in types if trait is None or trait in t.get("traits", []))


OBJECT_TYPES: Tuple[str, ...] = _names(MANIFEST["objects"])
POSITIONED_LIGHT_TYPES: Tuple[str, ...] = _names(MANIFEST["lights"], "positioned")
HOLED_TYPES: FrozenSet[str] = frozenset(_names(MANIFEST["objects"], "holed"))
TRANSPARENT_MATERIALS: FrozenSet[str] = frozenset(
    m["name"] for m in MANIFEST["materials"] if m.get("transparent")
)
# Light bends through these (session 5, F121: Film was taken for not refractive).
REFRACTIVE_MATERIALS: FrozenSet[str] = frozenset(
    m["name"] for m in MANIFEST["materials"] if m.get("transparent") and (m.get("ior") or 1) > 1
)
REPLACING_PRESETS: Dict[int, str] = {
    p["id"]: p["name"] for p in MANIFEST["proceduralPresets"] if p["effect"] == "replaces"
}
PRESET_NAMES: Dict[int, str] = {p["id"]: p["name"] for p in MANIFEST["proceduralPresets"]}
DSL_REVISION: str = MANIFEST["contentRevision"]

# Every type whose fields a scene sets by name, and so may carry bounds.
_FIELD_TYPES: List[dict] = (
    MANIFEST["objects"] + MANIFEST["lights"] + MANIFEST["sceneComposition"]
    + [MANIFEST["renderSettings"], MANIFEST["plane"], MANIFEST["cameraPath"]]
)

FIELD_BOUNDS: Dict[str, Dict[str, Bounds]] = {
    t["name"]: {
        f["name"]: (f.get("min"), f.get("max"))
        for f in t["fields"] if f.get("min") is not None or f.get("max") is not None
    }
    for t in _FIELD_TYPES
}
FIELD_BOUNDS = {name: fields for name, fields in FIELD_BOUNDS.items() if fields}

PRODUCT_CAPS: List[Tuple[str, Tuple[str, ...], float]] = [
    (c["type"], tuple(c["fields"]), c["max"]) for c in MANIFEST["productCaps"]
]

# What a scene may import: menger-app's compile surface, plus the Scala library parts the
# agent's policy admits (scala.math, implicit conversions) -- the language core is implicit.
IMPORT_ROOTS: Tuple[str, ...] = tuple(MANIFEST["compileSurface"]["packages"]) + (
    "scala.math", "scala.language.implicitConversions",
)

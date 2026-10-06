"""AD-4 resource-bound check (rule 3): numeric literals bound to known resource-affecting
fields are rejected above a fixed ceiling, before compile. These fields are reachable
through pure allowed-root DSL vocabulary with no allowlist bypass (the allowlist check in
`gauntlet/allowlist.py` cannot catch a `level = 50` resource bomb, since `level` is a
legitimate `menger.dsl` field), so this is a separate static pass over the same source
text.

Field names are the real `menger.dsl` object field names from `reference/dsl-manifest.json`:
`level` (Sponge/TesseractSponge/Sierpinski4D), `uSteps`/`vSteps` (ParametricSurface), and
`iterations` (LSystem's iteration-count field).

Usability review 2026-09 (T1#3): the ceilings below used to be hand-picked, independent of
`menger`'s own DSL/CLI limits (`menger.dsl.ResourceLimits`) -- a mismatch either rejected
something the renderer would happily accept, or (the sharper failure) accepted something the
renderer would reject only after a full compile+render round trip. They are now read from
`reference/dsl-manifest.json`'s per-field `max` (schema 1.2.0), the same manifest the
renderer's own `ManifestGenerator` derives from `ResourceLimits` -- one source, both sides of
the AD-9 boundary. `_FALLBACK_CEILINGS_BY_TYPE` (the old hand-picked values) is used only if
the manifest is missing or doesn't have a `max` for a tracked field -- defense-in-depth never
silently disables itself just because an artifact is stale or absent.

Review round 1 finding: an earlier version matched `field = literal` anywhere in the text,
with no constructor scoping. Run against the real 32-scene corpus, this false-positived on
`CausticsCanonical.scala`/`TwoSpheres.scala`'s `Caustics(iterations = 20, ...)` --
`Caustics.iterations` (a photon-mapping setting) shares a name with `LSystem.iterations` but
is a different, unrelated field. Every field is now only checked inside the specific
constructor call(s) it actually belongs to, mirroring `lint.py`'s constructor-scoped
extraction (`gauntlet/_scala_text.py`'s shared `find_call_bodies`/`named_args`).

Only plain numeric literals are checked. A `t`-driven expression (e.g. `level = progress *
3f`, `level = 1f + progress * 3f`) is not syntactically a literal and is silently skipped
for this check, per the I/O & Edge-Case Matrix's explicit "animated field -> skipped, not
failed" row -- stage 4 (story 5, renderer domain, where the expression can be evaluated at
a sampled `t`) is the right place for that rigor, not this one.
"""

from __future__ import annotations

from pathlib import Path

from adapters.artifacts import ArtifactError, load_manifest
from gauntlet._scala_text import as_float, find_call_bodies, line_of, named_args, strip_comments_and_strings
from gauntlet.manifest_bounds import discriminator_of, field_bounds, subtype_of
from gauntlet.types import Finding

_STAGE = "resource_bounds"

# AD-9: this package's own build-time artifact, resolved relative to this script's own
# location (never CWD) -- same reasoning `adapters/artifacts.py`'s docstring and `cli.py`'s
# `_MANIFEST_PATH` already give for owning this I/O.
_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "reference" / "dsl-manifest.json"

# Used only when the manifest can't be read, or doesn't carry a `max` for a tracked field --
# the pre-manifest hand-picked ceilings, kept as a defense-in-depth floor, not the primary
# source of truth any more.
_FALLBACK_CEILINGS_BY_TYPE: dict[str, dict[str, float]] = {
    "Sponge": {"level": 10},
    "TesseractSponge": {"level": 10},
    "Sierpinski4D": {"level": 10},
    "ParametricSurface": {"uSteps": 200, "vSteps": 200},
    "LSystem": {"iterations": 10},
}


def ceilings_from_manifest(
    manifest: dict, fallback: dict[str, dict[str, float]] = _FALLBACK_CEILINGS_BY_TYPE
) -> dict[str, dict[str, float]]:
    """Pure transform: `reference/dsl-manifest.json`'s per-field `max` -> the same
    `{type: {field: ceiling}}` shape `check_resource_bounds` has always used, scoped to
    exactly the (type, field) pairs `fallback` already tracks (this check's own scope is
    unchanged; only where the numbers come from is new). A tracked field the manifest
    doesn't carry a `max` for keeps its fallback value.
    """
    by_type = {obj.get("name"): obj.get("fields", []) for obj in manifest.get("objects", [])}
    result: dict[str, dict[str, float]] = {}
    for type_name, field_ceilings in fallback.items():
        fields_by_name = {f.get("name"): f for f in by_type.get(type_name, [])}
        result[type_name] = {}
        for field_name, fallback_ceiling in field_ceilings.items():
            manifest_field = fields_by_name.get(field_name, {})
            manifest_max = manifest_field.get("max")
            result[type_name][field_name] = manifest_max if manifest_max is not None else fallback_ceiling
    return result


def _load_ceilings() -> dict[str, dict[str, float]]:
    try:
        manifest = load_manifest(_MANIFEST_PATH)
    except ArtifactError:
        return _FALLBACK_CEILINGS_BY_TYPE
    return ceilings_from_manifest(manifest)


_FIELD_CEILINGS_BY_TYPE: dict[str, dict[str, float]] = _load_ceilings()


def subtype_fields_from_manifest(manifest: dict) -> dict[tuple[str, str], dict]:
    """(type, field) -> the manifest field entry, for the fields that carry `limitsBy`
    (schema 1.4.0): their bounds depend on another argument, e.g. `Sponge.level` on
    `spongeType`."""
    return {
        (obj.get("name"), f.get("name")): f
        for obj in manifest.get("objects", [])
        for f in obj.get("fields", [])
        if f.get("limitsBy")
    }


def _load_subtype_fields() -> dict[tuple[str, str], dict]:
    try:
        return subtype_fields_from_manifest(load_manifest(_MANIFEST_PATH))
    except ArtifactError:
        return {}


_SUBTYPE_FIELDS: dict[tuple[str, str], dict] = _load_subtype_fields()


def check_resource_bounds(scene_text: str) -> list[Finding]:
    """AD-4 rule 3: flags any resource-affecting field bound to a literal value above its
    fixed ceiling, scoped to the constructor call that actually declares the field. A
    non-literal (animated) value, or the same field name on an unrelated constructor
    (`Caustics.iterations`), is never extracted, so both are silently skipped rather than
    flagged, per the Edge-Case Matrix."""
    stripped = strip_comments_and_strings(scene_text)
    findings: list[Finding] = []
    for type_name, ceilings in _FIELD_CEILINGS_BY_TYPE.items():
        for offset, inner in find_call_bodies(stripped, type_name):
            args = named_args(inner)
            for field, ceiling in ceilings.items():
                if field not in args:
                    continue
                value = as_float(args[field])
                if value is None:
                    continue
                low, high = _bounds_for(type_name, field, ceiling, args)
                if high is not None and value > high:
                    message = (
                        f"{type_name}'s '{field}' = {args[field]} exceeds the "
                        f"resource-bound ceiling of {high:g} (defense-in-depth, AD-4 rule 3)"
                    )
                elif low is not None and value < low:
                    message = (
                        f"{type_name}'s '{field}' = {args[field]} is below the minimum of "
                        f"{low:g} the renderer accepts here"
                    )
                else:
                    continue
                findings.append(
                    Finding(
                        stage=_STAGE,
                        message=message,
                        field=field,
                        identifier=args[field],
                        line=line_of(scene_text, offset),
                    )
                )
    return findings


def _bounds_for(
    type_name: str, field: str, default_ceiling: float, args: dict[str, str]
) -> tuple[float | None, float | None]:
    """(minimum, ceiling) for one constructor call: the per-subtype bounds from `limitsBy`
    when the discriminating argument (`spongeType`) is a literal the manifest lists, else the
    conservative field-level ceiling and no minimum (usability review 2026-10, F59)."""
    manifest_field = _SUBTYPE_FIELDS.get((type_name, field))
    if manifest_field is None:
        return None, default_ceiling
    discriminator = discriminator_of(manifest_field)
    subtype = subtype_of(args.get(discriminator)) if discriminator else None
    by_value = (manifest_field.get("limitsBy") or {}).get("values") or {}
    if subtype not in by_value:
        return None, default_ceiling
    low, high, _ = field_bounds(manifest_field, subtype)
    return low, high

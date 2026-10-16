"""AD-4 resource-bound check (rule 3): numeric literals bound to known resource-affecting
fields are rejected above a fixed ceiling, before compile. These fields are reachable
through pure allowed-root DSL vocabulary with no allowlist bypass (the allowlist check in
`gauntlet/allowlist.py` cannot catch a `level = 50` resource bomb, since `level` is a
legitimate `menger.dsl` field), so this is a separate static pass over the same source
text.

The fields checked are every field `reference/dsl-manifest.json` gives a `min` or `max`
(menger's `FieldConstraints`), plus its `productCaps`.

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

import math

from core.vocabulary import FIELD_BOUNDS, MANIFEST, PRODUCT_CAPS
from gauntlet._scala_text import (
    as_float,
    find_call_bodies,
    line_of,
    named_args,
    strip_comments_and_strings,
)
from gauntlet.manifest_bounds import discriminator_of, field_bounds, subtype_of
from gauntlet.types import Finding

_STAGE = "resource_bounds"

# Review 2026-10-14 (msa#28): the check's *scope* is every bounded field in the manifest, not a
# hand list of five -- a field menger bounds (IBL.samples, RenderSettings.aaMaxDepth, ...) used to
# go unchecked here until someone added it. The hand-picked fallback ceilings are gone: the agent
# refuses to start without its manifest (core/generation.py), so there is nothing to fall back
# for. `productCaps` covers bounds per-field limits cannot express (`uSteps * vSteps`).


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


_SUBTYPE_FIELDS: dict[tuple[str, str], dict] = subtype_fields_from_manifest(MANIFEST)


def _finding(scene_text: str, offset: int, field: str, identifier: str, message: str) -> Finding:
    return Finding(stage=_STAGE, message=message, field=field, identifier=identifier,
                   line=line_of(scene_text, offset))


def check_resource_bounds(scene_text: str) -> list[Finding]:
    """AD-4 rule 3: flags any bounded field bound to a constant outside its bounds, scoped to
    the constructor call that actually declares the field. A non-constant (animated) value, or
    the same field name on an unrelated constructor, is never extracted, so both are silently
    skipped rather than flagged, per the Edge-Case Matrix."""
    stripped = strip_comments_and_strings(scene_text)
    findings: list[Finding] = []
    for type_name, bounds in FIELD_BOUNDS.items():
        for offset, inner in find_call_bodies(stripped, type_name):
            args = named_args(inner)
            for field, (field_min, field_max) in bounds.items():
                if field not in args:
                    continue
                value = as_float(args[field])
                if value is None:
                    continue
                low, high = _bounds_for(type_name, field, field_min, field_max, args)
                if high is not None and value > high:
                    message = (
                        f"{type_name}'s '{field}' = {args[field]} exceeds the "
                        f"resource-bound ceiling of {high:g} (defense-in-depth, AD-4 rule 3)"
                        + _alternatives_hint(type_name, field, value, args)
                    )
                elif low is not None and value < low:
                    message = (
                        f"{type_name}'s '{field}' = {args[field]} is below the minimum of "
                        f"{low:g} the renderer accepts here"
                    )
                else:
                    continue
                findings.append(_finding(scene_text, offset, field, args[field], message))
    for type_name, fields, cap in PRODUCT_CAPS:
        for offset, inner in find_call_bodies(stripped, type_name):
            args = named_args(inner)
            values = [as_float(args[f]) if f in args else None for f in fields]
            if any(v is None for v in values):
                continue
            product = math.prod(v for v in values if v is not None)
            if product > cap:
                shown = " * ".join(f"{f} ({args[f]})" for f in fields)
                findings.append(_finding(
                    scene_text, offset, fields[0], args[fields[0]],
                    f"{type_name}'s {shown} = {product:g} exceeds the cap of {cap:g}",
                ))
    return findings


def _bounds_for(
    type_name: str, field: str, field_min: float | None, field_max: float | None,
    args: dict[str, str],
) -> tuple[float | None, float | None]:
    """(minimum, ceiling) for one constructor call: the per-subtype bounds from `limitsBy`
    when the discriminating argument (`spongeType`) is a literal the manifest lists, else the
    field's own, conservative bounds (usability review 2026-10, F59). A field with per-subtype
    bounds checks no field-level minimum without a known subtype: RecursiveIAS starts at 1."""
    manifest_field = _SUBTYPE_FIELDS.get((type_name, field))
    if manifest_field is None:
        return field_min, field_max
    discriminator = discriminator_of(manifest_field)
    subtype = subtype_of(args.get(discriminator)) if discriminator else None
    by_value = (manifest_field.get("limitsBy") or {}).get("values") or {}
    if subtype not in by_value:
        return None, field_max
    low, high, _ = field_bounds(manifest_field, subtype)
    return low, high


def _alternatives_hint(type_name: str, field: str, value: float, args: dict[str, str]) -> str:
    """For a value above its subtype's ceiling, the other subtypes whose range admits it, as a
    repair hint (usability review 2026-10, session 4, F87: level 5.8 on VolumeFilling was
    repaired by lowering it, although RecursiveIAS accepts 1 to 13). Empty if there are none."""
    manifest_field = _SUBTYPE_FIELDS.get((type_name, field))
    if manifest_field is None:
        return ""
    discriminator = discriminator_of(manifest_field)
    current = subtype_of(args.get(discriminator)) if discriminator else None
    by_value = (manifest_field.get("limitsBy") or {}).get("values") or {}
    fitting = []
    for subtype in by_value:
        if subtype == current:
            continue
        low, high, _ = field_bounds(manifest_field, subtype)
        if (low is None or value >= low) and (high is None or value <= high):
            fitting.append(
                f"{subtype} ({(low or 0):g} to {high:g})" if high is not None else subtype
            )
    if not fitting:
        return ""
    return (
        f"; {discriminator} {', '.join(fitting)} accepts {value:g} -- switch to it, or ask, "
        "rather than lowering a value the user asked for"
    )

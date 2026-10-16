"""Per-subtype field bounds from the manifest's `limitsBy` (schema 1.4.0, usability review
2026-10, session 3, F59).

A manifest field such as `Sponge.level` carries one conservative `min`/`max`/`warnAt`, plus a
`limitsBy` block that gives the bounds for each value of a discriminating field (`spongeType`):
`{"field": "spongeType", "values": {"RecursiveIAS": {"min": 1, "max": 13}, ...}}`. Using the
single field-level `max` for every sponge type made the agent clamp a RecursiveIAS level of 5.8
to 5 although the engine accepts [1, 14).

Pure functions on plain dicts, no I/O, so `core/` and `gauntlet/` can both import them.
"""

from __future__ import annotations

import re
from typing import Optional

Bounds = tuple[Optional[float], Optional[float], Optional[float]]  # (min, max, warnAt)

_TRAILING_IDENTIFIER = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*$")


def subtype_of(argument_text: Optional[str]) -> Optional[str]:
    """The enum value named by a constructor argument: `RecursiveIAS` and
    `SpongeType.RecursiveIAS` both give `RecursiveIAS`; anything not an identifier gives None."""
    if argument_text is None:
        return None
    match = _TRAILING_IDENTIFIER.search(argument_text.strip())
    return match.group(1) if match else None


def field_bounds(field: dict, subtype: Optional[str]) -> Bounds:
    """The (min, max, warnAt) that apply to `field` for `subtype`. A subtype the manifest lists
    replaces the field-level bounds entirely (it may have no `warnAt`); an unknown or missing
    subtype keeps the field-level, conservative ones."""
    by_value = (field.get("limitsBy") or {}).get("values") or {}
    chosen = by_value.get(subtype) if subtype is not None else None
    if chosen is None:
        return field.get("min"), field.get("max"), field.get("warnAt")
    return chosen.get("min"), chosen.get("max"), chosen.get("warnAt")


def discriminator_of(field: dict) -> Optional[str]:
    """The name of the argument a field's `limitsBy` is keyed by, e.g. `spongeType`."""
    return (field.get("limitsBy") or {}).get("field")

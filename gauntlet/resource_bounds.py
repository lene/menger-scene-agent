"""AD-4 resource-bound check (rule 3): numeric literals bound to known resource-affecting
fields are rejected above a fixed ceiling, before compile. These fields are reachable
through pure allowed-root DSL vocabulary with no allowlist bypass (the allowlist check in
`gauntlet/allowlist.py` cannot catch a `level = 50` resource bomb, since `level` is a
legitimate `menger.dsl` field), so this is a separate static pass over the same source
text. Defense-in-depth pending a hard cap in `menger` itself (AD-4), not a substitute for
one -- the fixed ceilings below are chosen for that purpose, not exact production tuning.

Field names are the real `menger.dsl` object field names from `reference/dsl-manifest.json`:
`level` (Sponge/TesseractSponge/Sierpinski4D), `uSteps`/`vSteps` (ParametricSurface), and
`iterations` (LSystem's iteration-count field).

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

from gauntlet._scala_text import as_float, find_call_bodies, line_of, named_args, strip_comments_and_strings
from gauntlet.types import Finding

_STAGE = "resource_bounds"

# Fixed ceilings (defense-in-depth per AD-4 rule 3, not exact production tuning):
#  - `level` above 10: the corpus's visually-meaningful range tops out far below this (the
#    poc-run fixtures animate level 0.0 -> 3.0); a Menger/Sierpinski/tesseract sponge's
#    primitive count grows exponentially with level, so 10 is already deep into
#    "renderer-melting" territory.
#  - `uSteps`/`vSteps` above 200: a parametric-surface tessellation of 200x200 samples is
#    already far beyond what any reasonable render resolution can distinguish.
#  - `iterations` above 10: an L-System's segment count grows with the rule's branching
#    factor raised to the iteration count, so double-digit iterations is already a
#    plausible unbounded-expansion resource bomb.
#
# Each field is scoped to the constructor(s) that actually declare it -- see module
# docstring's Caustics.iterations collision finding.
_FIELD_CEILINGS_BY_TYPE: dict[str, dict[str, float]] = {
    "Sponge": {"level": 10},
    "TesseractSponge": {"level": 10},
    "Sierpinski4D": {"level": 10},
    "ParametricSurface": {"uSteps": 200, "vSteps": 200},
    "LSystem": {"iterations": 10},
}


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
                if value is None or value <= ceiling:
                    continue
                findings.append(
                    Finding(
                        stage=_STAGE,
                        message=(
                            f"{type_name}'s '{field}' = {args[field]} exceeds the "
                            f"resource-bound ceiling of {ceiling} (defense-in-depth, "
                            f"AD-4 rule 3)"
                        ),
                        field=field,
                        identifier=args[field],
                        line=line_of(scene_text, offset),
                    )
                )
    return findings

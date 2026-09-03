"""Shared result type for all four gauntlet stages built in this story.

AD-4 / Boundaries & Constraints: "Every stage returns a typed result (pass, or a list of
typed findings with a human-readable message) -- never raises, mirroring story 3's
`GenerationResult`/`ModelError` convention" (core/types.py, adapters/model.py). A clean
pass is the empty list; there is no separate "ok" sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Finding:
    """One static-analysis finding from a gauntlet check.

    `stage` names the check module that produced it ("allowlist", "resource_bounds",
    "lint", "clean_code"). `message` is a human-readable, standalone-actionable
    description. `field`, `identifier` and `line` are optional identifying context --
    populated whenever a check can attribute the finding cheaply (a named-parameter field,
    an offending import/type name, a 1-based source line), left `None` when the finding is
    file-level (e.g. clean-code's overall line-count bound) or the context genuinely isn't
    available (Boundaries & Constraints: "Findings carry enough context ... to be
    actionable without re-deriving the parse").
    """

    stage: str
    message: str
    field: Optional[str] = None
    identifier: Optional[str] = None
    line: Optional[int] = None

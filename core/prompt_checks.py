"""Deterministic reasons to ask before generating (usability review 2026-10, session 4, F93:
"it never asked - that is still a weak point"). Pure functions over the user's line.
"""

from __future__ import annotations

import re
from typing import Optional

# A line that ends mid-sentence: a comma, semicolon or colon, or a word that cannot end a
# request. Session 4, 8b: a copy-paste split "... the material cycles steel," from its rest.
_CUT_OFF_END = re.compile(
    r"([,;:]|\b(and|or|but|the|a|an|with|to|of|then|into|from|plus))\s*$", re.IGNORECASE
)
# A line that starts mid-sentence: leading whitespace or a comma -- the second half of a split
# paste ("  glass, diamond, ruby, gold"). A lower-case start alone is not enough: most
# requests start lower-case ("make it from wood").
_CUT_OFF_START = re.compile(r"^(\s+\S|[,;])")
_TAIL_WORDS = 4


def cut_off_reason(line: str) -> Optional[str]:
    """The question to ask when `line` looks like half of a request, else `None`."""
    if _CUT_OFF_END.search(line):
        tail = " ".join(line.split()[-_TAIL_WORDS:])
        return f"your request looks cut off after \"{tail}\" -- what is the rest?"
    if _CUT_OFF_START.match(line):
        head = " ".join(line.split()[:_TAIL_WORDS])
        return (
            f"your request looks like the second half of a line (\"{head} ...\") -- what "
            "came before it?"
        )
    return None


# Materials people name that the DSL has no preset for; the manifest's own preset names are
# subtracted at run time. Session 4, 8b: "steel" became chrome and "ruby" red glass, unasked.
_MATERIAL_WORDS = (
    "steel", "iron", "bronze", "brass", "aluminium", "aluminum", "silver", "titanium",
    "ruby", "emerald", "sapphire", "jade", "amethyst", "marble", "stone", "granite",
    "ceramic", "porcelain", "rubber", "leather", "fabric", "velvet", "obsidian",
)


def unknown_material_hint(prompt: str, manifest: dict) -> Optional[str]:
    """An instruction for the model when the request names a material without a preset. Not
    for a threaded answer (`run_turn(answered=True)`): repeated there, it asked again on every
    round (session 5, F104). A procedural preset's name (marble) is a look the DSL has."""
    presets = {m.get("name", "").lower() for m in manifest.get("materials", [])}
    presets |= {p.get("name", "").lower() for p in manifest.get("proceduralPresets", [])}
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    unknown = [w for w in _MATERIAL_WORDS if w in words and w not in presets]
    if not unknown:
        return None
    return (
        f"The request names {', '.join(unknown)}, which the DSL has no preset for. Unless the "
        "request already says how to approximate it, respond with NEEDS_CLARIFICATION asking "
        "once which approximation the user wants, offering two or three concrete compositions: "
        "a metal colour via `Material.metal(Color(...))`, Glass or Diamond with a colour or IOR, "
        "or a preset with a procedural pattern (session 4, F93; session 5, F104)."
    )


# A request that points at what the render window shows ("keep them like that", "the current
# view", "this angle"): the window's own state (4D rotation, camera) is then part of the
# context. Session 4, 4.11/4.12 (F96): "keep them like that" had nothing to keep.
_WINDOW_WORDS = re.compile(
    r"\b(window|view|like that|as it is|as is|this angle|current(ly)?|rotation|projection|"
    r"where i|what i see)\b",
    re.IGNORECASE,
)


def refers_to_window(prompt: str) -> bool:
    return _WINDOW_WORDS.search(prompt) is not None


def window_state_hint(description: str) -> str:
    """The instruction that carries the window's state into a generation prompt."""
    return (
        f"The render window currently shows: {description}. If the request refers to what the "
        "window shows (\"keep them like that\", \"this angle\"), write exactly these values "
        "into the scene file; otherwise ignore them (session 4, F96)."
    )

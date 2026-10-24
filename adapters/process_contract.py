"""menger's process contract (reference/process-result.json, a copy of menger's
contracts/process-result.json): the result every fast-exiting menger process prints, the exit
code each tag goes with, and the window-event and window-state versions (review 2026-10-14,
menger#82, #100, #116). The validator and the render window are both read against it, instead
of each parser encoding its own reading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "reference" / "process-result.json"
CONTRACT: dict = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))

RESULT_SCHEMA_VERSION: str = CONTRACT["schemaVersion"]
EXIT_CODES: dict = CONTRACT["exitCodes"]
WINDOW_EVENT_VERSION: int = CONTRACT["windowEvents"]["version"]
WINDOW_STATE_SCHEMA_VERSION: str = CONTRACT["windowStateSchemaVersion"]


def _major(version: str) -> str:
    return version.split(".", 1)[0]


def same_major(version: object, expected: str) -> bool:
    return isinstance(version, str) and bool(version) and _major(version) == _major(expected)


def result_problem(wire_tag: str, schema_version: object, returncode: int) -> Optional[str]:
    """Why a parsed result breaks the contract, or None. A result without a schemaVersion comes
    from a validator built before menger printed it; a newer major is one this agent cannot
    read; an exit code that disagrees with the tag means the process did not end the way its
    own output says."""
    if not isinstance(schema_version, str) or not schema_version:
        return ("the result has no schemaVersion: the validator image predates menger's result "
                "contract; rebuild it: menger/docker/scene-validator/build.sh")
    if not same_major(schema_version, RESULT_SCHEMA_VERSION):
        return (f"result schema {schema_version}, but this agent reads "
                f"{_major(RESULT_SCHEMA_VERSION)}.x")
    expected = EXIT_CODES.get(wire_tag)
    if expected is not None and expected != returncode:
        return f"exit code {returncode} does not match tag {wire_tag!r} (expected {expected})"
    return None

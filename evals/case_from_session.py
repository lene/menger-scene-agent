"""Cuts an eval case (evals/recorded.py) out of a session directory: the accepted turns, each
with the model's responses. A session with `model-calls.jsonl` (msa#31) gives the raw
responses as recorded, every call of the turn in order -- assigned to the turn by timestamp.
An older session has only the accepted scene and its readback summary: those become the two
responses (generation, readback). Expectations are left empty, to be written from the
request: what the scene must have, not what the model happened to write.

Usage: uv run python -m evals.case_from_session <session-dir> <case-id> [first-last] > case.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _turn_range(spec: Optional[str]) -> Tuple[int, int]:
    if spec is None:
        return 1, 10**9
    first, _, last = spec.partition("-")
    return int(first), int(last or first)


def case_from_session(session_dir: Path, case_id: str, turns: Optional[str] = None) -> dict:
    history = _read_jsonl(session_dir / "history.jsonl")
    calls = _read_jsonl(session_dir / "model-calls.jsonl")
    first, last = _turn_range(turns)
    out = []
    previous_ts = ""
    for entry in history:
        ts = entry.get("ts", "")
        if entry["outcome"] == "accepted" and first <= entry["ordinal"] <= last:
            if calls:
                responses = [
                    c.get("response", "") for c in calls if previous_ts < c["ts"] <= ts
                ]
            else:
                scene = (session_dir / entry["file"]).read_text(encoding="utf-8")
                responses = [f"```scala\n{scene}```", entry.get("readback_summary", "")]
            out.append({"prompt": entry["prompt"], "responses": responses, "expect": {}})
        previous_ts = ts or previous_ts
    return {"id": case_id, "source": f"{session_dir.name} turns {turns or 'all'}", "turns": out}


if __name__ == "__main__":
    session, case_id, *rest = sys.argv[1:]
    print(json.dumps(case_from_session(Path(session), case_id, rest[0] if rest else None),
                     indent=2))

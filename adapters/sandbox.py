"""The sandbox of a session (menger-scene-agent#58): whether the launchers are attested or the
user chose to run on the host, what the attestations said, and whether an image changed since
the last session. Its record goes into `session.json`.

THREAT MODEL: the launcher paths come from the user's own environment. All of this guards against
a wrong or stale configuration, which is the realistic failure, not against someone who controls
the user's shell (see `adapters/attestation.py`).

`MENGER_AGENT_UNSANDBOXED=1` is the one explicit opt-out for people who knowingly run on the
host. It covers both launchers (the validator compiles the scene, and compiling Scala runs code),
prints a warning at every start and is recorded in `session.json`. It replaces
`MENGER_AGENT_HOST_RENDER=1`, which named only the render window; the old name is rejected with a
hint.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional

from adapters.attestation import Attestation

UNSANDBOXED_ENV = "MENGER_AGENT_UNSANDBOXED"
LEGACY_HOST_RENDER_ENV = "MENGER_AGENT_HOST_RENDER"
SESSION_FILENAME = "session.json"

UNSANDBOXED_WARNING = (
    f"WARNING: {UNSANDBOXED_ENV}=1 -- the validator and the render window run generated scene "
    "code outside a verified sandbox: as you, possibly with network access. Nothing attests "
    "their launchers."
)


class SandboxConfigError(Exception):
    """A configuration the agent refuses to start with, with what to do about it."""


def unsandboxed_requested(environ: Mapping[str, str]) -> bool:
    if environ.get(LEGACY_HOST_RENDER_ENV):
        raise SandboxConfigError(
            f"{LEGACY_HOST_RENDER_ENV} is no longer read. To run generated scenes on this host, "
            f"unsandboxed, as you and possibly with network access, set {UNSANDBOXED_ENV}=1: it "
            "covers the validator as well as the render window. Otherwise unset it."
        )
    return environ.get(UNSANDBOXED_ENV) == "1"


def sandbox_record(
    unsandboxed: bool, attestations: Mapping[str, Attestation], digest_changes: List[dict]
) -> dict:
    """What `session.json` keeps of the sandbox (`sandbox` for a new session, one more entry of
    `sandbox_runs` for each resume): the attestations of this start, or that there
    were none because the user opted out; and the images that differ from the last session's."""
    return {
        "unsandboxed": unsandboxed,
        "attestations": {name: a.as_dict() for name, a in attestations.items()},
        "digest_changes": digest_changes,
    }


def _recorded_digests(session_json: dict) -> Dict[str, str]:
    """Per launcher, the digest of the latest run that attested it: the creation record, then
    each resumed run's (`sandbox_runs`), later ones overriding. A run that opted out attested
    nothing and leaves the earlier digests as they were."""
    runs = [session_json.get("sandbox") or {}, *(session_json.get("sandbox_runs") or [])]
    digests: Dict[str, str] = {}
    for run in runs:
        for name, a in ((run or {}).get("attestations") or {}).items():
            if isinstance(a, dict) and isinstance(a.get("image_digest"), str):
                digests[name] = a["image_digest"]
    return digests


def _read_session_json(session_dir: Path) -> Optional[dict]:
    try:
        data = json.loads((session_dir / SESSION_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def reference_session(base_dir: Path, resumed_id: Optional[str]) -> Optional[tuple]:
    """(session id, its session.json) a start is compared with: the session being resumed, or
    else the most recently created one. None when there is none."""
    if resumed_id is not None:
        data = _read_session_json(base_dir / resumed_id)
        return (resumed_id, data) if data is not None else None
    try:
        candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    except OSError:
        return None
    sessions = [(p.name, _read_session_json(p)) for p in candidates]
    dated = [(str(d.get("created", "")), name, d) for name, d in sessions if d is not None]
    if not dated:
        return None
    _, name, data = max(dated, key=lambda entry: entry[:2])
    return name, data


def digest_changes(
    reference: Optional[tuple], attestations: Mapping[str, Attestation]
) -> List[dict]:
    """The launchers whose image digest differs from the one the reference session recorded.
    Images are rebuilt often in development, so this is a warning and a record, not a refusal."""
    if reference is None:
        return []
    session_id, data = reference
    before = _recorded_digests(data)
    return [
        {"launcher": name, "was": before[name], "now": a.image_digest,
         "image": a.image, "since_session": session_id}
        for name, a in attestations.items()
        if name in before and before[name] != a.image_digest
    ]


def digest_warning(change: dict) -> str:
    return (f"warning: the {change['launcher']} image {change['image']} is not the one of session "
            f"{change['since_session']} ({change['was'][:19]} then, {change['now'][:19]} now)")

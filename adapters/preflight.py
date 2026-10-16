"""`menger-scene-agent --check`: does this agent work with the menger it is about to run
(review 2026-10-14, msa#32)? The agent is paired with menger through the validator and
render-window images and the two launcher scripts, never as a library. This checks each part
and says what to do when one does not fit:

- each image exists, is built from a supported menger version (`org.opencontainers.image.
  version` >= MIN_MENGER_VERSION) and from the DSL this agent's manifest describes
  (`menger.dsl-content-revision` == the reference manifest's `contentRevision`);
- each launcher script exists and is executable.

The manifest's own schema version is checked at every start (`validate_artifacts`).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.vocabulary import DSL_REVISION

# The first menger release with the agent's process contract (README: "menger 0.9.0 or later").
MIN_MENGER_VERSION = "0.9.0"
VALIDATOR_IMAGE = "menger-scene-validator:latest"  # run-sandboxed.sh's default
RENDER_IMAGE_ENV = "MENGER_RENDER_IMAGE"
RENDER_IMAGE = "menger-render-window:latest"  # run-render-window.sh's default
VERSION_LABEL = "org.opencontainers.image.version"
DSL_REVISION_LABEL = "menger.dsl-content-revision"
BUILD_HINT = "build them from a matching menger checkout: menger/docker/scene-validator/build.sh"

_DOCKER_TIMEOUT_SECONDS = 5


def image_labels(image: str) -> Optional[Dict[str, str]]:
    """The image's id and labels, or None when docker or the image is not there."""
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", image, "--format",
             "{{json .Config.Labels}} {{.Id}}"],
            capture_output=True, text=True, timeout=_DOCKER_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    labels_json, _, image_id = out.stdout.strip().rpartition(" ")
    try:
        labels = json.loads(labels_json) or {}
    except ValueError:
        return None
    return {**labels, "id": image_id}


def _version_tuple(version: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in version.split("-")[0].split(".") if part.isdigit())


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def check_image(name: str, image: str) -> Check:
    labels = image_labels(image)
    if labels is None:
        return Check(name, False, f"{image} not found (or docker unavailable); {BUILD_HINT}")
    version = labels.get(VERSION_LABEL, "")
    revision = labels.get(DSL_REVISION_LABEL, "")
    if not version or _version_tuple(version) < _version_tuple(MIN_MENGER_VERSION):
        return Check(name, False, f"{image} is menger {version or '<unlabelled>'}, needs >= "
                     f"{MIN_MENGER_VERSION}; {BUILD_HINT}")
    if revision != DSL_REVISION:
        return Check(name, False, f"{image} (menger {version}) was built from another DSL "
                     f"(revision {revision[:12] or '<none>'}, this agent: {DSL_REVISION[:12]}); "
                     f"{BUILD_HINT}, or use the agent release that matches it")
    return Check(name, True, f"{image}: menger {version}, DSL {revision[:12]}")


def check_launcher(name: str, path: Optional[str]) -> Check:
    if not path:
        return Check(name, False, "not configured")
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return Check(name, False, f"{path} is not an executable file")
    return Check(name, True, path)


def run_checks(validator_script: Optional[str], render_launcher: Optional[str]) -> List[Check]:
    return [
        check_launcher("validator script", validator_script),
        check_launcher("render launcher", render_launcher),
        check_image("validator image", VALIDATOR_IMAGE),
        check_image("render image", os.environ.get(RENDER_IMAGE_ENV) or RENDER_IMAGE),
    ]


def report(checks: List[Check]) -> str:
    return "\n".join(f"{'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}" for c in checks)

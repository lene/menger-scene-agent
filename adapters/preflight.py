"""`menger-scene-agent --check`: does this agent work with the menger it is about to run
(review 2026-10-14, msa#32)? The agent is paired with menger through the validator and
render-window images and the two launcher scripts, never as a library. This checks each part
and says what to do when one does not fit:

- each image exists, is built from a supported menger version (`org.opencontainers.image.
  version` >= MIN_MENGER_VERSION) and from the DSL this agent's manifest describes
  (`menger.dsl-content-revision` == the reference manifest's `contentRevision`);
- each launcher script exists, is executable and attests, with `--attest`, that it runs its
  container with no network and a read-only root (`adapters/attestation.py`, msa#58); the
  images checked are the ones the launchers attest, not a name assumed here.

This runs at every start, not only under `--check`: a failed check refuses to start, unless
`MENGER_AGENT_UNSANDBOXED=1` (`adapters/sandbox.py`), which skips the attestation and the images.

The manifest's own schema version is checked at every start (`validate_artifacts`).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from adapters.attestation import Attestation, AttestationFailure, attest_launcher
from core.vocabulary import DSL_REVISION

# The first menger release with the agent's process contract and DSL manifest 2.0.0 (README:
# "menger 0.9.1 or later"); 0.9.0 was released prematurely and has neither.
MIN_MENGER_VERSION = "0.9.1"
VALIDATOR_IMAGE = "menger-scene-validator:latest"  # run-sandboxed.sh's default
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


@dataclass(frozen=True)
class PreflightResult:
    checks: List[Check]
    attestations: Dict[str, Attestation]  # by launcher name ("validator", "render")

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


VALIDATOR = "validator"
RENDER = "render"
UNSANDBOXED_NOTE = (
    "UNSANDBOXED: launchers are not attested and images not checked; generated scene code runs "
    "wherever the launchers run it"
)


def _check_sandboxed_launcher(name: str, path: Optional[str]) -> Tuple[List[Check], Optional[Attestation]]:
    """The launcher's file check, its attestation, and the check of the image it attests."""
    exists = check_launcher(f"{name} launcher", path)
    if not exists.ok:
        return [exists], None
    attested = attest_launcher(str(path))
    if isinstance(attested, AttestationFailure):
        return [exists, Check(f"{name} sandbox", False, f"{path}: {attested.message}")], None
    sandbox = Check(
        f"{name} sandbox", True,
        f"{attested.sandbox}, network {attested.network}, root {attested.rootfs}, "
        f"{attested.image} {attested.image_digest[:19]}",
    )
    return [exists, sandbox, check_image(f"{name} image", attested.image)], attested


def run_checks(
    validator_script: Optional[str], render_launcher: Optional[str], unsandboxed: bool = False
) -> PreflightResult:
    if unsandboxed:
        return PreflightResult(
            [check_launcher("validator launcher", validator_script),
             check_launcher("render launcher", render_launcher),
             Check("sandbox", True, UNSANDBOXED_NOTE)],
            {},
        )
    checks: List[Check] = []
    attestations: Dict[str, Attestation] = {}
    for name, path in ((VALIDATOR, validator_script), (RENDER, render_launcher)):
        launcher_checks, attested = _check_sandboxed_launcher(name, path)
        checks += launcher_checks
        if attested is not None:
            attestations[name] = attested
    return PreflightResult(checks, attestations)


def report(checks: List[Check]) -> str:
    return "\n".join(f"{'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}" for c in checks)

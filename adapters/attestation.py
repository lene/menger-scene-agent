"""A launcher's self-attestation (menger-scene-agent#58, menger's contracts/process-result.json,
`launcherAttestation`): `<launcher> --attest` prints one JSON line saying how the launcher runs
its container, built from the very argument list it passes to `docker run`. The agent asserts
that line instead of trusting a launcher by its file name, as it did.

THREAT MODEL. The launcher paths come from the user's own environment. This guards against a
wrong or stale configuration -- a script that runs generated scene code on the host, a launcher
older than the flag, a renamed copy -- which is the realistic failure. It is not a defence
against someone who controls the user's shell or the menger checkout: a script that prints a
false attestation passes. What holds a hostile *scene* in is the container the attested
launcher starts (no network, read-only root, no capabilities), not this check.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Union

from adapters.child_env import child_environment
from adapters.process_contract import LAUNCHER_ATTESTATION_REQUIRED, single_document

ATTEST_FLAG = "--attest"
# docker asks the daemon for the image id, which is quick; a launcher that takes longer is not
# one the agent should wait for at every start.
_ATTEST_TIMEOUT_SECONDS = 30
_PREVIEW_LENGTH = 300

# What a launcher must attest: the contract's `launcherAttestation.required`.
REQUIRED_VALUES: Dict[str, str] = LAUNCHER_ATTESTATION_REQUIRED
_DIGEST_PREFIX = "sha256:"
_STRING_FIELDS = ("sandbox", "network", "rootfs", "image", "image_digest")


@dataclass(frozen=True)
class Attestation:
    sandbox: str
    network: str
    rootfs: str
    image: str
    image_digest: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AttestationFailure:
    """Why a launcher is not accepted, naming the field when one is wrong."""

    message: str


def attest_launcher(launcher: Union[str, Path]) -> Union[Attestation, AttestationFailure]:
    """Runs `<launcher> --attest` and returns what it attests, or why it is refused. Never
    raises for an expected outcome."""
    try:
        completed = subprocess.run(
            [str(launcher), ATTEST_FLAG], capture_output=True, text=True,
            timeout=_ATTEST_TIMEOUT_SECONDS, env=child_environment(), check=False,
        )
    except subprocess.TimeoutExpired:
        return AttestationFailure(f"{launcher} {ATTEST_FLAG} did not finish within "
                                  f"{_ATTEST_TIMEOUT_SECONDS} s")
    except (OSError, UnicodeDecodeError) as e:
        return AttestationFailure(f"could not run {launcher} {ATTEST_FLAG}: {e}")
    return parse_attestation(completed.stdout, completed.returncode)


def parse_attestation(stdout: str, returncode: int) -> Union[Attestation, AttestationFailure]:
    if returncode != 0:
        return AttestationFailure(
            f"the launcher does not attest a sandbox: `--attest` exited {returncode}"
            f"{_what_it_printed(stdout)}"
        )
    try:
        document = single_document(stdout)
    except ValueError as e:
        return AttestationFailure(
            f"the launcher does not attest a sandbox: `--attest` did not print exactly one JSON "
            f"line ({e}){_what_it_printed(stdout)}"
        )
    if not isinstance(document, dict):
        return AttestationFailure("the launcher does not attest a sandbox: its `--attest` "
                                  f"output is not a JSON object{_what_it_printed(stdout)}")
    return _check_fields(document)


def _check_fields(document: dict) -> Union[Attestation, AttestationFailure]:
    for name in _STRING_FIELDS:
        value = document.get(name)
        if not isinstance(value, str) or not value:
            return AttestationFailure(
                f"attestation field '{name}' is missing or not a non-empty string "
                f"(got {value!r}){_what_it_printed(_dump(document))}"
            )
    for name, required in REQUIRED_VALUES.items():
        if document[name] != required:
            return AttestationFailure(
                f"attestation field '{name}' is {document[name]!r}, must be {required!r}"
            )
    if not document["image_digest"].startswith(_DIGEST_PREFIX):
        return AttestationFailure(
            f"attestation field 'image_digest' is {document['image_digest']!r}, "
            f"must start with {_DIGEST_PREFIX!r}"
        )
    return Attestation(**{name: document[name] for name in _STRING_FIELDS})


def _dump(document: dict) -> str:
    return repr(document)


def _what_it_printed(stdout: str) -> str:
    preview = stdout.strip()[:_PREVIEW_LENGTH]
    return f" -- it printed {preview!r}" if preview else " -- it printed nothing"

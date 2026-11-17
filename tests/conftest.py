"""Every test runs with the real validator adapter wired into core.turn, as cli.py's main does
(core never imports it, msa#34); a test that fakes the validator replaces it as before.

The startup preflight (msa#58) looks at launcher files, runs `<launcher> --attest` and asks docker
for images. Tests of `cli.main` get an attested sandbox and fitting images instead, through the
three functions of `adapters.preflight` that touch the outside world; a test that wants the
real ones (`test_preflight.py`, `test_attestation.py`) is marked `real_preflight`, and one that
wants a failing preflight replaces the function itself."""

from __future__ import annotations

import pytest

import core.turn
from adapters import preflight
from adapters.attestation import Attestation
from adapters.scene_validator import validate_scene
from core.vocabulary import DSL_REVISION

FAKE_DIGEST = "sha256:" + "a" * 64


def attested(image: str = "menger-scene-validator:latest", digest: str = FAKE_DIGEST) -> Attestation:
    return Attestation("docker", "none", "read-only", image, digest)


@pytest.fixture(autouse=True)
def _real_validator_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core.turn, "validate_scene", validate_scene)


@pytest.fixture(autouse=True)
def _sandbox_that_attests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("real_preflight"):
        return
    labels = {preflight.VERSION_LABEL: "0.9.1", preflight.DSL_REVISION_LABEL: DSL_REVISION,
              "id": "x"}
    monkeypatch.setattr(preflight, "check_launcher",
                        lambda name, path: preflight.Check(name, True, str(path)))
    monkeypatch.setattr(preflight, "attest_launcher", lambda launcher: attested())
    monkeypatch.setattr(preflight, "image_labels", lambda image: labels)

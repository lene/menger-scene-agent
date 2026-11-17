"""`--check` (msa#32): does the installed menger fit this agent? Docker is faked."""

from __future__ import annotations

import json

import pytest

import cli
from adapters import preflight
from core.vocabulary import DSL_REVISION

# The real file check, attestation and image check: the autouse sandbox of conftest.py is off.
pytestmark = pytest.mark.real_preflight

ATTESTATION = {"sandbox": "docker", "network": "none", "rootfs": "read-only",
               "image": "menger-scene-validator:latest", "image_digest": "sha256:" + "e" * 64}


def _launcher(path, attestation=ATTESTATION):
    body = f"echo '{json.dumps(attestation)}'" if attestation is not None else ":"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return str(path)


def _labels(version="0.9.1", revision=DSL_REVISION):
    return {preflight.VERSION_LABEL: version, preflight.DSL_REVISION_LABEL: revision, "id": "x"}


@pytest.mark.parametrize("labels, ok, says", [
    (_labels(), True, "menger 0.9.1"),
    (_labels(version="0.10.2"), True, "menger 0.10.2"),
    (_labels(version="0.9.0"), False, "needs >= 0.9.1"),
    (_labels(version="0.8.1"), False, "needs >= 0.9.1"),
    (_labels(version=""), False, "<unlabelled>"),
    (_labels(revision="0" * 64), False, "another DSL"),
    (None, False, "not found"),
])
def test_image_check(monkeypatch, labels, ok, says):
    monkeypatch.setattr(preflight, "image_labels", lambda image: labels)
    check = preflight.check_image("validator image", "img:1")
    assert check.ok is ok
    assert says in check.detail
    if not ok:
        assert "build.sh" in check.detail  # always says what to do


def test_launcher_check(tmp_path):
    script = tmp_path / "run.sh"
    assert not preflight.check_launcher("x", None).ok
    assert not preflight.check_launcher("x", str(script)).ok
    script.write_text("#!/bin/sh\n")
    assert not preflight.check_launcher("x", str(script)).ok  # not executable
    script.chmod(0o755)
    assert preflight.check_launcher("x", str(script)).ok


def test_cli_check_exits_with_the_verdict(monkeypatch, tmp_path, capsys):
    script = _launcher(tmp_path / "run.sh")
    monkeypatch.delenv("MENGER_AGENT_UNSANDBOXED", raising=False)
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", script)
    monkeypatch.setenv("MENGER_RENDER_LAUNCHER", script)
    monkeypatch.setattr(preflight, "image_labels", lambda image: _labels())
    assert cli.main(["--check"]) == 0
    monkeypatch.setattr(preflight, "image_labels", lambda image: None)
    assert cli.main(["--check"]) == 1
    assert "FAIL validator image" in capsys.readouterr().out


def test_the_images_checked_are_the_ones_the_launchers_attest(monkeypatch, tmp_path):
    validator = _launcher(tmp_path / "v.sh", {**ATTESTATION, "image": "mine/validator:7"})
    render = _launcher(tmp_path / "r.sh", {**ATTESTATION, "image": "mine/window:7"})
    asked = []
    monkeypatch.setattr(preflight, "image_labels", lambda image: asked.append(image) or _labels())

    result = preflight.run_checks(validator, render)

    assert result.ok
    assert asked == ["mine/validator:7", "mine/window:7"]
    assert set(result.attestations) == {"validator", "render"}


def test_a_launcher_that_attests_nothing_fails_the_check_whatever_its_name(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "image_labels", lambda image: _labels())
    render = _launcher(tmp_path / "run-render-window.sh", attestation=None)

    result = preflight.run_checks(_launcher(tmp_path / "v.sh"), render)

    assert not result.ok
    (failed,) = [c for c in result.checks if not c.ok]
    assert failed.name == "render sandbox" and "does not attest" in failed.detail
    assert "render" not in result.attestations


def test_a_launcher_that_is_not_there_is_not_asked_to_attest(tmp_path):
    result = preflight.run_checks(str(tmp_path / "missing.sh"), None)

    assert not result.ok
    assert result.attestations == {}


def test_unsandboxed_checks_the_launcher_files_only(tmp_path):
    result = preflight.run_checks(
        _launcher(tmp_path / "a", attestation=None), _launcher(tmp_path / "b", attestation=None),
        unsandboxed=True,
    )

    assert result.ok and result.attestations == {}
    assert "UNSANDBOXED" in result.checks[-1].detail


def test_image_labels_without_docker(monkeypatch):
    def _no_docker(*a, **k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(preflight.subprocess, "run", _no_docker)
    assert preflight.image_labels("img:1") is None

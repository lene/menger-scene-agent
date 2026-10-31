"""`--check` (msa#32): does the installed menger fit this agent? Docker is faked."""

from __future__ import annotations

import pytest

import cli
from adapters import preflight
from core.vocabulary import DSL_REVISION


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
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setenv("MENGER_SCENE_VALIDATOR_SCRIPT", str(script))
    monkeypatch.setenv("MENGER_RENDER_LAUNCHER", str(script))
    monkeypatch.setattr(preflight, "image_labels", lambda image: _labels())
    assert cli.main(["--check"]) == 0
    monkeypatch.setattr(preflight, "image_labels", lambda image: None)
    assert cli.main(["--check"]) == 1
    assert "FAIL validator image" in capsys.readouterr().out


def test_image_labels_without_docker(monkeypatch):
    def _no_docker(*a, **k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(preflight.subprocess, "run", _no_docker)
    assert preflight.image_labels("img:1") is None

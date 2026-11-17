"""A launcher's `--attest` line (msa#58). The launchers are real scripts here, run for real: a
launcher is accepted for what it prints, never for what it is called."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from adapters import attestation
from adapters.attestation import Attestation, AttestationFailure, attest_launcher, parse_attestation

GOOD = {
    "sandbox": "docker", "network": "none", "rootfs": "read-only",
    "image": "menger-scene-validator:latest", "image_digest": "sha256:" + "d" * 64,
}


def _script(path: Path, body: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return str(path)


def _printing(path: Path, text: str, exit_code: int = 0) -> str:
    return _script(path, f"cat <<'EOF'\n{text}\nEOF\nexit {exit_code}")


def test_a_script_named_like_the_contained_launcher_that_prints_no_attestation_is_refused(tmp_path):
    result = attest_launcher(_script(tmp_path / "docker/scene-validator/run-render-window.sh", ":"))

    assert isinstance(result, AttestationFailure)
    assert "does not attest" in result.message


def test_a_script_of_any_name_that_attests_correctly_is_accepted(tmp_path):
    result = attest_launcher(_printing(tmp_path / "whatever", json.dumps(GOOD)))

    assert result == Attestation(**GOOD)


def test_the_launcher_is_called_with_attest_and_nothing_else(tmp_path):
    launcher = _script(
        tmp_path / "strict.sh",
        f"[ \"$*\" = \"--attest\" ] || exit 9\necho '{json.dumps(GOOD)}'",
    )

    assert isinstance(attest_launcher(launcher), Attestation)


@pytest.mark.parametrize("field, value", [
    ("network", "bridge"), ("network", "host"), ("rootfs", "writable"), ("sandbox", "none"),
    ("sandbox", "podman"),
])
def test_an_attestation_that_is_not_the_required_one_is_refused_naming_the_field(field, value):
    result = parse_attestation(json.dumps({**GOOD, field: value}), 0)

    assert isinstance(result, AttestationFailure)
    assert f"'{field}'" in result.message and value in result.message


@pytest.mark.parametrize("field", ["sandbox", "network", "rootfs", "image", "image_digest"])
def test_an_attestation_without_a_field_is_refused_naming_it(field):
    document = {k: v for k, v in GOOD.items() if k != field}

    result = parse_attestation(json.dumps(document), 0)

    assert isinstance(result, AttestationFailure)
    assert f"'{field}'" in result.message


def test_an_empty_or_non_string_digest_is_refused_naming_the_field():
    for digest in ("", None, 5):
        result = parse_attestation(json.dumps({**GOOD, "image_digest": digest}), 0)
        assert isinstance(result, AttestationFailure) and "'image_digest'" in result.message


def test_a_digest_that_is_not_a_sha256_is_refused():
    result = parse_attestation(json.dumps({**GOOD, "image_digest": "latest"}), 0)

    assert isinstance(result, AttestationFailure)
    assert "'image_digest'" in result.message and "sha256:" in result.message


def test_two_documents_are_refused_even_when_the_first_is_good():
    result = parse_attestation(json.dumps(GOOD) + "\n" + json.dumps({**GOOD, "network": "x"}), 0)

    assert isinstance(result, AttestationFailure)


def test_a_key_given_twice_is_refused_not_resolved_to_the_last_one():
    text = json.dumps(GOOD).replace('"network": "none"', '"network": "bridge", "network": "none"')

    assert isinstance(parse_attestation(text, 0), AttestationFailure)


@pytest.mark.parametrize("text", ["", "not json", "[]", '"docker"', json.dumps(GOOD) + " trailing"])
def test_anything_that_is_not_one_json_object_is_refused(text):
    assert isinstance(parse_attestation(text, 0), AttestationFailure)


def test_a_launcher_that_attests_and_fails_is_refused():
    result = parse_attestation(json.dumps(GOOD), 2)

    assert isinstance(result, AttestationFailure) and "exited 2" in result.message


def test_a_refused_result_instead_of_an_attestation_is_shown_in_the_message(tmp_path):
    refused = {"tag": "refused", "messages": ["cannot attest: image x not found"]}
    result = attest_launcher(_printing(tmp_path / "l.sh", json.dumps(refused), exit_code=2))

    assert isinstance(result, AttestationFailure)
    assert "image x not found" in result.message


def test_a_launcher_that_is_not_there_is_refused_not_raised(tmp_path):
    result = attest_launcher(str(tmp_path / "missing.sh"))

    assert isinstance(result, AttestationFailure)


def test_a_launcher_that_hangs_is_refused_after_the_timeout(monkeypatch):
    def _hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(attestation.subprocess, "run", _hang)

    result = attest_launcher("/x/launcher.sh")

    assert isinstance(result, AttestationFailure) and "did not finish" in result.message


def test_the_launcher_does_not_get_the_agents_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    launcher = _script(
        tmp_path / "env.sh",
        f"[ -z \"$ANTHROPIC_API_KEY\" ] || exit 9\necho '{json.dumps(GOOD)}'",
    )

    assert isinstance(attest_launcher(launcher), Attestation)

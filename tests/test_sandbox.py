"""The session's sandbox record (msa#58): the opt-out, the old name's rejection, and the image
digest compared with the last session's."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.attestation import Attestation
from adapters.sandbox import (
    SandboxConfigError,
    digest_changes,
    digest_warning,
    reference_session,
    sandbox_record,
    unsandboxed_requested,
)


def _attestation(digest: str, image: str = "img:1") -> Attestation:
    return Attestation("docker", "none", "read-only", image, digest)


def _session(base: Path, name: str, created: str, digests: dict) -> None:
    (base / name).mkdir(parents=True)
    attestations = {k: _attestation(v).as_dict() for k, v in digests.items()}
    (base / name / "session.json").write_text(
        json.dumps({"created": created, "sandbox": {"attestations": attestations}})
    )


def test_only_the_new_name_set_to_one_opts_out():
    assert unsandboxed_requested({"MENGER_AGENT_UNSANDBOXED": "1"}) is True
    assert unsandboxed_requested({}) is False
    assert unsandboxed_requested({"MENGER_AGENT_UNSANDBOXED": "0"}) is False


def test_the_old_name_is_rejected_with_a_hint_to_the_new_one():
    with pytest.raises(SandboxConfigError) as exc_info:
        unsandboxed_requested({"MENGER_AGENT_HOST_RENDER": "1"})

    assert "MENGER_AGENT_UNSANDBOXED=1" in str(exc_info.value)


def test_a_changed_digest_is_reported_per_launcher(tmp_path):
    _session(tmp_path, "old", "2026-11-01", {"validator": "sha256:1", "render": "sha256:2"})

    changes = digest_changes(
        reference_session(tmp_path, None),
        {"validator": _attestation("sha256:1"), "render": _attestation("sha256:3")},
    )

    assert [(c["launcher"], c["was"], c["now"], c["since_session"]) for c in changes] == [
        ("render", "sha256:2", "sha256:3", "old")
    ]
    assert "old" in digest_warning(changes[0])


def test_the_most_recently_created_session_is_the_reference(tmp_path):
    _session(tmp_path, "a", "2026-11-01", {"validator": "sha256:1"})
    _session(tmp_path, "b", "2026-11-09", {"validator": "sha256:2"})

    assert reference_session(tmp_path, None)[0] == "b"


def test_a_resumed_session_is_compared_with_itself_not_the_newest(tmp_path):
    _session(tmp_path, "a", "2026-11-01", {"validator": "sha256:1"})
    _session(tmp_path, "b", "2026-11-09", {"validator": "sha256:2"})

    assert reference_session(tmp_path, "a")[0] == "a"


def test_no_earlier_session_or_an_unreadable_one_gives_no_change(tmp_path):
    assert digest_changes(reference_session(tmp_path / "missing", None), {"validator": _attestation("x")}) == []
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "session.json").write_text("{not json")
    assert reference_session(tmp_path, None) is None


def test_an_earlier_session_that_was_unsandboxed_has_nothing_to_compare(tmp_path):
    (tmp_path / "u").mkdir()
    (tmp_path / "u" / "session.json").write_text(json.dumps(
        {"created": "2026-11-01", "sandbox": {"unsandboxed": True, "attestations": {}}}))

    assert digest_changes(reference_session(tmp_path, None), {"validator": _attestation("s")}) == []


def test_the_record_holds_the_opt_out_the_attestations_and_the_changes():
    record = sandbox_record(False, {"validator": _attestation("sha256:1")}, [{"x": 1}])

    assert record == {
        "unsandboxed": False,
        "attestations": {"validator": _attestation("sha256:1").as_dict()},
        "digest_changes": [{"x": 1}],
    }

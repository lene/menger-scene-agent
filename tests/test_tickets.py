"""Unit tests for adapters/tickets.py -- the ticket sink adapter. Filesystem-only (tmp_path),
no network access.

Covers every row of the story's I/O & Edge-Case Matrix: first request for a capability,
repeat request overwriting the existing draft, two distinct capabilities producing two
distinct files, a missing drafts_dir, an empty/whitespace-only missing_capability, and a
path-hostile missing_capability never escaping drafts_dir."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from adapters.tickets import write_draft
from core.types import TicketError

PROMPT = "render a torus knot with a checkerboard material"


# --- First request for a capability --------------------------------------------------------


def test_first_request_writes_a_new_draft_file(tmp_path):
    result = write_draft("torus knot geometry", PROMPT, tmp_path)

    assert isinstance(result, str)
    assert Path(result).is_file()


def test_returned_path_points_at_a_file_under_drafts_dir(tmp_path):
    result = write_draft("torus knot geometry", PROMPT, tmp_path)

    assert isinstance(result, str)
    path = Path(result)
    assert path.is_file()
    assert path.parent == tmp_path


def test_draft_contains_the_missing_capability_and_the_triggering_prompt(tmp_path):
    result = write_draft("torus knot geometry", PROMPT, tmp_path)

    text = Path(result).read_text(encoding="utf-8")
    assert "torus knot geometry" in text
    assert PROMPT in text


# --- Repeat request, same capability: overwrite not duplicate (AD-17) ----------------------


def test_repeat_request_same_capability_overwrites_rather_than_duplicates(tmp_path):
    first_result = write_draft("torus knot geometry", "first prompt", tmp_path)
    second_result = write_draft("torus knot geometry", "second prompt", tmp_path)

    assert first_result == second_result
    assert len(list(tmp_path.iterdir())) == 1


def test_repeat_request_overwritten_file_has_the_new_content(tmp_path):
    write_draft("torus knot geometry", "first prompt", tmp_path)
    result = write_draft("torus knot geometry", "second prompt", tmp_path)

    text = Path(result).read_text(encoding="utf-8")
    assert "second prompt" in text
    assert "first prompt" not in text


def test_repeat_request_differing_only_by_surrounding_whitespace_overwrites_not_duplicates(
    tmp_path,
):
    # Regression test: the digest used to be computed from the raw, unstripped
    # missing_capability while the stem was computed from the stripped text, so
    # "torus knot geometry" and " torus knot geometry " produced the same stem but a
    # different digest -- two files instead of one overwrite.
    first_result = write_draft("torus knot geometry", "first prompt", tmp_path)
    second_result = write_draft(" torus knot geometry ", "second prompt", tmp_path)

    assert first_result == second_result
    assert len(list(tmp_path.iterdir())) == 1
    text = Path(second_result).read_text(encoding="utf-8")
    assert "second prompt" in text
    assert "first prompt" not in text


# --- Two different capabilities: no collision -----------------------------------------------


def test_two_distinct_capabilities_produce_two_distinct_files(tmp_path):
    first_result = write_draft("torus knot geometry", PROMPT, tmp_path)
    second_result = write_draft("volumetric fog", PROMPT, tmp_path)

    assert first_result != second_result
    assert len(list(tmp_path.iterdir())) == 2


def test_two_capabilities_that_sanitize_to_the_same_stem_still_produce_distinct_files(tmp_path):
    # "a/b" and "a b" both collapse to the same sanitized stem under naive slugification --
    # the deterministic digest suffix is what keeps them from colliding.
    first_result = write_draft("a/b", PROMPT, tmp_path)
    second_result = write_draft("a b", PROMPT, tmp_path)

    assert first_result != second_result
    assert len(list(tmp_path.iterdir())) == 2


# --- drafts_dir missing ----------------------------------------------------------------------


def test_missing_drafts_dir_returns_a_typed_error_not_an_exception(tmp_path):
    missing = tmp_path / "does-not-exist"

    result = write_draft("torus knot geometry", PROMPT, missing)

    assert isinstance(result, TicketError)
    assert result.kind == "drafts_dir_unavailable"


def test_missing_drafts_dir_error_names_the_path(tmp_path):
    missing = tmp_path / "does-not-exist"

    result = write_draft("torus knot geometry", PROMPT, missing)

    assert isinstance(result, TicketError)
    assert str(missing) in result.message


def test_missing_drafts_dir_writes_no_file_anywhere_under_tmp_path(tmp_path):
    missing = tmp_path / "does-not-exist"

    write_draft("torus knot geometry", PROMPT, missing)

    assert list(tmp_path.rglob("*")) == []


def test_drafts_dir_that_is_a_file_not_a_directory_returns_drafts_dir_unavailable(tmp_path):
    not_a_dir = tmp_path / "drafts-is-a-file"
    not_a_dir.write_text("not a directory")

    result = write_draft("torus knot geometry", PROMPT, not_a_dir)

    assert isinstance(result, TicketError)
    assert result.kind == "drafts_dir_unavailable"


# --- drafts_dir exists but the write itself fails (permission, TOCTOU, disk full) ----------


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores directory write permission bits"
)
def test_unwritable_drafts_dir_returns_a_typed_write_failed_error_not_an_exception(tmp_path):
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    # Read + execute, no write: mkstemp() inside write_draft() must fail with PermissionError
    # rather than escaping the function, per its "never raises" contract.
    os.chmod(drafts_dir, stat.S_IRUSR | stat.S_IXUSR)

    try:
        result = write_draft("torus knot geometry", PROMPT, drafts_dir)
    finally:
        # Restore write permission so pytest can clean up tmp_path afterwards.
        os.chmod(drafts_dir, stat.S_IRWXU)

    assert isinstance(result, TicketError)
    assert result.kind == "write_failed"


# --- Empty/whitespace-only missing_capability -------------------------------------------------


def test_empty_missing_capability_returns_a_typed_error_not_an_exception(tmp_path):
    result = write_draft("", PROMPT, tmp_path)

    assert isinstance(result, TicketError)
    assert result.kind == "invalid_capability_text"


def test_whitespace_only_missing_capability_returns_a_typed_error(tmp_path):
    result = write_draft("   \n\t  ", PROMPT, tmp_path)

    assert isinstance(result, TicketError)
    assert result.kind == "invalid_capability_text"


def test_empty_missing_capability_writes_no_file(tmp_path):
    write_draft("", PROMPT, tmp_path)

    assert list(tmp_path.iterdir()) == []


# --- Path-hostile missing_capability: never escapes drafts_dir -----------------------------


def test_path_traversal_capability_never_escapes_drafts_dir(tmp_path):
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()

    result = write_draft("../../etc/passwd", PROMPT, drafts_dir)

    assert isinstance(result, str)
    path = Path(result)
    assert path.parent == drafts_dir
    assert path.is_relative_to(drafts_dir)
    # Nothing was written outside drafts_dir.
    assert list(tmp_path.iterdir()) == [drafts_dir]


def test_capability_with_embedded_slash_never_escapes_drafts_dir(tmp_path):
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()

    result = write_draft("a/b", PROMPT, drafts_dir)

    path = Path(result)
    assert path.parent == drafts_dir
    assert list(tmp_path.iterdir()) == [drafts_dir]

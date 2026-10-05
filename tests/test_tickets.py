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


def test_repeat_request_keeps_every_use_case_in_the_one_draft(tmp_path):
    # Session 3, F79: the draft collects every request that hit the capability (it used to
    # keep only the last).
    write_draft("torus knot geometry", "first prompt", tmp_path)
    result = write_draft("torus knot geometry", "second prompt", tmp_path)

    text = Path(result).read_text(encoding="utf-8")
    assert "- first prompt\n- second prompt\n" in text


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
    assert "- first prompt\n- second prompt\n" in text


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


def test_draft_names_the_nearest_possible_alternative_when_given(tmp_path):
    # Usability review 2026-09, session 2 (F49, msa#17): the decline's "nearest" belongs in
    # the ticket, so whoever files it sees what the user was offered instead.
    path = write_draft("glow around objects", "give it a glowing halo", tmp_path,
                       nearest="an emissive surface")

    text = Path(path).read_text(encoding="utf-8")
    assert "## Nearest possible today" in text
    assert "an emissive surface" in text


# --- Usability review 2026-10, session 3 (F79): titles, de-dupe, use cases, session -------

# The three declines session 3 turned into three drafts of one capability.
_GLOW_DECLINES = (
    "no glow or halo around an object, and no emission that falls off with distance",
    "there is no glow or halo effect in this DSL",
    "no glow or halo exists",
)


def test_the_title_names_the_capability_as_a_request_not_the_decline(tmp_path):
    result = write_draft(_GLOW_DECLINES[0], PROMPT, tmp_path)

    first_line = Path(result).read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "# Support glow or halo around an object"


def test_differently_worded_declines_of_one_capability_share_one_draft(tmp_path):
    paths = {
        write_draft(decline, f"prompt {i}", tmp_path, session_id=f"s{i}")
        for i, decline in enumerate(_GLOW_DECLINES)
    }

    assert len(paths) == 1
    text = Path(paths.pop()).read_text(encoding="utf-8")
    for i in range(3):
        assert f"- prompt {i} (session s{i})" in text


def test_unrelated_capabilities_stay_separate(tmp_path):
    write_draft("no camera shake", PROMPT, tmp_path)
    write_draft("no orbiting camera path", PROMPT, tmp_path)

    assert len(list(tmp_path.glob("*.md"))) == 2


def test_a_draft_says_when_it_is_done(tmp_path):
    text = Path(write_draft("no 5D penteract", PROMPT, tmp_path)).read_text(encoding="utf-8")

    assert "## Done when" in text

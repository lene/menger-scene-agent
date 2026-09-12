"""Unit tests for adapters/scene_store.py -- the session/scene-store adapter. Filesystem-only
(tmp_path), no network access, no real session-workspace paths.

Covers every row of the story's I/O & Edge-Case Matrix: first accepted turn, second accepted
turn (first file untouched), a rejected attempt, session-already-exists failing exclusivity,
reading the current scene with zero turns, and restoring an earlier ordinal without touching
later ones."""

from __future__ import annotations

import json
import threading

import pytest

from adapters.scene_store import SceneStore, SceneStoreError

# --- Session creation: exclusivity (AD-15) -------------------------------------------------


def test_create_session_makes_a_directory_under_base_path(tmp_path):
    store = SceneStore.create_session(tmp_path, slug="my-scene")

    assert store.session_dir.is_dir()
    assert store.session_dir.parent == tmp_path


def test_create_session_id_is_not_the_slug_alone(tmp_path):
    # AD-15: "A slug alone is not an identity" -- the directory name must carry more than
    # the human-chosen slug verbatim.
    store = SceneStore.create_session(tmp_path, slug="my-scene")

    assert store.session_dir.name != "my-scene"
    assert store.session_dir.name.startswith("my-scene")


def test_two_create_session_calls_with_the_same_slug_produce_different_sessions(tmp_path):
    first = SceneStore.create_session(tmp_path, slug="my-scene")
    second = SceneStore.create_session(tmp_path, slug="my-scene")

    assert first.session_dir != second.session_dir
    assert first.session_dir.is_dir()
    assert second.session_dir.is_dir()


def test_create_session_with_an_already_existing_session_id_fails(tmp_path):
    SceneStore.create_session(tmp_path, session_id="fixed-id")

    with pytest.raises(SceneStoreError, match="already exists"):
        SceneStore.create_session(tmp_path, session_id="fixed-id")


def test_create_session_failure_does_not_silently_reuse_the_existing_directory(tmp_path):
    original = SceneStore.create_session(tmp_path, session_id="fixed-id")
    original.accept("object A:\n  val scene = Scene()\n", "first prompt")

    with pytest.raises(SceneStoreError):
        SceneStore.create_session(tmp_path, session_id="fixed-id")

    # The original session's data must be untouched by the failed second attempt.
    assert original.current_scene() == "object A:\n  val scene = Scene()\n"


# --- current_scene() with zero accepted turns ----------------------------------------------


def test_current_scene_is_none_for_a_fresh_session(tmp_path):
    store = SceneStore.create_session(tmp_path)

    assert store.current_scene() is None


# --- current_scene_path() (spec-ai-scene-agent story 21) -----------------------------------


def test_current_scene_path_is_none_for_a_fresh_session(tmp_path):
    store = SceneStore.create_session(tmp_path)

    assert store.current_scene_path() is None


def test_current_scene_path_after_accept_returns_the_correct_ordinal_path(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")

    path = store.current_scene_path()

    assert path == store.session_dir / "001.scala"
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == "object First:\n  val scene = Scene()\n"


def test_current_scene_path_after_second_accept_returns_the_newest_ordinal_path(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")
    store.accept("object First:\n  val scene = Scene(objects = List(Sphere()))\n", "add a sphere")

    assert store.current_scene_path() == store.session_dir / "002.scala"


# --- First accepted turn --------------------------------------------------------------------


def test_accept_first_turn_writes_001_scala(tmp_path):
    store = SceneStore.create_session(tmp_path)

    ordinal = store.accept("object First:\n  val scene = Scene()\n", "make a scene")

    assert ordinal == 1
    written = store.session_dir / "001.scala"
    assert written.is_file()
    assert written.read_text(encoding="utf-8") == "object First:\n  val scene = Scene()\n"


def test_accept_first_turn_records_one_accepted_history_entry_naming_ordinal_1(tmp_path):
    store = SceneStore.create_session(tmp_path)

    store.accept("object First:\n  val scene = Scene()\n", "make a scene")

    lines = store.history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["ordinal"] == 1
    assert entry["outcome"] == "accepted"
    assert entry["prompt"] == "make a scene"
    # accept()'s docstring: when readback_summary is None, the key is left out of the
    # entry entirely rather than written as `null`.
    assert "readback_summary" not in entry


def test_current_scene_after_first_accept_returns_its_text(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")

    assert store.current_scene() == "object First:\n  val scene = Scene()\n"


# --- Second accepted turn: first file untouched ----------------------------------------------


def test_accept_second_turn_writes_002_scala_and_leaves_001_byte_identical(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")
    first_path = store.session_dir / "001.scala"
    first_bytes_before = first_path.read_bytes()

    ordinal = store.accept("object First:\n  val scene = Scene(objects = List(Sphere()))\n", "add a sphere")

    assert ordinal == 2
    assert (store.session_dir / "002.scala").is_file()
    assert first_path.read_bytes() == first_bytes_before


def test_accept_second_turn_grows_history_to_two_entries(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")
    store.accept("object First:\n  val scene = Scene(objects = List(Sphere()))\n", "add a sphere")

    lines = store.history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    second_entry = json.loads(lines[1])
    assert second_entry["ordinal"] == 2
    assert second_entry["outcome"] == "accepted"


def test_current_scene_after_second_accept_returns_the_newest_text(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")
    store.accept("object First:\n  val scene = Scene(objects = List(Sphere()))\n", "add a sphere")

    assert store.current_scene() == "object First:\n  val scene = Scene(objects = List(Sphere()))\n"


# --- Rejected attempt ------------------------------------------------------------------------


def test_record_rejected_writes_no_scene_file(tmp_path):
    store = SceneStore.create_session(tmp_path)

    store.record_rejected("do something impossible", "model refused")

    assert list(store.session_dir.glob("*.scala")) == []


def test_record_rejected_does_not_consume_an_ordinal(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")

    store.record_rejected("do something impossible", "model refused")
    next_ordinal = store.accept("object Second:\n  val scene = Scene()\n", "try again")

    # The rejected attempt must not have consumed ordinal 2 -- the next accepted attempt
    # after it is still ordinal 2, not 3.
    assert next_ordinal == 2


def test_record_rejected_still_appends_a_history_entry(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")

    store.record_rejected("do something impossible", "model refused")

    lines = store.history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rejected_entry = json.loads(lines[1])
    assert rejected_entry["outcome"] == "rejected"
    assert rejected_entry["ordinal"] is None
    assert rejected_entry["file"] is None
    assert rejected_entry["prompt"] == "do something impossible"
    assert rejected_entry["reason"] == "model refused"


def test_record_rejected_on_a_fresh_session_still_grows_history_by_one(tmp_path):
    store = SceneStore.create_session(tmp_path)

    store.record_rejected("bad prompt", "invalid output")

    lines = store.history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert store.current_scene() is None


# --- record_consult() (spec-ai-scene-agent story 19) ---------------------------------------


def test_record_consult_with_an_answer_writes_no_scene_file_and_consumes_no_ordinal(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")

    store.record_consult("where should the light go?", answer="upper-left, warm color")
    next_ordinal = store.accept("object Second:\n  val scene = Scene()\n", "try again")

    scala_files = sorted(p.name for p in store.session_dir.glob("*.scala"))
    assert scala_files == ["001.scala", "002.scala"]
    # The consult turn must not have consumed ordinal 2 -- the next accepted attempt after
    # it is still ordinal 2, not 3.
    assert next_ordinal == 2


def test_record_consult_with_an_answer_appends_a_consult_history_entry(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object First:\n  val scene = Scene()\n", "make a scene")

    store.record_consult("where should the light go?", answer="upper-left, warm color")

    lines = store.history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[1])
    assert entry["outcome"] == "consult"
    assert entry["ordinal"] is None
    assert entry["file"] is None
    assert entry["prompt"] == "where should the light go?"
    assert entry["answer"] == "upper-left, warm color"
    assert "error" not in entry


def test_record_consult_with_an_error_appends_a_consult_history_entry(tmp_path):
    store = SceneStore.create_session(tmp_path)

    store.record_consult("where should the light go?", error="model call failed")

    lines = store.history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["outcome"] == "consult"
    assert entry["ordinal"] is None
    assert entry["file"] is None
    assert entry["prompt"] == "where should the light go?"
    assert entry["error"] == "model call failed"
    assert "answer" not in entry
    assert store.current_scene() is None


def test_record_consult_on_a_fresh_session_still_grows_history_by_one(tmp_path):
    store = SceneStore.create_session(tmp_path)

    store.record_consult("a question", answer="an answer")

    lines = store.history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert store.current_scene() is None


def test_record_consult_rejects_both_answer_and_error(tmp_path):
    # edge-case-hunter review round: the docstring's "exactly one of answer/error" contract
    # is now enforced, not just documented.
    store = SceneStore.create_session(tmp_path)

    with pytest.raises(ValueError):
        store.record_consult("a question", answer="an answer", error="an error")


def test_record_consult_rejects_neither_answer_nor_error(tmp_path):
    store = SceneStore.create_session(tmp_path)

    with pytest.raises(ValueError):
        store.record_consult("a question")


# --- history.jsonl append-only shape ----------------------------------------------------------


def test_history_jsonl_is_one_json_object_per_line_never_rewritten(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object A:\n  val scene = Scene()\n", "turn 1")
    history_after_first = store.history_path.read_text(encoding="utf-8")

    store.accept("object A:\n  val scene = Scene(objects = List(Sphere()))\n", "turn 2")
    history_after_second = store.history_path.read_text(encoding="utf-8")

    # Never rewritten: the bytes recorded after turn 1 are an unmodified prefix of the
    # history after turn 2 -- appending only ever adds, never edits, existing lines.
    assert history_after_second.startswith(history_after_first)
    lines = history_after_second.splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)  # each line parses standalone as one JSON object
        assert "ordinal" in parsed
        assert "prompt" in parsed
        assert "outcome" in parsed


# --- Restoring an earlier ordinal -------------------------------------------------------------


def test_read_ordinal_returns_the_requested_ordinals_real_content(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object A:\n  val scene = Scene()\n", "turn 1")
    store.accept("object A:\n  val scene = Scene(objects = List(Sphere()))\n", "turn 2")
    store.accept("object A:\n  val scene = Scene(objects = List(Sphere(), Cube()))\n", "turn 3")

    assert store.read_ordinal(1) == "object A:\n  val scene = Scene()\n"
    assert store.read_ordinal(2) == "object A:\n  val scene = Scene(objects = List(Sphere()))\n"
    assert store.read_ordinal(3) == "object A:\n  val scene = Scene(objects = List(Sphere(), Cube()))\n"


def test_read_ordinal_does_not_touch_later_ordinals(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object A:\n  val scene = Scene()\n", "turn 1")
    store.accept("object A:\n  val scene = Scene(objects = List(Sphere()))\n", "turn 2")
    store.accept("object A:\n  val scene = Scene(objects = List(Sphere(), Cube()))\n", "turn 3")
    third_path = store.session_dir / "003.scala"
    third_bytes_before = third_path.read_bytes()

    store.read_ordinal(1)

    assert third_path.read_bytes() == third_bytes_before


def test_read_ordinal_raises_for_a_nonexistent_ordinal(tmp_path):
    store = SceneStore.create_session(tmp_path)
    store.accept("object A:\n  val scene = Scene()\n", "turn 1")

    with pytest.raises(SceneStoreError, match="does not exist"):
        store.read_ordinal(2)


def test_restoring_an_earlier_ordinal_and_using_it_as_the_next_prior_scene(tmp_path):
    # Restoring is reading an old ordinal for use as the next turn's prior_scene -- it never
    # un-writes a later ordinal (Boundaries & Constraints).
    store = SceneStore.create_session(tmp_path)
    store.accept("object A:\n  val scene = Scene()\n", "turn 1")
    store.accept("object A:\n  val scene = Scene(objects = List(Sphere()))\n", "turn 2")

    restored_prior_scene = store.read_ordinal(1)
    new_ordinal = store.accept(restored_prior_scene + "// derived from turn 1\n", "restore and refine")

    assert new_ordinal == 3
    assert store.read_ordinal(2) == "object A:\n  val scene = Scene(objects = List(Sphere()))\n"
    assert store.read_ordinal(3) == "object A:\n  val scene = Scene()\n// derived from turn 1\n"


# --- Concurrent accept() calls -- review round finding ---------------------------------------
#
# An earlier version computed the next ordinal, then wrote via `os.replace()`, which
# succeeds silently even when the target already exists. Two concurrent `accept()` calls
# (or two `SceneStore` instances pointed at the same directory) computing the same "next
# ordinal" meant the second writer's `os.replace()` silently discarded the first writer's
# content, with no exception and no on-disk trace beyond a duplicate-ordinal history.jsonl
# pair -- reproduced live during review with two barrier-synced threads. Fixed with an
# exclusive `os.link()`-based write plus a retry loop in `accept()`.


def test_concurrent_accepts_never_lose_a_turn_or_duplicate_an_ordinal(tmp_path):
    store = SceneStore.create_session(tmp_path)
    barrier = threading.Barrier(2)
    results: list = []
    errors: list = []

    def worker(label: str) -> None:
        try:
            # Both threads read the same "no turns yet" state, then race to write --
            # exactly the interleaving that used to let one writer silently vanish.
            barrier.wait(timeout=5)
            ordinal = store.accept(f"object {label}:\n  val scene = Scene()\n", f"prompt {label}")
            results.append((label, ordinal))
        except Exception as e:  # noqa: BLE001 -- captured for the assertion below, not swallowed
            errors.append(e)

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"Unexpected errors from concurrent accept() calls: {errors}"
    assert len(results) == 2

    # Both turns must have been assigned DIFFERENT ordinals -- neither is silently dropped.
    ordinals_assigned = sorted(ordinal for _, ordinal in results)
    assert ordinals_assigned == [1, 2]

    # Both scene files must exist with the correct, distinguishable content -- no clobbering.
    scene_files = sorted(store.session_dir.glob("*.scala"))
    assert len(scene_files) == 2
    contents = {p.read_text(encoding="utf-8") for p in scene_files}
    assert contents == {
        "object A:\n  val scene = Scene()\n",
        "object B:\n  val scene = Scene()\n",
    }

    # history.jsonl must have exactly two accepted entries with distinct, non-null ordinals.
    lines = store.history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    assert {e["ordinal"] for e in entries} == {1, 2}
    assert all(e["outcome"] == "accepted" for e in entries)


def test_reattaching_to_an_existing_session_directory_works(tmp_path):
    original = SceneStore.create_session(tmp_path, slug="reattach-test")
    original.accept("object A:\n  val scene = Scene()\n", "turn 1")

    reattached = SceneStore(session_dir=original.session_dir)

    assert reattached.current_scene() == "object A:\n  val scene = Scene()\n"


def test_constructing_a_scene_store_around_a_nonexistent_directory_fails_fast(tmp_path):
    with pytest.raises(SceneStoreError, match="does not exist"):
        SceneStore(session_dir=tmp_path / "never-created")


def test_create_session_rejects_a_session_id_containing_path_traversal(tmp_path):
    with pytest.raises(SceneStoreError, match="Invalid session_id"):
        SceneStore.create_session(tmp_path, session_id="../escape")


def test_create_session_rejects_an_absolute_path_as_session_id(tmp_path):
    with pytest.raises(SceneStoreError, match="Invalid session_id"):
        SceneStore.create_session(tmp_path, session_id="/etc/passwd")


def test_create_session_rejects_an_empty_session_id(tmp_path):
    with pytest.raises(SceneStoreError, match="Invalid session_id"):
        SceneStore.create_session(tmp_path, session_id="")


# --- Atomicity, as far as unit-testable (AD-14) ------------------------------------------------


def test_accept_leaves_no_temp_file_behind_after_a_successful_write(tmp_path):
    store = SceneStore.create_session(tmp_path)

    store.accept("object A:\n  val scene = Scene()\n", "turn 1")

    remaining = list(store.session_dir.iterdir())
    names = {p.name for p in remaining}
    assert names == {"001.scala", "history.jsonl"}
    assert not any(name.endswith(".tmp") for name in names)


def test_read_ordinal_never_sees_a_half_written_file_for_a_completed_accept(tmp_path):
    # The atomic write path (temp-name-then-rename) means a completed accept() call is only
    # ever observable as the single, complete final file -- there is no window in which a
    # reader (via read_ordinal/current_scene) could observe a partial one.
    store = SceneStore.create_session(tmp_path)
    text = "object A:\n" + ("  // padding line\n" * 200) + "  val scene = Scene()\n"

    store.accept(text, "turn 1")

    assert store.read_ordinal(1) == text

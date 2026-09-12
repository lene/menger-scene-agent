"""Persistent REPL session (spec-ai-scene-agent story 11): the first CLI entry point in this
repo. `core/turn.py`'s `run_turn()` (story 20) closes the whole generate->gauntlet->render
pipeline, but nothing invoked it from a shell -- this module is that shell.

Bootstraps a session (fresh via `SceneStore.create_session()`, or resumed via direct
`SceneStore(session_dir=...)` construction -- its own documented pattern, Code Map), loads
`reference/dsl-manifest.json`/`dsl-corpus.json` (resolved relative to this script's own
location, not CWD -- AD-9, this package's own build-time artifacts), constructs a model
adapter via `adapters.model_factory.get_model_adapter()`, then loops: read a line, call
`run_turn()` for plain text, print "Turn N: <outcome>" reusing `TurnResult.ordinal`/`.tag`
(Boundaries & Constraints: "Every run_turn() outcome prints its ordinal-or-none and tag,
never silently"). A `/ask`/`/question` prefix is recognized and routed to `core.consult`'s
`answer_consult()` (spec-ai-scene-agent story 19, PRD FR10) -- a consult turn never calls
`run_turn()`, never touches the scene file, and never consumes an ordinal; it is answered in
prose, printed, and logged to `history.jsonl` via `SceneStore.record_consult()` on both
success and failure.

`script_path` (AD-10, injected, never discovered) comes from a required env var,
`MENGER_SCENE_VALIDATOR_SCRIPT` -- unset is a clear startup error before any turn runs or
model adapter is constructed, mirroring `MissingAPIKeyError`'s construction-time-failure
pattern (Design Notes). The sessions base directory is `MENGER_AGENT_SESSIONS_DIR`,
defaulting to `./sessions` (relative to CWD) when unset -- a documented default, not a
silent filesystem search (Design Notes).

spec-ai-scene-agent story 21 ("render window refresh wiring"): after every `"accepted"`
`TurnResult`, `adapters.render_window.refresh_render_window()` is called with the
just-accepted file's path (`SceneStore.current_scene_path()`) and the previously tracked
`render_process`, updating `render_process` from the outcome -- the new handle on success,
`None` on failure (the adapter terminates `previous_process` unconditionally before
attempting the new launch, so the old handle is invalid either way once the call returns).
`launcher_path` comes from a required env var, `MENGER_RENDER_LAUNCHER` (AD-10, mirrors
`MENGER_SCENE_VALIDATOR_SCRIPT`'s existing pattern), checked alongside it before session
bootstrap.

spec-ai-scene-agent story 14 ("staleness pre-flight"): right after the manifest/corpus are
loaded, `core/generation.py`'s `validate_artifacts()` (public per this story -- a rename of
the private `_validate_artifacts`, no logic change) is called once here. A non-`None`
result exits with a clear message naming the stale artifact, before the model adapter is
constructed or a session directory is created -- the same schema-version check
`generate()`/`revise()` already ran on the first turn, just moved earlier so staleness is
caught at startup instead of after the first prompt.

spec-ai-scene-agent story 17 ("hand-edit fallback"): `core/turn.py`'s `check_hand_edit()` is
called once per loop iteration, right after `input()` and before `/ask` routing or
`run_turn()` dispatch. It compares the current on-disk scene against the loop's tracked
`prior_scene` (no filesystem watch, just a fresh re-read each iteration); a mismatch means
the user hand-edited the scene file directly. A clean edit is promoted to a new ordinal and
treated exactly like any other accepted turn -- `prior_scene` advances and the render window
refreshes via the same `_accept_and_refresh_render()` helper `run_turn()`'s own accepted
path uses (factored out, story 17, rather than duplicated). A dirty edit is reported and
changes nothing: `prior_scene` stays put and the render window is untouched. Either way, the
loop still goes on to process the line the user actually typed this iteration -- an
accepted hand edit and a "real" prompt can both be handled in the same iteration, the latter
seeing the former's freshly-promoted content as its `prior_scene`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from adapters.artifacts import ArtifactError, load_corpus, load_manifest
from adapters.model import MissingAPIKeyError, UnknownProviderError
from adapters.model_factory import get_model_adapter
from adapters.render_window import refresh_render_window
from adapters.scene_store import SceneStore, SceneStoreError
from core.consult import answer_consult
from core.generation import validate_artifacts
from core.turn import check_hand_edit, run_turn
from core.types import ConsultError, RenderWindowOutcome, RenderWindowResult, TurnResult

# AD-9: this package's own build-time artifacts, resolved relative to this script's own
# location (never CWD) -- same reasoning `adapters/artifacts.py`'s docstring already gives
# for owning this I/O (Design Notes: "Manifest/corpus paths are not injected").
_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _SCRIPT_DIR / "reference" / "dsl-manifest.json"
_CORPUS_PATH = _SCRIPT_DIR / "reference" / "dsl-corpus.json"

# AD-10: `script_path` names a path inside the sibling `menger` repo whose exact layout
# varies by checkout -- guessing it would violate AD-10 ("paths injected, never
# discovered"), so it is a required env var with no default (Design Notes).
_SCRIPT_PATH_ENV = "MENGER_SCENE_VALIDATOR_SCRIPT"

# New convention (no prior precedent) -- a documented default, not a silent filesystem
# search, so it doesn't violate AD-10's spirit the way guessing `script_path` would.
_SESSIONS_DIR_ENV = "MENGER_AGENT_SESSIONS_DIR"
_DEFAULT_SESSIONS_DIR = "./sessions"

# story 21: `launcher_path` names the staged `menger-app` binary (sibling repo, deployment-
# varying location) -- same AD-10 reasoning as `_SCRIPT_PATH_ENV` above, so it's a required
# env var with no default, checked alongside it before session bootstrap.
_RENDER_LAUNCHER_ENV = "MENGER_RENDER_LAUNCHER"

# Recognized verbatim as the first whitespace-separated token of a line -- story 11's
# original recognition logic, unchanged by story 19 (Design Notes: "story 19 replaces the
# stub's body, not this recognition logic").
_CONSULT_PREFIXES = ("/ask", "/question")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Persistent REPL session for the Menger AI scene agent.",
        epilog=(
            "Environment variables:\n"
            f"  {_SCRIPT_PATH_ENV}   required; path to the renderer-side scene validator "
            "script (menger repo, AD-10).\n"
            f"  {_RENDER_LAUNCHER_ENV}   required; path to the staged menger-app render "
            "window launcher binary (menger repo, AD-10).\n"
            f"  {_SESSIONS_DIR_ENV}  optional; sessions base directory (default: "
            f"'{_DEFAULT_SESSIONS_DIR}').\n"
            "\n"
            "Hand-editing is supported: you may edit the current scene file on disk "
            "directly between turns. Each loop iteration checks for such an edit (no model "
            "call) and either promotes it to a new ordinal, reports the lint violation that "
            "blocked it, or reports a storage failure if promoting a clean edit itself could "
            "not be persisted.\n"
            "\n"
            "/ask <question> or /question <question>: a consult turn -- a domain question "
            "answered in prose, grounded in the DSL manifest/corpus and the current scene. "
            "Never edits the scene, never gauntlet-checked, never consumes a turn ordinal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session",
        dest="session_id",
        default=None,
        help=(
            "Resume an existing session by id (its directory name under the sessions "
            "base dir). Omit to start a fresh session."
        ),
    )
    return parser.parse_args(argv)


def _require_script_path() -> str:
    """Reads `MENGER_SCENE_VALIDATOR_SCRIPT`. Missing is a clear startup error -- exits
    before any turn runs, before session bootstrap, and before a model adapter is
    constructed (mirrors `MissingAPIKeyError`'s construction-time-failure pattern)."""
    script_path = os.environ.get(_SCRIPT_PATH_ENV)
    if script_path is not None:
        script_path = script_path.strip()
    if not script_path:
        raise SystemExit(
            f"error: environment variable {_SCRIPT_PATH_ENV} is not set. It must name the "
            "renderer-side scene validator script (menger repo, AD-10: paths are injected, "
            "never discovered). Set it before starting the REPL."
        )
    return script_path


def _require_render_launcher() -> str:
    """Reads `MENGER_RENDER_LAUNCHER`. Missing is a clear startup error -- exits before any
    turn runs, before session bootstrap, and before a model adapter is constructed, checked
    alongside `_require_script_path()` (mirrors that function's pattern exactly, AD-10)."""
    launcher_path = os.environ.get(_RENDER_LAUNCHER_ENV)
    if launcher_path is not None:
        launcher_path = launcher_path.strip()
    if not launcher_path:
        raise SystemExit(
            f"error: environment variable {_RENDER_LAUNCHER_ENV} is not set. It must name "
            "the staged menger-app render window launcher binary (menger repo, AD-10: "
            "paths are injected, never discovered). Set it before starting the REPL."
        )
    return launcher_path


def _sessions_base_dir() -> Path:
    return Path(os.environ.get(_SESSIONS_DIR_ENV) or _DEFAULT_SESSIONS_DIR)


def _validate_session_id(session_id: Optional[str]) -> None:
    """Validates `--session` before any `SceneStore` construction is attempted (two
    independent reviewers flagged the unsanitized `base_dir / session_id` join as a
    path-escape risk): rejects an empty/whitespace-only value, an absolute path, and a
    value containing a `..` path-traversal segment -- each as a clear startup `SystemExit`
    naming the specific problem."""
    if session_id is None:
        return
    if not session_id.strip():
        raise SystemExit("error: --session must not be empty or whitespace-only.")
    candidate = Path(session_id)
    if candidate.is_absolute():
        raise SystemExit(
            f"error: --session {session_id!r} must not be an absolute path."
        )
    if ".." in candidate.parts:
        raise SystemExit(
            f"error: --session {session_id!r} must not contain '..' path-traversal "
            "segments."
        )


def _bootstrap_session(session_id: Optional[str]) -> SceneStore:
    """Fresh (`session_id is None`): `SceneStore.create_session()`. Resume: direct
    `SceneStore(session_dir=...)` construction against `base_dir/session_id` -- this
    module's own documented pattern (Code Map), no separate "open" factory exists. A
    resume id that doesn't name a real session directory is a clear startup error naming
    the id, never a raw stack trace. `session_id` is validated by `_validate_session_id()`
    before this is ever called."""
    base_dir = _sessions_base_dir()
    if session_id is None:
        return SceneStore.create_session(base_dir)

    session_dir = base_dir / session_id
    if not session_dir.is_dir():
        raise SystemExit(
            f"error: session '{session_id}' does not exist under '{base_dir}'. Check the "
            "--session id, or omit --session to start a fresh one."
        )
    return SceneStore(session_dir=session_dir)


def _format_history_entry(entry: dict) -> str:
    # spec-ai-scene-agent story 19 (review round, patch-level fix): a "consult" entry has
    # neither a "reason" key (that's rejected/failed-attempt shaped) nor an ordinal -- without
    # this branch, replaying a resumed session's prior /ask/-question turns collapsed to the
    # bare, content-free "Turn None: consult", silently dropping the recorded prompt/answer/
    # error that history.jsonl actually preserved.
    if entry.get("outcome") == "consult":
        prompt = entry.get("prompt")
        detail = entry.get("answer") if "answer" in entry else entry.get("error")
        return f"Consult: {prompt!r} -> {detail}"
    text = f"Turn {entry.get('ordinal')}: {entry.get('outcome')}"
    reason = entry.get("reason")
    if reason:
        text += f" - {reason}"
    return text


def _replay_history(store: SceneStore) -> None:
    """Prints one line per existing `history.jsonl` entry, in order, before the loop
    accepts new input (Code Map: "read via plain json.loads per line, no existing helper
    to reuse"). A line that fails to parse as JSON, or that parses to something other than
    a JSON object, is skipped with a one-line warning on stderr rather than crashing
    resume with a raw traceback -- `history.jsonl` is append-only (AD-7) so a later valid
    entry should still replay even if an earlier line is damaged."""
    if not store.history_path.is_file():
        return
    with open(store.history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"warning: skipping malformed history.jsonl line: {line!r}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(entry, dict):
                print(
                    "warning: skipping history.jsonl line that is not a JSON object "
                    f"(got {type(entry).__name__}): {line!r}",
                    file=sys.stderr,
                )
                continue
            print(_format_history_entry(entry))


def _format_turn_result(result: TurnResult) -> str:
    # Boundaries & Constraints: "Every run_turn() outcome prints its ordinal-or-none and
    # tag, never silently" -- `result.ordinal` is `None` on any non-accepted tag, printed
    # as-is. `result.messages` is always populated on a non-accepted tag (never empty on
    # `accepted`, per TurnResult's own docstring), so it's safe to fold in unconditionally.
    text = f"Turn {result.ordinal}: {result.tag}"
    if result.messages:
        text += " - " + "; ".join(result.messages)
    return text


def _format_render_outcome(outcome: RenderWindowOutcome) -> str:
    # story 21: "print a one-line status" -- success carries nothing else to report (the
    # live Popen handle isn't printable status), failure names its typed kind/message so a
    # refused/failed launch is never silent.
    if isinstance(outcome, RenderWindowResult):
        return "Render window: refreshed"
    return f"Render window: {outcome.kind} - {outcome.message}"


def _format_hand_edit_result(result: TurnResult) -> str:
    # Labeled distinctly from _format_turn_result -- this event isn't tied to a prompt the
    # user typed this iteration the way a normal "Turn N" line is (story 17). Shape is
    # "Hand edit: tag [(ordinal N)] [- messages]": the tag, then an "(ordinal N)" segment
    # when the edit was promoted (never present on a non-accepted tag, since `ordinal` is
    # None there), then any messages -- one extra segment `_format_turn_result` has no
    # equivalent for, since a normal turn's ordinal is already folded into its own "Turn N:"
    # prefix instead (review round, patch-level fix: previously documented as simply
    # mirroring `_format_turn_result`'s "tag [- messages]" shape, which omitted this segment).
    text = f"Hand edit: {result.tag}"
    if result.ordinal is not None:
        text += f" (ordinal {result.ordinal})"
    if result.messages:
        text += " - " + "; ".join(result.messages)
    return text


def _accept_and_refresh_render(
    store: SceneStore,
    render_launcher_path: str,
    render_process: Optional[subprocess.Popen[str]],
) -> Optional[subprocess.Popen[str]]:
    """Shared post-acceptance helper (story 17 -- factored out of the inline block story 21
    introduced, so the hand-edit-accepted path and the normal run_turn()-accepted path both
    call it instead of duplicating the refresh_render_window()/render_process-update dance).
    Called after ANY accepted TurnResult (from run_turn() or from check_hand_edit()); prints
    the one-line render status and returns the render_process value the caller should track
    from here on -- the new handle on success, `None` on failure (story 21: the adapter
    terminates `previous_process` unconditionally before attempting the new launch, so the
    old handle is invalid either way once this returns).

    Guarded exactly like the call site it replaces: current_scene_path() does a fresh
    filesystem scan (AD-13) and can theoretically raise SceneStoreError, and
    refresh_render_window's own contract doesn't cover exceptions raised before it's even
    called -- a single turn's post-processing must not be able to kill the whole REPL any
    more than run_turn() itself can."""
    try:
        scene_path = store.current_scene_path()
        if scene_path is None:
            # Should never happen: an "accepted" tag is only ever returned after
            # store.accept() has already written the ordinal this reads back -- but
            # current_scene_path()'s own return type is Optional, so this is handled rather
            # than silently trusted.
            raise RuntimeError(
                "accepted turn but current_scene_path() returned None -- this should be "
                "impossible; SceneStore/run_turn's accept-then-report invariant may be "
                "broken"
            )
        render_outcome = refresh_render_window(
            scene_path,
            render_launcher_path,
            previous_process=render_process,
        )
        print(_format_render_outcome(render_outcome))
        return (
            render_outcome.process
            if isinstance(render_outcome, RenderWindowResult)
            else None
        )
    except Exception as e:  # noqa: BLE001 -- a render-refresh failure must not kill the REPL
        print(f"Render window: error - {e}")
        return None


def _print_stage(stage: str) -> None:
    # spec-ai-scene-agent story 12 ("live status line"): a multi-second turn (a real model
    # call, later a sandboxed subprocess) prints nothing until fully done without this --
    # PRD FR3. Sequential printed lines, no carriage-return overwrite (Design Notes: exact
    # rendering format explicitly left open by the PRD, this is a documented implementation
    # choice, not a human-values judgment). Printed to stderr (not stdout), flushed
    # explicitly (patch-level fix, post-review): ephemeral progress output belongs on
    # stderr, keeping stdout clean for the actual "Turn N: tag" result line -- and under any
    # buffered/piped stdout, unflushed progress output would silently defeat the entire
    # point of a live status line (it would just look hung).
    print(f"... {stage}", file=sys.stderr, flush=True)


def _is_consult_input(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    first_word = stripped.split(None, 1)[0]
    return first_word in _CONSULT_PREFIXES


def _consult_question(line: str) -> str:
    """Strips the recognized `/ask`/`/question` prefix off `line`, returning the bare
    question text -- called only after `_is_consult_input(line)` is already `True`. Mirrors
    `_is_consult_input`'s own tokenization (`str.split(None, 1)`) so the two never disagree
    about where the prefix ends."""
    parts = line.strip().split(None, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _record_consult_safely(
    store: SceneStore,
    question: str,
    *,
    answer: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Attempts `store.record_consult(...)` but never lets a failure there propagate --
    guarded the same way `core/turn.py`'s `_record_rejected_safely` guards
    `store.record_rejected()` (Code Map): a `SceneStore` write failure while recording a
    consult turn is exactly as real, and as unpropagatable, as one while recording an
    accepted or rejected turn. Prints a storage-failure note; never crashes the REPL loop."""
    try:
        store.record_consult(question, answer=answer, error=error)
    except SceneStoreError as e:
        print(f"Consult history: storage_failed - could not record consult turn: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Always checked first: a missing script_path, missing render launcher path, or invalid
    # --session must exit before manifest/corpus loading, model adapter construction, or
    # session bootstrap (I/O & Edge-Case Matrix). None of these checks has a filesystem side
    # effect to clean up on failure.
    script_path = _require_script_path()
    render_launcher_path = _require_render_launcher()
    _validate_session_id(args.session_id)

    try:
        manifest = load_manifest(_MANIFEST_PATH)
    except ArtifactError as e:
        raise SystemExit(f"error: {e}") from e
    try:
        corpus = load_corpus(_CORPUS_PATH)
    except ArtifactError as e:
        raise SystemExit(f"error: {e}") from e

    # story 14 (staleness pre-flight): `core/generation.py`'s schema-version check
    # (already real, already fail-fast, already zero-model-call) used to only run
    # implicitly on the first generate()/revise() call inside a turn -- moved here so a
    # stale manifest/corpus is caught at startup, before any paid model call or session
    # directory is created (matches this function's existing anti-orphan ordering, AD-15).
    stale = validate_artifacts(manifest, corpus)
    if stale is not None:
        raise SystemExit(f"error: {stale.message}")

    try:
        adapter = get_model_adapter()
    except (MissingAPIKeyError, UnknownProviderError) as e:
        raise SystemExit(f"error: could not construct model adapter: {e}") from e

    # Session bootstrap is deliberately the LAST startup step (AD-15): everything else
    # that can fail (script_path, manifest/corpus, model adapter) has already succeeded by
    # this point, so a session directory is never created only to be orphaned -- never
    # cleaned up, never reused -- by a later startup failure.
    store = _bootstrap_session(args.session_id)

    if args.session_id is not None:
        _replay_history(store)

    # AD-13: "the current scene" is always the last file that actually exists on disk --
    # re-read here rather than assumed, matching every other reader of this value.
    prior_scene = store.current_scene()

    # story 21: tracks the render window launched for THIS session, threaded the same way
    # `prior_scene` is -- `None` until the first accepted turn, then the live `Popen` handle
    # (or `None` again on a failed refresh) after every subsequent accepted turn.
    render_process: Optional[subprocess.Popen[str]] = None

    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        # story 17: checked once per loop iteration, right after input() and before /ask
        # routing or run_turn() dispatch -- detects a hand edit made directly to the current
        # scene file since the last iteration. Guarded the same way run_turn() itself is: a
        # single bad check must not kill the REPL.
        try:
            hand_edit_result = check_hand_edit(store, prior_scene)
        except Exception as e:  # noqa: BLE001 -- a hand-edit check failure must not kill the REPL
            print(f"Hand edit check error: {e}")
            hand_edit_result = None

        if hand_edit_result is not None:
            print(_format_hand_edit_result(hand_edit_result))
            if hand_edit_result.tag == "accepted":
                # Treated exactly like a normal accepted turn -- prior_scene advances and
                # the render window refreshes via the same shared helper (Boundaries &
                # Constraints: "The render window only refreshes for a hand-edit that
                # actually passed the gate, using the same call path a normal accepted turn
                # already uses").
                prior_scene = store.current_scene()
                render_process = _accept_and_refresh_render(
                    store, render_launcher_path, render_process
                )
            # On any other tag (hand_edit_rejected, storage_failed): prior_scene/
            # render_process are left exactly as they were -- the loop still goes on to
            # process the line the user actually typed this iteration below.

        if _is_consult_input(line):
            # Never reaches run_turn() -- Boundaries & Constraints: "a /ask/-question
            # prefix never reaches run_turn()", "never calls generate()/revise(), never
            # runs any gauntlet check, and never consumes a turn ordinal" (story 19).
            question = _consult_question(line)
            if not question:
                # Review-round patch: a bare "/ask" or "/ask   " has no question text -- fail
                # fast with a clear message rather than sending an empty Question: to the
                # model, which would waste a real, paid model call on nothing.
                print("Consult error: empty question -- type a question after /ask or /question")
                continue
            consult_result = answer_consult(question, manifest, corpus, adapter, prior_scene)
            if isinstance(consult_result, ConsultError):
                print(f"Consult error: {consult_result.kind} - {consult_result.message}")
                _record_consult_safely(store, question, error=consult_result.message)
            else:
                print(consult_result)
                _record_consult_safely(store, question, answer=consult_result)
            continue

        if not line.strip():
            continue

        try:
            result = run_turn(
                line,
                prior_scene,
                manifest,
                corpus,
                adapter,
                store,
                script_path,
                on_stage=_print_stage,
            )
        except Exception as e:  # noqa: BLE001 -- a single bad turn must not kill the REPL
            print(f"Turn error: {e}")
            continue
        print(_format_turn_result(result))
        if result.tag == "accepted":
            prior_scene = store.current_scene()

            # story 21: refresh_render_window() is only ever called after an "accepted"
            # TurnResult (Boundaries & Constraints) -- never on any rejection tag. The
            # adapter terminates `render_process` unconditionally before attempting the new
            # launch, so the old handle is invalid either way once the call returns --
            # `render_process` is therefore always overwritten below, never conditionally
            # kept (Design Notes).
            render_process = _accept_and_refresh_render(
                store, render_launcher_path, render_process
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

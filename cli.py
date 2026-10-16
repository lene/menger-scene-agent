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

spec-ai-scene-agent story 22 ("interactive retry confirmation and /retry command"): PRD FR7
requires that a `generation_timeout` (story 15) never auto-retries, and that `/retry` "does
not blindly resend the last request" without either the scene having changed since the
failure or an explicit second confirmation. `main()`'s loop tracks a `_RetryState` (the
failed prompt, the scene at the moment of that failure, and whether a resend has already
been confirmed once) alongside `prior_scene`/`render_process`. The existing inline
"run_turn() -> print -> on-accept refresh" block is factored into `_execute_turn()` so both
the normal dispatch and `/retry`'s resend call the identical path (mirrors story 17's own
`_accept_and_refresh_render()` factoring). `/retry` (recognized as a dispatch branch
alongside `_is_consult_input()`, before blank-line/`run_turn()` dispatch) is handled by
`_handle_retry()`: no pending timeout reports "nothing to retry"; a changed scene resends
immediately; an unchanged scene arms a confirmation gate on the first `/retry` and resends
only on the second. `_next_retry_state()` is the single place either dispatch path
(re)arms the gate on a fresh `"generation_timeout"`. PRD FR7 narrowed retry to
`generation_timeout`; usability session 3 (F71) widened it: any other rejection arms an
ungated retry, since `/retry` is the natural word for "try that again".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import readline  # noqa: F401 -- side-effect only: wires GNU readline into input() (arrow

# keys, Home/End, Ctrl-W word-delete, in-session Up/Down history)
import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from adapters.artifacts import ArtifactError, load_corpus, load_manifest
from adapters.model import MissingAPIKeyError, UnknownProviderError
from adapters.model_factory import get_model_adapter, selected_provider
from adapters.preflight import report, run_checks
from adapters.provenance import (
    MODEL_CALLS_FILENAME,
    RecordingModelAdapter,
    collect_provenance,
    configure_session_logging,
    history_provenance,
    write_session_json,
)
from adapters.render_window import (
    close_render_window,
    crash_report,
    refresh_render_window,
    wait_for_window_event,
)
from adapters.scene_store import SceneStore
from core.consult import answer_consult
from core.generation import validate_artifacts
from core.turn import MAX_REPAIR_ROUNDS, check_hand_edit, open_items, run_turn
from repl import window
from repl.formatting import (
    _format_history_entry,
)
from repl.session import Context, Ports, run_repl
from repl.window import WindowPorts, _report_frame_build_failures

# AD-9: this package's own build-time artifacts, resolved relative to this script's own
# location (never CWD) -- same reasoning `adapters/artifacts.py`'s docstring already gives
# for owning this I/O (Design Notes: "Manifest/corpus paths are not injected").
_SCRIPT_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
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
_CONTAINED_LAUNCHER_NAME = "run-render-window.sh"
_HOST_RENDER_ENV = "MENGER_AGENT_HOST_RENDER"

def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="menger-scene-agent",
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
            "Never edits the scene, never gauntlet-checked, never consumes a turn ordinal.\n"
            "\n"
            "/retry: resends the request that most recently failed. After a rejection it "
            "resends at once. After a generation_timeout: if the scene has changed since the "
            "failure (a hand edit or another accepted turn), resends immediately; if unchanged, "
            "the first /retry explains nothing has changed and requires a second /retry (still "
            "unchanged) before actually resending -- never auto-retried, per PRD FR7. With "
            "nothing failed, reports there is nothing to retry.\n"
            "\n"
            "/open: lists every warning and Not done item that still holds for the current "
            "scene. A turn prints only the new ones and counts the rest as \"earlier ... still "
            "apply\".\n"
            "\n"
            "needs_clarification: when a turn is rejected for being too ambiguous, your "
            "next line is automatically threaded back into the original request (not sent "
            "alone) -- answer the question directly, e.g. just \"z\" for \"which axis?\". "
            "Compounds across repeated rounds; cleared as soon as a turn resolves any other "
            "way. The same applies after a rejected scene (compile_errors, lint_findings, "
            "local_finding, readback_failed): your next line is treated as a change to the "
            "rejected request, e.g. \"start at level 0 instead\", not as a new request. The "
            "prompt then reads \"(continuing) >\"; type /drop to abandon the rejected request "
            "instead.\n"
            "\n"
            "Self-repair: a scene rejected for compile_errors, lint_findings or local_finding "
            f"is sent back to the model with the error, at most {MAX_REPAIR_ROUNDS} times per "
            "turn, shown as \"... repairing (<reason>)\". Only if that fails is the rejection "
            "reported.\n"
            "\n"
            "unsupported: when a turn is rejected because the DSL genuinely cannot produce "
            "the requested effect at all (not just ambiguous), the rejection names the "
            "nearest thing that IS possible, but your next line is NOT threaded back -- "
            "nothing about a follow-up resolves \"this doesn't exist\", so it starts a fresh "
            "request. A ticket draft for the missing feature is written to the session's "
            "tickets/ directory and its path printed; filing it as an issue is a separate "
            "step outside the agent."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "check that this agent fits the installed menger -- both launcher scripts, both "
            "images' menger version and DSL revision -- print the result and exit (0: all "
            "fit, 1: something does not, with what to do)"
        ),
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
    if Path(launcher_path).name != _CONTAINED_LAUNCHER_NAME:
        # Review 2026-10-14 (menger#77): the window compiles and runs every accepted scene;
        # outside the container that is generated code running as the user, with network.
        if os.environ.get(_HOST_RENDER_ENV) != "1":
            raise SystemExit(
                f"error: {_RENDER_LAUNCHER_ENV} must name menger's contained render window, "
                f"docker/scene-validator/{_CONTAINED_LAUNCHER_NAME} (got {launcher_path}). "
                f"To run generated scenes on this host instead -- unsandboxed, as you, with "
                f"network access -- set {_HOST_RENDER_ENV}=1."
            )
        print(
            f"WARNING: {_HOST_RENDER_ENV}=1 -- the render window runs generated scene code on "
            "this host, outside the sandbox, as you, with network access."
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
            # F50: the replay shows what was asked, not only what came of it.
            if entry.get("outcome") != "consult" and entry.get("prompt"):
                print(f"> {entry['prompt']}")
            print(_format_history_entry(entry, store.session_dir))


def _window_ports() -> WindowPorts:
    """Built at call time from this module's names, the composition root's choice (tests
    replace them)."""
    return WindowPorts(refresh_render_window, wait_for_window_event, crash_report)


def _WindowWatcher(store: SceneStore, poll_seconds: float = window.WindowWatcher.POLL_SECONDS):
    return window.WindowWatcher(_window_ports(), store, poll_seconds)


def _report_render_crash(store, render_process, watcher=None):
    return window._report_render_crash(_window_ports(), store, render_process, watcher)


def _launch_window(window_path, render_launcher_path, previous_process, note=""):
    return window._launch_window(_window_ports(), window_path, render_launcher_path,
                                 previous_process, note)


def _after_reload(event, window_path, render_launcher_path, render_process):
    return window._after_reload(_window_ports(), event, window_path, render_launcher_path,
                                render_process)


def _accept_and_refresh_render(store, render_launcher_path, render_process):
    return window._accept_and_refresh_render(_window_ports(), store, render_launcher_path,
                                             render_process)


def _exit_on_termination_signals() -> None:
    """SIGHUP (terminal closed) and SIGTERM (`kill`) exit through the REPL's `finally`, which
    closes the render window; by default they ended the agent at once and left the window --
    in its own session, so untouched by the terminal's hangup -- holding the GPU (menger#104).
    Only from the main thread, where Python allows handlers (tests may call `main` elsewhere)."""
    if threading.current_thread() is not threading.main_thread():
        return

    def _exit(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, _exit)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.check:
        # msa#32: before the required-env checks below, so a fresh install can ask what is
        # missing instead of failing on the first thing.
        checks = run_checks(os.environ.get(_SCRIPT_PATH_ENV), os.environ.get(_RENDER_LAUNCHER_ENV))
        print(report(checks))
        return 0 if all(c.ok for c in checks) else 1

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
    # F50: where this session's files are, fresh or resumed. On stderr with the other status
    # lines ("... <stage>"); stdout carries the turn results.
    print(f"Session: {store.session_dir}", file=sys.stderr)
    # Which model this session talks to, so a session can be compared with others (a LiteLLM
    # proxy offers many). Adapters without the property (test fakes) print "unknown".
    model_name = getattr(adapter, "model_name", "unknown")
    print(f"Model: {selected_provider()} {model_name}", file=sys.stderr)

    # Provenance (review 2026-10-14, msa#31): session.json, agent.log, every model call, and
    # the run's identity on every history entry, so a session can be reconstructed.
    configure_session_logging(store.session_dir)
    provenance = collect_provenance(
        manifest, _MANIFEST_PATH, selected_provider(), model_name, script_path,
        render_launcher_path,
    )
    write_session_json(store.session_dir, provenance)
    store = replace(store, provenance=history_provenance(provenance))
    adapter = RecordingModelAdapter(adapter, store.session_dir / MODEL_CALLS_FILENAME)

    if args.session_id is not None:
        _replay_history(store)

    # The composition root (review 2026-10-14, msa#33/#34): what the loop runs with, and the
    # collaborators it calls, chosen here -- tests replace this module's names.
    ctx = Context(manifest, corpus, adapter, store, script_path, render_launcher_path)
    ports = Ports(
        run_turn=run_turn,
        check_hand_edit=check_hand_edit,
        answer_consult=answer_consult,
        open_items=open_items,
        accept_and_refresh_render=_accept_and_refresh_render,
        report_render_crash=_report_render_crash,
        report_frame_build_failures=_report_frame_build_failures,
        make_window_watcher=_WindowWatcher,
        close_render_window=close_render_window,
    )
    # SIGHUP and SIGTERM unwind through run_repl's `finally` too, which closes the window
    # (menger#104).
    _exit_on_termination_signals()
    return run_repl(ctx, ports)


if __name__ == "__main__":
    sys.exit(main())

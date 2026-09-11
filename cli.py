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
never silently"). A `/ask`/`/question` prefix is recognized and routed to a stub -- story 19
(Epic 3, not yet built) replaces the stub's body, not this recognition logic (Design Notes).

`script_path` (AD-10, injected, never discovered) comes from a required env var,
`MENGER_SCENE_VALIDATOR_SCRIPT` -- unset is a clear startup error before any turn runs or
model adapter is constructed, mirroring `MissingAPIKeyError`'s construction-time-failure
pattern (Design Notes). The sessions base directory is `MENGER_AGENT_SESSIONS_DIR`,
defaulting to `./sessions` (relative to CWD) when unset -- a documented default, not a
silent filesystem search (Design Notes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from adapters.artifacts import ArtifactError, load_corpus, load_manifest
from adapters.model import MissingAPIKeyError, UnknownProviderError
from adapters.model_factory import get_model_adapter
from adapters.scene_store import SceneStore
from core.turn import run_turn
from core.types import TurnResult

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

# Recognized verbatim as the first whitespace-separated token of a line -- Design Notes:
# "Recognizing the prefix and returning a clear 'not implemented' response satisfies FR2's
# routing requirement without calling into code that doesn't exist" (story 19 isn't built).
_CONSULT_PREFIXES = ("/ask", "/question")
_CONSULT_STUB_MESSAGE = (
    "Consult turns (/ask, /question) are not implemented yet -- story 19 (Epic 3) will "
    "replace this stub."
)


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Persistent REPL session for the Menger AI scene agent.",
        epilog=(
            "Environment variables:\n"
            f"  {_SCRIPT_PATH_ENV}   required; path to the renderer-side scene validator "
            "script (menger repo, AD-10).\n"
            f"  {_SESSIONS_DIR_ENV}  optional; sessions base directory (default: "
            f"'{_DEFAULT_SESSIONS_DIR}')."
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


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Always checked first: a missing script_path or invalid --session must exit before
    # manifest/corpus loading, model adapter construction, or session bootstrap (I/O &
    # Edge-Case Matrix). Neither check has a filesystem side effect to clean up on failure.
    script_path = _require_script_path()
    _validate_session_id(args.session_id)

    try:
        manifest = load_manifest(_MANIFEST_PATH)
    except ArtifactError as e:
        raise SystemExit(f"error: {e}") from e
    try:
        corpus = load_corpus(_CORPUS_PATH)
    except ArtifactError as e:
        raise SystemExit(f"error: {e}") from e

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

    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        if _is_consult_input(line):
            # Never reaches run_turn() -- Boundaries & Constraints: "a /ask/-question
            # prefix never reaches run_turn()".
            print(_CONSULT_STUB_MESSAGE)
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

    return 0


if __name__ == "__main__":
    sys.exit(main())

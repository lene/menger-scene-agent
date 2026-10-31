"""The REPL loop (review 2026-10-14, msa#33/#39/#34: split out of cli.py's `main()`).

`Context` is what a session runs with, `TurnState` what the loop carries from one line to the
next (it was four values threaded by hand through every call), `Ports` the collaborators a
turn depends on -- built by cli.py, the composition root, so they can be replaced. Slash
commands are a table of small handlers (`COMMANDS`, plus the `/ask` / `/question` consult
prefixes) instead of an inline if-ladder: a new command is one handler and one table entry.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from adapters import window_files
from adapters.render_window import (
    describe_window_state,
    read_window_state,
    render_window_scene_path,
    shown_4d_values,
)
from adapters.scene_export import ExportError
from adapters.scene_store import SceneStore, SceneStoreError, history_entry_problem
from adapters.tickets import (
    OPEN_ISSUES_FILE,
    issue_link,
    load_open_issues,
    related_issues,
    tracked_issue,
    write_draft,
)
from core.prompt_checks import cut_off_reason
from core.scene_facts import scene_duration
from core.types import ConsultError, TurnResult
from repl.formatting import (
    _format_hand_edit_result,
    _format_turn_result,
    _rejection_line,
    _wrap_for_terminal,
)
from repl.state import (
    _build_clarification_prompt,
    _ClarificationState,
    _next_clarification_state,
    _next_retry_state,
    _RetryState,
)

logger = logging.getLogger("cli")

# Recognized verbatim as the first whitespace-separated token of a line (story 11, 19).
CONSULT_PREFIXES = ("/ask", "/question")
# Takes arguments too, so it is matched by first word (msa#56).
EXPORT_COMMAND = "/export"
_EXPORT_USAGE = ("Export: /export image [t] (a PNG at t, default 0) or /export video [seconds] "
                 "(an mp4, default the scene's duration).")


@dataclass(frozen=True)
class Context:
    manifest: dict
    corpus: dict
    adapter: Any
    store: SceneStore
    script_path: str
    render_launcher_path: str


@dataclass(frozen=True)
class Ports:
    """What a turn calls that a test or another front end may replace."""

    run_turn: Callable[..., TurnResult]
    check_hand_edit: Callable[..., Optional[TurnResult]]
    answer_consult: Callable[..., Any]
    open_items: Callable[..., Tuple[list, list]]
    accept_and_refresh_render: Callable[..., Optional[subprocess.Popen[str]]]
    report_render_crash: Callable[..., Optional[subprocess.Popen[str]]]
    report_frame_build_failures: Callable[[SceneStore, int], int]
    make_window_watcher: Callable[[SceneStore], Any]
    close_render_window: Callable[[Optional[subprocess.Popen[str]]], None]
    export_scene: Callable[..., Any]


@dataclass
class TurnState:
    """`prior_scene`: the current scene (AD-13: re-read from disk, never assumed); the window
    launched for this session (story 21); a pending `/retry` (story 22); a pending
    clarification or threaded rejection (story 18, F4); how many frame failures were shown."""

    prior_scene: Optional[str]
    render_process: Optional[subprocess.Popen[str]] = None
    retry_state: Optional[_RetryState] = None
    clarification_state: Optional[_ClarificationState] = None
    frame_failures_reported: int = 0
    # The hand edit last rejected, so the idle check reports it once (msa#47).
    rejected_edit: Optional[str] = None
    # The last `/ask` (question, answer), for the next request only: "keep them like that"
    # after "/ask what are the 4D parameters now?" had no referent (session 6, msa#49).
    last_consult: Optional[Tuple[str, str]] = None


def _consult_note(consult: Tuple[str, str]) -> str:
    question, answer = consult
    return (f"The request follows this /ask, which is what words like \"them\" or \"that\" refer "
            f"to: question: {question} -- answer: {answer}")


# Once per REPL process: the validator could not evaluate a scene's values.
_SNAPSHOT_HINT_SHOWN: list[bool] = []

_FOLLOW_UP_HINT = "  Your next line continues this request; /drop abandons it."


# --- small helpers the turn uses ----------------------------------------------------------


def _window_state_text(store: SceneStore) -> Optional[str]:
    """The render window's current state as a sentence, or `None` (no window, or menger older
    than the state file; usability review 2026-10, session 4, F96)."""
    try:
        state = read_window_state(store.session_dir)
        return (describe_window_state(state, store.current_snapshots())
                if state is not None else None)
    except (KeyError, TypeError, ValueError):
        return None


def _window_state_kwargs(store: SceneStore) -> dict:
    """`{"window_state": text}` when a window state exists, else nothing."""
    text = _window_state_text(store)
    return {"window_state": text} if text is not None else {}


def _window_shown_kwargs(store: SceneStore) -> dict:
    """`{"window_shown": (t, values)}`: the 4D values the window shows at its t, which a "keep
    them like that" scene must still give there (session 6 follow-up, msa#49)."""
    try:
        state = read_window_state(store.session_dir)
        if state is None:
            return {}
        shown, _ = shown_4d_values(state, store.current_snapshots())
        return {"window_shown": (float(state.get("t") or 0.0), shown)} if shown else {}
    except (KeyError, TypeError, ValueError):
        return {}


def _format_open_items(ports: Ports, scene_text: Optional[str], manifest: dict) -> str:
    """`/open`: every warning and Not done item that holds for the current scene. Each turn
    prints only the new ones (usability review 2026-10, session 4, F91), so this is where the
    user sees them all again."""
    if scene_text is None:
        return "Open: no scene yet."
    warnings, not_done = ports.open_items(scene_text, manifest)
    if not warnings and not not_done:
        return "Open: no warnings or Not done items for the current scene."
    lines = ["Open for the current scene:"]
    lines += [f"  Warning: {w}" for w in warnings]
    lines += [f"  Not done: {part}" for part in not_done]
    return "\n".join(lines)


def _turn_paths(
    store: SceneStore, result: TurnResult, prompt: str
) -> Tuple[Optional[Path], Optional[Path]]:
    """(accepted scene file, kept rejected candidate) for the turn just run. The candidate is
    named only in the history entry `record_rejected()` just appended for this prompt (F32);
    a rejection that recorded nothing must not pick up an older entry's candidate. Best
    effort: a failed lookup only drops the path from the output, never the turn."""
    try:
        if result.ordinal is not None:
            return store.current_scene_path(), None
        lines = store.history_path.read_text(encoding="utf-8").splitlines()
        last = json.loads(lines[-1]) if lines else {}
        if (history_entry_problem(last) is None and last.get("outcome") == "rejected"
                and last.get("prompt") == prompt and last.get("file")):
            return None, store.session_dir / last["file"]
    except Exception:  # noqa: BLE001 -- output detail only, must not kill the REPL
        pass
    return None, None


def _write_ticket_draft(store: SceneStore, message: str, prompt: str) -> None:
    """F49 (msa#17): an "unsupported" decline, or a part of an accepted turn marked not done
    (F81), leaves a ticket draft in the session's `tickets/` directory, for the filing script
    outside the agent (AD-1: the agent never files anything itself). A failure to write it
    never disturbs the REPL. A capability an open issue already asks for gets that issue's
    link instead of a draft (session 5, F123, msa#55)."""
    missing, _, nearest = message.partition(" -- nearest: ")
    open_issues = load_open_issues(store.session_dir.parent / OPEN_ISSUES_FILE)
    issue = tracked_issue(missing or prompt, open_issues)
    if issue is not None:
        print(f"  Already tracked: {issue_link(issue)}")
        return
    related = related_issues(missing or prompt, open_issues, prompt=prompt)
    drafts_dir = store.session_dir / "tickets"
    try:
        drafts_dir.mkdir(exist_ok=True)
    except OSError:
        return
    draft = write_draft(
        missing or prompt, prompt, drafts_dir, nearest=nearest,
        session_id=store.session_dir.name, related=related,
    )
    if isinstance(draft, str):
        print(f"  Ticket draft: {draft}")
        for other in related:
            print(f"    Possibly tracked by: {issue_link(other)}")


_stage_line_open = False


@dataclass
class _StageClock:
    """Wall time per stage of one turn, logged with its result (menger#91: turn latency was
    never measured). A stage lasts until the next one starts, the last until the turn ends;
    "generating" includes the local checks, "validating" the sandboxed validator."""

    started: float = field(default_factory=time.monotonic)
    marks: list = field(default_factory=list)

    def __call__(self, stage: str) -> None:
        self.marks.append((stage, time.monotonic()))
        _print_stage(stage)

    def summary(self) -> str:
        end = time.monotonic()
        ends = [t for _, t in self.marks[1:]] + [end]
        parts = [f"{stage} {t1 - t0:.1f}s" for (stage, t0), t1 in zip(self.marks, ends)]
        return ", ".join(parts + [f"total {end - self.started:.1f}s"])


def _print_stage(stage: str) -> None:
    # Story 12 ("live status line"): a multi-second turn printed nothing until done. On stderr,
    # flushed, keeping stdout for the result line. Session 3 (msa#21): the stages of one turn
    # share a line, "... generating, validating, reading back"; `_end_stage_line()` closes it.
    global _stage_line_open
    print(f", {stage}" if _stage_line_open else f"... {stage}", end="", file=sys.stderr, flush=True)
    _stage_line_open = True


def _end_stage_line() -> None:
    global _stage_line_open
    if _stage_line_open:
        print(file=sys.stderr, flush=True)
    _stage_line_open = False


def _record_consult_safely(
    store: SceneStore, question: str, *, answer: Optional[str] = None, error: Optional[str] = None
) -> None:
    """Attempts `store.record_consult(...)` but never lets a failure there propagate -- a
    `SceneStore` write failure while recording a consult turn is as real, and as
    unpropagatable, as one while recording any other turn. Prints a storage-failure note."""
    try:
        store.record_consult(question, answer=answer, error=error)
    except SceneStoreError as e:
        print(f"Consult history: storage_failed - could not record consult turn: {e}")


# --- one turn ---------------------------------------------------------------------------


def execute_turn(
    ctx: Context, ports: Ports, state: TurnState, prompt: str, answered: bool = False
) -> Optional[TurnResult]:
    """Runs one turn, prints its result, and on "accepted" advances `state.prior_scene` and
    refreshes the window -- the one path a typed request, a `/retry` resend and nothing else
    take (story 22). A `run_turn()` exception is printed as "Turn error: ..." and returns
    `None` ("a single bad turn must not kill the REPL"); Ctrl-C abandons the turn (F51)."""
    consult, state.last_consult = state.last_consult, None
    clock = _StageClock()
    try:
        result = ports.run_turn(
            prompt,
            state.prior_scene,
            ctx.manifest,
            ctx.corpus,
            ctx.adapter,
            ctx.store,
            ctx.script_path,
            on_stage=clock,
            **_window_state_kwargs(ctx.store),
            **_window_shown_kwargs(ctx.store),
            **({"answered": True} if answered else {}),
            **({"context_note": _consult_note(consult)} if consult else {}),
        )
        _end_stage_line()
        logger.info("turn: %s, ordinal %s; %s", result.tag, result.ordinal, clock.summary())
    except Exception as e:  # noqa: BLE001 -- a single bad turn must not kill the REPL
        _end_stage_line()
        logger.exception("turn error; %s", clock.summary())
        print(f"Turn error: {e}")
        return None
    except KeyboardInterrupt:
        # F51: the scene is re-read in case the turn got as far as accepting.
        _end_stage_line()
        print("Turn interrupted (Ctrl-C).")
        state.prior_scene = ctx.store.current_scene()
        return None

    print(_wrap_for_terminal(_format_turn_result(result, *_turn_paths(ctx.store, result, prompt))))
    if result.snapshots_missing and not _SNAPSHOT_HINT_SHOWN:
        _SNAPSHOT_HINT_SHOWN.append(True)
        # An image too old to send snapshots is refused before this (menger#116), so empty
        # snapshots mean the validator could not evaluate this scene's values.
        print(
            "  Note: the validator could not evaluate this scene's values, so changes and "
            "camera checks fall back to parsing the scene text"
        )
    if result.tag == "unsupported":
        _write_ticket_draft(ctx.store, result.messages[0] if result.messages else "", prompt)
    for part in result.not_done:
        _write_ticket_draft(ctx.store, part, prompt)
    if result.tag == "accepted":
        state.prior_scene = ctx.store.current_scene()
        state.render_process = ports.accept_and_refresh_render(
            ctx.store, ctx.render_launcher_path, state.render_process
        )
    return result


# --- commands -----------------------------------------------------------------------------


def _retry(ctx: Context, ports: Ports, state: TurnState, _line: str) -> None:
    """`/retry` (story 22, PRD FR7): nothing pending -> says so; scene changed since the
    failure -> resends at once; unchanged after a timeout -> the first `/retry` arms a gate and
    only a second one resends. Any other rejection resends at once (F71). The resend settles a
    pending request like any turn."""
    pending = state.retry_state
    if pending is None:
        print("Retry: nothing to retry -- no failed request to resend.")
        return
    scene_changed = state.prior_scene != pending.scene_at_failure
    if not scene_changed and not pending.confirmed:
        print(
            "Retry: nothing has changed since the last generation_timeout. Type /retry "
            "again to resend the identical prompt."
        )
        state.retry_state = _RetryState(
            prompt=pending.prompt, scene_at_failure=pending.scene_at_failure, confirmed=True
        )
        return
    resend_scene = state.prior_scene
    print(f"Retry: resending: {pending.prompt}")
    result = execute_turn(ctx, ports, state, pending.prompt)
    if result is None:
        # run_turn() raised: no tag to reason about, so the pending retry stays as it was.
        return
    state.retry_state = _next_retry_state(result, pending.prompt, resend_scene)
    state.clarification_state = _next_clarification_state(result, pending.prompt)


def _open(ctx: Context, ports: Ports, state: TurnState, _line: str) -> None:
    print(_wrap_for_terminal(_format_open_items(ports, ctx.store.current_scene(), ctx.manifest)))


def _drop(ctx: Context, ports: Ports, state: TurnState, _line: str) -> None:
    """F48: abandon a pending clarification or threaded rejection."""
    if state.clarification_state is None:
        print("Drop: no pending request.")
    else:
        state.clarification_state = None
        print("Dropped the pending request; your next line starts a new one.")


def _consult(ctx: Context, ports: Ports, state: TurnState, line: str) -> None:
    """`/ask`, `/question` (story 19): a domain question answered in prose. Never reaches
    run_turn(), never runs a gauntlet check, never consumes an ordinal."""
    parts = line.strip().split(None, 1)
    question = parts[1].strip() if len(parts) > 1 else ""
    if not question:
        # A bare "/ask" would spend a paid model call on nothing.
        print("Consult error: empty question -- type a question after /ask or /question")
        return
    try:
        result = ports.answer_consult(
            question, ctx.manifest, ctx.corpus, ctx.adapter, state.prior_scene,
            **_window_state_kwargs(ctx.store),
        )
    except KeyboardInterrupt:  # F51: abandons the question, not the REPL
        print("Consult interrupted (Ctrl-C).")
        return
    if isinstance(result, ConsultError):
        print(f"Consult error: {result.kind} - {result.message}")
        _record_consult_safely(ctx.store, question, error=result.message)
    else:
        print(result)
        _record_consult_safely(ctx.store, question, answer=result)
        state.last_consult = (question, result)


def _export_request(line: str) -> Optional[Tuple[str, Optional[float]]]:
    """`(kind, value)` of an `/export` line, or `None` when it does not parse."""
    words = line.split()
    if len(words) not in (2, 3) or words[1] not in ("image", "video"):
        return None
    try:
        return words[1], float(words[2]) if len(words) == 3 else None
    except ValueError:
        return None


def _export(ctx: Context, ports: Ports, state: TurnState, line: str) -> None:
    """`/export image [t]`, `/export video [seconds]` (msa#56): the accepted scene rendered by
    menger in the contained launcher, without a model call."""
    request = _export_request(line)
    if request is None:
        print(_EXPORT_USAGE)
        return
    kind, value = request
    scene_file = render_window_scene_path(ctx.store.session_dir)
    if state.prior_scene is None or not window_files.is_file(scene_file):
        print("Export: no accepted scene yet.")
        return
    seconds = None
    if kind == "video":
        seconds = value if value is not None else scene_duration(window_files.read_text(scene_file))
        if seconds is None or seconds <= 0:
            print("Export: the scene declares no duration -- give the length: "
                  "/export video <seconds>")
            return
    print(f"... exporting {kind}")
    try:
        outcome = ports.export_scene(
            scene_file, ctx.render_launcher_path, kind,
            t=value if kind == "image" else None, seconds=seconds,
        )
    except KeyboardInterrupt:
        print("Export interrupted (Ctrl-C).")
        return
    if isinstance(outcome, ExportError):
        print(f"Export error: {outcome.message}")
    else:
        print(f"Exported: {outcome.path}")


def _new(ctx: Context, ports: Ports, state: TurnState, _line: str) -> None:
    # ponytail: a fresh scene inside a session breaks AD-13 (the last file on disk is the
    # current scene; the old one would come back as a "hand edit"), so a new session it is.
    print("New: a session holds one scene -- type /exit and start menger-scene-agent again.")


def _unknown(ctx: Context, ports: Ports, state: TurnState, line: str) -> None:
    """Session 5 (F107): `/exit`, `/new` and a pasted file path went to the model as requests."""
    names = sorted([*COMMANDS, *EXIT_COMMANDS, *CONSULT_PREFIXES, EXPORT_COMMAND])
    print(f"Unknown command {line.split(None, 1)[0]!r} (not sent to the model). "
          f"Commands: {', '.join(names)}.")


def _status_echo(ctx: Context, ports: Ports, state: TurnState, _line: str) -> None:
    print("Ignored: that is the agent's own status line, not a request.")


def _export_hint(ctx: Context, ports: Ports, state: TurnState, _line: str) -> None:
    print("Not sent to the model: a scene file cannot write an image or a video. "
          "To save the scene: /export video [seconds] (mp4) or /export image [t] (png).")


Command = Callable[[Context, Ports, TurnState, str], None]

# Exact stripped lines. The consult prefixes take arguments and are matched by first word.
COMMANDS: Dict[str, Command] = {"/retry": _retry, "/open": _open, "/drop": _drop, "/new": _new}
# End the loop, like Ctrl-D (run_repl).
EXIT_COMMANDS = ("/exit", "/quit")
# A pasted copy of `_print_stage`'s line ("... generating, validating"), session 5 (F107).
_STATUS_ECHO = re.compile(r"^\.\.\. (generating|validating|repairing|reading back)\b")
# "save the rendered animation as mp4": an export, which the model cannot do and declined with a
# ticket draft for `/export` itself (session 6, msa#56). A saving verb with an image or video
# word, or "as/to an mp4".
_EXPORT_REQUEST = re.compile(
    r"\b(save|export|write|record|store|download)\b.*\b(mp4|video|movie|gif|png|jpe?g|image|"
    r"picture)\b|\b(as|to|into) an? (mp4|png|gif|jpe?g|video|movie)\b",
    re.IGNORECASE,
)


def command_for(line: str) -> Optional[Command]:
    stripped = line.strip()
    if not stripped:
        return None
    first = stripped.split(None, 1)[0]
    if first in CONSULT_PREFIXES:
        return _consult
    if first == EXPORT_COMMAND:
        return _export
    if _STATUS_ECHO.match(stripped):
        return _status_echo
    if first.startswith("/"):
        return COMMANDS.get(stripped, _unknown)
    if _EXPORT_REQUEST.search(stripped):
        return _export_hint
    return None


# --- the loop ------------------------------------------------------------------------------


def _check_hand_edit(ctx: Context, ports: Ports, state: TurnState) -> None:
    """Story 17: once per line, before anything else -- a clean hand edit of the current scene
    file is promoted like an accepted turn (same window refresh); a rejected one is reported
    and the line the user typed is still processed."""
    try:
        result = ports.check_hand_edit(ctx.store, state.prior_scene, ctx.script_path)
    except Exception as e:  # noqa: BLE001 -- a hand-edit check failure must not kill the REPL
        print(f"Hand edit check error: {e}")
        return
    if result is None:
        return
    print(_format_hand_edit_result(result))
    if result.tag == "accepted":
        state.prior_scene = ctx.store.current_scene()
        state.render_process = ports.accept_and_refresh_render(
            ctx.store, ctx.render_launcher_path, state.render_process
        )
    else:
        state.rejected_edit = ctx.store.current_scene()


def _hand_edit_while_idle(ctx: Context, ports: Ports, state: TurnState, watcher: Any) -> bool:
    """Session 5 (F108, msa#47): a hand edit reloaded the window only after the next typed
    line. The watcher calls this while the REPL waits for input; True when it printed."""
    current = ctx.store.current_scene()
    if current in (state.prior_scene, state.rejected_edit):
        return False
    print()  # off the prompt line
    _check_hand_edit(ctx, ports, state)
    watcher.watch(state.render_process)
    return True


def _request(ctx: Context, ports: Ports, state: TurnState, line: str) -> None:
    """A plain line: a new request, or the continuation of a pending one (story 18, F4) -- it
    cannot tell an answer from an unrelated new request, so it assumes the answer; /drop
    abandons (F48). A line that looks like half a request gets a question before any model
    call (session 4, F93/F103)."""
    cut_off = cut_off_reason(line) if state.clarification_state is None else None
    if cut_off is not None:
        state.clarification_state = _ClarificationState(pending_prompt=line, reason=cut_off)
        print(_rejection_line("needs_clarification", cut_off))
        print(_FOLLOW_UP_HINT)
        return
    pending = state.clarification_state
    prompt = _build_clarification_prompt(pending, line) if pending is not None else line
    answered = pending is not None and pending.asked_for_clarification
    scene_before_turn = state.prior_scene
    result = execute_turn(ctx, ports, state, prompt, answered=answered)
    if result is None:
        return
    state.clarification_state = _next_clarification_state(result, prompt)
    if state.clarification_state is not None:
        print(_FOLLOW_UP_HINT)
    if result.tag != "accepted":
        # A fresh timeout re-arms the /retry gate from scratch, any other rejection arms an
        # ungated retry (F71). An accepted turn leaves a pending retry untouched: an
        # intervening turn must not erase an earlier generation_timeout (story 22).
        state.retry_state = _next_retry_state(result, prompt, scene_before_turn)


def run_repl(ctx: Context, ports: Ports) -> int:
    """Reads lines until Ctrl-D/Ctrl-C at the prompt. The window never outlives the REPL,
    however the loop ends (W4, menger#104)."""
    state = TurnState(prior_scene=ctx.store.current_scene())
    watcher = ports.make_window_watcher(ctx.store)
    watcher.on_idle(lambda: _hand_edit_while_idle(ctx, ports, state, watcher))
    try:
        with watcher.busy():
            while True:
                state.render_process = ports.report_render_crash(
                    ctx.store, state.render_process, watcher
                )
                watcher.watch(state.render_process)
                state.frame_failures_reported = ports.report_frame_build_failures(
                    ctx.store, state.frame_failures_reported
                )
                # F48: a visible marker while the next line is threaded onto a pending request.
                prompt = "> " if state.clarification_state is None else "(continuing) > "
                try:
                    with watcher.idle(prompt):
                        line = input(prompt)
                except (EOFError, KeyboardInterrupt):
                    break
                if line.strip() in EXIT_COMMANDS:
                    break
                _check_hand_edit(ctx, ports, state)
                command = command_for(line)
                if command is not None:
                    command(ctx, ports, state, line)
                elif line.strip():
                    _request(ctx, ports, state, line)
    finally:
        watcher.stop()
        ports.close_render_window(state.render_process)
    return 0


__all__ = [
    "Context", "Ports", "TurnState", "COMMANDS", "EXIT_COMMANDS", "command_for", "execute_turn",
    "run_repl",
]

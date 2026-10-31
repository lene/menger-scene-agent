"""How the REPL prints results: turns, rejections, replayed history, render-window status and
hand edits (review 2026-10-14, msa#33/#39: split out of cli.py). Pure functions of their
arguments, except `_wrap_for_terminal`, which reads the terminal width."""

from __future__ import annotations

import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Optional

from adapters.render_window import RenderWindowOutcome, RenderWindowResult
from core.turn import BUDGET_QUESTION
from core.types import TurnResult

# A rejection's history `reason` starts with its turn tag ("compile_errors: ..."), which
# `_format_history_entry` recovers so a replayed rejection shows its real tag (F38).
_REASON_TAG = re.compile(r"^([a-z_]+): (.*)$", re.DOTALL)


def _tag_label(tag: str) -> str:
    return tag.replace("_", " ")


def _rejection_line(tag: Optional[str], *messages: str) -> str:
    """One line for a single short reason; otherwise every reason, every line of a multi-line
    one (compiler output) and the nearest alternative on its own indented line (F80b)."""
    messages = tuple(m for m in messages if m)
    if tag == "unchanged" and messages:
        return "; ".join(messages)  # F76: not a rejection; it says "No change made: ..."
    if tag == "lint_findings" and messages and messages[0].startswith("scene-budget"):
        tag = "over_budget"  # session 5 (F109): the validator reports the budget as a finding
        # Session 6 (msa#48): only an identical repair asked; a changed one ended with no question.
        if not any(BUDGET_QUESTION in m for m in messages):
            messages += (BUDGET_QUESTION,)
    head = f"Rejected ({_tag_label(tag)})" if tag else "Rejected"
    lines = []
    for message in messages:
        reason, found, nearest = message.partition(" -- nearest: ")
        lines += reason.splitlines() + ([f"Nearest: {nearest}"] if found else [])
    if len(lines) <= 1:
        return f"{head}: {lines[0]}" if lines else head
    return f"{head}:" + "".join(f"\n  {line}" for line in lines)


def _format_history_entry(entry: dict, session_dir: Optional[Path] = None) -> str:
    # spec-ai-scene-agent story 19 (review round, patch-level fix): a "consult" entry has
    # neither a "reason" key (that's rejected/failed-attempt shaped) nor an ordinal -- without
    # this branch, replaying a resumed session's prior /ask/-question turns collapsed to the
    # bare, content-free "Turn None: consult", silently dropping the recorded prompt/answer/
    # error that history.jsonl actually preserved.
    if entry.get("outcome") == "consult":
        prompt = entry.get("prompt")
        detail = entry.get("answer") if "answer" in entry else entry.get("error")
        return f"Consult: {prompt!r} -> {detail}"
    file_name = entry.get("file")
    file_path = session_dir / file_name if session_dir is not None and file_name else file_name
    ordinal = entry.get("ordinal")
    if ordinal is not None:
        # Same shape as a live accepted turn (_format_turn_result), F50/F38.
        text = f"Turn {ordinal}: {entry.get('outcome')}"
        if file_path:
            text += f" → {file_path}"
        if entry.get("readback_summary"):
            text += f"\n  Readback: {entry['readback_summary']}"
        return text
    # F2: never a numberless "Turn None: ..." line; F38: the real tag, not "(rejected)".
    reason = entry.get("reason") or ""
    match = _REASON_TAG.match(reason)
    text = _rejection_line(*match.groups()) if match else _rejection_line(None, reason)
    if file_path:
        text += f"\n  Candidate kept: {file_path}"
    return text


_LABELLED_LINE = re.compile(r"^(\s*)([A-Z][\w ()]*: )")


def _wrap_for_terminal(text: str) -> str:
    """Wraps long lines at word boundaries to the terminal's width, continuation lines indented
    under the text after a "  Label: " prefix (usability review 2026-10, session 4, F103: the
    terminal broke readbacks and warnings in the middle of words). Paths and other long words
    are never split. Only for a terminal; a pipe or a log keeps whole lines."""
    if not sys.stdout.isatty():
        return text
    width = shutil.get_terminal_size().columns
    wrapped = []
    for line in text.splitlines():
        if len(line) <= width:
            wrapped.append(line)
            continue
        label = _LABELLED_LINE.match(line)
        indent = len(label.group(0)) if label else len(line) - len(line.lstrip())
        wrapped.append(textwrap.fill(
            line, width, subsequent_indent=" " * indent,
            break_long_words=False, break_on_hyphens=False,
        ))
    return "\n".join(wrapped)


def _format_turn_result(
    result: TurnResult,
    scene_path: Optional[Path] = None,
    candidate_path: Optional[Path] = None,
) -> str:
    # Boundaries & Constraints: "Every run_turn() outcome prints its ordinal-or-none and
    # tag, never silently". Usability review 2026-09 (F2): a non-accepted tag's `ordinal` is
    # always `None` (nothing was ever persisted to reject), and printing that as the literal
    # word "Turn None: ..." read as a bug -- a rejection now gets its own "Rejected (<tag>):
    # ..." shape instead, never a numberless "Turn" line. `result.messages` is always
    # populated on a non-accepted tag (never empty on `accepted`, per TurnResult's own
    # docstring), so it's safe to fold in unconditionally either way. Usability review 2026-09
    # (F50/F38, tester-approved layout): the accepted file, the readback and a kept rejected
    # candidate are shown on their own indented lines.
    if result.ordinal is not None:
        text = f"Turn {result.ordinal}: {result.tag}"
        if scene_path is not None:
            text += f" → {scene_path}"
        if result.messages:
            text += " - " + "; ".join(result.messages)
    else:
        text = _rejection_line(result.tag, *result.messages)
    if result.readback_summary:
        text += f"\n  Readback: {result.readback_summary}"
    if result.changes is not None:
        # Session 4 (F90): from the evaluated values, not the model's restatement.
        listed = "; ".join(result.changes) if result.changes else (
            "no value (only code layout or comments)"
        )
        text += f"\n  Changed: {listed}"
    for note in result.notes:
        text += f"\n  Note: {note}"
    if result.removed_properties:
        # F29: a turn that dropped earlier settings must say so, not do it silently.
        text += "\n  Note: this turn removed " + ", ".join(result.removed_properties)
    if result.warnings:
        # F8: a revise() turn that moved something or changed a material the request didn't
        # ask for must say so, not do it silently -- same spirit as F29 above.
        for warning in result.warnings:
            text += f"\n  Warning: {warning}"
    for part in result.not_done:
        text += f"\n  Not done: {part}"  # F81: not only in a doc comment
    for part in result.stale_not_done:
        # F91: its nearest is gone from the scene, so the line no longer told the truth.
        text += f"\n  Note: removed the outdated Not done line: {part}"
    if result.carried_over:
        # F91: reported on an earlier turn, still true -- counted instead of repeated.
        text += (
            f"\n  ({result.carried_over} earlier warning(s) / Not done item(s) still apply)"
        )
    if candidate_path is not None:
        text += f"\n  Candidate kept: {candidate_path}"
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

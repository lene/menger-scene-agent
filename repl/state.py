"""The REPL's two conversational state machines (review 2026-10-14, msa#33/#39: split out of
cli.py): a pending `/retry` and a pending clarification, which threads the user's next line
back into the request it answers. Immutable states and pure transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.types import TurnResult


@dataclass(frozen=True)
class _RetryState:
    """Tracks the most recent `generation_timeout` so `/retry` (story 22, PRD FR7) can tell
    "resend this" apart from "nothing to retry" and enforce its own confirmation gate.
    `prompt` is the exact text that failed; `scene_at_failure` is `prior_scene` as it stood
    at that moment (a plain equality compare against the loop's current `prior_scene` is how
    `_handle_retry()` detects "the scene changed since the failure" -- a hand edit or another
    accepted turn always refreshes `prior_scene`, so a stale `scene_at_failure` reliably means
    "changed"). `confirmed` is `False` until a first `/retry` with an unchanged scene arms the
    gate; only then does a second `/retry` (still unchanged) actually resend. Reassigned
    wholesale on every transition (never mutated in place) -- frozen like every other typed
    outcome in this module."""

    prompt: str
    scene_at_failure: Optional[str]
    confirmed: bool


def _next_retry_state(
    result: TurnResult, prompt: str, scene_at_failure: Optional[str]
) -> Optional[_RetryState]:
    """Computes the freshly-armed `_RetryState` for a `"generation_timeout"` tag -- `None`
    for every other tag. A fresh `"generation_timeout"` -- including one produced by a
    resend -- always re-arms the gate from scratch, unconfirmed (Boundaries & Constraints:
    "previous confirmation state never carries over across a new failure").

    Callers decide what a `None` return means for THEM, which differs by dispatch path:
    - The normal per-loop `run_turn()` dispatch (an unrelated prompt the user just typed)
      only calls this, and only assigns its result, when the tag actually IS
      `"generation_timeout"` -- an unrelated turn's `accepted`/rejection tag must NOT erase
      an existing pending retry, precisely so the I/O & Edge-Case Matrix's "scene changed
      since the failure (a hand edit or another accepted turn happened in between)" row is
      even reachable: that row's `/retry` still needs the ORIGINAL failed prompt remembered
      after such an intervening turn, not wiped by it.
    - `/retry`'s own resend (`_handle_retry`) always assigns this result verbatim, `None`
      included: once a pending retry has been acted on (resent), it is resolved either way
      -- a fresh timeout replaces it, anything else clears it, since there is nothing left
      to retry (a second `/retry` afterward must report "nothing to retry", never resend
      the same already-resolved prompt again)."""
    if result.tag == "generation_timeout":
        return _RetryState(prompt=prompt, scene_at_failure=scene_at_failure, confirmed=False)
    if result.tag != "accepted":
        # Session 3, F71: any rejected request can be tried again. The model answers anew each
        # time, so there is no identical-resend gate as for a timeout: `/retry` is the request.
        return _RetryState(prompt=prompt, scene_at_failure=scene_at_failure, confirmed=True)
    return None


@dataclass(frozen=True)
class _ClarificationState:
    """Tracks a pending `needs_clarification` (story 18) so the user's next plain-text line
    is threaded back into the original request instead of being sent to the model as a
    lone, context-free fragment -- mirrors `_RetryState`'s shape/lifecycle, scoped to
    `needs_clarification` instead of `generation_timeout`.

    `pending_prompt` is the full text that produced this clarification -- the user's raw
    line the first time, or a previously merged prompt if this is a second-or-later round
    (see `_next_clarification_state`). `reason` is the model's stated clarification message
    (`"; ".join(result.messages)`, matching `_format_turn_result`'s own join)."""

    pending_prompt: str
    reason: str
    # False for a rejected attempt (usability review 2026-09, F4) rather than a question the
    # model asked -- only the framing of the merged prompt differs.
    asked_for_clarification: bool = True
    # True for an "unsupported" decline (session 3, F77): the follow-up is a new request, the
    # declined one only context for what it refers to.
    declined: bool = False


# Rejections the user typically answers by adjusting the same request ("start at level 0
# instead"). Their next line is threaded back into the rejected request like a clarification
# answer; sent alone it lost everything the request had asked for (usability review 2026-09,
# F4: tesseract, glass, dark background and rotation all silently dropped).
# `generation_timeout` has /retry instead; infrastructure failures (refused, timeout,
# subprocess_failed, storage_failed, ...) say nothing about the request itself. "unsupported"
# (F16/msa#3) is deliberately excluded too, despite also being a request-shaped rejection: a
# follow-up doesn't resolve "the DSL cannot do this" the way it resolves an ambiguity or a
# fixable rejection, so it is never merged into it -- see TurnTag's own docstring in
# core/types.py. It is passed along as context instead (`declined`, F77).
_REJECTIONS_THREADED_INTO_FOLLOW_UP = frozenset(
    {"compile_errors", "lint_findings", "local_finding", "readback_failed"}
)


def _build_clarification_prompt(state: _ClarificationState, answer: str) -> str:
    """Composes the merged prompt sent to `run_turn()` when a `needs_clarification` (or a
    threaded rejection) is pending -- folds the original request, the stated reason, and the
    user's answer into one string. `core/generation.py`'s `prompt` parameter stays a single
    string throughout (Approach: no core/ signature changes), so this composition happens
    entirely here in the CLI."""
    if state.declined:
        return (
            f"{answer}\n\n"
            f"(Context: the previous request was declined as impossible -- "
            f"\"{state.pending_prompt}\": {state.reason}. The request above may refer to it, "
            f"e.g. accept its nearest alternative or mean the place or object it named; take "
            f"nothing else from it.)"
        )
    if state.asked_for_clarification:
        # Session 5 (F104, F105): "yes", "level 3.8" and "copper" were each asked about again.
        return (
            f"{state.pending_prompt}\n\n"
            f"(The system asked for clarification: {state.reason})\n"
            f"User's answer: {answer}\n"
            f"(This answer settles the question: a bare \"yes\" or \"ok\" accepts the option "
            f"the question offered first. If it is a different request instead (\"no, make "
            f"it aluminium\"), do that request. Do not respond NEEDS_CLARIFICATION again; if "
            f"something is still open, compose the nearest reading and state the assumption "
            f"in the doc comment.)"
        )
    return (
        f"{state.pending_prompt}\n\n"
        f"(The previous attempt at this request was rejected: {state.reason})\n"
        f"User's follow-up, to be applied to the request above -- keep everything else it "
        f"asked for: {answer}"
    )


def _next_clarification_state(
    result: TurnResult, pending_prompt: str
) -> Optional[_ClarificationState]:
    """`None` clears the pending clarification (any tag other than `needs_clarification`
    resolves it, accepted or otherwise-rejected alike). A fresh `needs_clarification` --
    including one produced by an already-merged prompt -- re-arms with `pending_prompt` set
    to the just-sent (possibly already-merged) text, so a second or third clarification
    round keeps compounding the whole history rather than only the latest answer."""
    if result.tag == "needs_clarification":
        return _ClarificationState(
            pending_prompt=pending_prompt, reason="; ".join(result.messages)
        )
    if result.tag in _REJECTIONS_THREADED_INTO_FOLLOW_UP:
        return _ClarificationState(
            pending_prompt=pending_prompt,
            reason=f"{result.tag}: " + "; ".join(result.messages),
            asked_for_clarification=False,
        )
    if result.tag == "unsupported":
        return _ClarificationState(
            pending_prompt=pending_prompt,
            reason="; ".join(result.messages),
            asked_for_clarification=False,
            declined=True,
        )
    return None

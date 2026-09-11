"""Turn pipeline orchestration (spec-ai-scene-agent story 20): the one place that wires
`generate()`/`revise()` (`core/generation.py`), the four `gauntlet.check_*()` stages, the
renderer-side adapter (`adapters.scene_validator.validate_scene`, story 10),
`semantic_readback()` (`core/readback.py`), and `SceneStore` (`adapters/scene_store.py`)
together into one agent turn. None of those modules wire themselves to each other -- each
one's own docstring says so explicitly (no pipeline wiring is their job). This module closes
PRD FR1: before this story, `generate()`/`revise()` produced scene text that nothing gated.

AD-1/AD-2/AD-3: agent-side checks always run first (Boundaries & Constraints) -- a renderer
request is never made while a local `gauntlet.check_*()` finding exists. `store.accept()`/
`store.record_rejected()` remain the only acceptance/rejection paths. `run_turn()` is typed-
result, never-raises-for-an-expected-outcome, the same contract every module it composes
already holds itself to.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from adapters.model import ModelAdapter
from adapters.scene_store import SceneStore, SceneStoreError
from adapters.scene_validator import validate_scene
from core.generation import generate, revise
from core.readback import semantic_readback
from core.types import (
    GenerationError,
    ReadbackError,
    TurnResult,
    ValidationError,
    ValidationResult,
)
from gauntlet.allowlist import check_allowlist
from gauntlet.clean_code import check_clean_code
from gauntlet.lint import check_lint
from gauntlet.resource_bounds import check_resource_bounds
from gauntlet.types import Finding

# Design Notes: candidate text is written to `session_dir/.candidate.scala` -- a name
# `SceneStore`'s `_ORDINAL_FILENAME` regex (`^(\d{3,})\.scala$`) never matches, so it's
# invisible to `current_scene()`/ordinal logic. Overwritten per turn, unconditionally deleted
# before `run_turn()` returns (success or failure) -- disposable scratch, not the durable
# artifact `store.accept()` produces.
_STAGING_FILENAME = ".candidate.scala"

# The four gauntlet stages (Code Map), run in a fixed order -- aggregated together (rather
# than short-circuiting on the first stage that finds something) so a rejected turn's
# `TurnResult.messages`/`findings` report everything wrong with the candidate at once.
_LOCAL_CHECKS = (check_allowlist, check_resource_bounds, check_clean_code, check_lint)


def _run_local_checks(scene_text: str) -> List[Finding]:
    findings: List[Finding] = []
    for check in _LOCAL_CHECKS:
        findings.extend(check(scene_text))
    return findings


def _local_finding_reason(findings: List[Finding]) -> str:
    return "; ".join(f"{finding.stage}: {finding.message}" for finding in findings)


def _validation_messages(result: ValidationResult) -> List[str]:
    # `messages` alone can be empty for a `lint_findings` result (story 10's I/O & Edge-Case
    # Matrix: that row populates `findings`, not `messages`) -- folding both in keeps the
    # rejection reason and `TurnResult.messages` informative regardless of which
    # renderer-domain tag produced it.
    messages = list(result.messages)
    messages.extend(f"{finding.invariant}: {finding.message}" for finding in result.findings)
    return messages


def _record_rejected_safely(
    store: SceneStore, prompt: str, reason: str, messages: List[str]
) -> List[str]:
    """Attempts `store.record_rejected(prompt, reason)` but never lets a failure there
    propagate (Boundaries & Constraints: `run_turn()` never raises for an expected
    outcome, and a `SceneStore` write failure while recording a *rejection* is exactly as
    real -- and as unpropagatable -- as one while recording an acceptance, see
    `storage_failed`). Returns `messages` unchanged when the write succeeds; when
    `record_rejected()` itself raises `SceneStoreError`, returns `messages` with the
    original rejection `reason` and a storage-failure note appended, so the caller's
    `TurnResult` folds both in rather than the exception propagating."""
    try:
        store.record_rejected(prompt, reason)
        return messages
    except SceneStoreError as e:
        return list(messages) + [reason, f"storage_failed: could not record rejection: {e}"]


def run_turn(
    prompt: str,
    prior_scene: Optional[str],
    manifest: dict,
    corpus: dict,
    adapter: ModelAdapter,
    store: SceneStore,
    script_path: Union[str, Path],
    image: Optional[str] = None,
    timeout: Optional[float] = None,
) -> TurnResult:
    """Runs one agent turn end to end: derive scene text (CAP-1 `generate()` when
    `prior_scene` is `None`, CAP-2 `revise()` otherwise), gate it through the agent-side
    gauntlet, then the renderer-side gauntlet (`validate_scene()`), then semantic readback,
    and only then persist it via `store.accept()` -- or reject at whichever stage first
    finds something, via `store.record_rejected()`. Never raises for an expected outcome.

    `script_path` is an explicit injected parameter (AD-10, "paths injected, never
    discovered") -- this function receives it; where an eventual CLI entry point sources it
    from is out of this story's scope. `image`/`timeout` are forwarded to `validate_scene()`
    verbatim -- this function has no opinion of its own about either."""
    generation_result = (
        generate(prompt, manifest, corpus, adapter)
        if prior_scene is None
        else revise(prompt, prior_scene, manifest, corpus, adapter)
    )

    if isinstance(generation_result, GenerationError):
        # Not one of the frozen I/O & Edge-Case Matrix's five rows, but a real precondition
        # for all of them: gauntlet.check_*() takes a `str`, so a generation failure must be
        # handled before any local check runs -- no staging file is ever written, no
        # renderer call is ever made, same as the local-finding path below.
        messages = _record_rejected_safely(
            store,
            prompt,
            f"generation_failed: {generation_result.message}",
            [generation_result.message],
        )
        return TurnResult(tag="generation_failed", messages=messages)

    scene_text = generation_result

    local_findings = _run_local_checks(scene_text)
    if local_findings:
        # Always: "a renderer request is never made when a local finding exists" -- no
        # staging file is written on this path at all.
        messages = _record_rejected_safely(
            store,
            prompt,
            _local_finding_reason(local_findings),
            [finding.message for finding in local_findings],
        )
        return TurnResult(tag="local_finding", messages=messages, findings=local_findings)

    staging_path = store.session_dir / _STAGING_FILENAME
    try:
        try:
            # Moved inside the try (review round, patch-level fix): a write failure here
            # (disk full, permission error) is exactly as real a failure mode as any other
            # storage failure this function guards against, and must become a typed
            # `TurnResult` rather than an unhandled `OSError` escaping to the caller.
            staging_path.write_text(scene_text, encoding="utf-8")
        except OSError as e:
            reason = f"storage_failed: could not write staging file: {e}"
            messages = _record_rejected_safely(store, prompt, reason, [reason])
            return TurnResult(tag="storage_failed", messages=messages)

        outcome = validate_scene(staging_path, script_path, image=image, timeout=timeout)

        if isinstance(outcome, ValidationError):
            messages = _record_rejected_safely(
                store, prompt, f"{outcome.kind}: {outcome.message}", [outcome.message]
            )
            return TurnResult(tag=outcome.kind, messages=messages)

        if outcome.tag != "ok":
            # `_validation_messages(outcome)` can itself be empty (a bare non-"ok" tag with
            # neither `messages` nor `findings` populated, e.g. a bare "refused") -- fall
            # back to the tag itself so `TurnResult.messages` is never empty on a
            # non-accepted tag, matching this module's own documented contract (review
            # round, patch-level fix; mirrors the fallback already used for the
            # `record_rejected()` reason string on this same branch).
            validation_messages = _validation_messages(outcome) or [outcome.tag]
            messages = _record_rejected_safely(
                store,
                prompt,
                "; ".join(validation_messages) or outcome.tag,
                validation_messages,
            )
            return TurnResult(tag=outcome.tag, messages=messages)

        readback_result = semantic_readback(scene_text, adapter)
        if isinstance(readback_result, ReadbackError):
            # I/O & Edge-Case Matrix: "turn is NOT accepted without its summary -- treated
            # as rejection, not a partial accept."
            messages = _record_rejected_safely(
                store,
                prompt,
                f"readback_failed: {readback_result.message}",
                [readback_result.message],
            )
            return TurnResult(tag="readback_failed", messages=messages)

        try:
            ordinal = store.accept(scene_text, prompt, readback_summary=readback_result)
        except SceneStoreError as e:
            # `store.accept()` can raise after exhausting its ordinal-claim retries under
            # contention (documented on `SceneStore.accept()`) -- a rare-but-real failure
            # that must become a typed `TurnResult`, not an unhandled exception (review
            # round, patch-level fix). Same pattern as every other failure tag: still try
            # to leave a `history.jsonl` trace via `record_rejected()`, itself guarded by
            # `_record_rejected_safely` in case the store is broken in a way that affects
            # that write too.
            reason = f"storage_failed: {e}"
            messages = _record_rejected_safely(store, prompt, reason, [reason])
            return TurnResult(tag="storage_failed", messages=messages)
        return TurnResult(
            tag="accepted", messages=[], readback_summary=readback_result, ordinal=ordinal
        )
    finally:
        # Unconditional (Boundaries & Constraints: "The staging file is always deleted
        # (success or failure) before run_turn returns -- never left in session_dir").
        staging_path.unlink(missing_ok=True)

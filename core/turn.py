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

import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from core.generation import generate, revise
from core.ports import ModelAdapter, SceneStore, SceneStoreError
from core.prompt_checks import (
    caustics_strength_question,
    play_once_note,
    pre_projection_colour_hint,
    refers_to_window,
    undirected_placement_hint,
    unknown_material_hint,
    window_state_hint,
)
from core.readback import semantic_readback
from core.scene_facts import (
    camera_gaze_warnings,
    caustics_caveats,
    caveat_warnings,
    extract_scene_facts,
    facts_diff,
    manifest_subtype_warn_levels,
    manifest_warn_levels,
    occlusion_warnings,
    scene_changes,
)
from core.snapshot_facts import (
    CheckResult,
    asks_to_keep_view,
    check_turn,
    kept_view_findings,
    snapshot_changes,
    unrequested_changes,
)
from core.types import (
    GenerationError,
    GenerationResult,
    ReadbackError,
    StageName,
    TurnResult,
    ValidationError,
    ValidationOutcome,
    ValidationResult,
)
from gauntlet._scala_text import STRING_LITERAL, strip_comments_and_strings
from gauntlet.allowlist import check_allowlist
from gauntlet.clean_code import check_clean_code
from gauntlet.lint import check_lint
from gauntlet.resource_bounds import check_resource_bounds
from gauntlet.types import Finding

# The renderer-side gauntlet -- menger's sandboxed validator process -- is a port (review
# 2026-10-14, msa#34: core imported the adapter). The composition roots set it once
# (`use_validator`: cli.py's main, evals/recorded.py); tests replace it like any port. Unset, a
# turn reports the missing configuration instead of running anything.
Validator = Callable[..., ValidationOutcome]


def _no_validator(*_args: object, **_kwargs: object) -> ValidationOutcome:
    return ValidationError(
        kind="subprocess_failed",
        message="no scene validator configured (core.turn.use_validator was never called)",
    )


validate_scene: Validator = _no_validator


def use_validator(validator: Validator) -> None:
    global validate_scene
    validate_scene = validator


def use_default_validator(validator: Validator) -> None:
    """Sets `validator` unless one is configured already -- a composition root's default, which
    an earlier explicit choice (a test, another front end) keeps."""
    if validate_scene is _no_validator:
        use_validator(validator)


# Design Notes: candidate text is written to `session_dir/.candidate.scala` -- a name
# `SceneStore`'s `_ORDINAL_FILENAME` regex (`^(\d{3,})\.scala$`) never matches, so it's
# invisible to `current_scene()`/ordinal logic. Overwritten per turn, unconditionally deleted
# before `run_turn()` returns (success or failure) -- disposable scratch, not the durable
# artifact `store.accept()` produces.
_STAGING_FILENAME = ".candidate.scala"

# F47: bounded self-repair of the agent's own compile/lint failures (PRD FR-7 amendment,
# decided by user 2026-10-02: "Approve, max 2 rounds").
MAX_REPAIR_ROUNDS = 2

# Rejections of the agent's own candidate that a revision with the failure fed back can fix
# (F47), and the stage shown while it tries. A refusal, a validator crash or a readback
# failure is never repaired automatically (PRD FR-7 amendment 2026-10-02).
_REPAIRABLE_TAGS: dict[str, StageName] = {
    "local_finding": "repairing (local finding)",
    "compile_errors": "repairing (compile errors)",
    "lint_findings": "repairing (lint findings)",
}

# The four gauntlet stages (Code Map), run in a fixed order -- aggregated together (rather
# than short-circuiting on the first stage that finds something) so a rejected turn's
# `TurnResult.messages`/`findings` report everything wrong with the candidate at once.
_LOCAL_CHECKS = (check_allowlist, check_resource_bounds, check_clean_code, check_lint)

# A part of the request the scene does not do, as generation_rules.md tells the model to mark
# it: `// NOT DONE: <part> -- nearest: <alternative>` (F81).
_NOT_DONE = re.compile(r"^[ \t]*//[ \t]*NOT DONE:(.+)$", re.MULTILINE)
_NOT_DONE_LINE = re.compile(r"^[ \t]*//[ \t]*NOT DONE:(.+)\n?", re.MULTILINE)

# A material preset a `// NOT DONE` line names as its nearest (`Material.Chrome`).
_MATERIAL_TOKEN = re.compile(r"\bMaterial\.[A-Z]\w*")


def not_done_parts(scene_text: Optional[str]) -> List[str]:
    """The scene's `// NOT DONE:` parts, stripped."""
    return [m.strip() for m in _NOT_DONE.findall(scene_text or "")]


def _same_code(a: str, b: str) -> bool:
    """Equal but for whitespace and comments: a reply that only adds `// NOT DONE` lines
    changes nothing (session 5, F118). String literals count, a texture path is code."""
    return (strip_comments_and_strings(a).split() == strip_comments_and_strings(b).split()
            and STRING_LITERAL.findall(a) == STRING_LITERAL.findall(b))


def split_not_done(parts: List[str]) -> Tuple[List[str], List[str]]:
    """(real Not done parts, caveats): a part without "-- nearest:" is not the
    `<part> -- nearest: <alternative>` form generation_rules.md asks for, but a remark the model
    filed there (session 4, 8b: "diamond, glass on this sponge -- ... renders as chaotic
    fragmented refraction; requested as asked", with its own ticket draft). Shown as a note,
    no ticket (F92)."""
    real = [p for p in parts if "-- nearest:" in p]
    return real, [p for p in parts if "-- nearest:" not in p]


def _generation_failure(store: SceneStore, prompt: str, error: GenerationError) -> TurnResult:
    """The tag for a `GenerationError` (stories 15, 18; F16): a timeout, the model's own
    question or decline, or any other failure -- each distinct end to end."""
    tag = {
        "needs_clarification": "needs_clarification",
        "unsupported": "unsupported",
        "model_call_timeout": "generation_timeout",
    }.get(error.kind, "generation_failed")
    messages = _record_rejected_safely(store, prompt, f"{tag}: {error.message}", [error.message])
    return TurnResult(tag=tag, messages=messages)


def _with_first_attempt(rejection: TurnResult, first_candidate: str, rounds_done: int) -> TurnResult:
    """A rejection after repair rounds keeps what the first attempt said it could not do (F100:
    8a's "there is no 5D object, so a penteract cannot be built" was lost behind the repair's
    syntax error)."""
    parts = not_done_parts(first_candidate) if rounds_done else []
    if not parts:
        return rejection
    return TurnResult(
        tag=rejection.tag,
        messages=rejection.messages + [f"The first attempt said: Not done: {p}" for p in parts],
        findings=rejection.findings,
    )


# Over budget, the repair keeps what the user named instead of lowering it (F87), so the user
# decides (session 5, F109); asked on every over-budget rejection (session 6, msa#48).
BUDGET_QUESTION = "Lower it (name a level), or /drop this request?"


def _unrepaired_note(feedback: List[str]) -> str:
    """What to say when the repair returned the same scene."""
    if any(f.startswith("scene-budget") for f in feedback):
        return ("The scene is over the budget above, and the automatic repair keeps the level "
                f"you asked for. {BUDGET_QUESTION}")
    return "The automatic repair returned the same scene, so it was stopped."


def scene_warnings(
    facts, scene_text: str, manifest: dict, snapshots: Sequence[dict] = ()
) -> List[str]:
    """Warnings that are properties of a scene itself, not of a turn's changes: an occluded
    object (F23), a slow level or glass on a 4D sponge (F36, #4c), an animated camera that ends
    up looking at nothing (F83)."""
    return (
        occlusion_warnings(facts, snapshots)
        + caveat_warnings(
            facts, manifest_warn_levels(manifest), manifest_subtype_warn_levels(manifest)
        )
        + camera_gaze_warnings(facts, strip_comments_and_strings(scene_text))
        + caustics_caveats(facts, strip_comments_and_strings(scene_text))
    )


def open_items(scene_text: str, manifest: dict) -> Tuple[List[str], List[str]]:
    """Everything still open for a scene, for the REPL's `/open` (usability review 2026-10,
    session 4: warnings are now reported once, so the user needs a way to see them all
    again): its scene warnings and its Not done parts."""
    facts = extract_scene_facts(strip_comments_and_strings(scene_text))
    real, caveats = split_not_done(not_done_parts(scene_text))
    return scene_warnings(facts, scene_text, manifest) + caveats, real


def drop_stale_not_done(scene_text: str) -> Tuple[str, List[str]]:
    """Removes `// NOT DONE` lines whose named material nearest is no longer in the code (8b:
    "nearest: Material.Chrome" stayed after chrome and steel were both dropped, and was
    reported as true on nine more turns -- usability review 2026-10, session 4, F91). A line
    that names no material is kept: only the model can tell whether it still applies."""
    code = strip_comments_and_strings(scene_text)
    dropped: List[str] = []

    def keep_or_drop(match: "re.Match[str]") -> str:
        tokens = _MATERIAL_TOKEN.findall(match.group(1))
        if tokens and not any(re.search(rf"\b{re.escape(t)}\b", code) for t in tokens):
            dropped.append(match.group(1).strip())
            return ""
        return match.group(0)

    return _NOT_DONE_LINE.sub(keep_or_drop, scene_text), dropped

# A named argument in a scene file: `proceduralType = 8`, `edgeRadius = Some(0.005f)`.
# `==` is excluded so comparisons don't count.
_NAMED_ARGUMENT = re.compile(r"\b([a-z][A-Za-z0-9]*)\s*=(?!=)")


def removed_properties(prior_scene: Optional[str], new_scene: str) -> List[str]:
    """Named arguments the prior scene set and the new one no longer does, sorted. Coarse by
    design (names only, not per object), which is enough to tell the user "this turn also
    removed proceduralType" -- usability review 2026-09, F29."""
    if prior_scene is None:
        return []
    before = _named_arguments(prior_scene)
    after = _named_arguments(new_scene)
    return sorted(before - after)


def _named_arguments(scene: str) -> set[str]:
    # A `val x =` is a definition, not an argument: `val scene` -> `def scene(t)` is no
    # removal (F75).
    return {
        match.group(1)
        for match in _NAMED_ARGUMENT.finditer(scene)
        if not re.search(r"\b(?:val|var)\s+$", scene[: match.start()])
    }


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
    # The validator also lists each finding in `messages` as "invariant: message" (F44 in
    # usability review session 2 showed it twice), so only add findings not already there.
    messages = list(result.messages)
    for finding in result.findings:
        text = f"{finding.invariant}: {finding.message}"
        if text not in messages:
            messages.append(text)
    return messages


def _record_rejected_safely(
    store: SceneStore,
    prompt: str,
    reason: str,
    messages: List[str],
    scene_text: Optional[str] = None,
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
        store.record_rejected(prompt, reason, scene_text=scene_text)
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
    *,
    on_stage: Optional[Callable[[StageName], None]] = None,
    repair_rounds: int = MAX_REPAIR_ROUNDS,
    window_state: Optional[str] = None,
    answered: bool = False,
    context_note: Optional[str] = None,
    window_shown: Optional[Tuple[float, List[Tuple[str, dict]]]] = None,
) -> TurnResult:
    """Runs one agent turn end to end: derive scene text (CAP-1 `generate()` when
    `prior_scene` is `None`, CAP-2 `revise()` otherwise), gate it through the agent-side
    gauntlet, then the renderer-side gauntlet (`validate_scene()`), then semantic readback,
    and only then persist it via `store.accept()` -- or reject at whichever stage first
    finds something, via `store.record_rejected()`. Never raises for an expected outcome.

    `script_path` is an explicit injected parameter (AD-10, "paths injected, never
    discovered") -- this function receives it; where an eventual CLI entry point sources it
    from is out of this story's scope. `image`/`timeout` are forwarded to `validate_scene()`
    verbatim -- this function has no opinion of its own about either.

    `on_stage` (spec-ai-scene-agent story 12, "live status line"), when provided, is called
    with a stage name at each transition this function actually reaches: `"generating"`
    (before `generate()`/`revise()`), `"validating"` (before the local gauntlet checks --
    this single call covers both agent-side and renderer-side validation as one
    user-facing phase, no second call before `validate_scene()`), `"reading back"` (before
    `semantic_readback()`, reached only on a renderer `ok`). Never fired for a stage that's
    short-circuited past. Defaults to `None`, a no-op, so every existing caller is
    unaffected. `on_stage` is called synchronously, inline, as part of this function's own
    execution (no threading/async involved) -- callbacks should be cheap and non-blocking.
    Any exception `on_stage` itself raises (including a `TypeError` from a non-callable
    value) is swallowed rather than propagated: its only job is cosmetic reporting, and it
    must never abort a turn that had otherwise succeeded up to that point.

    `answered`: `prompt` threads the user's answer to a question the model asked (session 5,
    F104/F105). The material hint is then left out, and a second question gets one round that
    asks for the scene instead: one clarification per request.

    `context_note`: what the request refers back to that the scene does not hold, e.g. the
    `/ask` question and answer just before "keep them like that" (session 6, msa#49).

    `window_shown`: the window's t and the 4D values it shows there; a request to keep the view
    gets a finding for each value the new scene no longer gives at that t (msa#49)."""

    def _emit(stage: StageName) -> None:
        if on_stage is None:
            return
        try:
            on_stage(stage)
        except Exception:  # noqa: BLE001 -- cosmetic reporting must never abort a turn
            pass

    question = None if answered else caustics_strength_question(prompt)
    if question is not None:
        return TurnResult(tag="needs_clarification", messages=[question])
    _emit("generating")
    # F93: a material with no preset is a question, not a silent approximation. Only the
    # model's copy of the request carries the hint; history and tickets keep the user's words.
    hints = [] if answered else [unknown_material_hint(prompt, manifest),
                                 undirected_placement_hint(prompt)]
    hints.append(pre_projection_colour_hint(prompt))
    model_prompt = "".join([prompt, *(f"\n\n({h})" for h in hints if h)])
    if context_note:
        model_prompt += f"\n\n({context_note})"
    if window_state and refers_to_window(prompt):
        # F96: "keep them like that" -- the window's own state (a description of the file
        # `<current.scala>.window-state.json`, passed by the REPL) is context for the model.
        model_prompt += f"\n\n({window_state_hint(window_state)})"

    def _derive(text: str) -> GenerationResult:
        if prior_scene is None:
            return generate(text, manifest, corpus, adapter)
        return revise(text, prior_scene, manifest, corpus, adapter)

    generation_result = _derive(model_prompt)
    if (answered and repair_rounds > 0 and isinstance(generation_result, GenerationError)
            and generation_result.kind == "needs_clarification"):
        _emit("repairing (asked again)")
        generation_result = _derive(
            f"{model_prompt}\n\nYou asked again ({generation_result.message}), but the user has "
            "already answered. Do not ask a second time: compose the scene now with the nearest "
            "reading of the request and the answer, and state the assumption in the doc comment."
        )

    if isinstance(generation_result, GenerationError):
        # Not one of the frozen I/O & Edge-Case Matrix's five rows, but a real precondition
        # for all of them: gauntlet.check_*() takes a `str`, so a generation failure must be
        # handled before any local check runs -- no staging file is ever written, no
        # renderer call is ever made, same as the local-finding path below.
        #
        # spec-ai-scene-agent story 15: a model-call timeout gets its own distinct tag
        # ("generation_timeout") rather than the generic "generation_failed" every other
        # GenerationError kind maps to -- a user seeing "Turn N: generation_failed" has no
        # way to tell a hung model-provider request apart from any other generation failure.
        #
        # spec-ai-scene-agent story 18: same reasoning for "needs_clarification" -- the model
        # itself judged the request too ambiguous or self-contradictory to compose (PRD FR5),
        # distinct from both "generation_failed" and "generation_timeout" end to end.
        #
        # usability review 2026-09 (F16, msa#3): same reasoning again for "unsupported" -- a
        # request the DSL genuinely cannot satisfy at all, distinct from "needs_clarification"
        # (see TurnTag's own docstring for why `cli.py` treats the two differently).
        return _generation_failure(store, prompt, generation_result)

    scene_text = generation_result

    unchanged = prior_scene is not None and scene_text.split() == prior_scene.split()
    # Session 5 (F117, F118): a reply that only added `// NOT DONE` lines changed nothing
    # either; its round asks for the nearest it named.
    only_not_done = (
        prior_scene is not None and not unchanged and _same_code(scene_text, prior_scene)
        and not_done_parts(scene_text) != not_done_parts(prior_scene)
    )
    if prior_scene is not None and (unchanged or only_not_done) and repair_rounds > 0:
        # Session 4, 4.3 (F92): an impossible request came back as the unchanged scene, and
        # the user got "No change made ... rephrase it" instead of a decline. One more round
        # asks for the change, a decline with a nearest, or a question.
        _emit("repairing (local finding)")
        if only_not_done:
            instruction = (
                "You marked part of the request NOT DONE but changed nothing else. Compose the "
                "nearest approximation you named now, say in the doc comment that it is an "
                "approximation, and keep the NOT DONE line for what is still missing."
            )
        else:
            instruction = (
                "You returned the current scene unchanged. If the request can be done, make "
                "the change. If it cannot be done at all, respond UNSUPPORTED naming what is "
                "not possible and the nearest thing that is. If it is unclear, respond "
                "NEEDS_CLARIFICATION with your question."
            )
        retried = revise(f"{prompt}\n\n{instruction}", prior_scene, manifest, corpus, adapter)
        if isinstance(retried, GenerationError):
            return _generation_failure(store, prompt, retried)
        scene_text = retried

    if prior_scene is not None and scene_text.split() == prior_scene.split():
        # F76 (recurrence of F56): the model returned the current scene. Not a new turn: say
        # so instead of presenting it with a readback as if something had been done.
        message = (
            "No change made: the model returned the current scene unchanged, so nothing was "
            "saved. If you asked for a change, rephrase it."
        )
        messages = _record_rejected_safely(store, prompt, f"unchanged: {message}", [message])
        return TurnResult(tag="unchanged", messages=messages)

    staging_path = store.session_dir / _STAGING_FILENAME
    # The prior turn's evaluated scene, if it is the scene this turn revises (session 4, F90).
    prior_snapshots = (
        store.current_snapshots()
        if prior_scene is not None and prior_scene == store.current_scene()
        else []
    )
    checks = CheckResult()
    try:
        # F47 (PRD FR-7 amendment 2026-10-02): a compile/lint failure of the agent's own
        # candidate gets up to `repair_rounds` revisions with the failure fed back; each round
        # is a visible stage and each failed candidate is kept by `_record_rejected_safely`.
        rounds_done = 0
        first_candidate = scene_text
        repair_reasons: List[str] = []
        while True:
            _emit("validating")
            from_checks = False  # the rejection below comes from the scene checks
            local_findings = _run_local_checks(scene_text)
            if local_findings:
                # Always: "a renderer request is never made when a local finding exists" --
                # no staging file is written on this path at all.
                feedback = [finding.message for finding in local_findings]
                messages = _record_rejected_safely(
                    store, prompt, _local_finding_reason(local_findings), feedback, scene_text
                )
                rejection = TurnResult(
                    tag="local_finding", messages=messages, findings=local_findings
                )
            else:
                try:
                    # Moved inside the try (review round, patch-level fix): a write failure
                    # here (disk full, permission error) is exactly as real a failure mode as
                    # any other storage failure this function guards against, and must become
                    # a typed `TurnResult` rather than an unhandled `OSError` escaping.
                    staging_path.write_text(scene_text, encoding="utf-8")
                except OSError as e:
                    reason = f"storage_failed: could not write staging file: {e}"
                    messages = _record_rejected_safely(store, prompt, reason, [reason])
                    return TurnResult(tag="storage_failed", messages=messages)

                outcome = validate_scene(staging_path, script_path, image=image, timeout=timeout)

                if isinstance(outcome, ValidationError):
                    messages = _record_rejected_safely(
                        store, prompt, f"{outcome.kind}: {outcome.message}", [outcome.message],
                        scene_text,
                    )
                    return TurnResult(tag=outcome.kind, messages=messages)

                if outcome.tag == "ok":
                    # Session 4 (F89, F102): checks on the evaluated scene; their findings go
                    # back to the model like a local finding, what is left becomes a warning.
                    checks = check_turn(
                        prior_snapshots, outcome.snapshots, outcome.camera_path, prompt, manifest
                    )
                    if window_shown is not None and asks_to_keep_view(prompt):
                        checks.findings.extend(kept_view_findings(*window_shown, outcome.snapshots))
                    if not checks.findings or rounds_done >= repair_rounds:
                        break
                    feedback = list(checks.findings)
                    rejection_tag = "local_finding"
                    from_checks = True
                else:
                    from_checks = False
                    # `_validation_messages(outcome)` can itself be empty (a bare non-"ok" tag
                    # with neither `messages` nor `findings` populated, e.g. a bare
                    # "refused") -- fall back to the tag itself so `TurnResult.messages` is
                    # never empty on a non-accepted tag (review round, patch-level fix;
                    # mirrors the fallback used for the `record_rejected()` reason string).
                    feedback = _validation_messages(outcome) or [outcome.tag]
                    rejection_tag = outcome.tag
                messages = _record_rejected_safely(
                    store, prompt, "; ".join(feedback) or rejection_tag, feedback, scene_text
                )
                rejection = TurnResult(tag=rejection_tag, messages=messages)

            if rejection.tag not in _REPAIRABLE_TAGS or rounds_done >= repair_rounds:
                return _with_first_attempt(rejection, first_candidate, rounds_done)
            _emit(_REPAIRABLE_TAGS[rejection.tag])
            repaired = revise(
                f"{prompt}\n\nThe scene below is your previous attempt at this request; it was "
                f"rejected with: {'; '.join(feedback)}. Fix exactly that and change nothing else.",
                scene_text, manifest, corpus, adapter,
            )
            unrepaired = isinstance(repaired, GenerationError) or repaired.split() == scene_text.split()
            if unrepaired and from_checks:
                # The scene itself validated; a scene-check finding the repair could not fix is
                # shown as a warning, never a reason to reject a valid scene.
                break
            if isinstance(repaired, GenerationError):
                return _with_first_attempt(rejection, first_candidate, rounds_done)
            if repaired.split() == scene_text.split():
                # F100: a repair that changes nothing cannot fix anything; stop at once (8a turn
                # 7 spent its last round on an identical file).
                return _with_first_attempt(
                    TurnResult(
                        tag=rejection.tag,
                        messages=rejection.messages + [_unrepaired_note(feedback)],
                        findings=rejection.findings,
                    ),
                    first_candidate,
                    rounds_done,
                )
            scene_text = repaired
            repair_reasons.extend(feedback)
            rounds_done += 1

        new_facts = extract_scene_facts(strip_comments_and_strings(scene_text))
        # F60: a repair may change what the request asked for (a level 5.8 clamped to 5 to fit
        # a limit); every fact it changed is reported, with the reason it was repaired.
        repair_warnings = [
            f"the automatic repair changed {change} (to fix: {'; '.join(repair_reasons)})"
            for change in scene_changes(
                extract_scene_facts(strip_comments_and_strings(first_candidate)), new_facts
            )
        ] if rounds_done else []
        prior_facts = (
            extract_scene_facts(strip_comments_and_strings(prior_scene))
            if prior_scene is not None
            else None
        )
        snapshots = outcome.snapshots
        # F90: with both turns' evaluated scenes, changes and unrequested-change warnings are
        # values; without (an older image, a hand-edited prior), the text-parsed facts.
        has_snapshots = bool(prior_snapshots) and bool(snapshots)
        if has_snapshots:
            turn_warnings = unrequested_changes(prior_snapshots, snapshots, prompt, manifest)
            changes: Optional[List[str]] = snapshot_changes(prior_snapshots, snapshots, manifest)
        else:
            turn_warnings = facts_diff(prior_facts, new_facts, prompt) if prior_facts else []
            changes = None
        # F89/F102 findings the repair rounds did not fix, and the checks' own warnings.
        turn_warnings = turn_warnings + checks.findings + checks.warnings

        # F91: a scene warning or a Not done part the previous scene already had was reported
        # then; it is counted, not repeated (session 4: one warning printed 8 times).
        current_warnings = scene_warnings(new_facts, scene_text, manifest, snapshots)
        earlier_warnings = (
            set(scene_warnings(prior_facts, prior_scene, manifest, prior_snapshots))
            if prior_scene is not None and prior_facts is not None
            else set()
        )
        new_scene_warnings = [w for w in current_warnings if w not in earlier_warnings]
        turn_warnings = repair_warnings + turn_warnings + new_scene_warnings

        scene_text, stale_not_done = drop_stale_not_done(scene_text)
        earlier_not_done = set(not_done_parts(prior_scene))
        current_not_done = not_done_parts(scene_text)
        new_not_done, caveats = split_not_done(
            [p for p in current_not_done if p not in earlier_not_done]
        )
        carried_over = (len(current_warnings) - len(new_scene_warnings)) + (
            len(current_not_done) - len(new_not_done) - len(caveats)
        )

        _emit("reading back")
        readback_result = semantic_readback(
            scene_text, adapter, request=prompt, prior_scene=prior_scene, changes=changes
        )
        if isinstance(readback_result, ReadbackError):
            # I/O & Edge-Case Matrix: "turn is NOT accepted without its summary -- treated
            # as rejection, not a partial accept."
            messages = _record_rejected_safely(
                store,
                prompt,
                f"readback_failed: {readback_result.message}",
                [readback_result.message],
                scene_text,
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
        if snapshots:
            store.write_snapshots(ordinal, snapshots)
        play_once = [n for n in [play_once_note(prompt, scene_text)] if n is not None]
        return TurnResult(
            tag="accepted",
            messages=[],
            readback_summary=readback_result,
            ordinal=ordinal,
            removed_properties=removed_properties(prior_scene, scene_text),
            warnings=turn_warnings,
            not_done=new_not_done,
            stale_not_done=stale_not_done,
            carried_over=carried_over,
            changes=changes,
            notes=checks.notes + [f"the scene notes: {c}" for c in caveats] + play_once,
            snapshots_missing=not snapshots,
        )
    finally:
        # Unconditional (Boundaries & Constraints: "The staging file is always deleted
        # (success or failure) before run_turn returns -- never left in session_dir").
        staging_path.unlink(missing_ok=True)


# spec-ai-scene-agent story 17 ("hand-edit fallback, lint-gated and ordinal-assigned"): the
# synthetic prompt recorded for a promoted hand edit -- names the source in history.jsonl the
# same way any other prompt would, without inventing a second history schema.
_HAND_EDIT_PROMPT = "<hand-edit>"


def check_hand_edit(
    store: SceneStore,
    known_scene: Optional[str],
    script_path: Optional[Union[str, Path]] = None,
) -> Optional[TurnResult]:
    """Detects and gates an out-of-band hand edit to the current scene file (spec-ai-scene-
    agent story 17): AD-13 defines "current scene" as whatever the last file on disk
    contains, but nothing checked a hand-edited file before this -- AD-4's lint/allowlist
    gate and AD-7/AD-8's "no in-place mutation" invariant both assumed content reaches disk
    only through `run_turn()`. A raw hand edit reaches neither.

    Detection is a plain value comparison, not a filesystem watch: re-reads
    `store.current_scene()` fresh from disk and compares it against `known_scene` (the
    caller's last-known value). Returns `None` when they match -- either nothing changed, or
    (both `None`) no turn has been accepted yet in this session -- there is nothing to
    compare or gate.

    When they differ, an external edit is assumed. Only the same *local* gauntlet
    aggregation `run_turn()` already uses (`_run_local_checks`/`_local_finding_reason`) runs
    against it -- per FR9's explicit Out of Scope, and unlike `run_turn()`, this never runs a
    model call and never invokes `validate_scene()` (AD-4 control 1, lint-only). A clean pass
    is persisted under a brand-new ordinal via `store.accept()` -- never an in-place rewrite
    of the edited file -- with a synthetic prompt (`_HAND_EDIT_PROMPT`) naming the source, and
    returned as an `"accepted"` `TurnResult`; callers should treat this exactly like any other
    accepted turn (refresh their tracked "current scene" value and the render window).

    A finding calls `store.record_rejected(_HAND_EDIT_PROMPT, reason)` (review round,
    patch-level fix: this used to be the one rejection path in this system with no
    `history.jsonl` trace at all -- every other rejection path, local-finding or
    renderer-side, already calls `record_rejected()`; a rejected hand edit is exactly as real
    an attempt and is now recorded the same way, using the same synthetic prompt the accepted
    case already uses) before returning a `"hand_edit_rejected"` `TurnResult` naming the
    violation (via `_local_finding_reason`, the same reason string `run_turn()`'s own
    local-finding path computes). The edit itself is never promoted to a new ordinal -- it is
    simply left un-promoted. The caller's `known_scene` is untouched by this function
    returning -- it just doesn't hand back an updated value, so the prior accepted content
    remains current for pipeline purposes even though the on-disk file itself still holds the
    failing edit until the user fixes or reverts it.

    Never raises for an expected outcome, mirroring `run_turn()`'s own contract: both the
    initial `store.current_scene()` read and the later `store.accept()` call can each raise
    `SceneStoreError` (the same "exhausted ordinal-claim retries under contention"/"session
    directory became inaccessible" failure modes `run_turn()` guards against -- review round,
    patch-level fix: the `current_scene()` read used to be unguarded here, able to escape this
    function's own never-raises contract if the session directory became inaccessible between
    calls). Both are reported as a `"storage_failed"` `TurnResult`, never an unhandled
    `SceneStoreError`.
    """
    try:
        current = store.current_scene()
    except SceneStoreError as e:
        return TurnResult(tag="storage_failed", messages=[str(e)])
    if current == known_scene:
        return None
    if current is None:
        # Unreachable through this function's own normal call paths (three independent
        # automated reviewers confirmed this, review round -- including one noting the
        # existing test named as covering it actually short-circuits earlier via the
        # `current == known_scene` check above instead): current_scene() returning None means
        # no ordinal file exists at all, which requires known_scene to also be None for any
        # caller following this module's own contract (nothing accepted yet in this
        # session) -- and current == known_scene (both None) already returns above before
        # this line is ever reached. Reaching here would require every ordinal file to vanish
        # from session_dir between the caller learning a non-None known_scene and this call;
        # AD-8 says this codebase itself never deletes an ordinal, so no code path of this
        # module's own can produce that -- only an external actor (someone manually deleting
        # files from the session directory) could. Kept, not deleted, as a defensive backstop
        # against exactly that external-tampering scenario: current_scene()'s own return type
        # is Optional, and _run_local_checks() below requires a str, so this must still be
        # handled rather than assumed away.
        return None

    findings = _run_local_checks(current)
    if findings:
        reason = _local_finding_reason(findings)
        messages = _record_rejected_safely(store, _HAND_EDIT_PROMPT, reason, [reason])
        return TurnResult(
            tag="hand_edit_rejected",
            messages=messages,
            findings=findings,
        )

    # Review 2026-10-14 (menger#88): a hand edit used to be promoted after local lint only,
    # although an accepted scene is what the render window runs. With a validator configured
    # it now passes the same sandboxed validation as a generated scene.
    if script_path is not None:
        rejection = _validate_hand_edit(store, current, script_path)
        if rejection is not None:
            return rejection

    try:
        ordinal = store.accept(current, _HAND_EDIT_PROMPT)
    except SceneStoreError as e:
        return TurnResult(tag="storage_failed", messages=[str(e)])

    return TurnResult(tag="accepted", messages=[], ordinal=ordinal)


def _validate_hand_edit(
    store: SceneStore, scene_text: str, script_path: Union[str, Path]
) -> Optional[TurnResult]:
    """The sandboxed validator's verdict on a hand edit: None when it is "ok", else the
    `"hand_edit_rejected"` (or error-kind) `TurnResult` naming why, recorded in history."""
    staging_path = store.session_dir / _STAGING_FILENAME
    try:
        staging_path.write_text(scene_text, encoding="utf-8")
    except OSError as e:
        return TurnResult(tag="storage_failed", messages=[f"could not write staging file: {e}"])
    try:
        outcome = validate_scene(staging_path, script_path)
    finally:
        staging_path.unlink(missing_ok=True)
    if isinstance(outcome, ValidationError):
        reason = f"{outcome.kind}: {outcome.message}"
        return TurnResult(
            tag=outcome.kind,
            messages=_record_rejected_safely(store, _HAND_EDIT_PROMPT, reason, [reason]),
        )
    if outcome.tag == "ok":
        return None
    feedback = _validation_messages(outcome) or [outcome.tag]
    return TurnResult(
        tag="hand_edit_rejected",
        messages=_record_rejected_safely(store, _HAND_EDIT_PROMPT, "; ".join(feedback), feedback),
    )

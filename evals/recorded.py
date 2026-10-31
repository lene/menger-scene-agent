"""Model-output eval on recorded responses (review 2026-10-14, msa#30; decided by user
2026-10-16: "Recorded first, live later").

A case is a usability-session request sequence with the model's recorded raw responses and,
per turn, objective expectations from the request itself: which object types, materials and
lights the scene must have, what its text must say. `run_case` replays it through the real
`run_turn()` -- prompt composition, extraction, both gauntlets, the scene checks, repair
rounds, readback -- with `ReplayModelAdapter` in place of the model, and each validator call
answered from the turn's recorded `validator` results (menger's result JSON, snapshots
included), so the agent's snapshot and budget checks run too (menger#78). A turn without them
gets the stub's bare "ok". So a change to the agent that breaks a turn the model once got
right fails here, without a model call or a GPU.

`--live` sends the prompts to the configured provider instead (paid: the user's call), the
recorded cases and `evals/script/` -- every usability-script request with its conditions, no
recorded responses (msa#57). `--repeat N` runs each case N times (models vary); pass counts
and the questions asked are kept per run in
`evals/live-results/<provider>-<model>/<UTC time>-<agent revision>.json`, the newest also as
`latest.json`. A case whose pass rate falls below its best in the model's last
`RATCHET_RUNS` kept runs by more than one run's share is flagged. `--compare A B` prints two
kept runs' pass rates per case. `--validator` takes menger's `run-sandboxed.sh` instead of the
stub; its answers must then equal the recorded ones (the nightly job in menger-toplevel's
agent-contract.yml). `--record` writes the validator results seen, and live the model's
responses, into each named case that passes: after a menger change, without `--live`; when the
agent now makes calls the recording lacks, with it.

Usage: uv run python -m evals.recorded [--live [--repeat N]] [--validator SCRIPT] [--record]
                                       [case.json ...]
       uv run python -m evals.recorded --compare A.json B.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import core.turn
from adapters.artifacts import load_corpus, load_manifest
from adapters.model import ModelError, ModelRequest, ModelResult
from adapters.model_factory import get_model_adapter, selected_provider
from adapters.process_contract import EXIT_CODES
from adapters.scene_store import SceneStore
from adapters.scene_validator import _parse_stdout, validate_scene
from core.ports import ModelAdapter
from core.scene_facts import extract_scene_facts
from core.turn import Validator, run_turn, use_default_validator
from core.types import ValidationError, ValidationOutcome
from gauntlet._scala_text import strip_comments_and_strings
from repl.state import _build_clarification_prompt, _ClarificationState

ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = Path(__file__).resolve().parent / "cases"
STUB_VALIDATOR = Path(__file__).resolve().parent / "stub_validator.py"
LIVE_RESULTS_DIR = Path(__file__).resolve().parent / "live-results"
LATEST = "latest.json"
# Kept runs a new live run's pass rates are held against.
RATCHET_RUNS = 3
# The usability script's requests with conditions and no recorded responses: live runs only.
SCRIPT_DIR = Path(__file__).resolve().parent / "script"
DARK = 0.15  # every background channel at or below this is "dark"
# An indented JSON array of plain values (no nested array or object): one per line as
# json.dumps writes it. A raw newline never occurs inside a JSON string, so joining at
# newlines changes only whitespace between tokens.
_SCALAR_ARRAY = re.compile(r"\[\n *([^\[\]{}\n]+(?:\n *[^\[\]{}\n]+)*)\n *\]")


@dataclass
class ReplayModelAdapter:
    """Answers each model call with the next recorded response. Running out means the agent
    now makes a call the recording does not have (an extra repair round, a new stage): the
    call fails, the turn is rejected and the case reports it."""

    responses: Iterator[str]
    model_name: str = "replay"
    calls: List[str] = field(default_factory=list)

    def complete(self, request: ModelRequest) -> ModelResult:
        self.calls.append(request.user_prompt)
        try:
            return next(self.responses)
        except StopIteration:
            return ModelError(kind="call_failed", message="no recorded response left for this call")


def _first_match(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(0) if match else None


def _scene_problems(expect: dict, scene: str, prior: str) -> List[str]:
    """The conditions on the scene file (the `expect` keys are listed in evals/script/README.md)."""
    facts = extract_scene_facts(strip_comments_and_strings(scene))
    problems = []
    types = [o.type_name for o in facts.objects]
    materials = [o.material for o in facts.objects]
    lights = [light.type_name for light in facts.lights]
    for name in expect.get("objects", []):
        if name not in types:
            problems.append(f"no {name} object (objects: {types})")
    for name in expect.get("no_objects", []):
        if name in types:
            problems.append(f"a {name} object is still there")
    for name, (lo, hi) in expect.get("object_count", {}).items():
        if not lo <= types.count(name) <= hi:
            problems.append(f"{types.count(name)} {name} objects, expected {lo} to {hi}")
    for name in expect.get("materials", []):
        if name not in materials:
            problems.append(f"no {name} material (materials: {materials})")
    for name in expect.get("lights", []):
        if name not in lights:
            problems.append(f"no {name} light (lights: {lights})")
    for name, (lo, hi) in expect.get("levels", {}).items():
        levels = [o.level for o in facts.objects if o.type_name == name]
        if not any(level is not None and lo <= level <= hi for level in levels):
            problems.append(f"{name} level {levels}, expected {lo} to {hi}")
    for upper, lower in expect.get("above", []):
        ys = {o.type_name: o.pos[1] for o in facts.objects if o.pos is not None}
        if not (upper in ys and lower in ys and ys[upper] > ys[lower]):
            problems.append(f"{upper} is not above {lower} ({ys})")
    for pattern in expect.get("contains", []):
        if not re.search(pattern, scene):
            problems.append(f"scene text does not match {pattern!r}")
    for pattern in expect.get("not_contains", []):
        if re.search(pattern, scene):
            problems.append(f"scene text matches {pattern!r}")
    for pattern, minimum in expect.get("min_count", {}).items():
        if len(re.findall(pattern, scene)) < minimum:
            problems.append(f"fewer than {minimum} matches of {pattern!r}")
    for pattern in expect.get("changed", []):
        if _first_match(pattern, scene) == _first_match(pattern, prior):
            problems.append(f"{pattern!r} unchanged: {_first_match(pattern, scene)!r}")
    if expect.get("camera_farther"):
        before = extract_scene_facts(strip_comments_and_strings(prior)).camera
        if facts.camera is None or before is None or (
            _distance(*facts.camera) <= _distance(*before)
        ):
            problems.append(f"camera {facts.camera} is not farther out than {before}")
    if expect.get("dark_background") and not (
        facts.background is not None and max(facts.background) <= DARK
    ):
        problems.append(f"background {facts.background} is not dark")
    return problems


def _distance(position: Tuple[float, ...], look_at: Tuple[float, ...]) -> float:
    return sum((a - b) ** 2 for a, b in zip(position, look_at)) ** 0.5


def _turn_problems(
    expect: dict, scene: str, tag: str, messages: Sequence[str] = (), questions: int = 0,
    prior: str = "",
) -> List[str]:
    problems = []
    if questions > expect.get("questions_at_most", 1):
        problems.append(f"{questions} question(s), at most {expect.get('questions_at_most', 1)}")
    tags = expect.get("tag", "accepted")
    if tag not in ([tags] if isinstance(tags, str) else tags):
        return problems + [f"tag {tag}, expected {tags}", f"messages: {list(messages)}"]
    for pattern in expect.get("message", []):
        if not re.search(pattern, " ".join(messages)):
            problems.append(f"messages do not match {pattern!r}: {list(messages)}")
    return problems + _scene_problems(expect, scene, prior)


@dataclass
class CaseResult:
    case_id: str
    problems: List[str]
    questions: List[str] = field(default_factory=list)
    recorded: Optional[List[dict]] = None  # `--record`: the case's turns as this run saw them

    @property
    def passed(self) -> bool:
        return not self.problems


@dataclass
class _QuestionCounter:
    """Passes calls on and keeps every response, and every question the model asked."""

    inner: ModelAdapter
    questions: List[str] = field(default_factory=list)
    responses: List[ModelResult] = field(default_factory=list)

    def complete(self, request: ModelRequest) -> ModelResult:
        result = self.inner.complete(request)
        self.responses.append(result)
        if isinstance(result, str) and result.lstrip().startswith("NEEDS_CLARIFICATION"):
            self.questions.append(result.strip())
        return result


def _as_entry(outcome: ValidationOutcome) -> dict:
    """A validator outcome as a case records it: menger's result JSON and its exit code."""
    if isinstance(outcome, ValidationError):
        return {"error": outcome.kind, "message": outcome.message}
    wire_tag = outcome.tag.replace("_", "-")
    return {"exit": EXIT_CODES[wire_tag], "result": {
        "tag": wire_tag, "messages": outcome.messages,
        "findings": [{"invariant": f.invariant, "message": f.message} for f in outcome.findings],
        "scene": outcome.scene, "schemaVersion": outcome.schema_version,
        "snapshots": outcome.snapshots, "cameraPath": outcome.camera_path}}


def _from_entry(entry: dict) -> ValidationOutcome:
    if "error" in entry:
        return ValidationError(kind=entry["error"], message=entry["message"])
    return _parse_stdout(json.dumps(entry["result"]), "", entry["exit"])


@dataclass
class _ValidatorTape:
    """One turn's validator calls (menger#78). With `recorded` it answers from the
    case's recorded results, snapshots included, so the agent's snapshot and budget checks run
    without menger; without, it runs `inner` and keeps what it said, for `--record` and for the
    drift check against the real validator."""

    inner: Validator
    recorded: Optional[List[dict]] = None
    entries: List[dict] = field(default_factory=list)

    def __call__(self, *args: object, **kwargs: object) -> ValidationOutcome:
        if self.recorded is None:
            outcome = self.inner(*args, **kwargs)
        elif len(self.entries) < len(self.recorded):
            outcome = _from_entry(self.recorded[len(self.entries)])
        else:
            outcome = ValidationError(kind="subprocess_failed",
                                      message="no recorded validator result left for this call")
        self.entries.append(_as_entry(outcome))
        return outcome


# What the result says about itself, not about the scene: an additive schema bump (menger#113's
# loadError made it 1.3.0) must not make every recording drift. Whether the agent can read the
# version is `adapters.process_contract.result_problem`'s job, on every real run.
_METADATA_KEYS = frozenset({"schemaVersion"})


def _drift(recorded: List[dict], seen: List[dict]) -> List[str]:
    """Where the validator's answers differ from the recorded ones: the case was recorded
    against a menger that validated differently, so it needs `--record` again."""
    if len(recorded) != len(seen):
        return [f"{len(seen)} validator call(s), {len(recorded)} recorded"]
    return [f"validator call {k}: {key} differs from the recording"
            for k, (old, new) in enumerate(zip(recorded, seen), start=1)
            for key in sorted(old.get("result", old).keys() | new.get("result", new).keys())
            if key not in _METADATA_KEYS
            if old.get("result", old).get(key) != new.get("result", new).get(key)]


def run_case(
    case: dict, manifest: dict, corpus: dict, live: Optional[ModelAdapter] = None,
    validator: Path = STUB_VALIDATOR, record: bool = False,
) -> CaseResult:
    """Runs the case's turns in one session. A turn with an `answer` gets it threaded like the
    REPL does (`repl.state`) when the model asks, once.

    Against the stub, a recorded replay answers each validator call from the turn's recorded
    `validator` results; against a real validator, those are what its answers must equal.
    With `record`, a passing case's turns get the validator results seen (and, live, the model
    responses) written into the returned `CaseResult.recorded`."""
    problems: List[str] = []
    questions: List[str] = []
    use_default_validator(validate_scene)  # the real adapter, by default against the stub
    inner = core.turn.validate_scene
    replay = live is None and validator == STUB_VALIDATOR and not record
    recorded_turns = copy.deepcopy(case["turns"])
    prior: Optional[str] = None
    with tempfile.TemporaryDirectory() as tmp:
        store = SceneStore.create_session(Path(tmp), session_id="eval")
        try:
            for n, (turn, out) in enumerate(zip(case["turns"], recorded_turns), start=1):
                adapter = _QuestionCounter(live or ReplayModelAdapter(iter(turn["responses"])))
                tape = _ValidatorTape(inner, turn.get("validator") if replay else None)
                core.turn.use_validator(tape)
                prompt = turn["prompt"]
                result = run_turn(prompt, prior, manifest, corpus, adapter, store, validator,
                                  answered=turn.get("answered", False))
                if result.tag == "needs_clarification" and "answer" in turn:
                    pending = _ClarificationState(pending_prompt=prompt,
                                                  reason="; ".join(result.messages))
                    result = run_turn(_build_clarification_prompt(pending, turn["answer"]), prior,
                                      manifest, corpus, adapter, store, validator, answered=True)
                scene = store.current_scene() or ""
                turn_problems = _turn_problems(turn.get("expect", {}), scene, result.tag,
                                               result.messages, len(adapter.questions), prior or "")
                if not replay and not record and live is None and "validator" in turn:
                    turn_problems += _drift(turn["validator"], tape.entries)
                problems += [f"turn {n} ({prompt[:40]!r}): {p}" for p in turn_problems]
                questions += adapter.questions
                out["validator"] = tape.entries
                if live is not None:
                    out["responses"] = adapter.responses
                if result.tag == "accepted":
                    prior = scene
        finally:
            core.turn.use_validator(inner)
    return CaseResult(case["id"], problems, questions,
                      recorded_turns if record and not problems else None)


def load_cases(paths: Optional[List[Path]] = None, live: bool = False) -> List[dict]:
    """Recorded cases; with `live`, also the usability script's (no recorded responses)."""
    dirs = [CASES_DIR, SCRIPT_DIR] if live else [CASES_DIR]
    files = paths or [p for d in dirs for p in sorted(d.glob("*.json"))]
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def _agent_revision() -> str:
    """The agent's commit, `-dirty` with uncommitted changes: what code the run measured."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_live_results(
    adapter: ModelAdapter, runs: Dict[str, List[CaseResult]],
    stamp: Optional[str] = None, revision: Optional[str] = None,
) -> Path:
    model = getattr(adapter, "model_name", "unknown")
    directory = LIVE_RESULTS_DIR / f"{selected_provider()}-{model}".replace("/", "_")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stamp}-{revision or _agent_revision()}.json"
    text = json.dumps([
        {"case": case_id, "passed": sum(r.passed for r in results), "runs": len(results),
         "questions": [r.questions for r in results], "problems": [r.problems for r in results]}
        for case_id, results in runs.items()
    ], indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    (directory / LATEST).write_text(text, encoding="utf-8")
    return path


def _entries(path: Path) -> Dict[str, Tuple[int, int]]:
    """`{case: (passed, runs)}` of a kept run."""
    return {e["case"]: (e["passed"], e["runs"])
            for e in json.loads(path.read_text(encoding="utf-8")) if e["runs"]}


def compare_runs(a: Path, b: Path) -> List[str]:
    """One line per case: its pass rate in `a`, in `b`, and the change."""
    before, after = _entries(a), _entries(b)

    def rate(entry: Optional[Tuple[int, int]]) -> str:
        return "   -" if entry is None else f"{entry[0] / entry[1]:4.0%}"

    lines = []
    for case in sorted(before.keys() | after.keys()):
        x, y = before.get(case), after.get(case)
        change = "" if x is None or y is None else f"  {y[0] / y[1] - x[0] / x[1]:+.0%}"
        lines.append(f"{rate(x)} -> {rate(y)}{change}  {case}")
    return lines


def previous_runs(path: Path) -> List[Path]:
    """The model's last `RATCHET_RUNS` kept runs before `path` (names start with the time)."""
    kept = sorted(p for p in path.parent.glob("*.json") if p.name != LATEST and p.name < path.name)
    return kept[-RATCHET_RUNS:]


def ratchet_drops(path: Path, previous: Sequence[Path]) -> List[str]:
    """Cases whose pass rate in `path` is below their best in `previous` by more than one
    run's share: with `--repeat 3`, 100% -> 33% is flagged, 100% -> 67% is noise."""
    best: Dict[str, float] = {}
    for p in previous:
        for case, (passed, runs) in _entries(p).items():
            best[case] = max(best.get(case, 0.0), passed / runs)
    drops = []
    for case, (passed, runs) in _entries(path).items():
        if case in best and passed / runs < best[case] - 1 / runs - 1e-9:
            drops.append(f"{case}: {passed}/{runs} runs, best of the last {len(previous)} kept "
                         f"run(s) {best[case]:.0%}")
    return drops


def _write_recording(path: Path, case: dict, turns: List[dict],
                     live: Optional[ModelAdapter]) -> None:
    case = {**case, "turns": turns}
    if live is not None:
        case["recorded_live"] = {
            "provider": selected_provider(), "model": getattr(live, "model_name", "unknown"),
            "date": datetime.now(timezone.utc).date().isoformat(), "agent": _agent_revision()}
    text = _SCALAR_ARRAY.sub(lambda m: "[" + re.sub(r"\n *", " ", m.group(1)) + "]",
                             json.dumps(case, indent=2))
    path.write_text(text + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.recorded")
    parser.add_argument("cases", nargs="*", type=Path)
    parser.add_argument("--live", action="store_true",
                        help="send the prompts to the configured provider (paid)")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case (live)")
    parser.add_argument("--validator", type=Path, default=STUB_VALIDATOR,
                        help="validator script, e.g. menger's run-sandboxed.sh (default: stub)")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"),
                        help="print two kept live runs' pass rates per case, and exit")
    parser.add_argument("--record", action="store_true",
                        help="write the validator results (and, live, the model responses) of "
                             "each named case that passes back into its file")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.compare:
        print("\n".join(compare_runs(*args.compare)))
        return 0
    if args.record and not args.cases:
        parser.error("--record rewrites case files: name them")
    live = get_model_adapter() if args.live else None
    manifest = load_manifest(ROOT / "reference" / "dsl-manifest.json")
    corpus = load_corpus(ROOT / "reference" / "dsl-corpus.json")
    runs: Dict[str, List[CaseResult]] = {}
    cases = load_cases(args.cases or None, live=args.live)
    for path, case in zip(args.cases or [None] * len(cases), cases):
        runs[case["id"]] = [run_case(case, manifest, corpus, live, args.validator, args.record)
                            for _ in range(args.repeat if args.live else 1)]
        recorded = next((r.recorded for r in runs[case["id"]] if r.recorded), None)
        if args.record and path is not None and recorded is not None:
            _write_recording(path, case, recorded, live)
            print(f"recorded: {path}")
    for case_id, results in runs.items():
        passed = sum(r.passed for r in results)
        asked = sum(len(r.questions) for r in results)
        print(f"{'PASS' if passed == len(results) else 'FAIL'} {case_id}"
              + (f" ({passed}/{len(results)} runs, {asked} question(s))" if live else ""))
        for problem in next((r.problems for r in results if r.problems), []):
            print(f"  {problem}")
    all_passed = sum(all(r.passed for r in results) for results in runs.values())
    print(f"{all_passed}/{len(runs)} cases pass")
    if live is not None and not args.record:  # a recording run measures nothing
        path = _write_live_results(live, runs)
        print(f"written: {path}")
        drops = ratchet_drops(path, previous_runs(path))
        for drop in drops:
            print(f"  dropped: {drop}")
        if drops:
            print(f"{len(drops)} case(s) below their recent best")
    return 0 if all_passed == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

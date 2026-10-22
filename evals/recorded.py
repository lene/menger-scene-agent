"""Model-output eval on recorded responses (review 2026-10-14, msa#30; decided by user
2026-10-16: "Recorded first, live later").

A case is a usability-session request sequence with the model's recorded raw responses and,
per turn, objective expectations from the request itself: which object types, materials and
lights the scene must have, what its text must say. `run_case` replays it through the real
`run_turn()` -- prompt composition, extraction, both gauntlets, the scene checks, repair
rounds, readback -- with `ReplayModelAdapter` in place of the model and a validator stub that
answers "ok" under menger's process contract. So a change to the agent that breaks a turn the
model once got right fails here, without a model call or a GPU.

`--live` sends the prompts to the configured provider instead (paid: the user's call), the
recorded cases and `evals/script/` -- every usability-script request with its conditions, no
recorded responses (msa#57). `--repeat N` runs each case N times (models vary); pass counts
and the questions asked go to `evals/live-results/<provider>-<model>.json`. `--validator`
takes menger's `run-sandboxed.sh` instead of the stub, so compile errors and the scene budget
count too.

Usage: uv run python -m evals.recorded [--live [--repeat N] [--validator SCRIPT]] [case.json ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from adapters.artifacts import load_corpus, load_manifest
from adapters.model import ModelError, ModelRequest, ModelResult
from adapters.model_factory import get_model_adapter, selected_provider
from adapters.scene_store import SceneStore
from adapters.scene_validator import validate_scene
from core.ports import ModelAdapter
from core.scene_facts import extract_scene_facts
from core.turn import run_turn, use_default_validator
from gauntlet._scala_text import strip_comments_and_strings
from repl.state import _build_clarification_prompt, _ClarificationState

ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = Path(__file__).resolve().parent / "cases"
STUB_VALIDATOR = Path(__file__).resolve().parent / "stub_validator.py"
LIVE_RESULTS_DIR = Path(__file__).resolve().parent / "live-results"
# The usability script's requests with conditions and no recorded responses: live runs only.
SCRIPT_DIR = Path(__file__).resolve().parent / "script"
DARK = 0.15  # every background channel at or below this is "dark"


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

    @property
    def passed(self) -> bool:
        return not self.problems


@dataclass
class _QuestionCounter:
    """Passes calls on and keeps every question the model asked."""

    inner: ModelAdapter
    questions: List[str] = field(default_factory=list)

    def complete(self, request: ModelRequest) -> ModelResult:
        result = self.inner.complete(request)
        if isinstance(result, str) and result.lstrip().startswith("NEEDS_CLARIFICATION"):
            self.questions.append(result.strip())
        return result


def run_case(
    case: dict, manifest: dict, corpus: dict, live: Optional[ModelAdapter] = None,
    validator: Path = STUB_VALIDATOR,
) -> CaseResult:
    """Runs the case's turns in one session. A turn with an `answer` gets it threaded like the
    REPL does (`repl.state`) when the model asks, once."""
    problems: List[str] = []
    questions: List[str] = []
    use_default_validator(validate_scene)  # the real adapter, by default against the stub
    prior: Optional[str] = None
    with tempfile.TemporaryDirectory() as tmp:
        store = SceneStore.create_session(Path(tmp), session_id="eval")
        for n, turn in enumerate(case["turns"], start=1):
            adapter = _QuestionCounter(live or ReplayModelAdapter(iter(turn["responses"])))
            prompt = turn["prompt"]
            result = run_turn(prompt, prior, manifest, corpus, adapter, store, validator,
                              answered=turn.get("answered", False))
            if result.tag == "needs_clarification" and "answer" in turn:
                pending = _ClarificationState(pending_prompt=prompt, reason="; ".join(result.messages))
                result = run_turn(_build_clarification_prompt(pending, turn["answer"]), prior,
                                  manifest, corpus, adapter, store, validator, answered=True)
            scene = store.current_scene() or ""
            turn_problems = _turn_problems(turn.get("expect", {}), scene, result.tag,
                                           result.messages, len(adapter.questions), prior or "")
            problems += [f"turn {n} ({prompt[:40]!r}): {p}" for p in turn_problems]
            questions += adapter.questions
            if result.tag == "accepted":
                prior = scene
    return CaseResult(case["id"], problems, questions)


def load_cases(paths: Optional[List[Path]] = None, live: bool = False) -> List[dict]:
    """Recorded cases; with `live`, also the usability script's (no recorded responses)."""
    dirs = [CASES_DIR, SCRIPT_DIR] if live else [CASES_DIR]
    files = paths or [p for d in dirs for p in sorted(d.glob("*.json"))]
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def _write_live_results(adapter: ModelAdapter, runs: Dict[str, List[CaseResult]]) -> Path:
    model = getattr(adapter, "model_name", "unknown")
    path = LIVE_RESULTS_DIR / f"{selected_provider()}-{model}.json".replace("/", "_")
    LIVE_RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps([
        {"case": case_id, "passed": sum(r.passed for r in results), "runs": len(results),
         "questions": [r.questions for r in results], "problems": [r.problems for r in results]}
        for case_id, results in runs.items()
    ], indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.recorded")
    parser.add_argument("cases", nargs="*", type=Path)
    parser.add_argument("--live", action="store_true",
                        help="send the prompts to the configured provider (paid)")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case (live)")
    parser.add_argument("--validator", type=Path, default=STUB_VALIDATOR,
                        help="validator script, e.g. menger's run-sandboxed.sh (default: stub)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    live = get_model_adapter() if args.live else None
    manifest = load_manifest(ROOT / "reference" / "dsl-manifest.json")
    corpus = load_corpus(ROOT / "reference" / "dsl-corpus.json")
    runs: Dict[str, List[CaseResult]] = {}
    for case in load_cases(args.cases or None, live=args.live):
        runs[case["id"]] = [run_case(case, manifest, corpus, live, args.validator)
                            for _ in range(args.repeat if args.live else 1)]
    for case_id, results in runs.items():
        passed = sum(r.passed for r in results)
        asked = sum(len(r.questions) for r in results)
        print(f"{'PASS' if passed == len(results) else 'FAIL'} {case_id}"
              + (f" ({passed}/{len(results)} runs, {asked} question(s))" if live else ""))
        for problem in next((r.problems for r in results if r.problems), []):
            print(f"  {problem}")
    all_passed = sum(all(r.passed for r in results) for results in runs.values())
    print(f"{all_passed}/{len(runs)} cases pass")
    if live is not None:
        print(f"written: {_write_live_results(live, runs)}")
    return 0 if all_passed == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

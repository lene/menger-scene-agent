"""Model-output eval on recorded responses (review 2026-10-14, msa#30; decided by user
2026-10-16: "Recorded first, live later").

A case is a usability-session request sequence with the model's recorded raw responses and,
per turn, objective expectations from the request itself: which object types, materials and
lights the scene must have, what its text must say. `run_case` replays it through the real
`run_turn()` -- prompt composition, extraction, both gauntlets, the scene checks, repair
rounds, readback -- with `ReplayModelAdapter` in place of the model and a validator stub that
answers "ok" under menger's process contract. So a change to the agent that breaks a turn the
model once got right fails here, without a model call or a GPU.

`--live` sends the same prompts to the configured provider instead (paid: the user's call)
and writes pass/fail and the questions the model asked per case to
`evals/live-results/<provider>-<model>.json` (session 5: the prompt side of F104/F105 only
shows with a real model).

Usage: uv run python -m evals.recorded [--live] [case.json ...]   (default: every case)
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from adapters.artifacts import load_corpus, load_manifest
from adapters.model import ModelError, ModelRequest, ModelResult
from adapters.model_factory import get_model_adapter, selected_provider
from adapters.scene_store import SceneStore
from adapters.scene_validator import validate_scene
from core.ports import ModelAdapter
from core.scene_facts import extract_scene_facts
from core.turn import run_turn, use_default_validator
from gauntlet._scala_text import strip_comments_and_strings

ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = Path(__file__).resolve().parent / "cases"
STUB_VALIDATOR = Path(__file__).resolve().parent / "stub_validator.py"
LIVE_RESULTS_DIR = Path(__file__).resolve().parent / "live-results"
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


def _turn_problems(expect: dict, scene: str, tag: str) -> List[str]:
    if tag != expect.get("tag", "accepted"):
        return [f"tag {tag}, expected {expect.get('tag', 'accepted')}"]
    facts = extract_scene_facts(strip_comments_and_strings(scene))
    problems = []
    types = [o.type_name for o in facts.objects]
    materials = [o.material for o in facts.objects]
    lights = [light.type_name for light in facts.lights]
    for name in expect.get("objects", []):
        if name not in types:
            problems.append(f"no {name} object (objects: {types})")
    for name in expect.get("materials", []):
        if name not in materials:
            problems.append(f"no {name} material (materials: {materials})")
    for name in expect.get("lights", []):
        if name not in lights:
            problems.append(f"no {name} light (lights: {lights})")
    for pattern in expect.get("contains", []):
        if not re.search(pattern, scene):
            problems.append(f"scene text does not match {pattern!r}")
    if expect.get("dark_background") and not (
        facts.background is not None and max(facts.background) <= DARK
    ):
        problems.append(f"background {facts.background} is not dark")
    return problems


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
    """Passes calls to a live adapter and keeps every question the model asked."""

    inner: ModelAdapter
    questions: List[str] = field(default_factory=list)

    def complete(self, request: ModelRequest) -> ModelResult:
        result = self.inner.complete(request)
        if isinstance(result, str) and result.lstrip().startswith("NEEDS_CLARIFICATION"):
            self.questions.append(result.strip())
        return result


def run_case(
    case: dict, manifest: dict, corpus: dict, live: Optional[ModelAdapter] = None
) -> CaseResult:
    problems: List[str] = []
    use_default_validator(validate_scene)  # the real adapter, run against the stub script
    prior: Optional[str] = None
    counter = _QuestionCounter(live) if live is not None else None
    with tempfile.TemporaryDirectory() as tmp:
        store = SceneStore.create_session(Path(tmp), session_id="eval")
        for n, turn in enumerate(case["turns"], start=1):
            adapter = counter or ReplayModelAdapter(iter(turn["responses"]))
            result = run_turn(
                turn["prompt"], prior, manifest, corpus, adapter, store, STUB_VALIDATOR,
                answered=turn.get("answered", False),
            )
            scene = store.current_scene() or ""
            turn_problems = _turn_problems(turn.get("expect", {}), scene, result.tag)
            if result.tag != "accepted" and result.messages:
                turn_problems.append(f"messages: {result.messages}")
            problems += [f"turn {n} ({turn['prompt'][:40]!r}): {p}" for p in turn_problems]
            if result.tag == "accepted":
                prior = scene
    return CaseResult(case["id"], problems, counter.questions if counter else [])


def load_cases(paths: Optional[List[Path]] = None) -> List[dict]:
    files = paths or sorted(CASES_DIR.glob("*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def _write_live_results(adapter: ModelAdapter, results: List[CaseResult]) -> Path:
    model = getattr(adapter, "model_name", "unknown")
    path = LIVE_RESULTS_DIR / f"{selected_provider()}-{model}.json".replace("/", "_")
    LIVE_RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps([
        {"case": r.case_id, "passed": r.passed, "questions": r.questions, "problems": r.problems}
        for r in results
    ], indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    live = get_model_adapter() if "--live" in args else None
    paths = [Path(a) for a in args if a != "--live"]
    manifest = load_manifest(ROOT / "reference" / "dsl-manifest.json")
    corpus = load_corpus(ROOT / "reference" / "dsl-corpus.json")
    results = [run_case(c, manifest, corpus, live) for c in load_cases(paths or None)]
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'} {r.case_id}"
              + (f" ({len(r.questions)} question(s))" if live is not None else ""))
        for p in r.problems:
            print(f"  {p}")
    passed = sum(r.passed for r in results)
    print(f"{passed}/{len(results)} cases pass")
    if live is not None:
        print(f"written: {_write_live_results(live, results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

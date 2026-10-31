"""Cold-start wall time of the sandboxed validator over the reference corpus (menger#91).

Every run is one `run-sandboxed.sh` call: a container start, a JVM start and a Scala compile.
A turn makes up to three of them (two repair rounds), so this is the per-turn cost menger
controls; the model's time depends on the provider and is not part of the budget.

Budget (arc42 section 10, P10; decided by user 2026-10-30): p50 <= 5 s, p90 <= 10 s per run.
Advisory: the exit code is 0 unless `--enforce` is given (the CI gate waits for the first
nightly run to say how fast its runner is).

    python3 scripts/time_validator.py --menger-dir ../menger [--image TAG] [--passes 3] [--enforce]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

P50_BUDGET_SECONDS = 5.0
P90_BUDGET_SECONDS = 10.0
DEFAULT_PASSES = 3
DEFAULT_IMAGE = "menger-scene-validator:latest"
_RUN_SANDBOXED = Path("docker/scene-validator/run-sandboxed.sh")
_CORPUS = Path(__file__).resolve().parent.parent / "reference" / "dsl-corpus.json"

# (wall seconds, result tag) of one validator run.
Runner = Callable[[Path], Tuple[float, str]]


@dataclass(frozen=True)
class Summary:
    runs: int
    p50: float
    p90: float
    maximum: float


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, q in (0, 1]."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarise(walls: Sequence[float]) -> Summary:
    return Summary(
        runs=len(walls),
        p50=statistics.median(walls),
        p90=percentile(walls, 0.9),
        maximum=max(walls),
    )


def over_budget(summary: Summary) -> List[str]:
    """What the summary exceeds; empty when it is within the budget."""
    problems = []
    if summary.p50 > P50_BUDGET_SECONDS:
        problems.append(f"p50 {summary.p50:.1f}s > {P50_BUDGET_SECONDS:.1f}s")
    if summary.p90 > P90_BUDGET_SECONDS:
        problems.append(f"p90 {summary.p90:.1f}s > {P90_BUDGET_SECONDS:.1f}s")
    return problems


def time_scenes(runner: Runner, scenes: Sequence[Path], passes: int) -> List[Tuple[Path, float, str]]:
    return [(scene, *runner(scene)) for _ in range(passes) for scene in scenes]


def sandboxed_runner(menger_dir: Path, image: str) -> Runner:
    script = menger_dir / _RUN_SANDBOXED

    def run(scene: Path) -> Tuple[float, str]:
        started = time.monotonic()
        result = subprocess.run(
            [str(script), str(scene), "--image", image], capture_output=True, text=True, check=False
        )
        wall = time.monotonic() - started
        try:
            return wall, str(json.loads(result.stdout)["tag"])
        except (ValueError, KeyError):
            return wall, "unparsed"

    return run


def report(timed: Sequence[Tuple[Path, float, str]]) -> Tuple[Summary, List[str]]:
    """The summary of the runs that returned `ok`, and a line for each that did not (a scene that
    fails to compile finishes early and would bias the statistics down)."""
    ok = [wall for _, wall, tag in timed if tag == "ok"]
    notes = sorted({f"{scene.name}: {tag}" for scene, _, tag in timed if tag != "ok"})
    return summarise(ok), notes


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--menger-dir", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    parser.add_argument("--enforce", action="store_true", help="exit 1 when over the budget")
    args = parser.parse_args(argv)

    scenes = json.loads(_CORPUS.read_text(encoding="utf-8"))["scenes"]
    with tempfile.TemporaryDirectory(prefix="validator-timing-") as tmp:
        files = []
        for i, scene in enumerate(scenes):
            path = Path(tmp) / f"{i:02d}-{scene['name']}.scala"
            path.write_text(scene["source"], encoding="utf-8")
            files.append(path)
        timed = time_scenes(sandboxed_runner(args.menger_dir, args.image), files, args.passes)

    summary, notes = report(timed)
    print(f"cold validator runs: {summary.runs} ok of {len(timed)} "
          f"({len(files)} scenes x {args.passes} passes)")
    print(f"p50 {summary.p50:.2f}s  p90 {summary.p90:.2f}s  max {summary.maximum:.2f}s "
          f"(budget p50 <= {P50_BUDGET_SECONDS:.0f}s, p90 <= {P90_BUDGET_SECONDS:.0f}s)")
    for note in notes:
        print(f"not timed: {note}")
    problems = over_budget(summary)
    for problem in problems:
        print(f"OVER BUDGET: {problem}")
    return 1 if problems and args.enforce else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

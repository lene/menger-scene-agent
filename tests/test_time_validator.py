"""scripts/time_validator.py: the statistics and the budget check (menger#91). The sandboxed run
itself needs Docker and is not exercised here."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "time_validator", Path(__file__).resolve().parent.parent / "scripts" / "time_validator.py"
)
assert _SPEC is not None and _SPEC.loader is not None
tv = importlib.util.module_from_spec(_SPEC)
# A dataclass looks its module up by name while it is created.
sys.modules["time_validator"] = tv
_SPEC.loader.exec_module(tv)


def test_percentile_is_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    assert tv.percentile(values, 0.9) == 9.0
    assert tv.percentile(values, 1.0) == 10.0
    assert tv.percentile(values, 0.1) == 1.0


def test_summarise_gives_median_p90_and_max():
    summary = tv.summarise([3.0, 3.2, 3.4, 3.6, 5.0])

    assert (summary.runs, summary.p50, summary.p90, summary.maximum) == (5, 3.4, 5.0, 5.0)


def test_within_the_budget_nothing_is_reported():
    assert tv.over_budget(tv.Summary(runs=10, p50=3.2, p90=3.6, maximum=5.1)) == []


def test_a_slow_median_and_a_slow_p90_are_each_reported():
    problems = tv.over_budget(tv.Summary(runs=10, p50=5.5, p90=12.0, maximum=20.0))

    assert problems == ["p50 5.5s > 5.0s", "p90 12.0s > 10.0s"]


def test_the_budget_edge_is_inclusive():
    assert tv.over_budget(tv.Summary(runs=1, p50=5.0, p90=10.0, maximum=10.0)) == []


def test_only_ok_runs_are_timed_and_the_others_are_named():
    scenes = [Path("a.scala"), Path("b.scala")]
    results = {"a.scala": (3.0, "ok"), "b.scala": (1.0, "compile-errors")}

    timed = tv.time_scenes(lambda p: results[p.name], scenes, passes=2)
    summary, notes = tv.report(timed)

    assert len(timed) == 4
    assert (summary.runs, summary.p50) == (2, 3.0)
    assert notes == ["b.scala: compile-errors"]


def test_every_scene_is_run_once_per_pass():
    calls = []

    def runner(path):
        calls.append(path.name)
        return 1.0, "ok"

    tv.time_scenes(runner, [Path("a.scala"), Path("b.scala")], passes=3)

    assert calls == ["a.scala", "b.scala"] * 3

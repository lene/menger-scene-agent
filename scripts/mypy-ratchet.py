"""mypy with a baseline that may only shrink (review 2026-10-14, menger-scene-agent#40).

The errors present when the gate was introduced are in mypy-baseline.txt, one per line, without
line numbers (so moving code does not count as a new error). Any error not in it fails.
`--update` rewrites the baseline from the current run; use it only to shrink it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent / "mypy-baseline.txt"
_ERROR = re.compile(r"^(?P<path>[^:]+):\d+(?::\d+)?: error: (?P<message>.*)$")


def current_errors() -> Counter[str]:
    run = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", "--no-pretty", "."],
        capture_output=True, text=True, check=False,
    )
    if run.returncode not in (0, 1):
        sys.exit(f"mypy itself failed (exit {run.returncode}):\n{run.stderr}")
    return Counter(
        f"{m['path']}: {m['message']}" for m in map(_ERROR.match, run.stdout.splitlines()) if m
    )


def main() -> int:
    errors = current_errors()
    if "--update" in sys.argv:
        BASELINE.write_text("".join(f"{e}\n" for e in sorted(errors.elements())))
        print(f"baseline: {sum(errors.values())} errors")
        return 0
    baseline = Counter(BASELINE.read_text().splitlines()) if BASELINE.exists() else Counter()
    new = errors - baseline
    fixed = baseline - errors
    for error in sorted(new.elements()):
        print(f"NEW mypy error: {error}")
    if fixed:
        print(f"{sum(fixed.values())} baselined errors are gone: shrink the baseline with "
              "`uv run python scripts/mypy-ratchet.py --update`")
    print(f"mypy: {sum(errors.values())} errors, baseline {sum(baseline.values())}, "
          f"new {sum(new.values())}")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())

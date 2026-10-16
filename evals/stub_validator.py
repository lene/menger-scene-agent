#!/usr/bin/env python3
"""The eval's stand-in for run-sandboxed.sh: every scene is "ok", printed as menger's process
contract has it (reference/process-result.json), exit 0. The eval tests the agent, not the
renderer; arguments (scene file, --expect-dsl-revision) are ignored."""

import json
from pathlib import Path

contract = json.loads(
    (Path(__file__).resolve().parent.parent / "reference" / "process-result.json").read_text()
)
print(json.dumps({"tag": "ok", "schemaVersion": contract["schemaVersion"], "messages": [],
                  "findings": []}))

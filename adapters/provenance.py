"""What a session ran with and what happened in it (review 2026-10-14, msa#31): a failed or
hostile public session could not be reconstructed. `history.jsonl` had no timestamps and no
model, version or image identity; raw model output and repair rounds were not kept; nothing
logged. This module writes:

- `session.json`: the provenance of the run that created the session (agent and Python
  version, provider and model, manifest version, DSL revision and sha, validator image).
- `model-calls.jsonl`: every model request and its raw response or error, with timing --
  generation, each repair round, readback and consults alike (`RecordingModelAdapter`).
- `agent.log`: the agent's `logging` output for the session (`configure_session_logging`).

Each `history.jsonl` entry carries `ts` and the run's `HISTORY_KEYS` (`SceneStore`).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Union

from adapters.model import ModelAdapter, ModelError, ModelRequest, ModelResult

SESSION_FILENAME = "session.json"
MODEL_CALLS_FILENAME = "model-calls.jsonl"
LOG_FILENAME = "agent.log"
DEFAULT_VALIDATOR_IMAGE = "menger-scene-validator:latest"  # run-sandboxed.sh's default
DSL_REVISION_LABEL = "menger.dsl-content-revision"

# The provenance keys every history entry repeats: enough to tell which run wrote it, when a
# resumed session switched model or agent version.
HISTORY_KEYS = ("agent_version", "provider", "model", "dsl_revision")

_DOCKER_TIMEOUT_SECONDS = 5

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def agent_version() -> str:
    try:
        return importlib.metadata.version("menger-scene-agent")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _image_identity(image: str) -> Dict[str, Optional[str]]:
    """The validator image's id and DSL-revision label, or None for each when docker or the
    image is not there (provenance is a record, never a reason to fail a start)."""
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", image, "--format",
             "{{.Id}} {{index .Config.Labels \"" + DSL_REVISION_LABEL + "\"}}"],
            capture_output=True, text=True, timeout=_DOCKER_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"id": None, "dsl_revision": None}
    if out.returncode != 0:
        return {"id": None, "dsl_revision": None}
    image_id, _, label = out.stdout.strip().partition(" ")
    return {"id": image_id or None, "dsl_revision": label or None}


def collect_provenance(
    manifest: dict,
    manifest_path: Path,
    provider: str,
    model: str,
    validator_script: Union[str, Path],
    render_launcher: Union[str, Path],
    image: Optional[str] = None,
) -> dict:
    validator_image = image or DEFAULT_VALIDATOR_IMAGE
    return {
        "agent_version": agent_version(),
        "python": platform.python_version(),
        "provider": provider,
        "model": model,
        "manifest_version": manifest.get("schemaVersion"),
        "dsl_revision": manifest.get("contentRevision"),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "validator_image": {"name": validator_image, **_image_identity(validator_image)},
        "validator_script": str(validator_script),
        "render_launcher": str(render_launcher),
    }


def history_provenance(provenance: dict) -> dict:
    return {key: provenance.get(key) for key in HISTORY_KEYS}


def write_session_json(session_dir: Path, provenance: dict) -> None:
    """Written once, by the run that created the session. A resumed run leaves it alone: its
    own identity is in every history entry it writes (`HISTORY_KEYS`) and in `agent.log`."""
    path = session_dir / SESSION_FILENAME
    if path.exists():
        logger.info("resumed session; run provenance: %s", json.dumps(provenance, sort_keys=True))
        return
    path.write_text(
        json.dumps({"created": utc_now(), **provenance}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass
class RecordingModelAdapter:
    """Wraps the session's model adapter and appends every call to `model-calls.jsonl`: the
    user prompt, the system prompt's sha256 and length (it is the manifest and corpus, the
    same on every call), and the raw response before any extraction -- or the error. A failed
    write is logged, never raised: recording must not break a turn."""

    inner: ModelAdapter
    path: Path

    @property
    def model_name(self) -> str:
        return getattr(self.inner, "model_name", "unknown")

    def complete(self, request: ModelRequest) -> ModelResult:
        started = time.monotonic()
        result = self.inner.complete(request)
        entry: dict = {
            "ts": utc_now(),
            "seconds": round(time.monotonic() - started, 3),
            "system_prompt_sha256": hashlib.sha256(request.system_prompt.encode()).hexdigest(),
            "system_prompt_chars": len(request.system_prompt),
            "user_prompt": request.user_prompt,
        }
        if isinstance(result, ModelError):
            entry["error"] = {"kind": result.kind, "message": result.message}
        else:
            entry["response"] = result
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError as e:
            logger.warning("could not record a model call in %s: %s", self.path, e)
        logger.info(
            "model call: %.1fs, %s", entry["seconds"],
            f"error {result.kind}" if isinstance(result, ModelError) else f"{len(result)} chars",
        )
        return result


_session_handler: Optional[logging.Handler] = None


def configure_session_logging(session_dir: Path) -> logging.Handler:
    """The agent's loggers write to `<session>/agent.log` (appended across resumed runs). The
    terminal stays as it is: the REPL prints its own status lines. A later call (another
    session in the same process, as in tests) replaces the previous session's handler."""
    global _session_handler
    root = logging.getLogger()
    if _session_handler is not None:
        root.removeHandler(_session_handler)
        _session_handler.close()
    _session_handler = logging.FileHandler(session_dir / LOG_FILENAME, encoding="utf-8")
    _session_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(_session_handler)
    root.setLevel(logging.INFO)
    return _session_handler

"""Loads the versioned manifest and corpus artifacts from disk.

AD-9: both are built in the renderer domain (`ManifestGenerator`, `CorpusExporter` in
`menger`) and mounted read-only here as versioned files under `reference/`. This module is
the only place in the agent repo that reads them -- `core/` never touches the filesystem,
and neither this module nor `core/` ever reads the `menger` repository directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union


class ArtifactError(RuntimeError):
    """Raised when a required artifact is missing or its on-disk JSON cannot be parsed --
    fails fast, with a clear message naming the artifact, before any model call
    (I/O & Edge-Case Matrix: missing manifest/corpus artifact)."""


def _load_json(path: Union[str, Path], label: str) -> dict:
    resolved = Path(path)
    if not resolved.is_file():
        raise ArtifactError(f"{label} artifact not found at '{resolved}'")
    try:
        text = resolved.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise ArtifactError(f"{label} artifact at '{resolved}' could not be read: {e}") from e
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ArtifactError(f"{label} artifact at '{resolved}' is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ArtifactError(
            f"{label} artifact at '{resolved}' must be a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def load_manifest(path: Union[str, Path]) -> dict:
    """Reads the DSL capability manifest JSON (story 1's `ManifestGenerator` output)."""
    return _load_json(path, "Manifest")


def load_corpus(path: Union[str, Path]) -> dict:
    """Reads the example-scene corpus JSON (this story's `CorpusExporter` output)."""
    return _load_json(path, "Corpus")

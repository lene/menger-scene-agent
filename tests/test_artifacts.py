"""Unit tests for adapters/artifacts.py -- the manifest/corpus loader.
Filesystem-only, no network access."""

from __future__ import annotations

import re

import pytest

from adapters.artifacts import ArtifactError, load_corpus, load_manifest
from core.types import EXPECTED_MANIFEST_SCHEMA_VERSION

REFERENCE_MANIFEST = "reference/dsl-manifest.json"
REFERENCE_CORPUS = "reference/dsl-corpus.json"


# --- Edge-Case Matrix: missing manifest/corpus artifact -----------------------------------


def test_load_manifest_fails_fast_with_a_clear_message_when_file_is_missing(tmp_path):
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(ArtifactError, match="Manifest .* not found"):
        load_manifest(missing)


def test_load_corpus_fails_fast_with_a_clear_message_when_file_is_missing(tmp_path):
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(ArtifactError, match="Corpus .* not found"):
        load_corpus(missing)


def test_load_manifest_names_the_missing_path_in_the_error(tmp_path):
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(ArtifactError) as excinfo:
        load_manifest(missing)

    assert str(missing) in str(excinfo.value)


def test_load_manifest_rejects_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")

    with pytest.raises(ArtifactError, match="not valid JSON"):
        load_manifest(bad)


# --- Reading the real reference artifacts (story 2's manifest, this story's corpus) -------


def test_load_manifest_reads_the_real_reference_artifact():
    manifest = load_manifest(REFERENCE_MANIFEST)

    assert manifest["schemaVersion"] == EXPECTED_MANIFEST_SCHEMA_VERSION
    assert "objects" in manifest
    assert manifest["conventions"], "the manifest must carry the DSL's conventions (F28)"


def test_load_corpus_reads_the_real_reference_artifact():
    corpus = load_corpus(REFERENCE_CORPUS)

    assert corpus["schemaVersion"] == "1.0.0"
    assert len(corpus["scenes"]) > 0
    assert all("source" in scene for scene in corpus["scenes"])


def test_the_reference_artifacts_have_no_shadows_setting():
    # menger#196: shadows are always on; the artifacts must come from a menger without the switch.
    manifest = load_manifest(REFERENCE_MANIFEST)
    corpus = load_corpus(REFERENCE_CORPUS)

    field_names = [f["name"] for f in manifest["renderSettings"]["fields"]]
    assert "shadows" not in field_names
    assert not [s["name"] for s in corpus["scenes"] if re.search(r"\bshadows\s*=", s["source"])]

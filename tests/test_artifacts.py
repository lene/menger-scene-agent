"""Unit tests for adapters/artifacts.py -- the manifest/corpus loader.
Filesystem-only, no network access."""

from __future__ import annotations

import pytest

from adapters.artifacts import ArtifactError, load_corpus, load_manifest

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

    assert manifest["schemaVersion"] == "1.0.0"
    assert "objects" in manifest


def test_load_corpus_reads_the_real_reference_artifact():
    corpus = load_corpus(REFERENCE_CORPUS)

    assert corpus["schemaVersion"] == "1.0.0"
    assert len(corpus["scenes"]) > 0
    assert all("source" in scene for scene in corpus["scenes"])
